/*--------------------------------------------------------------------
Copyright (c) 2026 Stefan Teleman <stefan.teleman@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Part of a fork of msieve (public domain, by Jason Papadopoulos).
--------------------------------------------------------------------*/

/* Root optimization with optimize_basic() on the GPU (see
   root_sieve_gpu.h).

   The engine has MSIEVE_ROOTOPT_THREADS threads (default: online
   CPUs), each with MSIEVE_ROOTOPT_SLOTS coroutine slots (default 256).
   A slot holds one polynomial's root optimization: its own
   poly_rootopt_t (about 200 KB) and a 1 MB stack, set up on first use.
   The slots of a thread are in two groups with a stream each. A thread
   loops: fill free slots from the job queue; for each group, collect
   its batch, run its runnable coroutines until each finishes or asks
   for optimize_basic() calls, and send those calls to the GPU as one
   batch, so one group's batch runs while the other group computes.
   All threads share one CUDA context.

   The coroutines run the batched root sieve (root_sieve_batch.c,
   MSIEVE_ROOTOPT_BATCHED=0 for the original): about 300 calls per
   request instead of one, and a few hundred requests per polynomial
   instead of 47,000. Its line sieves are requests too, done by ls_sieve
   (MSIEVE_GPU_LINESIEVE=0: on the CPU). A thread does an adaptive share
   of its batch's optimize_basic() calls itself while the GPU runs the
   rest (MSIEVE_ROOTOPT_CPU_SHARE), so a GPU busy with stage 1, or slow
   at double precision, does not hold the threads up.

   The last step, optimize_final_batch(), sends the root properties of
   a heap's rotations (alpha_<bits>, CGBN) and the Murphy E optimizations
   of its candidates (mf_opt_<degree>, on its own polled stream; by
   default only on GPUs with fast double precision).

   Measured (RTX PRO 4500, i9-13900K, the 6,273 polynomials of one
   155-digit leading coefficient, default E bound): 102.6 s with the
   root properties on the GPU, 107.0 s without them, against
   195.6 s for the original code on 32 CPU threads, the same best E and
   all the original's saved polynomials; identical results here and on
   the CPU with MSIEVE_ROOTOPT_BATCHED=1. With the original root sieve,
   64 of them took 449.6 s against 11.6 s: one minimize() takes ~5 ms on
   a GPU thread, and the original makes them one after the other. */

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "stage2.h"
#include "root_sieve_gpu.h"
#include <cuda_xface.h>
#include <stage1_core_gpu/sizeopt_core.h>

/* in stage1_sizeopt_gpu.c (whose header brings in stage 1's names) */
const char * so_ptx_name(uint32 compute_major, uint32 compute_minor);

#define RO_STACK_SIZE (1 << 20)
#define RO_THREADS_PER_BLOCK 128
#define RQ_LINES 1			/* kinds of request */
#define RQ_MURPHY 2
#define RQ_ALPHA 3
#define RQ_MURPHY_DEV 4			/* Murphy E of inputs on the GPU */
#define RQ_SETUP 5			/* the root sieve's setup */
#define RQ_ROOTS 6			/* its roots to the host */
#define RQ_LAT_XY 7			/* lattice choice: xy hits */
#define RQ_LAT_X 8			/* lattice choice: x-lines */
#define RQ_FINAL 9			/* kept polynomials, exactly */
#define RO_INIT_FINAL RF_MAX_JOBS	/* of them per group batch, at most */
#define RO_INIT_LT 32			/* lattice requests per group batch,
					   at most */
#define RO_HEAP ROOT_HEAP_SIZE		/* rotations per root heap */
#define RO_RA_THREADS 128		/* RA_THREADS in sizeopt_core.cu */
#define RO_MF_THREADS 128		/* MF_WARPS * 32 in sizeopt_core.cu */
#define RO_INIT_CALLS (64 * 1024)	/* first sizes of a group's buffers */
#define RO_INIT_LINES 2048		/* lines per group batch, at most:
					   the rest wait a round */
#define RO_INIT_SETUPS 32		/* root sieve setups per group batch,
					   at most: the rest wait a round
					   (no pinned reallocations) */
#define RO_INIT_ROOTS (512 * 1024)
#define RO_LS_THREADS 512		/* LS_THREADS in sizeopt_core.cu */
#define RO_LS_SMEM (LS_MAX_TABLE * 2 + 1536 * 8)	/* LS_SMEM there */

typedef enum {
	RO_FREE,		/* no polynomial */
	RO_NEW,			/* has a job, not started */
	RO_RUNNABLE,		/* started, or its request was answered */
	RO_WAITING,		/* asked for optimize_basic() calls */
	RO_SENT			/* its calls are on the GPU */
} ro_state_t;

typedef struct {
	double norm;
	double alpha;
	mpz_t a[MAX_POLY_DEGREE + 1];
	mpz_t r[2];
} ro_job_t;

typedef struct ro_thread ro_thread_t;

/* A thread's slots are in two groups, each with its own stream and
   batch buffers: while one group's batch is on the GPU, the thread
   runs the other group's coroutines. */
#define RO_GROUPS 2

/* a pinned host array and its device copy */
typedef struct {
	void *h;
	CUdeviceptr d;
	size_t cap;			/* bytes */
} ro_arr_t;

enum { RA_POLY, RA_PRIME, RA_POWER, RA_ROOT, RA_LINE, RA_OUT,
	RA_RUN, RA_CIN, RA_CAND, RA_NORM, RA_LCTR, RA_NUM };
/* RA_CIN: the host's copy of the candidates' polynomials, when their
   optimize_basic() is not done on the GPU (RA_NORM then unused) */

typedef struct {
	CUstream stream;
	ro_arr_t rs[4];			/* setups: polys, jobs, out, roots */
	ro_arr_t lt[6];			/* lattice: xy sets, hits, x reqs,
					   lats, outs; scratch (GPU only) */
	CUdeviceptr lt_scr_x, lt_scr_s;
	uint32_t lt_n[2];		/* xy, x requests in the batch */
	uint32_t lt_roots;		/* roots fetched in the batch */
	uint32_t rs_n[2];		/* at 512 and 1024 bits */
	ro_arr_t ra[3];			/* root properties: polys, rots, out */
	ro_arr_t ra_keep;		/* kept Murphy E inputs */
	CUdeviceptr d_keep_cnt;
	ro_arr_t rf[3];			/* kept polynomials: polys, jobs, out */
	uint32_t rf_n;
	CUdeviceptr ra_dd;		/* rotated polynomials, on the GPU */
	uint32_t ra_dd_cap;
	uint32_t ra_total;		/* kernel argument */
	CUdeviceptr d_ra_ctr;
	uint32_t ra_n[2];		/* rotations at 512 and 1024 bits */
	ro_arr_t ls[RA_NUM];		/* the line sieve batch */
	uint32_t ls_lines;		/* lines in it */
	uint32_t ls_nc;			/* candidate slots (kernel argument) */
	CUdeviceptr d_cin;		/* their optimize_basic() input */
	uint32_t cin_cap;
	uint32_t cap;			/* buffer size, calls */
	uint32_t n;			/* calls in the batch */
	uint32_t ng;			/* of them, on the GPU */
	CUdeviceptr d_in, d_out, d_ctr;
	so_basic_in_t *h_in;		/* pinned */
	so_basic_out_t *h_out;
	uint32_t *h_ctr;
} ro_buf_t;

typedef struct {
	ro_state_t state;
	ucontext_t ctx;
	void *stack;
	poly_rootopt_t ro;
	ro_job_t job;
	/* the pending request: optimize_basic() calls, or line sieves */
	uint32_t req_lines;		/* RQ_LINES or RQ_MURPHY, else 0 */
	const ra_poly_t *ra_poly;	/* root properties */
	const ra_rot_t *ra_rots;
	ra_res_t *ra_out;
	mf_in_t *ra_keep;		/* its kept Murphy E inputs */
	const ra_final_t *rf_jobs;	/* kept polynomials */
	ra_final_out_t *rf_out;
	uint32_t rf_offset;
	uint32_t ra_offset;
	const lt_set_t *lt_set;		/* lattice choice */
	const lt_xreq_t *lt_xreq;
	const lt_lat_t *lt_lats;
	void *lt_out;			/* lt_hit_t[] or lt_xout_t */
	uint32_t lt_offset, lt_lat0;
	bool lt_batched;
	const ra_poly_t *rs_poly;	/* the root sieve's setup */
	ls_root_t *rs_roots;
	rs_setup_out_t *rs_out;
	uint32_t rs_offset;
	bool rs_batched;		/* in this round's setup batch */
	bool roots_resident;		/* its roots are in the thread's
					   root area */
	uint32_t mf_dev_n;		/* its Murphy E inputs on the GPU: */
	uint32_t mf_dev_idx[RO_HEAP];	/* where, in the thread's area */
	uint32_t mf_dev_rot[RO_HEAP];	/* for which rotation */
	const mf_in_t *mf_in;		/* Murphy E optimizations */
	mf_out_t *mf_out;
	uint32_t mf_offset;
	const ls_poly_t *ls_poly;
	const ls_prime_t *ls_primes;
	const ls_power_t *ls_powers;
	const ls_root_t *ls_roots;
	const ls_line_t *ls_lines;
	ls_out_t *ls_out;
	const ls_run_t *ls_runs;
	uint32_t ls_num_runs;
	uint32_t ls_run0;		/* its runs in the batch */
	bool ls_wait;			/* not in this round's batch */
	const ls_cand_t **ls_cand;	/* where to say where its candidates */
	const basic_result_t **ls_norm;	/* and their norms are */
	const so_basic_in_t **ls_cpoly;	/* or, without, their polynomials */
	uint32_t ls_num_roots;
	uint32_t ls_offset[3];		/* its powers, roots, lines in the
					   batch */
	dpoly_t *req_in;
	basic_result_t *req_out;
	uint32_t req_n;
	uint32_t offset;		/* their place in the batch */
	uint32_t group;			/* which half of the thread's slots */
	bool ready;			/* stack and ro set up (on first use) */
	double t_start;			/* when its job was taken */
	dpoly_t one_in;			/* a single optimize_basic() */
	basic_result_t one_out;
} ro_slot_t;

struct ro_thread {
	ro_gpu_t *g;
	pthread_t tid;
	ucontext_t main_ctx;
	ro_slot_t *slots;
	ro_slot_t *current;
	uint32_t num_slots;

	ro_buf_t buf[RO_GROUPS];

	uint64_t batches, requests, polys, line_batches, lines, cpu_calls;
	uint64_t murphy, murphy_cpu, alphas, setups;
	uint32_t max_lines;		/* most lines in one batch */
	uint64_t lt_xys, lt_xs, root_fetches;
	uint64_t finals;		/* kept polynomials made exactly */

	/* Murphy E optimizations: one batch at a time on their own stream,
	   polled, as one takes ~0.25 s (each candidate is a chain of
	   ~200 integrals) and would hold up the groups' batches */
	CUstream mf_stream;
	ro_arr_t mf[3];			/* in, out, counters */
	uint32_t mf_n;			/* in the batch */
	uint32_t mf_ng;			/* of them, on the GPU */
	uint32_t mf_busy;		/* a batch is on the GPU */
	assess_t mf_assess;		/* for the CPU's share */
	CUdeviceptr roots_area;		/* each slot's roots, RS_MAX_ROOTS */
	CUdeviceptr mf_area;		/* Murphy E inputs made on the GPU,
					   RO_HEAP per slot */
	ro_arr_t mf_idx;		/* their indices, for a batch */
	uint32_t mf_nh;			/* batch: inputs from the host */
	uint32_t mf_nd;			/* and from the area */
	uint32_t finished;		/* jobs done since the last fill */
	double job_time;		/* a job's average time, seconds */
	double cpu_share;		/* of each batch's calls, done here */
	double t_cpu, t_gpu, t_idle;	/* seconds running coroutines,
					   waiting for batches, for jobs */
	double t_setup, t_teardown;
};

struct ro_gpu {
	msieve_obj *obj;
	poly_rootopt_t *proto;
	uint32_t degree;

	CUcontext ctx;
	CUmodule module;
	gpu_launch_t basic;
	CUfunction ls_sieve;		/* NULL: line sieve on the CPU */
	CUfunction mf_opt;		/* NULL: Murphy E on the CPU */
	CUfunction alpha[2];		/* 512, 1024 bits; NULL: CPU */
	CUfunction bs_size;		/* NULL: size scores on the CPU */
	CUfunction rs_setup[2];		/* 512, 1024 bits; NULL: CPU */
	CUfunction lt_xy, lt_x;		/* NULL: lattice choice on the CPU */
	CUfunction rf_final;		/* NULL: kept polynomials with GMP */
	CUdeviceptr d_rs_powers;
	CUdeviceptr d_ra_tables;
	CUdeviceptr d_mf_tables;
	uint32_t mf_max_blocks;
	uint32_t cand_gpu;		/* line candidates' optimize_basic()
					   on the GPU too */
	uint32_t bs_max_blocks;
	uint32_t num_sm;
	CUdevice device;
	uint32_t max_blocks;		/* resident blocks of the kernel */
	uint32_t batched;		/* the batched root sieve */
	uint32_t auto_share;		/* adapt the CPU's share of calls */
	double cpu_share;		/* its first value */

	uint32_t num_threads;
	uint32_t slots_per_thread;
	ro_thread_t *threads;

	/* every GPU has its own engine (context, kernels, threads); they
	   take jobs from one queue. A job stays on the GPU whose thread
	   took it: its roots and tables live there */
	struct ro_queue *q;
	ro_gpu_t **lanes;		/* all engines (lanes[0]: this handle) */
	uint32_t num_lanes;
	uint32_t lane;			/* this one */
	uint32_t device_index;		/* its CUDA device */
};

/* jobs waiting for a slot, shared by every GPU's engine */
typedef struct ro_queue {
	pthread_mutex_t lock;
	pthread_cond_t cv_job;
	pthread_cond_t cv_space;
	ro_job_t *queue;
	uint32_t q_cap, q_head, q_count;
	uint32_t active;		/* jobs in the threads' slots */
	uint32_t lane_active[MAX_GPU];	/* of them, on each GPU */
	uint32_t num_lanes;
	uint32_t total_threads;		/* of all engines */
	bool finishing;
} ro_queue_t;

/* the engine thread running on this OS thread */
static _Thread_local ro_thread_t *ro_self;

/*------------------------------------------------------------------------*/
static double
ro_now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint32_t
ro_env(const char *name, uint32_t dflt)
{
	const char *e = getenv(name);
	long v = (e != NULL) ? strtol(e, NULL, 10) : (long)dflt;

	return (uint32_t)MAX(1, MIN(v, 65536));
}

static void
ro_job_init(ro_job_t *j)
{
	for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_init(j->a[i]);
	mpz_init(j->r[0]);
	mpz_init(j->r[1]);
}

static void
ro_job_clear(ro_job_t *j)
{
	for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_clear(j->a[i]);
	mpz_clear(j->r[0]);
	mpz_clear(j->r[1]);
}

static void
ro_job_copy(ro_job_t *dst, const ro_job_t *src, uint32_t degree)
{
	for (uint32_t i = 0; i <= degree; i++)
		mpz_set(dst->a[i], src->a[i]);
	mpz_set(dst->r[0], src->r[0]);
	mpz_set(dst->r[1], src->r[1]);
	dst->norm = src->norm;
	dst->alpha = src->alpha;
}

/*------------------------------------------------------------------------*/
/* inside a coroutine: optimize_basic_batch() lands here */

static void
ro_hook_batch(dpoly_t *in, uint32 n, basic_result_t *out)
{
	ro_thread_t *t = ro_self;
	ro_slot_t *s = t->current;

	if (n == 0)
		return;

	if (in[0].degree != t->g->degree) {
		/* not what the kernel was built for: do it here */
		optimize_basic_hook_t h = optimize_basic_hook;

		optimize_basic_hook = NULL;
		for (uint32 i = 0; i < n; i++)
			out[i].norm = optimize_basic(in + i, &out[i].skewness,
						&out[i].translation);
		optimize_basic_hook = h;
		return;
	}

	s->req_lines = 0;
	s->req_in = in;
	s->req_out = out;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &t->main_ctx);
	/* resumed with the answers in out[] */
}

/* inside a coroutine: rs_line_flush() lands here */

static void
ro_hook_lines(const ls_poly_t *poly, const ls_prime_t *primes,
		const ls_power_t *powers, const ls_root_t *roots,
		const ls_line_t *lines, uint32 n, ls_out_t *out,
		const ls_run_t *runs, uint32 num_runs,
		const ls_cand_t **cand, const basic_result_t **norm,
		const so_basic_in_t **cand_poly)
{
	ro_slot_t *s = ro_self->current;
	uint32_t num_roots = 0;

	if (n == 0)
		return;
	for (uint32_t j = 0; j < poly->num_powers; j++)
		num_roots = MAX(num_roots,
				powers[j].root0 + powers[j].num_roots);

	s->req_lines = RQ_LINES;
	s->ls_poly = poly;
	s->ls_primes = primes;
	s->ls_powers = powers;
	s->ls_roots = roots;
	s->ls_lines = lines;
	s->ls_out = out;
	s->ls_runs = runs;
	s->ls_num_runs = num_runs;
	s->ls_cand = cand;
	s->ls_norm = norm;
	s->ls_cpoly = cand_poly;
	s->ls_num_roots = num_roots;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

/* inside a coroutine: the kept polynomials land here */

static void
ro_hook_final(const ra_poly_t *pl, const ra_final_t *jobs, uint32 n,
		ra_final_out_t *out)
{
	ro_slot_t *s = ro_self->current;

	s->req_lines = RQ_FINAL;
	s->ra_poly = pl;
	s->rf_jobs = jobs;
	s->rf_out = out;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

/* inside a coroutine: the lattice choice and the roots' fetch land
   here */

static void
ro_hook_lat_xy(const lt_set_t *set, lt_hit_t *hits)
{
	ro_slot_t *s = ro_self->current;

	s->req_lines = RQ_LAT_XY;
	s->lt_set = set;
	s->lt_out = hits;
	s->req_n = 1;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

static void
ro_hook_lat_x(const lt_xreq_t *req, const lt_lat_t *lats, lt_xout_t *out)
{
	ro_slot_t *s = ro_self->current;

	s->req_lines = RQ_LAT_X;
	s->lt_xreq = req;
	s->lt_lats = lats;
	s->lt_out = out;
	s->req_n = 1;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

static void
ro_hook_roots(ls_root_t *roots, uint32 n)
{
	ro_slot_t *s = ro_self->current;

	s->req_lines = RQ_ROOTS;
	s->rs_roots = roots;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

/* inside a coroutine: the root sieve's setup lands here */

static void
ro_hook_setup(const ra_poly_t *pl, ls_root_t *roots, rs_setup_out_t *out)
{
	ro_slot_t *s = ro_self->current;

	s->req_lines = RQ_SETUP;
	s->rs_poly = pl;
	s->rs_roots = roots;
	s->rs_out = out;
	s->req_n = 1;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

/* inside a coroutine: the root properties of a heap land here */

static void
ro_hook_alpha(const ra_poly_t *pl, const ra_rot_t *rots, uint32 n,
		ra_res_t *alpha, mf_in_t *keep)
{
	ro_slot_t *s = ro_self->current;

	s->ra_keep = keep;
	s->req_lines = RQ_ALPHA;
	s->ra_poly = pl;
	s->ra_rots = rots;
	s->ra_out = alpha;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);
}

/* inside a coroutine: optimize_final_batch() lands here */

static void
ro_hook_murphy(const mf_in_t *in, uint32 n, mf_out_t *out)
{
	ro_slot_t *s = ro_self->current;

	if (n == 0)
		return;
	if (in[0].degree != ro_self->g->degree) {
		/* not what the kernel was built for */
		murphy_batch_hook = NULL;
		optimize_final_batch_cpu(NULL, in, n, out);
		murphy_batch_hook = ro_hook_murphy;
		return;
	}
	s->req_lines = RQ_MURPHY;
	s->mf_in = in;
	s->mf_out = out;
	s->req_n = n;
	s->state = RO_WAITING;
	swapcontext(&s->ctx, &ro_self->main_ctx);

	/* the host counted the real roots */
	for (uint32 i = 0; i < n; i++)
		out[i].num_real_roots = in[i].num_real_roots;
}

/* and a single optimize_basic(), from code that is not batched */

static double
ro_hook(dpoly_t *apoly, double *best_skewness, double *best_translation)
{
	ro_slot_t *s = ro_self->current;

	s->one_in = *apoly;
	ro_hook_batch(&s->one_in, 1, &s->one_out);
	*best_skewness = s->one_out.skewness;
	*best_translation = s->one_out.translation;
	return s->one_out.norm;
}

static void
ro_coro_main(void)
{
	ro_thread_t *t = ro_self;
	ro_slot_t *s = t->current;

	poly_rootopt_run(&s->ro, s->job.a, s->job.r,
			s->job.norm, s->job.alpha);
	{
		double d = ro_now() - s->t_start;

		t->job_time = (t->job_time > 0) ?
				0.9 * t->job_time + 0.1 * d : d;
	}
	s->state = RO_FREE;
	t->polys++;
	t->finished++;
	/* returning resumes uc_link, the scheduler */
}

/*------------------------------------------------------------------------*/
/* a new coroutine for the job in slot s (its own function: getcontext()
   would otherwise make the compiler wary of the caller's variables) */

static void __attribute__((noinline))
ro_start(ro_thread_t *t, ro_slot_t *s)
{
	if (!s->ready) {
		poly_rootopt_t *p = t->g->proto;
		poly_rootopt_t *r = &s->ro;

		s->stack = mmap(NULL, RO_STACK_SIZE, PROT_READ | PROT_WRITE,
				MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
				-1, 0);
		if (s->stack == MAP_FAILED) {
			printf("error: cannot map a coroutine stack\n");
			exit(-1);
		}
		poly_rootopt_init(r, p->obj, p->callback, p->callback_data);
		mpz_set(r->gmp_N, p->gmp_N);
		r->degree = p->degree;
		r->murphy_p_bound = p->murphy_p_bound;
		r->max_sizeopt_norm = p->max_sizeopt_norm;
		r->min_e = p->min_e;
		r->min_e_bernstein = p->min_e_bernstein;
		s->ready = true;
	}

	getcontext(&s->ctx);
	s->ctx.uc_stack.ss_sp = s->stack;
	s->ctx.uc_stack.ss_size = RO_STACK_SIZE;
	s->ctx.uc_link = &t->main_ctx;
	makecontext(&s->ctx, ro_coro_main, 0);
	s->roots_resident = false;	/* a new polynomial */
	s->state = RO_RUNNABLE;
}

/*------------------------------------------------------------------------*/
/* take queued jobs into free slots; returns the number of busy slots */

static uint32_t
ro_fill(ro_thread_t *t, bool *done)
{
	ro_gpu_t *g = t->g;
	ro_queue_t *q = g->q;
	uint32_t busy = 0;

	pthread_mutex_lock(&q->lock);
	q->active -= t->finished;
	q->lane_active[g->lane] -= t->finished;
	t->finished = 0;
	for (;;) {
		/* at most a fair share of all the jobs, queued and running:
		   first each GPU's, then each of its threads', so that the
		   first threads to look do not take them all */
		uint32_t mine = 0, share, take, lane_share, lane_take;
		uint32_t all = q->active + q->q_count;

		for (uint32_t i = 0; i < t->num_slots; i++)
			mine += (t->slots[i].state != RO_FREE);
		lane_share = (all + q->num_lanes - 1) / q->num_lanes;
		lane_take = (lane_share > q->lane_active[g->lane]) ?
				lane_share - q->lane_active[g->lane] : 0;
		share = (lane_share + g->num_threads - 1) / g->num_threads;
		take = (share > mine) ? share - mine : 0;
		take = MIN(take, lane_take);

		/* the time limit: past it the queued jobs are dropped; near
		   it only jobs that can end before it start (a job lasts
		   its slot's whole batch, so this thread's average of them
		   is the measure). A degree-5 c110 ran 306 s past its limit
		   finishing the jobs it had started */
		if (rootopt_wall_end > 0) {
			struct timespec rt;
			double now;

			/* the limit is wall-clock time (gettimeofday) */
			clock_gettime(CLOCK_REALTIME, &rt);
			now = (double)rt.tv_sec + 1e-9 * (double)rt.tv_nsec;

			if (now > rootopt_wall_end) {
				__atomic_fetch_add(&rootopt_dropped,
						q->q_count, __ATOMIC_RELAXED);
				q->q_head = (q->q_head + q->q_count) %
						q->q_cap;
				q->q_count = 0;
				take = 0;
			}
			else if (t->job_time > 0 &&
				 now + t->job_time > rootopt_wall_end) {
				take = 0;
			}
		}

		busy = 0;
		for (uint32_t i = 0; i < t->num_slots; i++) {
			ro_slot_t *s = t->slots + i;

			if (s->state == RO_FREE && q->q_count > 0 &&
			    take > 0) {
				take--;
				ro_job_copy(&s->job, q->queue + q->q_head,
						g->degree);
				q->q_head = (q->q_head + 1) % q->q_cap;
				q->q_count--;
				q->active++;
				q->lane_active[g->lane]++;
				s->state = RO_NEW;	/* started below */
				s->t_start = ro_now();
			}
			if (s->state != RO_FREE)
				busy++;
		}
		pthread_cond_broadcast(&q->cv_space);

		if (busy > 0)
			break;
		if (q->finishing && q->q_count == 0) {
			*done = true;
			break;
		}
		if (rootopt_wall_end > 0) {
			/* look again within a second: the time limit can
			   pass while nothing signals (a producer waiting
			   for space while no thread may take a job) */
			struct timespec ts;

			clock_gettime(CLOCK_REALTIME, &ts);
			ts.tv_sec += 1;
			pthread_cond_timedwait(&q->cv_job, &q->lock, &ts);
		}
		else {
			pthread_cond_wait(&q->cv_job, &q->lock);
		}
	}
	pthread_mutex_unlock(&q->lock);

	/* outside the lock: a slot's first use sets it up */
	for (uint32_t i = 0; i < t->num_slots; i++) {
		if (t->slots[i].state == RO_NEW)
			ro_start(t, t->slots + i);
	}
	return busy;
}

/* make room for n calls in a batch buffer */

static void
ro_reserve(ro_buf_t *b, uint32_t n)
{
	if (n <= b->cap)
		return;
	if (b->cap > 0) {
		CUDA_TRY(cuMemFreeHost(b->h_in))
		CUDA_TRY(cuMemFreeHost(b->h_out))
		CUDA_TRY(cuMemFree(b->d_in))
		CUDA_TRY(cuMemFree(b->d_out))
	}
	b->cap = MAX(n, 2 * b->cap);
	CUDA_TRY(cuMemAlloc(&b->d_in, b->cap * sizeof(so_basic_in_t)))
	CUDA_TRY(cuMemAlloc(&b->d_out, b->cap * sizeof(so_basic_out_t)))
	CUDA_TRY(cuMemAllocHost((void **)&b->h_in,
				b->cap * sizeof(so_basic_in_t)))
	CUDA_TRY(cuMemAllocHost((void **)&b->h_out,
				b->cap * sizeof(so_basic_out_t)))
}

static void
ro_arr_reserve(ro_arr_t *a, size_t bytes)
{
	if (bytes <= a->cap)
		return;
	if (a->cap > 0) {
		CUDA_TRY(cuMemFreeHost(a->h))
		CUDA_TRY(cuMemFree(a->d))
	}
	a->cap = MAX(bytes, 2 * a->cap);
	CUDA_TRY(cuMemAllocHost(&a->h, a->cap))
	CUDA_TRY(cuMemAlloc(&a->d, a->cap))
}

static void
ro_arr_free(ro_arr_t *a)
{
	if (a->cap > 0) {
		CUDA_TRY(cuMemFreeHost(a->h))
		CUDA_TRY(cuMemFree(a->d))
	}
	a->cap = 0;
}

/* the line sieves of group grp's waiting coroutines: one ls_sieve
   block per line, every polynomial's data in one set of arrays */

static void
ro_launch_lines(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t np = 0, nw = 0, nr = 0, nl = 0, nruns = 0;

	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != RQ_LINES)
			continue;
		if (nl + s->req_n > RO_INIT_LINES) {
			s->ls_wait = true;	/* next round */
			continue;
		}
		s->ls_wait = false;
		s->ls_offset[0] = nw;
		s->ls_offset[1] = nr;
		s->ls_offset[2] = nl;
		s->ls_run0 = nruns;
		nruns += s->ls_num_runs;
		np++;
		nw += s->ls_poly->num_powers;
		nr += s->ls_num_roots;
		nl += s->req_n;
	}
	b->ls_lines = nl;
	if (nl == 0)
		return;

	ro_arr_reserve(b->ls + RA_POLY, np * sizeof(ls_poly_t));
	ro_arr_reserve(b->ls + RA_PRIME, np * LS_NUM_PRIMES *
					sizeof(ls_prime_t));
	ro_arr_reserve(b->ls + RA_POWER, nw * sizeof(ls_power_t));
	ro_arr_reserve(b->ls + RA_LINE, nl * sizeof(ls_line_t));
	ro_arr_reserve(b->ls + RA_OUT, nl * sizeof(ls_out_t));
	ro_arr_reserve(b->ls + RA_RUN, MAX(nruns, 1) * sizeof(ls_run_t));
	if (nl * LS_TOP > b->cin_cap) {
		/* the candidates' optimize_basic() input: GPU only */
		if (b->cin_cap > 0)
			CUDA_TRY(cuMemFree(b->d_cin))
		b->cin_cap = MAX(nl * LS_TOP, 2 * b->cin_cap);
		CUDA_TRY(cuMemAlloc(&b->d_cin, b->cin_cap *
					sizeof(so_basic_in_t)))
	}
	t->max_lines = MAX(t->max_lines, nl);
	ro_arr_reserve(b->ls + RA_CAND, nl * LS_TOP * sizeof(ls_cand_t));
	ro_arr_reserve(b->ls + RA_NORM, nl * LS_TOP * sizeof(so_basic_out_t));
	ro_arr_reserve(b->ls + RA_LCTR, SO_BASIC_NUM_STATS * sizeof(uint32_t));

	{
		ls_poly_t *polys = (ls_poly_t *)b->ls[RA_POLY].h;
		ls_prime_t *primes = (ls_prime_t *)b->ls[RA_PRIME].h;
		ls_power_t *powers = (ls_power_t *)b->ls[RA_POWER].h;
		ls_line_t *lines = (ls_line_t *)b->ls[RA_LINE].h;
		uint32_t k = 0;

		for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
			ro_slot_t *s = t->slots + i;
			ls_poly_t *pl = polys + k;

			if (s->state != RO_WAITING || s->req_lines != RQ_LINES ||
			    s->ls_wait)
				continue;
			*pl = *s->ls_poly;
			pl->prime0 = k * LS_NUM_PRIMES;
			pl->power0 = s->ls_offset[0];
			memcpy(primes + pl->prime0, s->ls_primes,
				LS_NUM_PRIMES * sizeof(ls_prime_t));
			for (uint32_t j = 0; j < pl->num_powers; j++) {
				ls_power_t *w = powers + pl->power0 + j;

				*w = s->ls_powers[j];
				w->root0 += i * RS_MAX_ROOTS;
			}
			/* the slot's roots are in the root area: put there
			   by its setup on the GPU, or else now */
			if (!s->roots_resident) {
				CUDA_TRY(cuMemcpyHtoDAsync(t->roots_area +
					(size_t)i * RS_MAX_ROOTS *
						sizeof(ls_root_t),
					s->ls_roots, MIN(s->ls_num_roots,
						RS_MAX_ROOTS) *
						sizeof(ls_root_t), b->stream))
				s->roots_resident = true;
			}
			memcpy((ls_run_t *)b->ls[RA_RUN].h + s->ls_run0,
				s->ls_runs, s->ls_num_runs * sizeof(ls_run_t));
			for (uint32_t j = 0; j < s->req_n; j++) {
				ls_line_t *l = lines + s->ls_offset[2] + j;

				*l = s->ls_lines[j];
				l->poly = k;
				l->run += s->ls_run0;
			}
			s->state = RO_SENT;
			k++;
		}
	}

	CUDA_TRY(cuMemcpyHtoDAsync(b->ls[RA_POLY].d, b->ls[RA_POLY].h,
			np * sizeof(ls_poly_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->ls[RA_PRIME].d, b->ls[RA_PRIME].h,
			np * LS_NUM_PRIMES * sizeof(ls_prime_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->ls[RA_POWER].d, b->ls[RA_POWER].h,
			nw * sizeof(ls_power_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->ls[RA_LINE].d, b->ls[RA_LINE].h,
			nl * sizeof(ls_line_t), b->stream))
	if (nruns > 0) {
		CUDA_TRY(cuMemcpyHtoDAsync(b->ls[RA_RUN].d, b->ls[RA_RUN].h,
				nruns * sizeof(ls_run_t), b->stream))
	}
	{
		void *args[9] = { &b->ls[RA_POLY].d, &b->ls[RA_PRIME].d,
				&b->ls[RA_POWER].d, &t->roots_area,
				&b->ls[RA_LINE].d, &b->ls[RA_OUT].d,
				&b->ls[RA_RUN].d, &b->d_cin,
				&b->ls[RA_CAND].d };

		CUDA_TRY(cuLaunchKernel(g->ls_sieve, nl, 1, 1,
				RO_LS_THREADS, 1, 1, RO_LS_SMEM, b->stream,
				args, NULL))
	}
	if (g->cand_gpu) {
		/* the candidates' optimize_basic(), on the GPU's own
		   candidates (unused slots are skipped) */
		uint32_t nc = nl * LS_TOP;
		uint32_t threads = MIN(RO_THREADS_PER_BLOCK,
				(uint32_t)g->basic.threads_per_block);
		uint32_t blocks = MIN((nc + threads - 1) / threads,
					g->max_blocks);
		void *args[4] = { &b->d_cin, &b->ls[RA_NORM].d,
				&b->ls_nc, &b->ls[RA_LCTR].d };

		b->ls_nc = nc;
		CUDA_TRY(cuMemsetD8Async(b->ls[RA_LCTR].d, 0,
				SO_BASIC_NUM_STATS * sizeof(uint32_t),
				b->stream))
		CUDA_TRY(cuLaunchKernel(g->basic.kernel_func, blocks, 1, 1,
				threads, 1, 1, 0, b->stream, args, NULL))
	}
	CUDA_TRY(cuMemcpyDtoHAsync(b->ls[RA_OUT].h, b->ls[RA_OUT].d,
			nl * sizeof(ls_out_t), b->stream))
	CUDA_TRY(cuMemcpyDtoHAsync(b->ls[RA_CAND].h, b->ls[RA_CAND].d,
			nl * LS_TOP * sizeof(ls_cand_t), b->stream))
	if (g->cand_gpu) {
		CUDA_TRY(cuMemcpyDtoHAsync(b->ls[RA_NORM].h, b->ls[RA_NORM].d,
				nl * LS_TOP * sizeof(so_basic_out_t),
				b->stream))
		CUDA_TRY(cuMemcpyDtoHAsync(b->ls[RA_LCTR].h,
				b->ls[RA_LCTR].d,
				SO_BASIC_NUM_STATS * sizeof(uint32_t),
				b->stream))
	}
	else {
		/* the polynomials back, for the batches the CPU shares */
		CUDA_TRY(cuMemcpyDtoHAsync(b->ls[RA_CIN].h, b->d_cin,
				nl * LS_TOP * sizeof(so_basic_in_t),
				b->stream))
	}
	t->line_batches++;
	t->lines += nl;
}

/* hand a finished Murphy E batch to its coroutines; with wait, wait
   for it */

static void
ro_murphy_poll(ro_thread_t *t, bool wait)
{
	if (!t->mf_busy)
		return;
	if (wait) {
		CUDA_TRY(cuStreamSynchronize(t->mf_stream))
	}
	else {
		CUresult r = cuStreamQuery(t->mf_stream);

		if (r == CUDA_ERROR_NOT_READY)
			return;
		CUDA_TRY(r)
	}

	for (uint32_t i = 0; i < t->num_slots; i++) {
		ro_slot_t *s = t->slots + i;
		const mf_out_t *o = (mf_out_t *)t->mf[1].h + s->mf_offset;

		if (s->state != RO_SENT)
			continue;
		if (s->req_lines == RQ_MURPHY) {
			memcpy(s->mf_out, o, s->req_n * sizeof(mf_out_t));
			s->state = RO_RUNNABLE;
		}
		else if (s->req_lines == RQ_MURPHY_DEV) {
			for (uint32_t k = 0; k < s->mf_dev_n; k++) {
				ra_res_t *r = s->ra_out + s->mf_dev_rot[k];

				r->score = o[k].score;
				r->skewness = o[k].skewness;
				r->translation = o[k].translation;
				r->flags |= RA_MURPHY;
			}
			s->state = RO_RUNNABLE;
		}
	}
	t->mf_busy = 0;
}

/* the Murphy E optimizations whose inputs bs_size_<deg> left in the
   thread's area: an index list up, the results (from out offset nh)
   down, nothing else crosses */

static void
ro_murphy_launch_dev(ro_thread_t *t, uint32_t nh, uint32_t nd)
{
	ro_gpu_t *g = t->g;
	uint32_t *idx;
	uint32_t k = 0;
	uint32_t blocks = MIN((nd + RO_MF_THREADS / 32 - 1) /
				(RO_MF_THREADS / 32), g->mf_max_blocks);
	CUdeviceptr out = t->mf[1].d + nh * sizeof(mf_out_t);
	void *args[6] = { &g->d_mf_tables, &t->mf_area, &t->mf_idx.d,
			&out, &t->mf_nd, &t->mf[2].d };

	ro_arr_reserve(&t->mf_idx, nd * sizeof(uint32_t));
	idx = (uint32_t *)t->mf_idx.h;
	for (uint32_t i = 0; i < t->num_slots; i++) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != RQ_MURPHY_DEV)
			continue;
		memcpy(idx + k, s->mf_dev_idx, s->mf_dev_n * sizeof(uint32_t));
		k += s->mf_dev_n;
		s->state = RO_SENT;
	}
	t->mf_nd = nd;

	CUDA_TRY(cuMemcpyHtoDAsync(t->mf_idx.d, t->mf_idx.h,
			nd * sizeof(uint32_t), t->mf_stream))
	CUDA_TRY(cuMemsetD8Async(t->mf[2].d, 0, 2 * sizeof(uint32_t),
			t->mf_stream))
	CUDA_TRY(cuLaunchKernel(g->mf_opt, blocks, 1, 1, RO_MF_THREADS, 1, 1,
			0, t->mf_stream, args, NULL))
	CUDA_TRY(cuMemcpyDtoHAsync((char *)t->mf[1].h +
			nh * sizeof(mf_out_t), out,
			nd * sizeof(mf_out_t), t->mf_stream))
	t->murphy += nd;
}

/* the waiting Murphy E optimizations of all the thread's coroutines,
   if the last batch is done: the GPU takes one warp per candidate, and
   the thread does the last cpu_share of them itself */

static void
ro_murphy_launch(ro_thread_t *t)
{
	ro_gpu_t *g = t->g;
	uint32_t n = 0;

	uint32_t nd = 0;

	if (t->mf_busy)
		return;
	for (uint32_t i = 0; i < t->num_slots; i++) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != RQ_MURPHY)
			continue;
		s->mf_offset = n;
		n += s->req_n;
	}
	t->mf_nh = n;
	for (uint32_t i = 0; i < t->num_slots; i++) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != RQ_MURPHY_DEV)
			continue;
		s->mf_offset = n + nd;
		nd += s->mf_dev_n;
	}
	if (n + nd == 0)
		return;
	ro_arr_reserve(t->mf + 1, (n + nd) * sizeof(mf_out_t));
	if (nd > 0)
		ro_murphy_launch_dev(t, n, nd);
	if (n == 0) {
		t->mf_busy = 1;
		return;
	}

	ro_arr_reserve(t->mf + 0, n * sizeof(mf_in_t));
	ro_arr_reserve(t->mf + 1, n * sizeof(mf_out_t));
	for (uint32_t i = 0; i < t->num_slots; i++) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != RQ_MURPHY)
			continue;
		memcpy((mf_in_t *)t->mf[0].h + s->mf_offset, s->mf_in,
				s->req_n * sizeof(mf_in_t));
		s->state = RO_SENT;
	}
	t->mf_n = n;
	t->mf_ng = n - (uint32_t)(t->cpu_share * n);

	if (t->mf_ng > 0) {
		uint32_t blocks = MIN((t->mf_ng + RO_MF_THREADS / 32 - 1) /
				(RO_MF_THREADS / 32), g->mf_max_blocks);
		CUdeviceptr none = 0;
		void *args[6] = { &g->d_mf_tables, &t->mf[0].d, &none,
				&t->mf[1].d, &t->mf_ng, &t->mf[2].d };

		CUDA_TRY(cuMemcpyHtoDAsync(t->mf[0].d, t->mf[0].h,
				t->mf_ng * sizeof(mf_in_t), t->mf_stream))
		CUDA_TRY(cuMemsetD8Async(t->mf[2].d, 0, 2 * sizeof(uint32_t),
				t->mf_stream))
		CUDA_TRY(cuLaunchKernel(g->mf_opt, blocks, 1, 1,
				RO_MF_THREADS, 1, 1, 0, t->mf_stream, args,
				NULL))
		CUDA_TRY(cuMemcpyDtoHAsync(t->mf[1].h, t->mf[1].d,
				t->mf_ng * sizeof(mf_out_t), t->mf_stream))
	}
	t->mf_busy = 1;

	/* the CPU's share, while the GPU works */
	if (t->mf_ng < n) {
		murphy_batch_hook = NULL;
		optimize_final_batch_cpu(&t->mf_assess,
				(mf_in_t *)t->mf[0].h + t->mf_ng,
				n - t->mf_ng,
				(mf_out_t *)t->mf[1].h + t->mf_ng);
		murphy_batch_hook = ro_hook_murphy;
		t->murphy_cpu += n - t->mf_ng;
	}
	t->murphy += n;
	if (t->mf_ng == 0 && nd == 0)
		ro_murphy_poll(t, true);	/* all done here */
}

/* the root sieve setups of group grp's waiting coroutines: one block
   per polynomial; the roots go to the slots' parts of the root area */

static void
ro_launch_setup(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t n = 0;

	b->rs_n[0] = b->rs_n[1] = 0;
	for (uint32_t w = 0; w < 2; w++) {
		for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
			ro_slot_t *s = t->slots + i;

			if (s->state != RO_WAITING ||
			    s->req_lines != RQ_SETUP ||
			    (s->rs_poly->pad == 1024) != w ||
			    n == RO_INIT_SETUPS)
				continue;
			s->rs_offset = n++;
			s->rs_batched = true;
			b->rs_n[w]++;
		}
	}
	if (n == 0)
		return;

	ro_arr_reserve(b->rs + 0, n * sizeof(ra_poly_t));
	ro_arr_reserve(b->rs + 1, n * sizeof(rs_job_t));
	ro_arr_reserve(b->rs + 2, n * sizeof(rs_setup_out_t));
	ro_arr_reserve(b->rs + 3, (size_t)n * RS_MAX_ROOTS *
				sizeof(ls_root_t));
	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;
		rs_job_t *j = (rs_job_t *)b->rs[1].h + s->rs_offset;

		if (s->state != RO_WAITING || s->req_lines != RQ_SETUP ||
		    !s->rs_batched)
			continue;
		s->rs_batched = false;
		((ra_poly_t *)b->rs[0].h)[s->rs_offset] = *s->rs_poly;
		j->poly = s->rs_offset;
		j->base = i * RS_MAX_ROOTS;
		s->state = RO_SENT;
	}

	CUDA_TRY(cuMemcpyHtoDAsync(b->rs[0].d, b->rs[0].h,
			n * sizeof(ra_poly_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->rs[1].d, b->rs[1].h,
			n * sizeof(rs_job_t), b->stream))
	for (uint32_t w = 0, off = 0; w < 2; w++) {
		CUdeviceptr jobs = b->rs[1].d + off * sizeof(rs_job_t);
		CUdeviceptr out = b->rs[2].d + off * sizeof(rs_setup_out_t);
		void *args[5] = { &g->d_rs_powers, &b->rs[0].d, &jobs,
				&t->roots_area, &out };

		if (b->rs_n[w] > 0) {
			CUDA_TRY(cuLaunchKernel(g->rs_setup[w], b->rs_n[w],
					1, 1, RS_THREADS, 1, 1, 0, b->stream,
					args, NULL))
		}
		off += b->rs_n[w];
	}
	CUDA_TRY(cuMemcpyDtoHAsync(b->rs[2].h, b->rs[2].d,
			n * sizeof(rs_setup_out_t), b->stream))

	/* the host's copy of the roots, once per polynomial */
	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_SENT || s->req_lines != RQ_SETUP ||
		    s->rs_poly->pad2[0])
			continue;
		CUDA_TRY(cuMemcpyDtoHAsync((ls_root_t *)b->rs[3].h +
				(size_t)s->rs_offset * RS_MAX_ROOTS,
				t->roots_area + (size_t)i * RS_MAX_ROOTS *
					sizeof(ls_root_t),
				RS_MAX_ROOTS * sizeof(ls_root_t), b->stream))
	}
	t->setups += n;
}

/* the lattice choices of group grp's waiting coroutines (at most
   RO_INIT_LT of each kind; the rest wait a round), and roots fetched
   for the host */

static void
ro_launch_lattice(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t nxy = 0, nx = 0, nlat = 0;

	b->lt_n[0] = b->lt_n[1] = 0;
	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		s->lt_batched = false;
		if (s->state != RO_WAITING)
			continue;
		if (s->req_lines == RQ_LAT_XY && nxy < RO_INIT_LT) {
			lt_set_t *st = (lt_set_t *)b->lt[0].h + nxy;

			*st = *s->lt_set;
			st->area_base = i * RS_MAX_ROOTS;
			s->lt_offset = nxy++;
			s->lt_batched = true;
		}
		else if (s->req_lines == RQ_LAT_X && nx < RO_INIT_LT &&
			 nlat + s->lt_xreq->num_lattices <= LT_MAX_LATS) {
			lt_xreq_t *rq = (lt_xreq_t *)b->lt[2].h + nx;

			*rq = *s->lt_xreq;
			rq->set.area_base = i * RS_MAX_ROOTS;
			rq->lat0 = nlat;
			memcpy((lt_lat_t *)b->lt[3].h + nlat, s->lt_lats,
				rq->num_lattices * sizeof(lt_lat_t));
			nlat += rq->num_lattices;
			s->lt_offset = nx++;
			s->lt_batched = true;
		}
		else if (s->req_lines == RQ_ROOTS) {
			/* the slot's roots, straight to its buffer */
			t->root_fetches++;
			CUDA_TRY(cuMemcpyDtoHAsync(s->rs_roots,
				t->roots_area + (size_t)i * RS_MAX_ROOTS *
					sizeof(ls_root_t),
				MIN(s->req_n, RS_MAX_ROOTS) *
					sizeof(ls_root_t), b->stream))
			s->lt_batched = true;
		}
		else {
			continue;
		}
		s->state = RO_SENT;
	}
	b->lt_n[0] = nxy;
	b->lt_n[1] = nx;
	t->lt_xys += nxy;
	t->lt_xs += nx;
	b->lt_roots = 0;
	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		if (t->slots[i].state == RO_SENT &&
		    t->slots[i].req_lines == RQ_ROOTS)
			b->lt_roots = 1;
	}

	if (nxy > 0) {
		void *args[3] = { &b->lt[0].d, &t->roots_area, &b->lt[1].d };

		CUDA_TRY(cuMemcpyHtoDAsync(b->lt[0].d, b->lt[0].h,
				nxy * sizeof(lt_set_t), b->stream))
		CUDA_TRY(cuLaunchKernel(g->lt_xy, nxy, 1, 1, 32, 1, 1, 0,
				b->stream, args, NULL))
		CUDA_TRY(cuMemcpyDtoHAsync(b->lt[1].h, b->lt[1].d,
				nxy * LT_MAX_PRIMES * sizeof(lt_hit_t),
				b->stream))
	}
	if (nx > 0) {
		void *args[6] = { &b->lt[2].d, &b->lt[3].d, &t->roots_area,
				&b->lt_scr_x, &b->lt_scr_s, &b->lt[4].d };

		CUDA_TRY(cuMemcpyHtoDAsync(b->lt[2].d, b->lt[2].h,
				nx * sizeof(lt_xreq_t), b->stream))
		CUDA_TRY(cuMemcpyHtoDAsync(b->lt[3].d, b->lt[3].h,
				nlat * sizeof(lt_lat_t), b->stream))
		CUDA_TRY(cuLaunchKernel(g->lt_x, nx, 1, 1, LT_THREADS, 1, 1,
				0, b->stream, args, NULL))
		CUDA_TRY(cuMemcpyDtoHAsync(b->lt[4].h, b->lt[4].d,
				nx * sizeof(lt_xout_t), b->stream))
	}
}

/* the kept polynomials of group grp's waiting coroutines (at most
   RO_INIT_FINAL; the rest wait a round): one 32-thread group each */

static void
ro_launch_final(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t n = 0, np = 0;

	b->rf_n = 0;
	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;
		ra_final_t *j = (ra_final_t *)b->rf[1].h + n;

		if (s->state != RO_WAITING || s->req_lines != RQ_FINAL ||
		    n + s->req_n > RO_INIT_FINAL)
			continue;
		((ra_poly_t *)b->rf[0].h)[np] = *s->ra_poly;
		memcpy(j, s->rf_jobs, s->req_n * sizeof(ra_final_t));
		for (uint32_t k = 0; k < s->req_n; k++)
			j[k].poly = np;
		s->rf_offset = n;
		n += s->req_n;
		np++;
		s->state = RO_SENT;
	}
	b->rf_n = n;
	if (n == 0)
		return;
	t->finals += n;

	CUDA_TRY(cuMemcpyHtoDAsync(b->rf[0].d, b->rf[0].h,
			np * sizeof(ra_poly_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->rf[1].d, b->rf[1].h,
			n * sizeof(ra_final_t), b->stream))
	{
		void *args[3] = { &b->rf[0].d, &b->rf[1].d, &b->rf[2].d };

		CUDA_TRY(cuLaunchKernel(g->rf_final, n, 1, 1, 32, 1, 1, 0,
				b->stream, args, NULL))
	}
	CUDA_TRY(cuMemcpyDtoHAsync(b->rf[2].h, b->rf[2].d,
			n * sizeof(ra_final_out_t), b->stream))
}

/* the root properties of group grp's waiting coroutines: one block per
   rotation, the 512-bit ones first, then the 1024-bit ones */

static void
ro_launch_alpha(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t np = 0, n = 0;

	b->ra_n[0] = b->ra_n[1] = 0;
	for (uint32_t w = 0; w < 2; w++) {
		for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
			ro_slot_t *s = t->slots + i;

			if (s->state != RO_WAITING ||
			    s->req_lines != RQ_ALPHA ||
			    (s->ra_poly->pad == 1024) != w)
				continue;
			s->ra_offset = n;
			n += s->req_n;
			b->ra_n[w] += s->req_n;
			np++;
		}
	}
	if (n == 0)
		return;

	ro_arr_reserve(b->ra + 0, np * sizeof(ra_poly_t));
	ro_arr_reserve(b->ra + 1, n * sizeof(ra_rot_t));
	ro_arr_reserve(b->ra + 2, n * sizeof(ra_res_t));
	if (n > b->ra_dd_cap) {
		if (b->ra_dd_cap > 0)
			CUDA_TRY(cuMemFree(b->ra_dd))
		b->ra_dd_cap = MAX(n, 2 * b->ra_dd_cap);
		CUDA_TRY(cuMemAlloc(&b->ra_dd,
				b->ra_dd_cap * sizeof(ra_dd_t)))
	}
	np = 0;
	for (uint32_t w = 0; w < 2; w++) {
		for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
			ro_slot_t *s = t->slots + i;
			ra_rot_t *r = (ra_rot_t *)b->ra[1].h + s->ra_offset;

			if (s->state != RO_WAITING ||
			    s->req_lines != RQ_ALPHA ||
			    (s->ra_poly->pad == 1024) != w)
				continue;
			((ra_poly_t *)b->ra[0].h)[np] = *s->ra_poly;
			s->rf_offset = np;	/* its kept inputs */
			memcpy(r, s->ra_rots, s->req_n * sizeof(ra_rot_t));
			for (uint32_t k = 0; k < s->req_n; k++) {
				r[k].poly = np;
				r[k].mf_index = (t->mf_area != 0 &&
						k < RO_HEAP) ?
						i * RO_HEAP + k : RA_NO_MF;
			}
			np++;
			s->state = RO_SENT;
		}
	}

	CUDA_TRY(cuMemcpyHtoDAsync(b->ra[0].d, b->ra[0].h,
			np * sizeof(ra_poly_t), b->stream))
	CUDA_TRY(cuMemcpyHtoDAsync(b->ra[1].d, b->ra[1].h,
			n * sizeof(ra_rot_t), b->stream))
	for (uint32_t w = 0, off = 0; w < 2; w++) {
		CUdeviceptr rots = b->ra[1].d + off * sizeof(ra_rot_t);
		CUdeviceptr out = b->ra[2].d + off * sizeof(ra_res_t);
		CUdeviceptr dd = b->ra_dd + off * sizeof(ra_dd_t);
		void *args[5] = { &g->d_ra_tables, &b->ra[0].d, &rots, &out,
				&dd };

		if (b->ra_n[w] > 0) {
			CUDA_TRY(cuLaunchKernel(g->alpha[w], b->ra_n[w], 1, 1,
					RO_RA_THREADS, 1, 1, 0, b->stream,
					args, NULL))
		}
		off += b->ra_n[w];
	}
	if (g->bs_size != NULL) {
		/* the size scores of the rotations that pass alpha */
		uint32_t blocks = MIN((n + 3) / 4, g->bs_max_blocks);
		void *args[10] = { &g->d_mf_tables, &b->ra[0].d, &b->ra[1].d,
				&b->ra_dd, &b->ra[2].d, &b->ra_total,
				&b->d_ra_ctr, &t->mf_area, &b->ra_keep.d,
				&b->d_keep_cnt };

		b->ra_total = n;
		CUDA_TRY(cuMemsetD8Async(b->d_keep_cnt, 0,
				np * sizeof(uint32_t), b->stream))
		CUDA_TRY(cuMemsetD8Async(b->d_ra_ctr, 0, sizeof(uint32_t),
				b->stream))
		CUDA_TRY(cuLaunchKernel(g->bs_size, blocks, 1, 1, 128, 1, 1,
				0, b->stream, args, NULL))
	}
	CUDA_TRY(cuMemcpyDtoHAsync(b->ra[2].h, b->ra[2].d,
			n * sizeof(ra_res_t), b->stream))
	if (g->bs_size != NULL) {
		CUDA_TRY(cuMemcpyDtoHAsync(b->ra_keep.h, b->ra_keep.d,
				np * RA_KEEP * sizeof(mf_in_t), b->stream))
	}
	t->alphas += n;
}

/* send the optimize_basic() calls of group grp's waiting coroutines to
   the GPU; does not wait for them */

static void
ro_launch(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;
	uint32_t n = 0;

	ro_launch_setup(t, grp);
	ro_launch_lattice(t, grp);
	ro_launch_lines(t, grp);
	ro_launch_alpha(t, grp);
	ro_launch_final(t, grp);

	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != 0)
			continue;
		s->offset = n;
		n += s->req_n;
	}
	b->n = n;
	if (n == 0)
		return;
	ro_reserve(b, n);

	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_WAITING || s->req_lines != 0)
			continue;
		for (uint32_t k = 0; k < s->req_n; k++) {
			so_basic_in_t *r = b->h_in + s->offset + k;

			for (uint32_t j = 0; j <= SO_MAX_DEG; j++)
				r->apoly[j] = (j <= g->degree) ?
						s->req_in[k].coeff[j] : 0;
		}
		s->state = RO_SENT;
	}

	/* the last cpu_share of the calls are done here, while the GPU
	   does the others; the results are the same either way */
	b->ng = n - (uint32_t)(t->cpu_share * n);

	if (b->ng > 0) {
		uint32_t ng = b->ng;
		uint32_t threads = MIN(RO_THREADS_PER_BLOCK,
				(uint32_t)g->basic.threads_per_block);
		uint32_t blocks = MIN((ng + threads - 1) / threads,
					g->max_blocks);
		void *args[4] = { &b->d_in, &b->d_out, &b->ng, &b->d_ctr };

		CUDA_TRY(cuMemcpyHtoDAsync(b->d_in, b->h_in,
				ng * sizeof(so_basic_in_t), b->stream))
		CUDA_TRY(cuMemsetD8Async(b->d_ctr, 0,
				SO_BASIC_NUM_STATS * sizeof(uint32_t),
				b->stream))
		/* each GPU thread takes calls until none are left */
		CUDA_TRY(cuLaunchKernel(g->basic.kernel_func, blocks, 1, 1,
				threads, 1, 1, 0, b->stream, args, NULL))
		CUDA_TRY(cuMemcpyDtoHAsync(b->h_out, b->d_out,
				ng * sizeof(so_basic_out_t), b->stream))
		CUDA_TRY(cuMemcpyDtoHAsync(b->h_ctr, b->d_ctr,
				SO_BASIC_NUM_STATS * sizeof(uint32_t),
				b->stream))
	}

	if (b->ng < n) {
		optimize_basic_hook_t h = optimize_basic_hook;

		optimize_basic_hook = NULL;
		for (uint32_t i = b->ng; i < n; i++) {
			dpoly_t p;

			p.degree = g->degree;
			for (uint32_t j = 0; j <= MAX_POLY_DEGREE; j++)
				p.coeff[j] = (j <= SO_MAX_DEG) ?
						b->h_in[i].apoly[j] : 0;
			b->h_out[i].norm = optimize_basic(&p,
						&b->h_out[i].skewness,
						&b->h_out[i].translation);
		}
		optimize_basic_hook = h;
		t->cpu_calls += n - b->ng;
	}
	t->batches++;
	t->requests += n;
}

/* wait for group grp's batch and hand the results to its coroutines */

static void
ro_collect(ro_thread_t *t, uint32_t grp)
{
	ro_gpu_t *g = t->g;
	ro_buf_t *b = t->buf + grp;

	uint32_t nrs = b->rs_n[0] + b->rs_n[1] + b->lt_n[0] + b->lt_n[1] +
			b->lt_roots + b->rf_n;

	if (b->n == 0 && b->ls_lines == 0 && b->ra_n[0] + b->ra_n[1] == 0 &&
	    nrs == 0)
		return;
	if (b->ng > 0 || b->ls_lines > 0 || b->ra_n[0] + b->ra_n[1] > 0 ||
	    nrs > 0) {
		double t0 = ro_now(), w;

		CUDA_TRY(cuStreamSynchronize(b->stream))

		/* waited: do more of the calls here next time; did not:
		   fewer */
		w = ro_now() - t0;
		if (g->auto_share && b->ng > 0) {
			if (w > 0.0005)
				t->cpu_share = MIN(0.95, t->cpu_share + 0.02);
			else
				t->cpu_share = MAX(0.0, t->cpu_share - 0.01);
		}
	}

	/* what the CPU code prints for these */
	if (b->ls_lines > 0 && g->cand_gpu) {
		uint32_t *c = (uint32_t *)b->ls[RA_LCTR].h;

		for (uint32_t i = 0; i < c[SO_BASIC_LINE_ITERS]; i++)
			printf(":");
	}
	if (b->ng > 0) {
		for (uint32_t i = 0; i < b->h_ctr[SO_BASIC_LINE_ITERS]; i++)
			printf(":");
	}

	for (uint32_t i = grp; i < t->num_slots; i += RO_GROUPS) {
		ro_slot_t *s = t->slots + i;

		if (s->state != RO_SENT || s->req_lines == RQ_MURPHY ||
		    s->req_lines == RQ_MURPHY_DEV)
			continue;
		if (s->req_lines == RQ_FINAL) {
			memcpy(s->rf_out, (ra_final_out_t *)b->rf[2].h +
				s->rf_offset,
				s->req_n * sizeof(ra_final_out_t));
			s->state = RO_RUNNABLE;
			continue;
		}
		if (s->req_lines == RQ_LAT_XY) {
			memcpy(s->lt_out, (lt_hit_t *)b->lt[1].h +
				(size_t)s->lt_offset * LT_MAX_PRIMES,
				LT_MAX_PRIMES * sizeof(lt_hit_t));
			s->state = RO_RUNNABLE;
			continue;
		}
		if (s->req_lines == RQ_LAT_X) {
			memcpy(s->lt_out, (lt_xout_t *)b->lt[4].h +
				s->lt_offset, sizeof(lt_xout_t));
			s->state = RO_RUNNABLE;
			continue;
		}
		if (s->req_lines == RQ_ROOTS) {
			s->state = RO_RUNNABLE;
			continue;
		}
		if (s->req_lines == RQ_SETUP) {
			const rs_setup_out_t *o = (rs_setup_out_t *)
						b->rs[2].h + s->rs_offset;

			*s->rs_out = *o;
			s->rs_out->pad = s->rs_poly->pad2[0];
			if (!s->rs_poly->pad2[0])
				memcpy(s->rs_roots, (ls_root_t *)b->rs[3].h +
					(size_t)s->rs_offset * RS_MAX_ROOTS,
					MIN(o->total, RS_MAX_ROOTS) *
						sizeof(ls_root_t));
			s->roots_resident = (o->total <= RS_MAX_ROOTS);
			s->state = RO_RUNNABLE;
			continue;
		}
		if (s->req_lines == RQ_ALPHA) {
			ra_res_t *res = s->ra_out;

			memcpy(res, (ra_res_t *)b->ra[2].h + s->ra_offset,
					s->req_n * sizeof(ra_res_t));
			if (g->bs_size != NULL)
				memcpy(s->ra_keep, (mf_in_t *)b->ra_keep.h +
					(size_t)s->rf_offset * RA_KEEP,
					RA_KEEP * sizeof(mf_in_t));
			s->state = RO_RUNNABLE;

			/* the rotations that passed have their Murphy E
			   inputs on the GPU already: on to Murphy E,
			   without the host */
			s->mf_dev_n = 0;
			for (uint32_t k = 0; t->mf_area != 0 &&
					k < MIN(s->req_n, RO_HEAP); k++) {
				if (!(res[k].flags & RA_PASSED))
					continue;
				s->mf_dev_idx[s->mf_dev_n] = i * RO_HEAP + k;
				s->mf_dev_rot[s->mf_dev_n++] = k;
			}
			if (s->mf_dev_n > 0) {
				s->req_lines = RQ_MURPHY_DEV;
				s->state = RO_WAITING;
			}
			continue;
		}
		if (s->req_lines == RQ_LINES) {
			*s->ls_cand = (const ls_cand_t *)b->ls[RA_CAND].h +
					(size_t)s->ls_offset[2] * LS_TOP;
			*s->ls_norm = !g->cand_gpu ? NULL :
				(const basic_result_t *)
					((so_basic_out_t *)b->ls[RA_NORM].h +
					(size_t)s->ls_offset[2] * LS_TOP);
			*s->ls_cpoly = g->cand_gpu ? NULL :
				(const so_basic_in_t *)b->ls[RA_CIN].h +
					(size_t)s->ls_offset[2] * LS_TOP;
			memcpy(s->ls_out, (ls_out_t *)b->ls[RA_OUT].h +
					s->ls_offset[2],
					s->req_n * sizeof(ls_out_t));
			s->state = RO_RUNNABLE;
			continue;
		}
		for (uint32_t k = 0; k < s->req_n; k++) {
			so_basic_out_t *o = b->h_out + s->offset + k;

			s->req_out[k].norm = o->norm;
			s->req_out[k].skewness = o->skewness;
			s->req_out[k].translation = o->translation;
		}
		s->state = RO_RUNNABLE;
	}
	b->n = 0;
	b->ls_lines = 0;
	b->ra_n[0] = b->ra_n[1] = 0;
	b->rs_n[0] = b->rs_n[1] = 0;
	b->lt_n[0] = b->lt_n[1] = 0;
	b->lt_roots = 0;
	b->rf_n = 0;
}

/*------------------------------------------------------------------------*/
/* a thread's streams, buffers and slots. All of them are made before
   any thread starts, and freed after all have stopped: CUDA allocations
   wait for the GPU and hold the driver's lock, so made while other
   threads run they cost seconds */

static void
ro_thread_alloc(ro_thread_t *t)
{
	ro_gpu_t *g = t->g;
	size_t n = t->num_slots;
	double t_start = ro_now();

	for (uint32_t k = 0; k < RO_GROUPS; k++) {
		ro_buf_t *b = t->buf + k;

		{
			int lo = 0, hi = 0;

			CUDA_TRY(cuCtxGetStreamPriorityRange(&lo, &hi))
			CUDA_TRY(cuStreamCreateWithPriority(&b->stream,
					CU_STREAM_NON_BLOCKING, hi))
		}

		/* room for typical batches from the start: every
		   reallocation of pinned memory waits for the whole GPU
		   and holds up the other threads' CUDA calls */
		ro_reserve(b, RO_INIT_CALLS);
		ro_arr_reserve(b->ls + RA_POLY, 256 * sizeof(ls_poly_t));
		ro_arr_reserve(b->ls + RA_PRIME, 256 * LS_NUM_PRIMES *
						sizeof(ls_prime_t));
		ro_arr_reserve(b->ls + RA_POWER, 256 * LS_MAX_POWERS *
						sizeof(ls_power_t));
		if (g->lt_x != NULL) {
			ro_arr_reserve(b->lt + 0, RO_INIT_LT * sizeof(lt_set_t));
			ro_arr_reserve(b->lt + 1, RO_INIT_LT * LT_MAX_PRIMES *
						sizeof(lt_hit_t));
			ro_arr_reserve(b->lt + 2, RO_INIT_LT * sizeof(lt_xreq_t));
			ro_arr_reserve(b->lt + 3, LT_MAX_LATS *
						sizeof(lt_lat_t));
			ro_arr_reserve(b->lt + 4, RO_INIT_LT * sizeof(lt_xout_t));
			CUDA_TRY(cuMemAlloc(&b->lt_scr_x, (size_t)RO_INIT_LT *
				LT_THREADS * LT_MAX_X_LATTICES * sizeof(uint64)))
			CUDA_TRY(cuMemAlloc(&b->lt_scr_s, (size_t)RO_INIT_LT *
				LT_THREADS * LT_MAX_X_LATTICES * sizeof(uint16)))
		}
		ro_arr_reserve(&b->ra_keep, t->num_slots * RA_KEEP * sizeof(mf_in_t));
		CUDA_TRY(cuMemAlloc(&b->d_keep_cnt, t->num_slots * sizeof(uint32_t)))
		ro_arr_reserve(b->rf + 0, RO_INIT_FINAL * sizeof(ra_poly_t));
		ro_arr_reserve(b->rf + 1, RO_INIT_FINAL * sizeof(ra_final_t));
		ro_arr_reserve(b->rf + 2, RO_INIT_FINAL *
						sizeof(ra_final_out_t));
		ro_arr_reserve(b->rs + 0, RO_INIT_SETUPS * sizeof(ra_poly_t));
		ro_arr_reserve(b->rs + 1, RO_INIT_SETUPS * sizeof(rs_job_t));
		ro_arr_reserve(b->rs + 2, RO_INIT_SETUPS * sizeof(rs_setup_out_t));
		ro_arr_reserve(b->rs + 3, RO_INIT_SETUPS * RS_MAX_ROOTS *
						sizeof(ls_root_t));
		ro_arr_reserve(b->ls + RA_RUN, 1024 * sizeof(ls_run_t));
		ro_arr_reserve(b->ls + RA_CAND, RO_INIT_LINES * LS_TOP *
						sizeof(ls_cand_t));
		ro_arr_reserve(b->ls + RA_NORM, RO_INIT_LINES * LS_TOP *
						sizeof(so_basic_out_t));
		ro_arr_reserve(b->ls + RA_LCTR, SO_BASIC_NUM_STATS *
						sizeof(uint32_t));
		if (!g->cand_gpu)
			ro_arr_reserve(b->ls + RA_CIN, RO_INIT_LINES * LS_TOP *
						sizeof(so_basic_in_t));
		b->cin_cap = RO_INIT_LINES * LS_TOP;
		CUDA_TRY(cuMemAlloc(&b->d_cin, b->cin_cap *
					sizeof(so_basic_in_t)))
		ro_arr_reserve(b->ls + RA_LINE, RO_INIT_LINES *
						sizeof(ls_line_t));
		ro_arr_reserve(b->ls + RA_OUT, RO_INIT_LINES *
						sizeof(ls_out_t));
		ro_arr_reserve(b->ra + 0, 256 * sizeof(ra_poly_t));
		ro_arr_reserve(b->ra + 1, 256 * 200 * sizeof(ra_rot_t));
		ro_arr_reserve(b->ra + 2, 256 * 200 * sizeof(ra_res_t));
		b->ra_dd_cap = 256 * 200;
		CUDA_TRY(cuMemAlloc(&b->ra_dd, b->ra_dd_cap * sizeof(ra_dd_t)))
		CUDA_TRY(cuMemAlloc(&b->d_ra_ctr, sizeof(uint32_t)))

		CUDA_TRY(cuMemAlloc(&b->d_ctr,
				SO_BASIC_NUM_STATS * sizeof(uint32_t)))
		CUDA_TRY(cuMemAllocHost((void **)&b->h_ctr,
				SO_BASIC_NUM_STATS * sizeof(uint32_t)))
	}

	t->slots = (ro_slot_t *)xcalloc(n, sizeof(ro_slot_t));
	for (uint32_t i = 0; i < n; i++) {
		ro_slot_t *s = t->slots + i;

		ro_job_init(&s->job);
		s->state = RO_FREE;
		s->group = i % RO_GROUPS;
	}

	{
		/* the short batches first: Murphy E blocks run long */
		int lo = 0, hi = 0;

		CUDA_TRY(cuCtxGetStreamPriorityRange(&lo, &hi))
		CUDA_TRY(cuStreamCreateWithPriority(&t->mf_stream,
				CU_STREAM_NON_BLOCKING, lo))
	}
	ro_arr_reserve(t->mf + 0, 4096 * sizeof(mf_in_t));
	ro_arr_reserve(t->mf + 1, 4096 * sizeof(mf_out_t));
	ro_arr_reserve(t->mf + 2, 2 * sizeof(uint32_t));
	assess_init(&t->mf_assess);
	if (g->ls_sieve != NULL)
		CUDA_TRY(cuMemAlloc(&t->roots_area, (size_t)t->num_slots *
				RS_MAX_ROOTS * sizeof(ls_root_t)))
	if (g->mf_opt != NULL && g->bs_size != NULL) {
		CUDA_TRY(cuMemAlloc(&t->mf_area, (size_t)t->num_slots *
				RO_HEAP * sizeof(mf_in_t)))
		ro_arr_reserve(&t->mf_idx, 4096 * sizeof(uint32_t));
	}
	t->t_setup = ro_now() - t_start;
}

static void
ro_thread_release(ro_thread_t *t)
{
	size_t n = t->num_slots;
	double t_start = ro_now();

	for (uint32_t i = 0; i < n; i++) {
		ro_slot_t *s = t->slots + i;

		if (s->ready) {
			poly_rootopt_free(&s->ro);
			munmap(s->stack, RO_STACK_SIZE);
		}
		ro_job_clear(&s->job);
	}
	free(t->slots);
	for (uint32_t k = 0; k < RO_GROUPS; k++) {
		ro_buf_t *b = t->buf + k;

		CUDA_TRY(cuMemFreeHost(b->h_in))
		CUDA_TRY(cuMemFreeHost(b->h_out))
		CUDA_TRY(cuMemFreeHost(b->h_ctr))
		CUDA_TRY(cuMemFree(b->d_in))
		CUDA_TRY(cuMemFree(b->d_out))
		CUDA_TRY(cuMemFree(b->d_ctr))
		for (uint32_t k2 = 0; k2 < RA_NUM; k2++)
			ro_arr_free(b->ls + k2);
		for (uint32_t k2 = 0; k2 < 3; k2++)
			ro_arr_free(b->ra + k2);
		for (uint32_t k2 = 0; k2 < 4; k2++)
			ro_arr_free(b->rs + k2);
		for (uint32_t k2 = 0; k2 < 5; k2++)
			ro_arr_free(b->lt + k2);
		for (uint32_t k2 = 0; k2 < 3; k2++)
			ro_arr_free(b->rf + k2);
		ro_arr_free(&b->ra_keep);
		CUDA_TRY(cuMemFree(b->d_keep_cnt))
		if (b->lt_scr_x != 0) {
			CUDA_TRY(cuMemFree(b->lt_scr_x))
			CUDA_TRY(cuMemFree(b->lt_scr_s))
		}
		CUDA_TRY(cuMemFree(b->ra_dd))
		CUDA_TRY(cuMemFree(b->d_ra_ctr))
		if (b->cin_cap > 0)
			CUDA_TRY(cuMemFree(b->d_cin))

		CUDA_TRY(cuStreamDestroy(b->stream))
	}
	for (uint32_t k = 0; k < 3; k++)
		ro_arr_free(t->mf + k);
	ro_arr_free(&t->mf_idx);
	if (t->roots_area != 0)
		CUDA_TRY(cuMemFree(t->roots_area))
	if (t->mf_area != 0)
		CUDA_TRY(cuMemFree(t->mf_area))
	CUDA_TRY(cuStreamDestroy(t->mf_stream))
	assess_free(&t->mf_assess);
	t->t_teardown = ro_now() - t_start;
}

/*------------------------------------------------------------------------*/
static void *
ro_thread_main(void *arg)
{
	ro_thread_t *t = (ro_thread_t *)arg;
	ro_gpu_t *g = t->g;
	bool done = false;

	CUDA_TRY(cuCtxSetCurrent(g->ctx))
	ro_self = t;
	optimize_basic_hook = ro_hook;
	optimize_basic_batch_hook = ro_hook_batch;
	if (g->ls_sieve != NULL)
		line_sieve_batch_hook = ro_hook_lines;
	if (g->mf_opt != NULL)
		murphy_batch_hook = ro_hook_murphy;
	if (g->alpha[0] != NULL)
		alpha_batch_hook = ro_hook_alpha;
	if (g->rf_final != NULL)
		final_batch_hook = ro_hook_final;
	if (g->rs_setup[0] != NULL && g->ls_sieve != NULL) {
		rs_setup_hook = ro_hook_setup;
		roots_fetch_hook = ro_hook_roots;
		if (g->lt_x != NULL) {
			lattice_xy_hook = ro_hook_lat_xy;
			lattice_x_hook = ro_hook_lat_x;
		}
	}
	root_sieve_batched = g->batched;

	for (;;) {
		double t0 = ro_now(), t1;
		uint32_t busy = ro_fill(t, &done);

		t1 = ro_now();
		t->t_idle += t1 - t0;
		if (busy == 0 && done)
			break;

		for (uint32_t k = 0; k < RO_GROUPS; k++) {
			/* this group's last batch, then its coroutines,
			   while the other group's batch runs */
			t0 = ro_now();
			ro_collect(t, k);
			t1 = ro_now();
			t->t_gpu += t1 - t0;

			for (uint32_t i = k; i < t->num_slots; i += RO_GROUPS) {
				ro_slot_t *s = t->slots + i;

				if (s->state != RO_RUNNABLE)
					continue;
				t->current = s;
				swapcontext(&t->main_ctx, &s->ctx);
			}
			ro_launch(t, k);
			t->t_cpu += ro_now() - t1;
		}

		/* Murphy E batches: collect a finished one, send the next;
		   wait only if nothing else is left to do */
		t0 = ro_now();
		ro_murphy_poll(t, false);
		ro_murphy_launch(t);
		if (t->mf_busy) {
			bool other = false;

			for (uint32_t i = 0; i < t->num_slots && !other; i++) {
				ro_slot_t *s = t->slots + i;

				other = (s->state != RO_FREE &&
					s->req_lines != RQ_MURPHY &&
					s->req_lines != RQ_MURPHY_DEV) ||
					s->state == RO_RUNNABLE ||
					s->state == RO_NEW;
			}
			if (!other)
				ro_murphy_poll(t, true);
		}
		t->t_gpu += ro_now() - t0;
	}

	optimize_basic_hook = NULL;
	optimize_basic_batch_hook = NULL;
	line_sieve_batch_hook = NULL;
	murphy_batch_hook = NULL;
	alpha_batch_hook = NULL;
	final_batch_hook = NULL;
	rs_setup_hook = NULL;
	roots_fetch_hook = NULL;
	lattice_xy_hook = NULL;
	lattice_x_hook = NULL;
	root_sieve_batched = 0;
	ro_self = NULL;

	return NULL;
}

/*------------------------------------------------------------------------*/
/* ls_sieve's constants: where each prime's table ends, its size and
   the step of a thread's index, laid out as rs_line_poly() lays out
   every polynomial's tables (root_sieve_init() fixes the sizes) */

static uint32_t
ro_ls_tables_set(CUmodule module)
{
	root_sieve_t *rs = (root_sieve_t *)xmalloc(sizeof(root_sieve_t));
	uint32_t end[LS_NUM_PRIMES], size[LS_NUM_PRIMES], step[LS_NUM_PRIMES];
	uint32_t table = 0, ok;
	CUdeviceptr d;
	size_t bytes;

	root_sieve_init(rs);
	ok = (rs->num_primes == LS_NUM_PRIMES);
	for (uint32_t i = 0; ok && i < LS_NUM_PRIMES; i++) {
		size[i] = rs->primes[i].contrib_array_size;
		table += size[i];
		end[i] = table;
		step[i] = RO_LS_THREADS % size[i];
	}
	root_sieve_free(rs);
	free(rs);
	if (!ok || table > LS_MAX_TABLE)
		return 0;

	CUDA_TRY(cuModuleGetGlobal(&d, &bytes, module, "ls_c_end"))
	CUDA_TRY(cuMemcpyHtoD(d, end, sizeof(end)))
	CUDA_TRY(cuModuleGetGlobal(&d, &bytes, module, "ls_c_size"))
	CUDA_TRY(cuMemcpyHtoD(d, size, sizeof(size)))
	CUDA_TRY(cuModuleGetGlobal(&d, &bytes, module, "ls_c_step"))
	CUDA_TRY(cuMemcpyHtoD(d, step, sizeof(step)))
	return 1;
}

/*------------------------------------------------------------------------*/
/* the integrator's rule and Dickman's tables, as the CPU code has them,
   for the Murphy E and size score kernels; 0 if they do not fit */

static uint32_t
ro_mf_tables_init(ro_gpu_t *g)
{
	mf_tables_t *tab;
	integrate_t integ;
	dickman_t dk;
	double *rule;
	uint32_t ok;

	if (g->d_mf_tables != 0)
		return 1;

	tab = (mf_tables_t *)xcalloc(1, sizeof(mf_tables_t));
	rule = (double *)xmalloc(3 * (MF_MAX_POINTS << MF_TRAP_LEVELS) *
				sizeof(double));
	integrate_init(&integ, SIZE_EPS, double_exponential);
	ok = integrate_de_rule(&integ, MF_MAX_POINTS, &tab->num_points,
			&tab->relative_error, &tab->center_weight,
			&tab->center_residual_weight, rule);
	integrate_free(&integ);
	for (uint32_t i = 0; ok && i < (tab->num_points << MF_TRAP_LEVELS);
	     i++) {
		tab->rule[i].abscissa = rule[3 * i];
		tab->rule[i].weight = rule[3 * i + 1];
		tab->rule[i].residual_weight = rule[3 * i + 2];
	}
	free(rule);

	dickman_init(&dk);
	for (uint32_t i = 2; ok && i < MF_DICKMAN_LINES; i++) {
		dickman_line_t *l = dk.lines + i;

		if (l->coeff_offset + l->num_coeffs > MF_DICKMAN_COEFFS) {
			ok = 0;
			break;
		}
		tab->dk_num_coeffs[i] = l->num_coeffs;
		tab->dk_offset[i] = l->coeff_offset;
		memcpy(tab->dk_coeffs + l->coeff_offset,
			dk.coeffs + l->coeff_offset,
			l->num_coeffs * sizeof(double));
	}
	dickman_free(&dk);
	tab->log_rfb_limit = log(5000000.0);	/* analyze_poly_murphy() */
	tab->log_afb_limit = log(10000000.0);
	tab->jt_cos94 = cos(94 * M_PI / 180);	/* jenkins_traub() */
	tab->jt_sin94 = sin(94 * M_PI / 180);

	if (ok) {
		CUDA_TRY(cuMemAlloc(&g->d_mf_tables, sizeof(mf_tables_t)))
		CUDA_TRY(cuMemcpyHtoD(g->d_mf_tables, tab,
				sizeof(mf_tables_t)))
	}
	free(tab);
	return ok;
}

/* the Murphy E kernel: on by default, MSIEVE_GPU_MURPHY=0 for the CPU
   code. The default is for the target hardware (RTX PRO 6000 Blackwell
   and up), not for the FP64 ratio: it used to be on only where double
   precision is fast (ratio <= 4). On a small card the GPU is already
   the limit with optimize_basic() and the line sieve, and Murphy E
   there slows the rest (RTX PRO 4500, 1/64: one c155 coefficient
   107.0 s with Murphy E on the CPU, 170-183 s with it on the GPU,
   though the kernel alone beats 32 CPU threads: 1512 against 1263
   candidates/s) */

static void
ro_murphy_init(ro_gpu_t *g, CUmodule module)
{
	const char *e = getenv("MSIEVE_GPU_MURPHY");
	char buf[32];
	CUfunction f;
	int per_sm = 0;

	if (e != NULL && strcmp(e, "0") == 0)
		return;
	sprintf(buf, "mf_opt_%u", g->degree);
	if (cuModuleGetFunction(&f, module, buf) != CUDA_SUCCESS ||
	    !ro_mf_tables_init(g))
		return;

	CUDA_TRY(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,
			f, RO_MF_THREADS, 0))
	g->mf_max_blocks = MAX(1, (uint32_t)per_sm) * g->num_sm;
	g->mf_opt = f;
}

/* the root sieve setup kernels and their table of prime powers, the
   ones root_sieve_init() makes (MSIEVE_GPU_RSSETUP=0: on the CPU) */

static void
ro_setup_init(ro_gpu_t *g, CUmodule module)
{
	const char *e = getenv("MSIEVE_GPU_RSSETUP");
	root_sieve_t *rs;
	rs_powers_t *pw;
	CUfunction f[2];
	uint32_t ok = 1;

	if ((e != NULL && strcmp(e, "0") == 0) ||
	    cuModuleGetFunction(f + 0, module, "rs_setup_512") !=
	    		CUDA_SUCCESS ||
	    cuModuleGetFunction(f + 1, module, "rs_setup_1024") !=
	    		CUDA_SUCCESS)
		return;

	rs = (root_sieve_t *)xmalloc(sizeof(root_sieve_t));
	pw = (rs_powers_t *)xcalloc(1, sizeof(rs_powers_t));
	root_sieve_init(rs);
	for (uint32_t i = 0; ok && i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		for (uint32_t j = 0; j < sp->num_powers; j++) {
			if (pw->num_powers == RS_MAX_POWERS) {
				ok = 0;
				break;
			}
			pw->prime[pw->num_powers] = sp->prime;
			pw->power[pw->num_powers++] = sp->powers[j].power;
		}
	}
	root_sieve_free(rs);
	free(rs);

	if (ok) {
		CUDA_TRY(cuMemAlloc(&g->d_rs_powers, sizeof(rs_powers_t)))
		CUDA_TRY(cuMemcpyHtoD(g->d_rs_powers, pw,
				sizeof(rs_powers_t)))
		g->rs_setup[0] = f[0];
		g->rs_setup[1] = f[1];
	}
	free(pw);
}

/* the size score kernel, after alpha (MSIEVE_GPU_BERNSTEIN=0: on the
   CPU) */

static void
ro_bernstein_init(ro_gpu_t *g, CUmodule module)
{
	const char *e = getenv("MSIEVE_GPU_BERNSTEIN");
	char buf[32];
	CUfunction f;
	int per_sm = 0;

	if ((e != NULL && strcmp(e, "0") == 0) || g->alpha[0] == NULL)
		return;
	sprintf(buf, "bs_size_%u", g->degree);
	if (cuModuleGetFunction(&f, module, buf) != CUDA_SUCCESS ||
	    !ro_mf_tables_init(g))
		return;

	CUDA_TRY(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,
			f, 128, 0))
	g->bs_max_blocks = MAX(1, (uint32_t)per_sm) * g->num_sm;
	g->bs_size = f;
}

/*------------------------------------------------------------------------*/
/* the root property kernels and their table of primes
   (MSIEVE_GPU_ALPHA=0: on the CPU) */

static void
ro_alpha_init(ro_gpu_t *g, CUmodule module)
{
	const char *e = getenv("MSIEVE_GPU_ALPHA");
	ra_tables_t *tab;
	CUfunction f[2];
	uint32_t p = 0;

	if ((e != NULL && strcmp(e, "0") == 0) ||
	    g->proto->murphy_p_bound != PRIME_BOUND ||
	    cuModuleGetFunction(f + 0, module, "alpha_512") !=
	    		CUDA_SUCCESS ||
	    cuModuleGetFunction(f + 1, module, "alpha_1024") !=
	    		CUDA_SUCCESS)
		return;

	/* the primes of analyze_poly_roots(), with its powers and logs */
	tab = (ra_tables_t *)xcalloc(1, sizeof(ra_tables_t));
	for (uint32_t i = 0; i < RA_NUM_PRIMES; i++) {
		uint32_t power;

		p += prime_delta[i];
		tab->prime[i] = p;
		tab->log_prime[i] = log((double)p);
		power = p;
		if (i < RA_NUM_SMALL) {
			while (power * p <= 65000)
				power *= p;
		}
		tab->modulus[i] = power;
	}
	if (p >= PRIME_BOUND || p + prime_delta[RA_NUM_PRIMES] <
				PRIME_BOUND) {
		free(tab);	/* not the primes below 2000 */
		return;
	}

	/* get_zeros_rec() recurses, and get_root_freq() keeps two root
	   buffers on the stack: more than the default 1 KB per thread */
	{
		size_t stack = 0;

		CUDA_TRY(cuCtxGetLimit(&stack, CU_LIMIT_STACK_SIZE))
		if (stack < 6144)
			CUDA_TRY(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 6144))
	}
	CUDA_TRY(cuMemAlloc(&g->d_ra_tables, sizeof(ra_tables_t)))
	CUDA_TRY(cuMemcpyHtoD(g->d_ra_tables, tab, sizeof(ra_tables_t)))
	free(tab);
	g->alpha[0] = f[0];
	g->alpha[1] = f[1];
}

/*------------------------------------------------------------------------*/
/* one GPU's engine: context, module, kernels and tables (its threads
   are made by ro_gpu_init) */
static ro_gpu_t *
ro_engine_new(msieve_obj *obj, poly_rootopt_t *proto, gpu_info_t *info,
		const char **name_out)
{
	const char *name;
	char buf[32];
	CUcontext ctx;
	CUmodule module;

	name = so_ptx_name(info->compute_version_major,
			info->compute_version_minor);
	*name_out = name;
	if (name == NULL)
		return NULL;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000 /* API change */
	CUDA_TRY(cuCtxCreate(&ctx, NULL, CU_CTX_BLOCKING_SYNC,
			info->device_handle))
#else
	CUDA_TRY(cuCtxCreate(&ctx, CU_CTX_BLOCKING_SYNC,
			info->device_handle))
#endif
	if (cuModuleLoad(&module, name) != CUDA_SUCCESS) {
		logprintf(obj, "GPU root optimization off: cannot load %s\n",
				name);
		CUDA_TRY(cuCtxDestroy(ctx))
		return NULL;
	}

	ro_gpu_t *g = (ro_gpu_t *)xcalloc(1, sizeof(ro_gpu_t));
	g->obj = obj;
	g->proto = proto;
	g->degree = proto->degree;
	g->ctx = ctx;
	g->module = module;
	g->num_sm = (uint32_t)info->num_compute_units;
	g->device = info->device_handle;
	sprintf(buf, "so_basic_%u", g->degree);
	gpu_launch_init(module, buf, &g->basic);
	{
		int per_sm = 0;
		uint32_t threads = MIN(RO_THREADS_PER_BLOCK,
				(uint32_t)g->basic.threads_per_block);

		CUDA_TRY(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,
				g->basic.kernel_func, (int)threads, 0))
		g->max_blocks = MAX(1, (uint32_t)per_sm) * g->num_sm;
	}
	{
		const char *e = getenv("MSIEVE_ROOTOPT_BATCHED");

		g->batched = !(e != NULL && strcmp(e, "0") == 0);

		/* MSIEVE_ROOTOPT_CPU_SHARE: a fixed share of the
		   optimize_basic() calls done on the CPU, or by default one
		   adapted to keep the threads from waiting for the GPU */
		e = getenv("MSIEVE_ROOTOPT_CPU_SHARE");
		g->auto_share = (e == NULL || strcmp(e, "auto") == 0);
		g->cpu_share = g->auto_share ? 0 :
				MAX(0.0, MIN(1.0, atof(e)));
	}
	{
		/* the line sieve kernel, if the GPU has the shared memory
		   for it and it is wanted (MSIEVE_GPU_LINESIEVE=0: CPU) */
		const char *e = getenv("MSIEVE_GPU_LINESIEVE");
		int optin = 0;
		CUfunction f;

		CUDA_TRY(cuDeviceGetAttribute(&optin,
			CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
			info->device_handle))
		if (!(e != NULL && strcmp(e, "0") == 0) &&
		    optin >= RO_LS_SMEM + 4096 &&
		    cuModuleGetFunction(&f, module, "ls_sieve") ==
		    		CUDA_SUCCESS) {
			CUDA_TRY(cuFuncSetAttribute(f,
				CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
				RO_LS_SMEM))
			if (ro_ls_tables_set(module))
				g->ls_sieve = f;
		}
	}
	ro_murphy_init(g, module);
	{
		/* the candidates' optimize_basic() on the GPU: on by
		   default, as Murphy E (for the target hardware),
		   MSIEVE_GPU_LINECAND=0 for the CPU share. On the RTX
		   PRO 4500 (FP64 1/64) the GPU is the limit already,
		   and calls that stay there cannot be shared with the
		   CPU: a c155 coefficient took 112-117 s against 104 s */
		const char *e = getenv("MSIEVE_GPU_LINECAND");

		g->cand_gpu = (e == NULL) || (strcmp(e, "0") != 0);
	}
	ro_alpha_init(g, module);
	if (cuModuleGetFunction(&g->rf_final, module, "rf_final") !=
			CUDA_SUCCESS)
		g->rf_final = NULL;
	ro_bernstein_init(g, module);
	ro_setup_init(g, module);
#if defined(RS_GPU_LATTICE)
	if (g->rs_setup[0] != NULL && g->ls_sieve != NULL &&
	    (g->degree == 4 || g->degree == 5) &&
	    cuModuleGetFunction(&g->lt_xy, module, "lt_xy") ==
	    		CUDA_SUCCESS &&
	    cuModuleGetFunction(&g->lt_x, module, "lt_x") == CUDA_SUCCESS) {
		/* lt_x's stack: its sieve and hit lists */
		size_t stack = 0;

		CUDA_TRY(cuCtxGetLimit(&stack, CU_LIMIT_STACK_SIZE))
		if (stack < 6144)
			CUDA_TRY(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 6144))
	}
	else {
		g->lt_xy = g->lt_x = NULL;
	}
#endif

	/* this thread goes back to having no context */
	CUDA_TRY(cuCtxPopCurrent(NULL))
	return g;
}

static void ro_engine_free(ro_gpu_t *g);

/* the root optimization engines: one per GPU, all visible GPUs unless
   -g names one (obj->which_gpu, MSIEVE_ALL_GPUS: all). The threads
   (MSIEVE_ROOTOPT_THREADS, default: online CPUs) are divided among
   them. MSIEVE_SIMULATE_GPUS=n (testing): n engines on the GPU(s) as
   i % (devices); they share a card, so no speedup */
ro_gpu_t *
ro_gpu_init(msieve_obj *obj, poly_rootopt_t *proto)
{
	gpu_config_t config;
	uint32_t devs[MAX_GPU], ndev = 0, nlanes, total;
	const char *name = NULL;
	ro_gpu_t **lanes;
	ro_queue_t *q;

	if (proto->degree < 4 || proto->degree > 6)
		return NULL;

	gpu_init(&config);
	{
		uint32 lr = 0, ls = 1;

#ifdef HAVE_MPI
		lr = obj->mpi_local_rank;
		ls = obj->mpi_local_size;
#endif
		ndev = gpu_select(&config, obj->which_gpu, lr, ls, devs,
				NULL);
	}
	if (ndev == 0)
		return NULL;
	nlanes = ro_env("MSIEVE_SIMULATE_GPUS", ndev);
	nlanes = MAX(1, MIN(nlanes, MAX_GPU));

	lanes = (ro_gpu_t **)xcalloc(nlanes, sizeof(ro_gpu_t *));
	for (uint32_t i = 0; i < nlanes; i++) {
		const char *nm;

		lanes[i] = ro_engine_new(obj, proto,
				config.info + devs[i % ndev], &nm);
		if (lanes[i] == NULL) {
			for (uint32_t k = 0; k < i; k++)
				ro_engine_free(lanes[k]);
			free(lanes);
			return NULL;
		}
		lanes[i]->device_index = devs[i % ndev];
		if (i == 0)
			name = nm;
	}

	q = (ro_queue_t *)xcalloc(1, sizeof(ro_queue_t));
	pthread_mutex_init(&q->lock, NULL);
	pthread_cond_init(&q->cv_job, NULL);
	pthread_cond_init(&q->cv_space, NULL);

	{
		/* the node's CPUs, shared by its MPI processes */
		uint32_t cpus = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);

#ifdef HAVE_MPI
		cpus = MAX(1, cpus / MAX(1, obj->mpi_local_size));
#endif
		total = ro_env("MSIEVE_ROOTOPT_THREADS", cpus);
	}
	total = MAX(total, nlanes);
	q->total_threads = 0;
	q->num_lanes = nlanes;
	for (uint32_t i = 0; i < nlanes; i++) {
		ro_gpu_t *g = lanes[i];

		g->q = q;
		g->lanes = lanes;
		g->num_lanes = nlanes;
		g->lane = i;
		g->num_threads = total / nlanes + (i < total % nlanes);
		g->slots_per_thread = ro_env("MSIEVE_ROOTOPT_SLOTS", 256);
		q->total_threads += g->num_threads;
	}
	q->q_cap = 4 * lanes[0]->slots_per_thread * nlanes;
	q->queue = (ro_job_t *)xcalloc(q->q_cap, sizeof(ro_job_t));
	for (uint32_t i = 0; i < q->q_cap; i++)
		ro_job_init(q->queue + i);

	for (uint32_t l = 0; l < nlanes; l++) {
		ro_gpu_t *g = lanes[l];

		g->threads = (ro_thread_t *)xcalloc(g->num_threads,
						sizeof(ro_thread_t));
		for (uint32_t i = 0; i < g->num_threads; i++) {
			ro_thread_t *t = g->threads + i;

			t->g = g;
			t->num_slots = g->slots_per_thread;
			t->cpu_share = g->cpu_share;
		}
		CUDA_TRY(cuCtxPushCurrent(g->ctx))
		for (uint32_t i = 0; i < g->num_threads; i++)
			ro_thread_alloc(g->threads + i);
		CUDA_TRY(cuCtxPopCurrent(NULL))
	}
	for (uint32_t l = 0; l < nlanes; l++) {
		ro_gpu_t *g = lanes[l];

		for (uint32_t i = 0; i < g->num_threads; i++) {
			ro_thread_t *t = g->threads + i;

			if (pthread_create(&t->tid, NULL, ro_thread_main,
					t) != 0) {
				printf("error: cannot start a root "
					"optimization thread\n");
				exit(-1);
			}
		}
	}

	{
		ro_gpu_t *g = lanes[0];

		if (nlanes > 1 || ndev > 1)
			logprintf(obj, "GPU root optimization on %u GPU%s"
				"%s: %u threads in all\n", nlanes,
				nlanes > 1 ? "s" : "",
				nlanes > ndev ? " (simulated)" : "",
				q->total_threads);
		logprintf(obj, "GPU root optimization on (%s): %u threads x "
			"%u polynomials, %s root sieve, line sieve on the %s, "
			"lattice choice on the %s, root properties on the %s, "
			"size scores on the %s, "
			"Murphy E on the %s\n",
			name, g->num_threads, g->slots_per_thread,
			g->batched ? "batched" : "original",
			(g->batched && g->ls_sieve != NULL) ? "GPU" : "CPU",
			(g->batched && g->lt_x != NULL) ? "GPU" : "CPU",
			(g->batched && g->alpha[0] != NULL) ? "GPU" : "CPU",
			(g->batched && g->bs_size != NULL) ? "GPU" : "CPU",
			(g->batched && g->mf_opt != NULL) ? "GPU" : "CPU");
		return g;
	}
}

/*------------------------------------------------------------------------*/
void
ro_gpu_submit(ro_gpu_t *g, mpz_t *alg_coeffs, mpz_t *rat_coeffs,
		double sizeopt_norm, double projective_alpha)
{
	ro_queue_t *q = g->q;

	pthread_mutex_lock(&q->lock);
	while (q->q_count == q->q_cap)
		pthread_cond_wait(&q->cv_space, &q->lock);

	ro_job_t *j = q->queue + (q->q_head + q->q_count) % q->q_cap;
	for (uint32_t i = 0; i <= g->degree; i++)
		mpz_set(j->a[i], alg_coeffs[i]);
	mpz_set(j->r[0], rat_coeffs[0]);
	mpz_set(j->r[1], rat_coeffs[1]);
	j->norm = sizeopt_norm;
	j->alpha = projective_alpha;
	q->q_count++;

	/* every waiting thread: one of a GPU at its share would not
	   take the job */
	pthread_cond_broadcast(&q->cv_job);
	pthread_mutex_unlock(&q->lock);
}

/*------------------------------------------------------------------------*/
/* an engine's device tables, module and context */
static void
ro_engine_free(ro_gpu_t *g)
{
	CUDA_TRY(cuCtxPushCurrent(g->ctx))
	if (g->d_mf_tables != 0)
		CUDA_TRY(cuMemFree(g->d_mf_tables))
	if (g->d_ra_tables != 0)
		CUDA_TRY(cuMemFree(g->d_ra_tables))
	if (g->d_rs_powers != 0)
		CUDA_TRY(cuMemFree(g->d_rs_powers))
	CUDA_TRY(cuModuleUnload(g->module))
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuCtxDestroy(g->ctx))
	free(g);
}

void
ro_gpu_finish(ro_gpu_t *g0)
{
	uint64_t batches = 0, requests = 0, polys = 0, lines = 0, cpu_calls = 0;
	uint64_t murphy = 0, murphy_cpu = 0, alphas = 0, setups = 0;
	uint64_t lt_xys = 0, lt_xs = 0, root_fetches = 0, finals = 0;
	double t_cpu = 0, t_gpu = 0, t_idle = 0;
	ro_queue_t *q;
	ro_gpu_t **lanes;
	uint32_t nlanes;
	char per_lane[512];
	size_t pl = 0;

	if (g0 == NULL)
		return;
	q = g0->q;
	lanes = g0->lanes;
	nlanes = g0->num_lanes;

	pthread_mutex_lock(&q->lock);
	q->finishing = true;
	pthread_cond_broadcast(&q->cv_job);
	pthread_mutex_unlock(&q->lock);

	for (uint32_t l = 0; l < nlanes; l++) {
		ro_gpu_t *g = lanes[l];

		for (uint32_t i = 0; i < g->num_threads; i++)
			pthread_join(g->threads[i].tid, NULL);
	}
	per_lane[0] = 0;
	for (uint32_t l = 0; l < nlanes; l++) {
		ro_gpu_t *g = lanes[l];
		uint64_t lane_polys = 0;

		CUDA_TRY(cuCtxPushCurrent(g->ctx))
		for (uint32_t i = 0; i < g->num_threads; i++)
			ro_thread_release(g->threads + i);
		CUDA_TRY(cuCtxPopCurrent(NULL))

		for (uint32_t i = 0; i < g->num_threads; i++) {
			ro_thread_t *t = g->threads + i;

			batches += t->batches;
			requests += t->requests;
			polys += t->polys;
			lane_polys += t->polys;
			lines += t->lines;
			cpu_calls += t->cpu_calls;
			murphy += t->murphy;
			murphy_cpu += t->murphy_cpu;
			alphas += t->alphas;
			setups += t->setups;
			lt_xys += t->lt_xys;
			finals += t->finals;
			lt_xs += t->lt_xs;
			root_fetches += t->root_fetches;
			t_cpu += t->t_cpu;
			t_gpu += t->t_gpu;
			t_idle += t->t_idle;
			if (getenv("MSIEVE_ROOTOPT_STATS") != NULL)
				printf("rootopt GPU %u thread %u: %" PRIu64
					" polynomials, %" PRIu64 " batches, "
					"%.1f s CPU, %.1f s GPU, %.1f s idle, "
					"%.1f s setup, %.1f s teardown, "
					"%u lines at most\n",
					l, i, t->polys, t->batches, t->t_cpu,
					t->t_gpu, t->t_idle, t->t_setup,
					t->t_teardown, t->max_lines);
		}
		if (pl + 40 < sizeof(per_lane))
			pl += sprintf(per_lane + pl, "%s%" PRIu64, l ? " " : "",
					lane_polys);
	}

	logprintf(g0->obj, "GPU root optimization: %" PRIu64 " polynomials, "
			"%" PRIu64 " optimize_basic calls in %" PRIu64
			" batches (%" PRIu64 " on the CPU), %" PRIu64 " lines "
			"sieved, %" PRIu64 " root sieve setups, %" PRIu64
			" root property sets, %" PRIu64
			" Murphy E optimizations (%" PRIu64 " on the CPU); "
			"thread time %.1f s CPU, %.1f s waiting for the GPU, "
			"%.1f s for work\n", polys, requests, batches,
			cpu_calls, lines, setups, alphas, murphy, murphy_cpu,
			t_cpu,
			t_gpu, t_idle);
	if (nlanes > 1)
		logprintf(g0->obj, "GPU root optimization: polynomials per "
				"GPU: %s\n", per_lane);

	if (g0->lt_x != NULL)
		logprintf(g0->obj, "GPU lattice choice: %" PRIu64 " xy, %" PRIu64
			" x; roots fetched for the CPU %" PRIu64 " times\n",
			lt_xys, lt_xs, root_fetches);
	if (g0->rf_final != NULL)
		logprintf(g0->obj, "GPU final polynomials: %" PRIu64
			" rotated and translated in CGBN\n", finals);

	for (uint32_t i = 0; i < q->q_cap; i++)
		ro_job_clear(q->queue + i);
	free(q->queue);
	pthread_mutex_destroy(&q->lock);
	pthread_cond_destroy(&q->cv_job);
	pthread_cond_destroy(&q->cv_space);
	free(q);
	for (uint32_t l = 0; l < nlanes; l++) {
		free(lanes[l]->threads);
		ro_engine_free(lanes[l]);
	}
	free(lanes);
}
