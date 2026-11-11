/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* The NFS line siever on the GPU (kernels: sieve_core.cu). The GPU
   sieves batches of lines and finds the candidates that msieve's line
   siever (sieve_line.c) would trial factor, with their resieved
   primes; host threads finish the factoring as do_one_factoring()
   does. The candidates are the CPU siever's, so the relations are too
   (MSIEVE_SIEVE_DUMP writes them for comparison). MSIEVE_GPU_SIEVE=0:
   the CPU siever. */

#include "sieve_line_int.h"
#include <cuda_xface.h>
#include <pthread.h>
#include <unistd.h>
#include "sieve_core.h"
#include <sys/time.h>

static double
ls_now(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static double ls_t_gpu, ls_t_host, ls_t_batch;	/* seconds, for the log */
static uint64 ls_n_cand;

#define LS_LINES 16		/* lines per batch, at most */
#define LS_MAX_CANDS (1u << 21)	/* candidates per chunk pass; one line's
				   chunk always fits (b = 1 of a c95 had
				   80,000) */

typedef struct {
	CUdeviceptr p, r, rem0, logp, pw_p, pw_r, pw_orig, pw_rem0, pw_logp;
	CUdeviceptr side;		/* ls_side_t */
	CUdeviceptr start;		/* per line and entry */
	CUdeviceptr bucket, count;
	uint8 *h_logp;			/* entries then powers */
	uint32 n, npw;
	ls_side_t s;
} gpu_side_t;

typedef struct {
	CUcontext ctx;
	CUmodule module;
	CUfunction k_starts, k_bucket, k_segment;
	uint32 num_sm;
	gpu_side_t side[2];		/* 0 rational, 1 algebraic */
	CUdeviceptr batch, lines, ltot, proj_r, proj_a;
	CUdeviceptr cand[2], ncand[2], overflow[2];	/* double buffered */
	ls_batch_t *h_batch;
	ls_cand_t *h_cand[2];
	uint32 max_cands;
} gpu_sieve_t;

/*------------------------------------------------------------------*/
static const char *
ls_ptx_name(int32 major, int32 minor)
{
	if (major >= 12) return "sieve_core_sm120.ptx";
	if (major >= 10) return "sieve_core_sm100.ptx";
	if (major == 9) return "sieve_core_sm90.ptx";
	if (major == 8)
		return (minor < 6) ? "sieve_core_sm80.ptx" :
			(minor < 9) ? "sieve_core_sm86.ptx" :
				      "sieve_core_sm89.ptx";
	if (major == 7)
		return (minor < 5) ? "sieve_core_sm70.ptx" :
				     "sieve_core_sm75.ptx";
	if (major == 6)
		return (minor < 1) ? "sieve_core_sm60.ptx" :
				     "sieve_core_sm61.ptx";
	return NULL;
}

/* f as two's complement words; 0 if a coefficient does not fit */
static uint32
ls_poly(mpz_poly_t *f, ls_poly_t *out)
{
	memset(out, 0, sizeof(*out));
	if (f->degree > LS_MAX_DEG)
		return 0;
	out->degree = f->degree;
	for (uint32 i = 0; i <= f->degree; i++) {
		uint32 w[LS_POLY_WORDS] = {0};
		size_t cnt = 0;
		mpz_t t;

		if (mpz_sizeinbase(f->coeff[i], 2) >= 32 * LS_POLY_WORDS - 1)
			return 0;
		mpz_init(t);
		mpz_abs(t, f->coeff[i]);
		mpz_export(w, &cnt, -1, sizeof(uint32), 0, 0, t);
		mpz_clear(t);
		if (mpz_sgn(f->coeff[i]) < 0) {
			uint32 carry = 1;
			for (uint32 k = 0; k < LS_POLY_WORDS; k++) {
				uint64 v = (uint64)(uint32)~w[k] + carry;
				w[k] = (uint32)v;
				carry = (uint32)(v >> 32);
			}
		}
		memcpy(out->c[i], w, sizeof(w));
	}
	return 1;
}

/* 1 if every norm |f(a, b)| of the job fits the kernel's evaluation
   (LS_EVAL_WORDS words, signed): sum |c_i| A^i B^(d-i) < 2^(bits-2) */
static uint32
ls_norms_fit(mpz_poly_t *f, sieve_job_t *job)
{
	uint64 A = (uint64)MAX(job->min_a < 0 ? -job->min_a : job->min_a,
				job->max_a < 0 ? -job->max_a : job->max_a);
	mpz_t sum, term, t;
	uint32 ok;

	mpz_init(sum);
	mpz_init(term);
	mpz_init(t);
	for (uint32 i = 0; i <= f->degree; i++) {
		mpz_abs(term, f->coeff[i]);
		uint64_2gmp(A, t);
		mpz_pow_ui(t, t, i);
		mpz_mul(term, term, t);
		uint64_2gmp(job->max_b, t);
		mpz_pow_ui(t, t, f->degree - i);
		mpz_mul(term, term, t);
		mpz_add(sum, sum, term);
	}
	ok = (mpz_sizeinbase(sum, 2) < 32 * LS_EVAL_WORDS - 2);
	mpz_clear(sum);
	mpz_clear(term);
	mpz_clear(t);
	return ok;
}

static CUdeviceptr
ls_upload(const void *h, size_t bytes)
{
	CUdeviceptr d;

	CUDA_TRY(cuMemAlloc(&d, MAX(bytes, 4)))
	if (bytes)
		CUDA_TRY(cuMemcpyHtoD(d, h, bytes))
	return d;
}

/* one side's factor base on the GPU; 0 if it is not usable (unsorted,
   or too large for the packed bucket updates) */
static uint32
ls_side_init(gpu_side_t *g, sieve_t *fb, int64 min_a, uint32 num_lines)
{
	uint32 n = fb->fb_size, npw = fb->powers_fb_size;
	uint32 *p = (uint32 *)xmalloc(n * sizeof(uint32));
	uint32 *r = (uint32 *)xmalloc(n * sizeof(uint32));
	uint32 *rem0 = (uint32 *)xmalloc(n * sizeof(uint32));
	uint32 *pp = (uint32 *)xmalloc(MAX(npw, 1) * sizeof(uint32));
	uint32 *pr = (uint32 *)xmalloc(MAX(npw, 1) * sizeof(uint32));
	uint32 *po = (uint32 *)xmalloc(MAX(npw, 1) * sizeof(uint32));
	uint32 *prem = (uint32 *)xmalloc(MAX(npw, 1) * sizeof(uint32));
	uint32 nd = 0;

	for (uint32 i = 0; i < n; i++) {
		int64 rem;

		p[i] = fb->fb_entries[i].p;
		r[i] = fb->fb_entries[i].r;
		if (i > 0 && p[i] < p[i - 1])
			return 0;
		if (p[i] >= (1u << 31) / 2)
			return 0;
		if (p[i] <= LS_SEG)
			nd = i + 1;
		rem = (int64)p[i] - min_a % (int64)p[i];
		if (rem >= p[i])
			rem -= p[i];
		rem0[i] = (uint32)rem;
	}
	for (uint32 i = 0; i < npw; i++) {
		int64 rem;

		pp[i] = fb->fb_powers[i].p;
		pr[i] = fb->fb_powers[i].r;
		po[i] = fb->fb_powers[i].orig_p;
		rem = (int64)pp[i] - min_a % (int64)pp[i];
		if (rem >= pp[i])
			rem -= pp[i];
		prem[i] = (uint32)rem;
	}
	g->n = n;
	g->npw = npw;
	g->p = ls_upload(p, n * sizeof(uint32));
	g->r = ls_upload(r, n * sizeof(uint32));
	g->rem0 = ls_upload(rem0, n * sizeof(uint32));
	g->pw_p = ls_upload(pp, npw * sizeof(uint32));
	g->pw_r = ls_upload(pr, npw * sizeof(uint32));
	g->pw_orig = ls_upload(po, npw * sizeof(uint32));
	g->pw_rem0 = ls_upload(prem, npw * sizeof(uint32));
	CUDA_TRY(cuMemAlloc(&g->logp, MAX(n, 4)))
	CUDA_TRY(cuMemAlloc(&g->pw_logp, MAX(npw, 4)))
	g->h_logp = (uint8 *)xmalloc(n + npw + 4);
	CUDA_TRY(cuMemAlloc(&g->start, (size_t)num_lines * (n + npw) *
					sizeof(uint32)))
	CUDA_TRY(cuMemAlloc(&g->bucket, (size_t)num_lines * LS_SEGS *
				LS_BUCKET_CAP * sizeof(uint64)))
	CUDA_TRY(cuMemAlloc(&g->count, (size_t)num_lines * LS_SEGS *
					sizeof(uint32)))

	memset(&g->s, 0, sizeof(g->s));
	g->s.p = g->p;
	g->s.r = g->r;
	g->s.rem0 = g->rem0;
	g->s.logp = g->logp;
	g->s.pw_p = g->pw_p;
	g->s.pw_r = g->pw_r;
	g->s.pw_orig = g->pw_orig;
	g->s.pw_rem0 = g->pw_rem0;
	g->s.pw_logp = g->pw_logp;
	g->s.num_entries = n;
	g->s.num_powers = npw;
	g->s.num_direct = nd;
	g->s.small_size = fb->small_fb_size;
	g->side = ls_upload(&g->s, sizeof(g->s));
	free(p); free(r); free(rem0); free(pp); free(pr); free(po); free(prem);
	return 1;
}

static void
ls_side_free(gpu_side_t *g)
{
	CUdeviceptr *d[] = { &g->p, &g->r, &g->rem0, &g->logp, &g->pw_p,
			&g->pw_r, &g->pw_orig, &g->pw_rem0, &g->pw_logp,
			&g->side, &g->start, &g->bucket, &g->count };

	for (uint32 i = 0; i < sizeof(d) / sizeof(d[0]); i++)
		if (*d[i])
			CUDA_TRY(cuMemFree(*d[i]))
	free(g->h_logp);
}

/*------------------------------------------------------------------*/
/* host factoring of the candidates: do_one_factoring() of sieve_line.c
   with the GPU's resieved primes */

typedef struct {
	sieve_job_t *job;
	const ls_cand_t *cand;
	uint32 num_cand;
	const uint64 *lines;
	uint32 **small_start[2];	/* per line: starts of the small
					   entries, LS_SKIP if not sieved */
	uint32 next;			/* the next candidate to take */
	uint32 rels;
	pthread_mutex_t lock;		/* next, rels, the output */
} ls_work_t;

typedef struct {
	ls_work_t *w;
	mpz_t res[2], tmp;
	mpz_t e1, e2, e3;		/* ls_eval */
	pthread_t tid;
} ls_thread_t;

/* eval_poly() with this thread's temporaries (eval_poly keeps them in
   the polynomial, which the threads share) */
static void
ls_eval(ls_thread_t *t, mpz_t res, int64 a, uint64 b, mpz_poly_t *poly)
{
	uint32 d = poly->degree;

	int64_2gmp(a, t->e1);
	uint64_2gmp(b, t->e2);
	mpz_set(t->e3, t->e2);
	mpz_set(res, poly->coeff[d]);
	while (--d) {
		mpz_mul(res, res, t->e1);
		mpz_addmul(res, poly->coeff[d], t->e2);
		mpz_mul(t->e2, t->e2, t->e3);
	}
	mpz_mul(res, res, t->e1);
	mpz_addmul(res, poly->coeff[d], t->e2);
}

static void
ls_div(mpz_t res, mpz_t tmp, uint32 p)
{
	while (mpz_tdiv_q_ui(tmp, res, p) == 0)
		mpz_swap(res, tmp);
}

/* the start of entry i of a side on line b (ls_start of the kernel) */
static uint32
ls_host_start(sieve_t *fb, uint32 i, int64 min_a, uint64 b)
{
	uint32 p = fb->fb_entries[i].p, r = fb->fb_entries[i].r;
	uint32 bp = (uint32)(b % p);
	uint32 br = (uint32)((uint64)r * bp % p);
	int64 rem;

	if (br == 0 && (r == p || bp == 0))
		return LS_SKIP;
	rem = (int64)p - min_a % (int64)p;
	if (rem >= p)
		rem -= p;
	rem += br;
	if (rem >= p)
		rem -= p;
	if (b % 2 == 0) {
		if (rem % 2 == 0)
			rem += p;
		rem /= 2;
	}
	return (uint32)rem;
}

/* the trial factoring of one side (do_one_tf) */
static uint32
ls_tf(ls_thread_t *t, sieve_t *fb, uint32 side, const ls_cand_t *c,
	uint64 b, const uint32 *small_start, uint32 *factors, uint32 *num_out)
{
	sieve_job_t *job = t->w->job;
	mpz_ptr res = t->res[side];
	uint32 nf = 0, nlp = (side ? (c->nlp >> 4) : c->nlp) & 15;

	for (uint32 i = 0; i < fb->small_fb_size; i++) {
		uint32 p = fb->fb_entries[i].p, st = small_start[i];

		if (st == LS_SKIP || c->pos % p != st % p)
			continue;
		if (p >= MAX_SKIPPED_FACTOR)
			factors[nf++] = p;
		ls_div(res, t->tmp, p);
	}
	if (c->nf[side] <= LS_CAND_FACTORS) {
		for (uint32 i = 0; i < c->nf[side]; i++) {
			factors[nf++] = c->f[side][i];
			ls_div(res, t->tmp, c->f[side][i]);
		}
	}
	else {
		/* more than the GPU kept: find them here */
		for (uint32 i = fb->small_fb_size; i < fb->fb_size; i++) {
			uint32 p = fb->fb_entries[i].p;
			uint32 st = ls_host_start(fb, i, job->min_a, b);

			if (st == LS_SKIP || c->pos < st ||
			    (c->pos - st) % p != 0)
				continue;
			if (nf < 100)
				factors[nf++] = p;
			ls_div(res, t->tmp, p);
		}
	}
	for (uint32 i = 0; i < fb->proj_fb_size; i++) {
		uint32 p = fb->proj_entries[i].p;

		if (b % p == 0) {
			if (p >= MAX_SKIPPED_FACTOR)
				factors[nf++] = p;
			ls_div(res, t->tmp, p);
		}
	}

	if (nlp == 3) {
		if (mpz_cmp(res, fb->LP3_max) > 0)
			return 0;
		if (mpz_cmp(res, fb->LP3_min) < 0 &&
		    mpz_cmp(res, fb->LP2_max) > 0)
			return 0;
	}
	else {
		if (mpz_cmp(res, fb->LP2_max) > 0)
			return 0;
	}
	if (mpz_cmp_ui(res, fb->LP1_max) > 0 &&
	    mpz_cmp(res, fb->LP2_min) < 0)
		return 0;
	*num_out = nf;
	return 1;
}

static uint32
ls_composite_ok(mpz_ptr res)
{
	/* a base-2 pseudoprime test, as do_one_factoring: a cofactor
	   above 2^32 - 1 that passes is a prime: no use */
	if (mpz_cmp_ui(res, (uint32)(-1)) > 0) {
		mpz_t two, e;
		uint32 prp;

		mpz_init_set_ui(two, 2);
		mpz_init(e);
		mpz_sub_ui(e, res, 1);
		mpz_powm(two, two, e, res);
		prp = (mpz_cmp_ui(two, 1) == 0);
		mpz_clear(two);
		mpz_clear(e);
		if (prp)
			return 0;
	}
	return 1;
}

static void
ls_factor_one(ls_thread_t *t, const ls_cand_t *c)
{
	ls_work_t *w = t->w;
	sieve_job_t *job = w->job;
	sieve_t *rfb = &job->sieve_rfb, *afb = &job->sieve_afb;
	uint64 b = w->lines[c->line];
	int64 a = (b % 2 == 0) ? job->min_a + 2 * (int64)c->pos + 1 :
				 job->min_a + (int64)c->pos;
	uint32 factors_r[100], factors_a[100], nr = 0, na = 0;
	mpz_ptr small, large;

	ls_eval(t, t->res[1], a, b, afb->poly);
	mpz_abs(t->res[1], t->res[1]);
	if (!ls_tf(t, afb, 1, c, b, w->small_start[1][c->line],
			factors_a, &na))
		return;
	ls_eval(t, t->res[0], a, b, rfb->poly);
	mpz_abs(t->res[0], t->res[0]);
	if (!ls_tf(t, rfb, 0, c, b, w->small_start[0][c->line],
			factors_r, &nr))
		return;

	small = t->res[0];
	large = t->res[1];
	if (mpz_cmp(small, large) > 0) {
		small = t->res[1];
		large = t->res[0];
	}
	if (!ls_composite_ok(small) || !ls_composite_ok(large))
		return;

	pthread_mutex_lock(&w->lock);
	if (mpz_cmp_ui(t->res[0], (uint32)(-1)) < 0 &&
	    mpz_cmp_ui(t->res[1], (uint32)(-1)) < 0) {
		uint32 lp_r[MAX_LARGE_PRIMES], lp_a[MAX_LARGE_PRIMES];

		lp_r[0] = mpz_get_ui(t->res[0]);
		lp_a[0] = mpz_get_ui(t->res[1]);
		for (uint32 i = 1; i < MAX_LARGE_PRIMES; i++)
			lp_r[i] = lp_a[i] = 1;
		print_relation(&job->obj->savefile, a, b, factors_r, nr, lp_r,
				factors_a, na, lp_a);
		w->rels++;
	}
	else {
		relation_batch_add(a, b, factors_r, nr, t->res[0],
				factors_a, na, t->res[1],
				&job->relation_batch);
		if (job->relation_batch.num_relations >=
				job->relation_batch.target_relations) {
			double tb = ls_now();

			w->rels += relation_batch_run(&job->relation_batch);
			ls_t_batch += ls_now() - tb;
		}
	}
	pthread_mutex_unlock(&w->lock);
}

static void *
ls_thread_main(void *arg)
{
	ls_thread_t *t = (ls_thread_t *)arg;
	ls_work_t *w = t->w;

	for (;;) {
		uint32 i0, i1;

		pthread_mutex_lock(&w->lock);
		i0 = w->next;
		i1 = MIN(i0 + 256, w->num_cand);
		w->next = i1;
		pthread_mutex_unlock(&w->lock);
		if (i0 >= i1)
			break;
		for (uint32 i = i0; i < i1; i++)
			ls_factor_one(t, w->cand + i);
	}
	return NULL;
}

/*------------------------------------------------------------------*/
/* positions of line b: the CPU's whole blocks */
static uint64
ls_line_positions(sieve_job_t *job, uint64 b)
{
	uint64 step = (b % 2 == 0) ? 2 * LS_BLOCK : LS_BLOCK;
	uint64 len = (uint64)(job->max_a - job->min_a);

	return (len + step - 1) / step * LS_BLOCK;
}

/* the logarithm base of one side for lines from b (init_one_sieve) */
static void
ls_side_base(sieve_job_t *job, sieve_t *fb, gpu_side_t *g, uint64 b,
		int32 *tab, int32 *sc2, int32 *sc3)
{
	double base = get_log_base(fb->poly, job->min_a, job->max_a,
				b + LOG_UPDATE_RATE - 1);
	double lb = log(base);

	fb->base = base;
	fb->log_base = lb;
	*sc2 = (uint32)((double)fb->cutoff2 * M_LN2 / lb + 0.5);
	*sc3 = (uint32)((double)fb->cutoff3 * M_LN2 / lb + 0.5);
	fb->scaled_cutoff2 = *sc2;
	fb->scaled_cutoff3 = *sc3;
	for (uint32 i = 0; i < g->n; i++)
		g->h_logp[i] = (uint8)fplog(fb->fb_entries[i].p, lb);
	for (uint32 i = 0; i < g->npw; i++)
		g->h_logp[g->n + i] = (uint8)fplog(fb->fb_powers[i].orig_p, lb);
	CUDA_TRY(cuMemcpyHtoD(g->logp, g->h_logp, g->n))
	if (g->npw)
		CUDA_TRY(cuMemcpyHtoD(g->pw_logp, g->h_logp + g->n, g->npw))
	for (uint32 k = 0; k < LS_TAB; k++)
		tab[k] = (uint32)((double)k * M_LN2 / lb);
}

static uint8
ls_proj(sieve_t *fb, uint64 b)
{
	mpz_t t, g;
	uint32 common;

	mpz_init(t);
	mpz_init(g);
	uint64_2gmp(b, t);
	mpz_gcd(g, fb->poly->coeff[fb->poly->degree], t);
	common = (uint32)mpz_get_ui(g);
	mpz_clear(t);
	mpz_clear(g);
	return (uint8)fplog(common, fb->log_base);
}

/*------------------------------------------------------------------*/
/* the segment kernel for lines [line0, line0 + n) of chunk c0, into
   candidate buffer k, without waiting */
static void
ls_segments_launch(gpu_sieve_t *G, uint32 k, uint32 line0, uint32 n,
		uint64 c0, uint32 reset_overflow)
{
	void *args[14] = { &G->side[0].side, &G->side[1].side, &G->batch,
			&line0, &G->side[0].start, &G->side[1].start, &c0,
			&G->side[0].bucket, &G->side[0].count,
			&G->side[1].bucket, &G->side[1].count, &G->cand[k],
			&G->ncand[k], &G->overflow[k] };

	CUDA_TRY(cuMemsetD32Async(G->ncand[k], 0, 1, 0))
	if (reset_overflow)
		CUDA_TRY(cuMemsetD32Async(G->overflow[k], 0, 1, 0))
	CUDA_TRY(cuLaunchKernel(G->k_segment, n * LS_SEGS, 1, 1, LS_THREADS,
			1, 1, LS_SEG * sizeof(uint32), 0, args, NULL))
}

/* wait for it: the overflow code (3: the candidates did not fit), the
   candidates copied to host buffer k */
static uint32
ls_segments_wait(gpu_sieve_t *G, uint32 k, uint32 *ncand)
{
	uint32 ovf = 0;
	double t0 = ls_now();

	CUDA_TRY(cuMemcpyDtoH(ncand, G->ncand[k], sizeof(uint32)))
	CUDA_TRY(cuMemcpyDtoH(&ovf, G->overflow[k], sizeof(uint32)))
	*ncand = MIN(*ncand, G->max_cands);
	if (ovf == 0 && *ncand)
		CUDA_TRY(cuMemcpyDtoH(G->h_cand[k], G->cand[k],
				(size_t)*ncand * sizeof(ls_cand_t)))
	ls_t_gpu += ls_now() - t0;
	return ovf;
}

/* factor candidates on the host threads (and dump them) */
static void
ls_factor(ls_work_t *w, ls_thread_t *threads, uint32 nthreads,
	  const ls_cand_t *cand, uint32 ncand, FILE *dump)
{
	sieve_job_t *job = w->job;
	double t0 = ls_now();

	if (ncand == 0)
		return;
	ls_n_cand += ncand;
	if (dump != NULL) {
		for (uint32 k = 0; k < ncand; k++) {
			const ls_cand_t *c = cand + k;
			uint64 b = w->lines[c->line];
			int64 a = (b % 2 == 0) ?
				job->min_a + 2 * (int64)c->pos + 1 :
				job->min_a + (int64)c->pos;

			fprintf(dump, "%" PRIu64 " %" PRId64 " %u %u\n", b, a,
				c->nlp & 15, c->nlp >> 4);
		}
	}
	w->cand = cand;
	w->num_cand = ncand;
	w->next = 0;
	for (uint32 k = 0; k < nthreads; k++) {
		threads[k].w = w;
		pthread_create(&threads[k].tid, NULL, ls_thread_main,
				threads + k);
	}
	for (uint32 k = 0; k < nthreads; k++)
		pthread_join(threads[k].tid, NULL);
	ls_t_host += ls_now() - t0;
}

/*------------------------------------------------------------------*/
uint32
line_sieve_gpu(sieve_job_t *job, uint32 *relations_found,
		uint32 max_relations, uint64 *last_b)
{
	msieve_obj *obj = job->obj;
	const char *e = getenv("MSIEVE_GPU_SIEVE");
	const char *dump_name = getenv("MSIEVE_SIEVE_DUMP");
	FILE *dump = NULL;
	gpu_config_t config;
	uint32 devs[MAX_GPU], ndev, lr = 0, ls = 1, nthreads;
	gpu_info_t *info;
	const char *name;
	gpu_sieve_t G;
	ls_thread_t *threads;
	uint64 b0 = job->min_b, done = 0;
	uint32 rels = *relations_found;
	double t0 = ls_now();

	/* off as the siever choice in sieve_line.c reads it (atoi: "0",
	   "no", "off"); with strcmp "0" alone, "no" gave this GPU line
	   siever instead of msieve's CPU siever */
	if (e != NULL && atoi(e) == 0)
		return 0;
	memset(&G, 0, sizeof(G));
	gpu_init(&config);
#ifdef HAVE_MPI
	lr = obj->mpi_local_rank;
	ls = obj->mpi_local_size;
#endif
	ndev = gpu_select(&config, obj->which_gpu, lr, ls, devs, NULL);
	if (ndev == 0)
		return 0;
	info = config.info + devs[0];
	name = ls_ptx_name(info->compute_version_major,
			info->compute_version_minor);
	if (name == NULL)
		return 0;
	{
		int optin = 0;

		CUDA_TRY(cuDeviceGetAttribute(&optin,
			CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
			info->device_handle))
		if (optin < LS_SEG * 4 + 9000)
			return 0;
	}
	{
		ls_poly_t pr, pa;

		if (!ls_poly(job->sieve_rfb.poly, &pr) ||
		    !ls_poly(job->sieve_afb.poly, &pa) ||
		    !ls_norms_fit(job->sieve_rfb.poly, job) ||
		    !ls_norms_fit(job->sieve_afb.poly, job)) {
			logprintf(obj, "line sieving on the CPU: polynomial "
					"or norms too large for the GPU\n");
			return 0;
		}
		G.h_batch = (ls_batch_t *)xcalloc(1, sizeof(ls_batch_t));
		G.h_batch->poly_r = pr;
		G.h_batch->poly_a = pa;
	}

	CUDA_TRY(cuDevicePrimaryCtxRetain(&G.ctx, info->device_handle))
	CUDA_TRY(cuCtxPushCurrent(G.ctx))
	if (cuModuleLoad(&G.module, name) != CUDA_SUCCESS) {
		logprintf(obj, "line sieving on the CPU: cannot load %s\n",
				name);
		CUDA_TRY(cuCtxPopCurrent(NULL))
		CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
		free(G.h_batch);
		return 0;
	}
	CUDA_TRY(cuModuleGetFunction(&G.k_starts, G.module, "ls_starts"))
	CUDA_TRY(cuModuleGetFunction(&G.k_bucket, G.module, "ls_bucket"))
	CUDA_TRY(cuModuleGetFunction(&G.k_segment, G.module, "ls_segment"))
	CUDA_TRY(cuFuncSetAttribute(G.k_segment,
			CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
			LS_SEG * sizeof(uint32)))
	G.num_sm = (uint32)info->num_compute_units;

	if (!ls_side_init(&G.side[0], &job->sieve_rfb, job->min_a, LS_LINES) ||
	    !ls_side_init(&G.side[1], &job->sieve_afb, job->min_a, LS_LINES)) {
		logprintf(obj, "line sieving on the CPU: factor base not "
				"usable on the GPU\n");
		ls_side_free(&G.side[0]);
		ls_side_free(&G.side[1]);
		CUDA_TRY(cuModuleUnload(G.module))
		CUDA_TRY(cuCtxPopCurrent(NULL))
		CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
		free(G.h_batch);
		return 0;
	}

	G.max_cands = LS_MAX_CANDS;
	CUDA_TRY(cuMemAlloc(&G.batch, sizeof(ls_batch_t)))
	CUDA_TRY(cuMemAlloc(&G.lines, LS_LINES * sizeof(uint64)))
	CUDA_TRY(cuMemAlloc(&G.ltot, LS_LINES * sizeof(uint64)))
	CUDA_TRY(cuMemAlloc(&G.proj_r, LS_LINES))
	CUDA_TRY(cuMemAlloc(&G.proj_a, LS_LINES))
	for (uint32 k = 0; k < 2; k++) {
		CUDA_TRY(cuMemAlloc(&G.cand[k], (size_t)G.max_cands *
						sizeof(ls_cand_t)))
		CUDA_TRY(cuMemAlloc(&G.ncand[k], sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&G.overflow[k], sizeof(uint32)))
		CUDA_TRY(cuMemAllocHost((void **)&G.h_cand[k],
				(size_t)G.max_cands * sizeof(ls_cand_t)))
	}

	{
		const char *et = getenv("MSIEVE_SIEVE_THREADS");
		long ncpu = sysconf(_SC_NPROCESSORS_ONLN);

#ifdef HAVE_MPI
		ncpu /= MAX(1, (long)obj->mpi_local_size);
#endif
		nthreads = (et != NULL) ? (uint32)atoi(et) :
				(uint32)MAX(1, ncpu);
		nthreads = MAX(1, nthreads);
	}
	threads = (ls_thread_t *)xcalloc(nthreads, sizeof(ls_thread_t));
	for (uint32 k = 0; k < nthreads; k++) {
		mpz_init(threads[k].res[0]);
		mpz_init(threads[k].res[1]);
		mpz_init(threads[k].tmp);
		mpz_init(threads[k].e1);
		mpz_init(threads[k].e2);
		mpz_init(threads[k].e3);
	}
	if (dump_name != NULL)
		dump = fopen(dump_name, "w");

	logprintf(obj, "line sieving on GPU %u (%s), %u lines per batch, "
			"%u host threads\n", devs[0], info->name, LS_LINES,
			nthreads);

	{
		/* two batches of host line data: the pending candidates'
		   batch may be the previous one */
		uint64 lines[2][LS_LINES], ltot[LS_LINES];
		uint8 proj_r[LS_LINES], proj_a[LS_LINES];
		uint32 *small_mem[2][2];
		uint32 *small_start[2][2][LS_LINES];	/* [batch][side][line] */
		uint32 ns[2] = { job->sieve_rfb.small_fb_size,
				 job->sieve_afb.small_fb_size };
		ls_work_t w[2];			/* per batch buffer */
		uint32 bp = 0;			/* this batch's buffer */
		uint32 kbuf = 0;		/* the candidate buffer */
		uint32 pend = 0, pend_k = 0, pend_n = 0, pend_bp = 0;
		uint32 stop = 0;

		for (uint32 x = 0; x < 2; x++) {
			for (uint32 sd = 0; sd < 2; sd++) {
				small_mem[x][sd] = (uint32 *)xmalloc(LS_LINES *
						MAX(ns[sd], 1) * sizeof(uint32));
				for (uint32 l = 0; l < LS_LINES; l++)
					small_start[x][sd][l] =
						small_mem[x][sd] +
						(size_t)l * ns[sd];
			}
			memset(w + x, 0, sizeof(w[x]));
			pthread_mutex_init(&w[x].lock, NULL);
			w[x].job = job;
			w[x].lines = lines[x];
			w[x].small_start[0] = small_start[x][0];
			w[x].small_start[1] = small_start[x][1];
		}

		while (b0 + done <= job->max_b && !stop) {
			uint32 nl = LS_LINES;
			uint64 maxlt = 0;
			ls_batch_t *hb = G.h_batch;

			/* one logarithm base per batch, recomputed every
			   LOG_UPDATE_RATE lines as init_one_sieve does */
			nl = MIN(nl, LOG_UPDATE_RATE - (uint32)(done %
							LOG_UPDATE_RATE));
			if (job->max_b - (b0 + done) + 1 < nl)
				nl = (uint32)(job->max_b - (b0 + done) + 1);
			if (done % LOG_UPDATE_RATE == 0) {
				ls_side_base(job, &job->sieve_rfb, &G.side[0],
					b0 + done, hb->tab_r, &hb->sc2_r,
					&hb->sc3_r);
				ls_side_base(job, &job->sieve_afb, &G.side[1],
					b0 + done, hb->tab_a, &hb->sc2_a,
					&hb->sc3_a);
			}
			for (uint32 l = 0; l < nl; l++) {
				uint64 b = b0 + done + l;

				lines[bp][l] = b;
				ltot[l] = ls_line_positions(job, b);
				maxlt = MAX(maxlt, ltot[l]);
				proj_r[l] = ls_proj(&job->sieve_rfb, b);
				proj_a[l] = ls_proj(&job->sieve_afb, b);
				for (uint32 i = 0; i < ns[0]; i++)
					small_start[bp][0][l][i] = ls_host_start(
						&job->sieve_rfb, i, job->min_a,
						b);
				for (uint32 i = 0; i < ns[1]; i++)
					small_start[bp][1][l][i] = ls_host_start(
						&job->sieve_afb, i, job->min_a,
						b);
			}
			hb->num_lines = nl;
			hb->max_cands = G.max_cands;
			hb->min_a = job->min_a;
			hb->lines = G.lines;
			hb->ltot = G.ltot;
			hb->proj_r = G.proj_r;
			hb->proj_a = G.proj_a;
			CUDA_TRY(cuMemcpyHtoD(G.batch, hb, sizeof(ls_batch_t)))
			CUDA_TRY(cuMemcpyHtoD(G.lines, lines[bp],
						nl * sizeof(uint64)))
			CUDA_TRY(cuMemcpyHtoD(G.ltot, ltot, nl * sizeof(uint64)))
			CUDA_TRY(cuMemcpyHtoD(G.proj_r, proj_r, nl))
			CUDA_TRY(cuMemcpyHtoD(G.proj_a, proj_a, nl))

			for (uint32 sd = 0; sd < 2; sd++) {
				gpu_side_t *g = G.side + sd;
				uint32 nt = (g->n + g->npw) * nl;
				uint32 grid = MIN((nt + 255) / 256,
						G.num_sm * 32);
				void *args[4] = { &g->side, &G.lines, &nl,
						&g->start };

				CUDA_TRY(cuLaunchKernel(G.k_starts, grid, 1, 1,
					256, 1, 1, 0, 0, args, NULL))
			}
			for (uint64 c0 = 0; c0 < maxlt; c0 += LS_CHUNK) {
				uint32 ncand = 0, ovf;

				/* this chunk on the GPU ... */
				CUDA_TRY(cuMemsetD32Async(G.overflow[kbuf], 0,
							1, 0))
				for (uint32 sd = 0; sd < 2; sd++) {
					gpu_side_t *g = G.side + sd;
					uint32 nt = (g->n - g->s.num_direct) * nl;
					uint32 grid = MAX(1, MIN((nt + 255) / 256,
							G.num_sm * 32));
					void *args[7] = { &g->side, &G.batch,
						&g->start, &c0, &g->bucket,
						&g->count, &G.overflow[kbuf] };

					CUDA_TRY(cuMemsetD32Async(g->count, 0,
						LS_LINES * LS_SEGS, 0))
					CUDA_TRY(cuLaunchKernel(G.k_bucket,
						grid, 1, 1, 256, 1, 1, 0, 0,
						args, NULL))
				}
				ls_segments_launch(&G, kbuf, 0, nl, c0, 0);

				/* ... while the host factors the last one */
				if (pend) {
					ls_factor(w + pend_bp, threads, nthreads,
						G.h_cand[pend_k], pend_n, dump);
					rels += w[pend_bp].rels;
					w[pend_bp].rels = 0;
					pend = 0;
				}

				ovf = ls_segments_wait(&G, kbuf, &ncand);
				if (ovf == 1) {
					printf("error: GPU line sieve bucket "
						"overflow at b = %" PRIu64 "\n",
						b0 + done);
					exit(-1);
				}
				if (ovf == 3) {
					/* too many candidates: the lines one
					   at a time, without overlap */
					for (uint32 l = 0; l < nl; l++) {
						ls_segments_launch(&G, kbuf, l,
								1, c0, 1);
						if (ls_segments_wait(&G, kbuf,
								&ncand)) {
							printf("error: GPU line "
							"sieve candidate "
							"overflow\n");
							exit(-1);
						}
						ls_factor(w + bp, threads,
							nthreads,
							G.h_cand[kbuf], ncand,
							dump);
						rels += w[bp].rels;
						w[bp].rels = 0;
					}
				}
				else {
					pend = 1;
					pend_k = kbuf;
					pend_n = ncand;
					pend_bp = bp;
					kbuf ^= 1;
				}
			}
			done += nl;
			bp ^= 1;

			if (obj->flags & (MSIEVE_FLAG_USE_LOGFILE |
					  MSIEVE_FLAG_LOG_TO_STDOUT)) {
				fprintf(stderr, "b = %" PRIu64 ", %u complete / "
					"%u batched relations (need %u)\r",
					b0 + done - 1, rels,
					job->relation_batch.num_relations,
					max_relations);
				fflush(stderr);
			}
			if (rels >= max_relations ||
			    (obj->flags & MSIEVE_FLAG_STOP_SIEVING))
				stop = 1;
		}
		if (pend) {
			ls_factor(w + pend_bp, threads, nthreads,
				G.h_cand[pend_k], pend_n, dump);
			rels += w[pend_bp].rels;
		}

		if (job->relation_batch.num_relations > 0) {
			double tb = ls_now();

			rels += relation_batch_run(&job->relation_batch);
			ls_t_batch += ls_now() - tb;
		}
		if (obj->flags & (MSIEVE_FLAG_USE_LOGFILE |
				  MSIEVE_FLAG_LOG_TO_STDOUT))
			fprintf(stderr, "\n");
		logprintf(obj, "completed b = %" PRIu64 ", found %u "
				"relations\n", b0 + done - 1, rels);
		logprintf(obj, "GPU line sieving: %" PRIu64 " lines in "
				"%.1f s (waiting for the GPU %.1f s, %" PRIu64
				" candidates factored in %.1f s, batch "
				"factoring %.1f s)\n", done, ls_now() - t0,
				ls_t_gpu, ls_n_cand, ls_t_host, ls_t_batch);
		for (uint32 x = 0; x < 2; x++) {
			pthread_mutex_destroy(&w[x].lock);
			free(small_mem[x][0]);
			free(small_mem[x][1]);
		}
	}

	for (uint32 k = 0; k < nthreads; k++) {
		mpz_clear(threads[k].res[0]);
		mpz_clear(threads[k].res[1]);
		mpz_clear(threads[k].tmp);
		mpz_clear(threads[k].e1);
		mpz_clear(threads[k].e2);
		mpz_clear(threads[k].e3);
	}
	free(threads);
	if (dump != NULL)
		fclose(dump);
	ls_side_free(&G.side[0]);
	ls_side_free(&G.side[1]);
	CUDA_TRY(cuMemFree(G.batch))
	CUDA_TRY(cuMemFree(G.lines))
	CUDA_TRY(cuMemFree(G.ltot))
	CUDA_TRY(cuMemFree(G.proj_r))
	CUDA_TRY(cuMemFree(G.proj_a))
	for (uint32 k = 0; k < 2; k++) {
		CUDA_TRY(cuMemFree(G.cand[k]))
		CUDA_TRY(cuMemFree(G.ncand[k]))
		CUDA_TRY(cuMemFree(G.overflow[k]))
		CUDA_TRY(cuMemFreeHost(G.h_cand[k]))
	}
	CUDA_TRY(cuModuleUnload(G.module))
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
	free(G.h_batch);

	*relations_found = rels;
	*last_b = b0 + done - 1;
	return 1;
}
