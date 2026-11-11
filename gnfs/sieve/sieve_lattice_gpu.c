/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* The NFS lattice siever on the GPU (kernels: lasieve_core.cu). msieve
   has no lattice siever of its own; this one sieves special-q lattices
   on the algebraic side, q from the -ns range (default: above the
   algebraic factor base), each root r of f mod q a lattice
   a = a0 i + a1 j, b = b0 i + b1 j (skew-reduced), I x J positions.
   The GPU sieves batches of lattices and returns the candidates with
   their primes >= LA_RESIEVE_MIN; host threads divide them out (and q,
   and the primes below LA_RESIEVE_MIN by trial division) and keep the
   relations whose cofactors pass msieve's large prime bounds, as the
   line siever does. Relations go to the savefile in msieve's format;
   duplicates between special-q are removed by msieve's filtering.

   Settings: MSIEVE_LATTICE_LOGI (log2 of I, default 12; J = I / 2),
   MSIEVE_LATTICE_TINY (primes below it are not sieved, default 32;
   their expected logs lower the threshold), MSIEVE_LATTICE_FILTER=0
   (no rational test on the GPU),
   MSIEVE_LATTICE_BATCH (lattices per batch), MSIEVE_SIEVE_THREADS,
   MSIEVE_LATTICE_CHECK=1 (compare the GPU's sieve sums of the first
   lattice with a direct computation on the host),
   MSIEVE_LATTICE_VERIFY=1 (check every cofactor for a factor base
   prime the GPU did not report), MSIEVE_BATCH_DUMP=<file> (the
   cofactors sent to batch factoring, for test/sieve/batchreplay.c) */

#include "sieve_line_int.h"
#include <cuda_xface.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include "lasieve_core.h"
#include "relstore.h"

/* candidate buffers, per lattice: the c95 had ~31,000 candidates per
   lattice (I = 4096) and ~6 resieved primes per candidate; a batch that
   overflows is split */
#define LA_CANDS 49152

static double
la_now(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static const char *
la_ptx_name(int32 major, int32 minor)
{
	if (major >= 12) return "lasieve_core_sm120.ptx";
	if (major >= 10) return "lasieve_core_sm100.ptx";
	if (major == 9) return "lasieve_core_sm90.ptx";
	if (major == 8)
		return (minor < 6) ? "lasieve_core_sm80.ptx" :
			(minor < 9) ? "lasieve_core_sm86.ptx" :
				      "lasieve_core_sm89.ptx";
	if (major == 7)
		return (minor < 5) ? "lasieve_core_sm70.ptx" :
				     "lasieve_core_sm75.ptx";
	if (major == 6)
		return (minor < 1) ? "lasieve_core_sm60.ptx" :
				     "lasieve_core_sm61.ptx";
	return NULL;
}

/*------------------------------------------------------------------*/
typedef struct {
	CUdeviceptr p, r, logp, side;
	la_side_t s;
	uint32 *h_p, *h_r;
	uint16 *h_logp;
} la_gside_t;

typedef struct {
	CUcontext ctx;
	CUmodule module;
	CUfunction k_roots, k_large, k_tile, k_rowlogs;
	uint32 num_sm;
	la_gside_t side[2];		/* 0 rational, 1 algebraic */
	CUdeviceptr d_sides;		/* la_side_t[2] */
	CUdeviceptr batch, lat, rho, sieve, bitmap, count, cands, out, out2;
	CUdeviceptr hkey, hval, rowlog;
	la_batch_t h_batch;
	la_cand_t *h_cands;		/* the slot la_gpu_batch fills */
	uint32 rat_filter;		/* la_filter usable at all */
	double r0_abs, r1_abs;		/* |rational coefficients| */
	CUfunction k_filter, k_cofactor;
	uint32 cof_filter;		/* la_cofactor usable at all */
	double acoef_abs[LA_MAX_DEG + 1];	/* |algebraic coefficients| */
	uint32 max_lat, logI, J;
	uint32 tiny;			/* primes below this are not sieved */
	uint64 positions;		/* per lattice */
} la_gpu_t;

static uint32
la_side_init(la_gside_t *g, sieve_t *fb, uint32 I, uint32 tiny)
{
	double bias = 0;

	uint32 n = fb->fb_size;		/* the entries, as init_one_fb kept them */

	g->h_p = (uint32 *)xmalloc(MAX(n, 1) * sizeof(uint32));
	g->h_r = (uint32 *)xmalloc(MAX(n, 1) * sizeof(uint32));
	g->h_logp = (uint16 *)xmalloc(MAX(n, 1) * sizeof(uint16));
	memset(&g->s, 0, sizeof(g->s));
	for (uint32 i = 0; i < n; i++) {
		uint32 p = fb->fb_entries[i].p;

		if ((i > 0 && p < g->h_p[i - 1]) || p >= (1u << 31))
			return 0;
		g->h_p[i] = p;
		g->h_r[i] = fb->fb_entries[i].r;
		g->h_logp[i] = (uint16)floor(LA_SCALE * log2((double)p));
		if (p < I)
			g->s.n_small = i + 1;
		if (p < LA_RESIEVE_MIN)
			g->s.n_noresieve = i + 1;
		if (p < tiny) {
			/* a root divides a random coprime (a, b) with
			   probability p / (p^2 - 1) */
			g->s.n_tiny = i + 1;
			bias += g->h_logp[i] * (double)p / ((double)p * p - 1);
		}
	}
	g->s.n = n;
	g->s.tiny_bias = (int32_t)(bias + 0.5);
	CUDA_TRY(cuMemAlloc(&g->p, MAX(n, 1) * sizeof(uint32)))
	CUDA_TRY(cuMemAlloc(&g->r, MAX(n, 1) * sizeof(uint32)))
	CUDA_TRY(cuMemAlloc(&g->logp, MAX(n, 1) * sizeof(uint16)))
	CUDA_TRY(cuMemcpyHtoD(g->p, g->h_p, n * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(g->r, g->h_r, n * sizeof(uint32)))
	CUDA_TRY(cuMemcpyHtoD(g->logp, g->h_logp, n * sizeof(uint16)))
	g->s.p = g->p;
	g->s.r = g->r;
	g->s.logp = g->logp;
	return 1;
}

/*------------------------------------------------------------------*/
/* the special-q lattice of (q, r): a basis of {(a, b): a = r b mod q},
   Gauss-reduced for the norm a^2 + (s b)^2 */
static void
la_reduce(uint32 q, uint32 r, double skew, la_lat_t *L)
{
	int64 u[2] = { (int64)q, 0 }, v[2] = { (int64)r, 1 };
	double s2 = skew * skew;

	for (;;) {
		double nu = (double)u[0] * u[0] + s2 * (double)u[1] * u[1];
		double nv = (double)v[0] * v[0] + s2 * (double)v[1] * v[1];
		double d;
		int64 mu;

		if (nv < nu) {
			int64 t0 = u[0], t1 = u[1];

			u[0] = v[0]; u[1] = v[1];
			v[0] = t0; v[1] = t1;
			d = nu; nu = nv; nv = d;
		}
		d = (double)u[0] * v[0] + s2 * (double)u[1] * v[1];
		mu = (int64)floor(d / nu + 0.5);
		if (mu == 0)
			break;
		v[0] -= mu * u[0];
		v[1] -= mu * u[1];
	}
	L->a0 = u[0];
	L->b0 = u[1];
	L->a1 = v[0];
	L->b1 = v[1];
	L->q = q;
	L->r = r;
	L->log2q = log2((double)q);
}

static uint32
la_is_prime(uint32 n)
{
	if (n < 2)
		return 0;
	if (n % 2 == 0)
		return n == 2;
	for (uint32 d = 3; (uint64)d * d <= n; d += 2) {
		if (n % d == 0)
			return 0;
	}
	return 1;
}

/*------------------------------------------------------------------*/
/* host factoring of the candidates */

static uint32 la_small_primes[64], la_num_small;	/* < LA_RESIEVE_MIN */

/* MSIEVE_LATTICE_VERIFY=1: every cofactor is checked for a factor base
   prime >= LA_RESIEVE_MIN the GPU did not report (there must be none) */
static mpz_t la_vfy_prod[2];
static uint32 la_vfy_on, la_vfy_bad;

/* MSIEVE_BATCH_DUMP=<file>: every cofactor pair sent to batch factoring
   ("a b rational algebraic"), for test/sieve/batchreplay.c */
static FILE *la_dump;
static double la_t_gpu, la_t_host, la_t_batch, la_t_wait;
static uint64 la_n_cand, la_n_out, la_n_rat;	/* found, to the host,
						   past la_filter */
static uint64 la_stat[6];	/* rejected: rational LP, algebraic LP,
				   Fermat; printed; batched; - */

typedef struct {
	sieve_job_t *job;
	const la_lat_t *lat;
	const la_cand_t *cand;
	const uint32 *fbp[2];	/* the factor base primes, for candidates
				   whose resieved primes did not all fit */
	uint32 fbn[2];
	uint32 num_cand;
	uint32 logI;
	uint32 next;
	uint32 rels;
	pthread_mutex_t lock;
} la_work_t;

typedef struct {
	la_work_t *w;
	mpz_t res[2], tmp, e1, e2, e3;
	pthread_t tid;
} la_thread_t;

static void
la_eval(la_thread_t *t, mpz_t res, int64 a, uint64 b, mpz_poly_t *poly)
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

static uint32
la_div(mpz_t res, mpz_t tmp, uint32 p)
{
	uint32 k = 0;

	while (mpz_tdiv_q_ui(tmp, res, p) == 0) {
		mpz_swap(res, tmp);
		k++;
	}
	return k;
}

/* the cofactor of one side against msieve's large prime bounds */
static uint32
la_lp_ok(sieve_t *fb, mpz_ptr res, uint32 nlp)
{
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
	return 1;
}

static uint32
la_composite_ok(mpz_ptr res)
{
	/* a base-2 pseudoprime cofactor above 2^32 - 1 is a prime that
	   is too large: no use */
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

/* batch factoring in the background: when the batch is full it goes
   to la_bq_main and the host goes on with a second one (waiting only if
   the previous run has not finished). Relations from the host threads
   and from the batch runs reach the savefile under one lock */
static pthread_mutex_t la_save_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t la_bq_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t la_bq_cond = PTHREAD_COND_INITIALIZER;
static relation_batch_t la_spare;	/* the other batch */
static relation_batch_t *la_bq;		/* running (la_spare), or NULL */
static uint32 la_bq_quit, la_bq_rels;
static uint64 la_bq_in, la_bq_out;	/* cofactors run, relations found */
static pthread_t la_bq_tid;

static void
la_print_locked(savefile_t *savefile, int64 a, uint64 b,
		uint32 *factors_r, uint32 num_factors_r,
		uint32 lp_r[MAX_LARGE_PRIMES],
		uint32 *factors_a, uint32 num_factors_a,
		uint32 lp_a[MAX_LARGE_PRIMES])
{
	char buf[LINE_BUF_SIZE];

	format_relation(buf, a, b, factors_r, num_factors_r, lp_r,
			factors_a, num_factors_a, lp_a);
	pthread_mutex_lock(&la_save_lock);
	savefile_write_line(savefile, buf);
	relstore_add_line((relstore_t *)savefile->relstore, buf);
	pthread_mutex_unlock(&la_save_lock);
}

static void *
la_bq_main(void *arg)
{
	(void)arg;
	pthread_mutex_lock(&la_bq_lock);
	for (;;) {
		relation_batch_t *rb;
		double t0;
		uint32 n;

		while (la_bq == NULL && !la_bq_quit)
			pthread_cond_wait(&la_bq_cond, &la_bq_lock);
		if (la_bq == NULL)
			break;
		rb = la_bq;
		la_bq_in += rb->num_relations;
		pthread_mutex_unlock(&la_bq_lock);
		t0 = la_now();
		n = relation_batch_run(rb);
		pthread_mutex_lock(&la_bq_lock);
		la_bq_rels += n;
		la_bq_out += n;
		la_t_batch += la_now() - t0;
		la_bq = NULL;
		pthread_cond_broadcast(&la_bq_cond);
	}
	pthread_mutex_unlock(&la_bq_lock);
	return NULL;
}

static void
la_bq_start(sieve_job_t *job)
{
	relation_batch_t *rb = &job->relation_batch;

	la_spare = *rb;		/* shares the prime product (read only) */
	la_spare.num_relations = la_spare.num_factors = 0;
	la_spare.relations = (cofactor_t *)xmalloc(rb->num_relations_alloc *
						sizeof(cofactor_t));
	la_spare.factors = (uint32 *)xmalloc(rb->num_factors_alloc *
						sizeof(uint32));
	rb->print_relation = la_spare.print_relation = la_print_locked;
	la_bq = NULL;
	la_bq_quit = 0;
	la_bq_rels = 0;
	pthread_create(&la_bq_tid, NULL, la_bq_main, NULL);
}

/* the full job->relation_batch to the background, the empty one back */
static void
la_bq_submit(sieve_job_t *job)
{
	relation_batch_t tmp;

	pthread_mutex_lock(&la_bq_lock);
	while (la_bq != NULL)
		pthread_cond_wait(&la_bq_cond, &la_bq_lock);
	tmp = la_spare;
	la_spare = job->relation_batch;
	job->relation_batch = tmp;
	la_bq = &la_spare;
	pthread_cond_broadcast(&la_bq_cond);
	pthread_mutex_unlock(&la_bq_lock);
}

/* the relations of the finished background runs since the last call
   (wait: after the running one finishes) */
static uint32
la_bq_take(uint32 wait)
{
	uint32 n;

	pthread_mutex_lock(&la_bq_lock);
	while (wait && la_bq != NULL)
		pthread_cond_wait(&la_bq_cond, &la_bq_lock);
	n = la_bq_rels;
	la_bq_rels = 0;
	pthread_mutex_unlock(&la_bq_lock);
	return n;
}

/* the relations the batched cofactors not run yet should give, at the
   rate of the runs so far (0 before the first) */
static double
la_bq_estimate(sieve_job_t *job)
{
	double est;

	pthread_mutex_lock(&la_bq_lock);
	est = (double)job->relation_batch.num_relations +
		(la_bq != NULL ? la_spare.num_relations : 0);
	est *= la_bq_in ? (double)la_bq_out / la_bq_in : 0;
	pthread_mutex_unlock(&la_bq_lock);
	return est;
}

static void
la_bq_stop(void)
{
	pthread_mutex_lock(&la_bq_lock);
	la_bq_quit = 1;
	pthread_cond_broadcast(&la_bq_cond);
	pthread_mutex_unlock(&la_bq_lock);
	pthread_join(la_bq_tid, NULL);
	free(la_spare.relations);
	free(la_spare.factors);
}

static void
la_factor_one(la_thread_t *t, uint32 k)
{
	la_work_t *w = t->w;
	sieve_job_t *job = w->job;
	sieve_t *sfb[2] = { &job->sieve_rfb, &job->sieve_afb };
	const la_cand_t *c = w->cand + k;
	const la_lat_t *L = w->lat + (uint32)(c->key >> 32);
	uint32 factors[2][100], nf[2] = { 0, 0 };

	for (uint32 s = 0; s < 2; s++) {
		mpz_ptr res = t->res[s];

		la_eval(t, res, c->a, (uint64)c->b, sfb[s]->poly);
		mpz_abs(res, res);
		if (mpz_sgn(res) == 0)
			return;
		if (s == 1) {
			if (la_div(res, t->tmp, L->q) == 0)
				return;		/* cannot happen */
			factors[1][nf[1]++] = L->q;
		}
		if (c->nf <= LA_CF &&	/* i = 0: see la_filter */
		    ((uint32)c->key & ((1u << w->logI) - 1)) !=
		    (1u << w->logI) / 2) {
			for (uint32 h = 0; h < c->nf; h++) {
				uint32 p = c->f[h] & 0x7fffffff;

				if ((c->f[h] >> 31) != s)
					continue;
				if (la_div(res, t->tmp, p) && nf[s] < 90)
					factors[s][nf[s]++] = p;
			}
		}
		else {		/* every factor base prime >= LA_RESIEVE_MIN */
			for (uint32 i = 0; i < w->fbn[s]; i++) {
				uint32 p = w->fbp[s][i];

				if (p < LA_RESIEVE_MIN ||
				    (i > 0 && p == w->fbp[s][i - 1]))
					continue;
				if (mpz_divisible_ui_p(res, p) &&
				    la_div(res, t->tmp, p) && nf[s] < 90)
					factors[s][nf[s]++] = p;
			}
		}
		{
			/* the u = 0 entries (rows j = 0 mod p) are not
			   resieved: their prime divides j */
			uint32 pj = la_jprime((uint32)c->key >> w->logI);

			if (pj > 1 && la_div(res, t->tmp, pj) && nf[s] < 90)
				factors[s][nf[s]++] = pj;
		}
		for (uint32 i = 0; i < la_num_small; i++)
			la_div(res, t->tmp, la_small_primes[i]);
		if (!la_lp_ok(sfb[s], res, (c->nlp >> (4 * s)) & 15)) {
			__sync_fetch_and_add(la_stat + s, 1);
			return;
		}
	}
	{
		mpz_ptr small = t->res[0], large = t->res[1];

		if (mpz_cmp(small, large) > 0) {
			small = t->res[1];
			large = t->res[0];
		}
		if (!la_composite_ok(small) || !la_composite_ok(large)) {
			__sync_fetch_and_add(la_stat + 2, 1);
			return;
		}
	}

	if (la_vfy_on) {
		for (uint32 s = 0; s < 2; s++) {
			mpz_gcd(t->tmp, la_vfy_prod[s], t->res[s]);
			if (mpz_cmp_ui(t->tmp, 1) != 0 &&
			    __sync_fetch_and_add(&la_vfy_bad, 1) < 5)
				gmp_fprintf(stderr, "lattice verify: q %u r %u "
					"j %u x %u side %u: cofactor has "
					"factor base prime(s) %Zd\n", L->q,
					L->r, (uint32)c->key >> w->logI,
					(uint32)c->key & ((1u << w->logI) - 1),
					s, t->tmp);
		}
	}
	pthread_mutex_lock(&w->lock);
	if (mpz_cmp_ui(t->res[0], (uint32)(-1)) < 0 &&
	    mpz_cmp_ui(t->res[1], (uint32)(-1)) < 0) {
		uint32 lp_r[MAX_LARGE_PRIMES], lp_a[MAX_LARGE_PRIMES];

		lp_r[0] = mpz_get_ui(t->res[0]);
		lp_a[0] = mpz_get_ui(t->res[1]);
		for (uint32 i = 1; i < MAX_LARGE_PRIMES; i++)
			lp_r[i] = lp_a[i] = 1;
		la_print_locked(&job->obj->savefile, c->a, (uint64)c->b,
				factors[0], nf[0], lp_r,
				factors[1], nf[1], lp_a);
		w->rels++;
		la_stat[3]++;
	}
	else {
		la_stat[4]++;
		if (la_dump != NULL)
			gmp_fprintf(la_dump, "%" PRId64 " %" PRIu64 " %Zd %Zd\n",
					c->a, (uint64)c->b, t->res[0],
					t->res[1]);
		relation_batch_add(c->a, (uint64)c->b, factors[0], nf[0],
				t->res[0], factors[1], nf[1], t->res[1],
				&job->relation_batch);
		if (job->relation_batch.num_relations >=
				job->relation_batch.target_relations)
			la_bq_submit(job);
	}
	pthread_mutex_unlock(&w->lock);
}

static void *
la_thread_main(void *arg)
{
	la_thread_t *t = (la_thread_t *)arg;
	la_work_t *w = t->w;

	for (;;) {
		uint32 k0, k1;

		pthread_mutex_lock(&w->lock);
		k0 = w->next;
		w->next = MIN(w->num_cand, k0 + 64);
		k1 = w->next;
		pthread_mutex_unlock(&w->lock);
		if (k0 >= k1)
			break;
		for (uint32 k = k0; k < k1; k++)
			la_factor_one(t, k);
	}
	return NULL;
}

/*------------------------------------------------------------------*/
/* sieve lattices [0, n) of lat on the GPU; 0 if the candidates did
   not fit. The candidates passing la_filter are in G->h_cands */
static uint32
la_gpu_batch(la_gpu_t *G, const la_lat_t *lat, uint32 n,
		uint32 *ncand, uint32 *nout, uint32 debug)
{
	la_batch_t *hb = &G->h_batch;
	uint32 cnt[LA_N_COUNTS];
	uint32 tpos = MAX(LA_TILE_POS, 1u << G->logI);
	uint32 tiles = G->J / (tpos >> G->logI);
	uint32 resieve;
	double t0 = la_now();

	hb->num_lat = n;
	/* half the old 16384 positions: three blocks of la_tile an SM
	   instead of one (64 KB of shared memory each) */
	hb->tile_pos = tpos;
	hb->debug_sums = debug;
	hb->rat_filter = G->rat_filter;
	if (hb->rat_filter) {
		/* the rational norms must stay below 2^126 */
		double A = 0, B = 0, I2 = (double)(1u << G->logI) / 2;

		for (uint32 k = 0; k < n; k++) {
			A = MAX(A, fabs((double)lat[k].a0) * I2 +
					fabs((double)lat[k].a1) * G->J);
			B = MAX(B, fabs((double)lat[k].b0) * I2 +
					fabs((double)lat[k].b1) * G->J);
		}
		if (G->r1_abs * A + G->r0_abs * B > ldexp(1.0, 125))
			hb->rat_filter = 0;

		/* la_cofactor: |F(a, b)| <= sum |c_i| A^i B^(d-i) must stay
		   below 2^(LA_COF_BITS - 3) */
		hb->cof_filter = 0;
		if (G->cof_filter && hb->rat_filter) {
			double s = 0;

			for (uint32 i = 0; i <= hb->deg[1]; i++)
				s += G->acoef_abs[i] * pow(A, i) *
					pow(B, hb->deg[1] - i);
			hb->cof_filter = (s < ldexp(1.0, LA_COF_BITS - 3));
		}
	}
	hb->one_sieve = !debug;
	CUDA_TRY(cuMemcpyHtoD(G->lat, lat, n * sizeof(la_lat_t)))
	CUDA_TRY(cuMemcpyHtoD(G->batch, hb, sizeof(la_batch_t)))
	CUDA_TRY(cuMemsetD32(G->sieve, 0, (hb->one_sieve ? 1 : n) *
				G->positions))
	CUDA_TRY(cuMemsetD32(G->bitmap, 0, n * (G->positions / 32)))
	CUDA_TRY(cuMemsetD32(G->hkey, 0xffffffff, 2 * (hb->hmask + 1)))
	CUDA_TRY(cuMemsetD32(G->count, 0, LA_N_COUNTS))
	{
		void *args[2] = { &G->d_sides, &G->batch };

		CUDA_TRY(cuLaunchKernel(G->k_roots, G->num_sm * 16, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
		CUDA_TRY(cuLaunchKernel(G->k_rowlogs, G->num_sm * 4, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
	}
	resieve = 0;
	for (uint32 k = 0; k < n; k++) {	/* one lattice at a time */
		uint32 one = 1;
		void *args[5] = { &G->d_sides, &G->batch, &resieve, &k, &one };

		CUDA_TRY(cuLaunchKernel(G->k_large, G->num_sm * 16, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
		if (hb->one_sieve) {	/* its small primes, while in L2 */
			void *targs[3] = { &G->d_sides, &G->batch, &k };

			CUDA_TRY(cuLaunchKernel(G->k_tile, tiles, 1, 1,
					LA_THREADS, 1, 1,
					tpos * sizeof(uint32), 0, targs,
					NULL))
		}
	}
	if (!hb->one_sieve) {
		uint32 zero = 0;
		void *args[3] = { &G->d_sides, &G->batch, &zero };

		CUDA_TRY(cuLaunchKernel(G->k_tile, n * tiles, 1, 1,
				LA_THREADS, 1, 1, tpos * sizeof(uint32), 0,
				args, NULL))
	}
	resieve = 1;
	{
		uint32 zero = 0;
		void *args[5] = { &G->d_sides, &G->batch, &resieve, &zero, &n };

		CUDA_TRY(cuLaunchKernel(G->k_large, G->num_sm * 16, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
	}
	{
		void *args[1] = { &G->batch };

		CUDA_TRY(cuLaunchKernel(G->k_filter, G->num_sm * 8, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
	}
	if (hb->cof_filter) {
		void *args[1] = { &G->batch };

		CUDA_TRY(cuLaunchKernel(G->k_cofactor, G->num_sm * 16, 1, 1,
				256, 1, 1, 0, 0, args, NULL))
	}
	CUDA_TRY(cuMemcpyDtoH(cnt, G->count, sizeof(cnt)))
	la_t_gpu += la_now() - t0;
	if (cnt[LA_N_OVERFLOW])
		return 0;
	*ncand = cnt[LA_N_CANDS];
	la_n_rat += cnt[LA_N_OUT];
	if (hb->cof_filter) {
		*nout = cnt[LA_N_OUT2];
		la_stat[1] += cnt[LA_N_ALG];
		la_stat[2] += cnt[LA_N_FERMAT];
	}
	else {
		*nout = cnt[LA_N_OUT];
	}
	if (*nout)
		CUDA_TRY(cuMemcpyDtoH(G->h_cands,
				hb->cof_filter ? G->out2 : G->out,
				*nout * sizeof(la_cand_t)))
	return 1;
}

/* the host side runs while the GPU sieves the next batch: two slots
   (candidates, a copy of the lattices, the work), one host job at a
   time (they share the batch factoring and the savefile), the GPU
   filling the slot the running job does not use */
typedef struct {
	la_cand_t *cands;		/* pinned */
	la_lat_t *lat;
	la_work_t w;
	double t0;
} la_slot_t;

static la_slot_t la_slots[2];
static int la_running = -1;		/* slot of the running job */
static uint32 la_fill;			/* slot la_gpu_batch fills next */
static la_thread_t *la_threads;
static uint32 la_nthreads;

/* wait for the running host job; its relations */
static uint32
la_host_join(void)
{
	la_slot_t *sl;
	double t0 = la_now();

	if (la_running < 0)
		return 0;
	sl = la_slots + la_running;
	for (uint32 k = 0; k < la_nthreads; k++)
		pthread_join(la_threads[k].tid, NULL);
	pthread_mutex_destroy(&sl->w.lock);
	la_t_wait += la_now() - t0;
	la_t_host += la_now() - sl->t0;
	la_running = -1;
	return sl->w.rels;
}

/* start the host threads on the candidates la_gpu_batch just put in
   slot la_fill (after la_host_join) */
static void
la_host_start(la_gpu_t *G, sieve_job_t *job, const la_lat_t *lat,
		uint32 nlat, uint32 ncand, uint32 nout)
{
	la_slot_t *sl = la_slots + la_fill;
	la_work_t *w = &sl->w;

	la_n_cand += ncand;
	la_n_out += nout;
	memcpy(sl->lat, lat, nlat * sizeof(la_lat_t));
	memset(w, 0, sizeof(*w));
	pthread_mutex_init(&w->lock, NULL);
	w->job = job;
	w->logI = G->logI;
	w->lat = sl->lat;
	w->cand = sl->cands;
	w->num_cand = nout;
	for (uint32 s = 0; s < 2; s++) {
		w->fbp[s] = G->side[s].h_p;
		w->fbn[s] = G->side[s].s.n;
	}
	sl->t0 = la_now();
	for (uint32 k = 0; k < la_nthreads; k++) {
		la_threads[k].w = w;
		pthread_create(&la_threads[k].tid, NULL, la_thread_main,
				la_threads + k);
	}
	la_running = (int)la_fill;
	la_fill ^= 1;
	G->h_cands = la_slots[la_fill].cands;
}

/* la_filter's parameters: the rational polynomial and the bounds of
   la_lp_ok in 128 bits; off if any does not fit
   (MSIEVE_LATTICE_FILTER=0: off) */
static void
la_u128_set(uint64 *w, mpz_t v)
{
	mpz_t t;

	mpz_init(t);
	mpz_tdiv_r_2exp(t, v, 64);
	w[0] = mpz_get_ui(t);		/* unsigned long is 64 bits here */
	mpz_tdiv_q_2exp(t, v, 64);
	w[1] = mpz_get_ui(t);
	mpz_clear(t);
}

/* v (|v| < 2^127) as 128-bit two's complement */
static void
la_s128_set(uint64 *w, mpz_t v)
{
	mpz_t t;

	mpz_init(t);
	mpz_abs(t, v);
	la_u128_set(w, t);
	if (mpz_sgn(v) < 0) {		/* negate */
		w[0] = ~w[0] + 1;
		w[1] = ~w[1] + (w[0] == 0);
	}
	mpz_clear(t);
}

static void
la_filter_setup(la_gpu_t *G, sieve_job_t *job)
{
	la_batch_t *hb = &G->h_batch;
	sieve_t *fb = &job->sieve_rfb;
	mpz_poly_t *rp = &job->fb->rfb.poly;
	const char *e = getenv("MSIEVE_LATTICE_FILTER");

	G->rat_filter = (sizeof(unsigned long) == 8 && rp->degree == 1 &&
			mpz_sizeinbase(rp->coeff[0], 2) < 120 &&
			mpz_sizeinbase(rp->coeff[1], 2) < 120 &&
			mpz_sizeinbase(fb->LP3_max, 2) < 127 &&
			mpz_sizeinbase(fb->LP2_max, 2) < 127 &&
			(e == NULL || atoi(e) != 0));
	if (!G->rat_filter)
		return;
	la_s128_set(hb->r0, rp->coeff[0]);
	la_s128_set(hb->r1, rp->coeff[1]);
	G->r0_abs = fabs(mpz_get_d(rp->coeff[0]));
	G->r1_abs = fabs(mpz_get_d(rp->coeff[1]));
	hb->lp1max = fb->LP1_max;
	la_u128_set(hb->lp2min, fb->LP2_min);
	la_u128_set(hb->lp2max, fb->LP2_max);
	la_u128_set(hb->lp3min, fb->LP3_min);
	la_u128_set(hb->lp3max, fb->LP3_max);

	/* la_cofactor (MSIEVE_LATTICE_COFACTOR=0: off) */
	{
		sieve_t *afb = &job->sieve_afb;
		mpz_poly_t *ap = &job->fb->afb.poly;
		const char *ec = getenv("MSIEVE_LATTICE_COFACTOR");
		uint32 ok = (ap->degree <= LA_MAX_DEG &&
			mpz_sizeinbase(afb->LP3_max, 2) < 127 &&
			mpz_sizeinbase(afb->LP2_max, 2) < 127 &&
			(ec == NULL || atoi(ec) != 0));

		for (uint32 i = 0; ok && i <= ap->degree; i++)
			ok = (mpz_sizeinbase(ap->coeff[i], 2) < LA_COF_BITS - 8);
		G->cof_filter = ok;
		if (!ok)
			return;
		for (uint32 i = 0; i <= ap->degree; i++) {
			mpz_t v;
			size_t cnt;

			memset(hb->acoef[i], 0, sizeof(hb->acoef[i]));
			mpz_init(v);
			mpz_abs(v, ap->coeff[i]);
			mpz_export(hb->acoef[i], &cnt, -1, sizeof(uint32), 0, 0,
					v);
			if (mpz_sgn(ap->coeff[i]) < 0) {	/* negate */
				uint32 carry = 1;

				for (uint32 k = 0; k < LA_COF_BITS / 32; k++) {
					uint64 s = (uint64)(~hb->acoef[i][k]) +
							carry;

					hb->acoef[i][k] = (uint32)s;
					carry = (uint32)(s >> 32);
				}
			}
			mpz_clear(v);
			G->acoef_abs[i] = fabs(mpz_get_d(ap->coeff[i]));
		}
		hb->lpa1max = afb->LP1_max;
		la_u128_set(hb->lpa2min, afb->LP2_min);
		la_u128_set(hb->lpa2max, afb->LP2_max);
		la_u128_set(hb->lpa3min, afb->LP3_min);
		la_u128_set(hb->lpa3max, afb->LP3_max);
	}
}

/*------------------------------------------------------------------*/
/* MSIEVE_LATTICE_CHECK: the sieve sums of one lattice at sampled
   positions, directly from a = R b mod p (independent of rho and of
   the reduced bases), against the GPU's */
static void
la_check(msieve_obj *obj, la_gpu_t *G, const la_lat_t *L)
{
	const uint32 I = 1u << G->logI, J = G->J;
	uint32 *sv = (uint32 *)xmalloc(G->positions * sizeof(uint32));
	uint32 nsamp = 0, bad = 0, ncand, nout;

	if (!la_gpu_batch(G, L, 1, &ncand, &nout, 1)) {
		logprintf(obj, "lattice check: buffer overflow\n");
		free(sv);
		return;
	}
	CUDA_TRY(cuMemcpyDtoH(sv, G->sieve, G->positions * sizeof(uint32)))
	srand(12345);
	for (uint32 k = 0; k < I + 3000; k++) {
		uint32 x, j, want[2] = { 0, 0 };
		int64 i, a, b;

		if (k < I) {		/* one whole row */
			x = k;
			j = J / 2 + 1;
		}
		else {
			x = (uint32)rand() % I;
			j = 1 + (uint32)rand() % (J - 1);
		}
		i = (int64)x - I / 2;
		a = L->a0 * i + L->a1 * (int64)j;
		b = L->b0 * i + L->b1 * (int64)j;
		for (uint32 s = 0; s < 2; s++) {
			la_gside_t *g = G->side + s;

			for (uint32 e = g->s.n_tiny; e < g->s.n; e++) {
				int64 p = g->h_p[e], R = g->h_r[e];
				int64 u, v, am = ((a % p) + p) % p;
				int64 bm = ((b % p) + p) % p;
				int64 b0 = ((L->b0 % p) + p) % p;
				int64 b1 = ((L->b1 % p) + p) % p;
				uint32 hit;

				if (R == p) {
					u = b0;
					v = b1;
					hit = (bm == 0);
				}
				else {
					int64 a0 = ((L->a0 % p) + p) % p;
					int64 a1 = ((L->a1 % p) + p) % p;

					u = ((a0 - (int64)((uint64)R * b0 % p)) %
							p + p) % p;
					v = ((a1 - (int64)((uint64)R * b1 % p)) %
							p + p) % p;
					hit = ((am - (int64)((uint64)R * bm % p))
							% p == 0);
				}
				if ((u == 0 && v == 0) ||
				    (u != 0 && p >= I && v == 0))
					continue;	/* p = q; FK degenerate */
				if (hit)
					want[s] += g->h_logp[e];
			}
		}
		nsamp++;
		if (sv[((uint64)j << G->logI) | x] !=
				(want[0] | (want[1] << 16))) {
			if (bad < 5)
				logprintf(obj, "lattice check: x %u j %u: GPU "
					"%u/%u, direct %u/%u\n", x, j,
					sv[((uint64)j << G->logI) | x] & 0xffff,
					sv[((uint64)j << G->logI) | x] >> 16,
					want[0], want[1]);
			bad++;
		}
	}
	logprintf(obj, "lattice check (q %u, r %u): %u of %u sampled sieve "
			"sums differ; %u candidates, %u past the GPU filter\n",
			L->q, L->r, bad, nsamp, ncand, nout);
	free(sv);
}

/*------------------------------------------------------------------*/
/* sieve a batch, splitting it if its candidates do not fit; the
   relations of the previous host job (this batch's are counted by the
   next call, or la_host_join at the end) */
static uint32
la_run(msieve_obj *obj, la_gpu_t *G, sieve_job_t *job, const la_lat_t *lat,
	uint32 n)
{
	uint32 ncand, nout;

	if (!la_gpu_batch(G, lat, n, &ncand, &nout, 0)) {
		if (n == 1) {
			logprintf(obj, "warning: lattice q %u r %u has too many "
				"candidates, skipped\n", lat->q, lat->r);
			return 0;
		}
		return la_run(obj, G, job, lat, n / 2) +
			la_run(obj, G, job, lat + n / 2, n - n / 2);
	}
	{
		/* this batch's host job starts once the previous one ends,
		   which ran while the GPU sieved this batch */
		uint32 rels = la_host_join();

		la_host_start(G, job, lat, n, ncand, nout);
		return rels;
	}
}

uint32
lattice_sieve_gpu(sieve_job_t *job, double skew, uint32 *relations_found,
		uint32 max_relations, uint64 *last_q)
{
	msieve_obj *obj = job->obj;
	const char *e;
	gpu_config_t config;
	uint32 devs[MAX_GPU], ndev, lr = 0, ls = 1, nthreads;
	gpu_info_t *info;
	const char *name;
	la_gpu_t G;
	la_thread_t *threads;
	la_lat_t *lat;
	uint32 nlat = 0, rels = *relations_found, num_lat_done = 0;
	uint64 q, q0, q1;
	double t0 = la_now();

	memset(&G, 0, sizeof(G));
	gpu_init(&config);
#ifdef HAVE_MPI
	lr = obj->mpi_local_rank;
	ls = obj->mpi_local_size;
#endif
	ndev = gpu_select(&config, obj->which_gpu, lr, ls, devs, NULL);
	if (ndev == 0) {
		logprintf(obj, "lattice sieving needs a GPU\n");
		return 0;
	}
	info = config.info + devs[0];
	name = la_ptx_name(info->compute_version_major,
			info->compute_version_minor);
	if (name == NULL)
		return 0;

	G.logI = 12;
	if ((e = getenv("MSIEVE_LATTICE_LOGI")) != NULL)
		G.logI = (uint32)atoi(e);
	G.logI = MAX(9, MIN(14, G.logI));
	G.J = 1u << (G.logI - 1);
	G.positions = (uint64)G.J << G.logI;

	CUDA_TRY(cuDevicePrimaryCtxRetain(&G.ctx, info->device_handle))
	CUDA_TRY(cuCtxPushCurrent(G.ctx))
	if (cuModuleLoad(&G.module, name) != CUDA_SUCCESS) {
		logprintf(obj, "lattice sieving: cannot load %s\n", name);
		CUDA_TRY(cuCtxPopCurrent(NULL))
		CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
		return 0;
	}
	{
		CUdeviceptr ap;
		size_t asz;
		uint32 abi[2] = { 0, 0 };

		if (cuModuleGetGlobal(&ap, &asz, G.module, "la_abi") !=
				CUDA_SUCCESS || asz != sizeof(abi) ||
		    cuMemcpyDtoH(abi, ap, sizeof(abi)) != CUDA_SUCCESS ||
		    abi[0] != LA_ABI || abi[1] != sizeof(la_batch_t)) {
			logprintf(obj, "lattice sieving: %s does not match "
				"this msieve (an older file in the current "
				"directory? copy the .ptx files of this build)"
				"\n", name);
			CUDA_TRY(cuModuleUnload(G.module))
			CUDA_TRY(cuCtxPopCurrent(NULL))
			CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
			return 0;
		}
	}
	CUDA_TRY(cuModuleGetFunction(&G.k_roots, G.module, "la_roots"))
	CUDA_TRY(cuModuleGetFunction(&G.k_large, G.module, "la_large"))
	CUDA_TRY(cuModuleGetFunction(&G.k_tile, G.module, "la_tile"))
	CUDA_TRY(cuModuleGetFunction(&G.k_rowlogs, G.module, "la_rowlogs"))
	CUDA_TRY(cuModuleGetFunction(&G.k_filter, G.module, "la_filter"))
	CUDA_TRY(cuModuleGetFunction(&G.k_cofactor, G.module, "la_cofactor"))
	CUDA_TRY(cuFuncSetAttribute(G.k_tile,
			CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
			LA_TILE_POS_MAX * sizeof(uint32)))
	{
		int v;

		CUDA_TRY(cuDeviceGetAttribute(&v,
				CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
				info->device_handle))
		G.num_sm = (uint32)v;
	}

	{
		uint32 tiny = 32;	/* c95: 2.3x faster la_tile, as many
					   relations (0: sieve every prime) */

		if ((e = getenv("MSIEVE_LATTICE_TINY")) != NULL)
			tiny = (uint32)atoi(e);
		G.tiny = MIN(tiny, LA_RESIEVE_MIN);
	}
	if (!la_side_init(G.side + 0, &job->sieve_rfb, 1u << G.logI, G.tiny) ||
	    !la_side_init(G.side + 1, &job->sieve_afb, 1u << G.logI, G.tiny)) {
		logprintf(obj, "lattice sieving: factor base unusable\n");
		return 0;
	}
	{
		la_side_t s2[2] = { G.side[0].s, G.side[1].s };

		CUDA_TRY(cuMemAlloc(&G.d_sides, sizeof(s2)))
		CUDA_TRY(cuMemcpyHtoD(G.d_sides, s2, sizeof(s2)))
	}

	/* batch size from the free memory */
	{
		size_t fr = 0, tot = 0;
		uint64 per = G.positions * 4 + G.positions / 8 +
			(uint64)(G.side[0].s.n + G.side[1].s.n) * 4 +
			LA_CANDS * (2 * sizeof(la_cand_t) + 4 * 12);

		CUDA_TRY(cuMemGetInfo(&fr, &tot))
		G.max_lat = (uint32)MIN(64, (uint64)(0.4 * fr) / per);
		if ((e = getenv("MSIEVE_LATTICE_BATCH")) != NULL)
			G.max_lat = (uint32)atoi(e);
		G.max_lat = MAX(LA_MAX_DEG, G.max_lat);	/* a q's roots */
	}
	{
		la_batch_t *hb = &G.h_batch;
		sieve_t *sfb[2] = { &job->sieve_rfb, &job->sieve_afb };
		mpz_poly_t *pl[2] = { &job->fb->rfb.poly, &job->fb->afb.poly };

		memset(hb, 0, sizeof(*hb));
		hb->logI = G.logI;
		hb->J = G.J;
		hb->max_cands = G.max_lat * LA_CANDS;
		hb->hmask = 1;
		while (hb->hmask < 2 * hb->max_cands)
			hb->hmask *= 2;
		hb->hmask--;
		for (uint32 s = 0; s < 2; s++) {
			hb->deg[s] = pl[s]->degree;
			for (uint32 k = 0; k <= pl[s]->degree; k++)
				hb->coef[s][k] = mpz_get_d(pl[s]->coeff[k]);
			hb->cut2[s] = sfb[s]->cutoff2;
			hb->cut3[s] = sfb[s]->cutoff3;
		}
		hb->bits3[0] = 185;	/* as the line siever */
		hb->bits3[1] = 150;
		CUDA_TRY(cuMemAlloc(&G.batch, sizeof(la_batch_t)))
		CUDA_TRY(cuMemAlloc(&G.lat, G.max_lat * sizeof(la_lat_t)))
		CUDA_TRY(cuMemAlloc(&G.rho, (size_t)G.max_lat *
			(G.side[0].s.n + G.side[1].s.n) * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&G.sieve, (size_t)G.max_lat *
					G.positions * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&G.bitmap, (size_t)G.max_lat *
					(G.positions / 8)))
		CUDA_TRY(cuMemAlloc(&G.count, LA_N_COUNTS * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&G.cands, (size_t)hb->max_cands *
					sizeof(la_cand_t)))
		CUDA_TRY(cuMemAlloc(&G.out, (size_t)hb->max_cands *
					sizeof(la_cand_t)))
		CUDA_TRY(cuMemAlloc(&G.out2, (size_t)hb->max_cands *
					sizeof(la_cand_t)))
		CUDA_TRY(cuMemAlloc(&G.hkey, (size_t)(hb->hmask + 1) *
					sizeof(uint64)))
		CUDA_TRY(cuMemAlloc(&G.hval, (size_t)(hb->hmask + 1) *
					sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&G.rowlog, (size_t)G.max_lat * G.J *
					sizeof(uint32)))
		for (uint32 k = 0; k < 2; k++) {
			CUDA_TRY(cuMemAllocHost((void **)&la_slots[k].cands,
				(size_t)hb->max_cands * sizeof(la_cand_t)))
			la_slots[k].lat = (la_lat_t *)xmalloc(G.max_lat *
						sizeof(la_lat_t));
		}
		la_running = -1;
		la_fill = 0;
		G.h_cands = la_slots[0].cands;
		hb->lat = G.lat;
		hb->rho = G.rho;
		hb->sieve = G.sieve;
		hb->bitmap = G.bitmap;
		hb->count = G.count;
		hb->cands = G.cands;
		hb->out = G.out;
		hb->out2 = G.out2;
		hb->hkey = G.hkey;
		hb->hval = G.hval;
		hb->rowlog = G.rowlog;
		la_filter_setup(&G, job);
	}

	if ((e = getenv("MSIEVE_BATCH_DUMP")) != NULL &&
	    (la_dump = fopen(e, "w")) == NULL)
		logprintf(obj, "lattice sieving: cannot open %s\n", e);
	if ((e = getenv("MSIEVE_LATTICE_VERIFY")) != NULL && atoi(e)) {
		la_vfy_on = 1;
		for (uint32 s = 0; s < 2; s++) {
			mpz_init_set_ui(la_vfy_prod[s], 1);
			for (uint32 i = 0; i < G.side[s].s.n; i++) {
				uint32 p = G.side[s].h_p[i];

				if (p >= LA_RESIEVE_MIN &&
				    (i == 0 || p != G.side[s].h_p[i - 1]))
					mpz_mul_ui(la_vfy_prod[s],
						la_vfy_prod[s], p);
			}
		}
	}
	la_num_small = 0;
	for (uint32 p = 2; p < LA_RESIEVE_MIN; p++) {
		if (la_is_prime(p))
			la_small_primes[la_num_small++] = p;
	}
	{
		long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
		const char *et = getenv("MSIEVE_SIEVE_THREADS");

#ifdef HAVE_MPI
		ncpu /= MAX(1, (long)obj->mpi_local_size);
#endif
		nthreads = (et != NULL) ? (uint32)atoi(et) :
				(uint32)MAX(1, ncpu);
		nthreads = MAX(1, nthreads);
	}
	threads = (la_thread_t *)xcalloc(nthreads, sizeof(la_thread_t));
	la_threads = threads;
	la_bq_start(job);
	la_nthreads = nthreads;
	for (uint32 k = 0; k < nthreads; k++) {
		mpz_init(threads[k].res[0]);
		mpz_init(threads[k].res[1]);
		mpz_init(threads[k].tmp);
		mpz_init(threads[k].e1);
		mpz_init(threads[k].e2);
		mpz_init(threads[k].e3);
	}

	q0 = MAX(job->min_b, (uint64)job->fb->afb.max_prime + 1);
	q1 = MIN(job->max_b, (uint64)0xffffffff);
	logprintf(obj, "lattice sieving on GPU %u (%s): special-q %" PRIu64
			" to %" PRIu64 ", I = %u, J = %u, skew %.1f, %u "
			"lattices per batch, %u host threads\n", devs[0],
			info->name, q0, q1, 1u << G.logI, G.J, skew,
			G.max_lat, nthreads);

	lat = (la_lat_t *)xmalloc(G.max_lat * sizeof(la_lat_t));
	if ((e = getenv("MSIEVE_LATTICE_CHECK")) != NULL && atoi(e)) {
		for (q = q0; q <= q1; q++) {
			uint32 roots[LA_MAX_DEG], hc, nr;

			if (!la_is_prime((uint32)q))
				continue;
			nr = poly_get_zeros(roots, &job->fb->afb.poly,
					(uint32)q, &hc, 0);
			if (nr == 0)
				continue;
			la_reduce((uint32)q, roots[0], skew, lat);
			la_check(obj, &G, lat);
			break;
		}
	}

	for (q = q0; q <= q1; q++) {
		uint32 roots[LA_MAX_DEG], hc, nr = 0;

		if (la_is_prime((uint32)q))
			nr = poly_get_zeros(roots, &job->fb->afb.poly,
					(uint32)q, &hc, 0);
		else if (q < q1)
			continue;	/* (at q1: run the last batch) */
		for (uint32 k = 0; k < nr; k++)
			la_reduce((uint32)q, roots[k], skew, lat + nlat++);
		if (nlat + LA_MAX_DEG <= G.max_lat && q < q1)
			continue;
		if (nlat == 0)
			continue;

		rels += la_run(obj, &G, job, lat, nlat);
		rels += la_bq_take(0);
		num_lat_done += nlat;
		nlat = 0;
		*last_q = q;
		if (obj->flags & (MSIEVE_FLAG_USE_LOGFILE |
				  MSIEVE_FLAG_LOG_TO_STDOUT)) {
			fprintf(stderr, "q = %" PRIu64 ", %u complete / %u "
				"batched relations (need %u)\r", q, rels,
				job->relation_batch.num_relations,
				max_relations);
			fflush(stderr);
		}
		/* the host job and the batch runs lag behind: once the
		   estimate says enough, finish them and check */
		if (rels + la_bq_estimate(job) >= max_relations ||
		    (obj->flags & MSIEVE_FLAG_STOP_SIEVING)) {
			rels += la_host_join();
			rels += la_bq_take(1);
			if (job->relation_batch.num_relations > 0) {
				double tb = la_now();

				rels += relation_batch_run(&job->relation_batch);
				la_t_batch += la_now() - tb;
			}
			if (rels >= max_relations ||
			    (obj->flags & MSIEVE_FLAG_STOP_SIEVING))
				break;
		}
	}
	rels += la_host_join();
	rels += la_bq_take(1);
	la_bq_stop();
	if (job->relation_batch.num_relations > 0) {
		double tb = la_now();

		rels += relation_batch_run(&job->relation_batch);
		la_t_batch += la_now() - tb;
	}
	if (obj->flags & (MSIEVE_FLAG_USE_LOGFILE | MSIEVE_FLAG_LOG_TO_STDOUT))
		fprintf(stderr, "\n");
	logprintf(obj, "completed q = %" PRIu64 ", found %u relations\n",
			*last_q, rels);
	logprintf(obj, "GPU lattice sieving: %u lattices in %.1f s (GPU %.1f "
			"s; %" PRIu64 " candidates factored in %.1f s, batch "
			"factoring %.1f s of it, alongside the GPU; waiting "
			"for the host %.1f s)\n", num_lat_done, la_now() - t0,
			la_t_gpu, la_n_out, la_t_host, la_t_batch, la_t_wait);
	logprintf(obj, "lattice candidates: %" PRIu64 " found, %" PRIu64
			" past the GPU's rational test%s, %" PRIu64 " past its "
			"algebraic and Fermat tests%s (to the host)\n",
			la_n_cand, la_n_rat, G.rat_filter ? "" : " (off)",
			la_n_out, G.cof_filter ? "" : " (off)");
	logprintf(obj, "lattice candidates: %" PRIu64 " failed the rational "
			"bounds, %" PRIu64 " the algebraic, %" PRIu64 " the "
			"Fermat test; %" PRIu64 " relations direct, %" PRIu64
			" to batch factoring\n", la_stat[0], la_stat[1],
			la_stat[2], la_stat[3], la_stat[4]);
	if (la_dump != NULL) {
		fclose(la_dump);
		la_dump = NULL;
	}
	if (la_vfy_on) {
		logprintf(obj, "lattice verify: %u cofactors with a factor "
				"base prime the GPU did not report\n",
				la_vfy_bad);
		mpz_clear(la_vfy_prod[0]);
		mpz_clear(la_vfy_prod[1]);
	}
	*relations_found = rels;

	for (uint32 k = 0; k < nthreads; k++) {
		mpz_clear(threads[k].res[0]);
		mpz_clear(threads[k].res[1]);
		mpz_clear(threads[k].tmp);
		mpz_clear(threads[k].e1);
		mpz_clear(threads[k].e2);
		mpz_clear(threads[k].e3);
	}
	free(threads);
	free(lat);
	for (uint32 s = 0; s < 2; s++) {
		CUDA_TRY(cuMemFree(G.side[s].p))
		CUDA_TRY(cuMemFree(G.side[s].r))
		CUDA_TRY(cuMemFree(G.side[s].logp))
		free(G.side[s].h_p);
		free(G.side[s].h_r);
		free(G.side[s].h_logp);
	}
	{
		CUdeviceptr *d[] = { &G.d_sides, &G.batch, &G.lat, &G.rho,
			&G.sieve, &G.bitmap, &G.count, &G.cands, &G.out,
			&G.hkey, &G.hval, &G.rowlog, &G.out2 };

		for (uint32 k = 0; k < sizeof(d) / sizeof(d[0]); k++)
			CUDA_TRY(cuMemFree(*d[k]))
	}
	for (uint32 k = 0; k < 2; k++) {
		CUDA_TRY(cuMemFreeHost(la_slots[k].cands))
		free(la_slots[k].lat);
	}
	CUDA_TRY(cuModuleUnload(G.module))
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
	return 1;
}
