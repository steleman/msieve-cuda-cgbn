/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.

$Id: sieve_util.c 1076 2026-04-23 03:36:32Z jasonp_sf $
--------------------------------------------------------------------*/

#include "sieve.h"
#include "relstore.h"
#if !defined(WIN32) && !defined(_WIN64)
#include <pthread.h>
#include <unistd.h>
#endif

/*------------------------------------------------------------------*/
void format_relation(char *buf, int64 a, uint64 b, 
			uint32 *factors_r, uint32 num_factors_r, 
			uint32 large_prime_r[MAX_LARGE_PRIMES],
			uint32 *factors_a, uint32 num_factors_a, 
			uint32 large_prime_a[MAX_LARGE_PRIMES]) {
	
	uint32 i, j;
	char *tmp = buf;

	tmp += sprintf(buf, "%" PRId64 ",%" PRIu64, a, b);
	for (i = 0; i < num_factors_r; i++) {
		if (i == 0)
			tmp += sprintf(tmp, ":%x", factors_r[i]);
		else
			tmp += sprintf(tmp, ",%x", factors_r[i]);
	}
	for (j = 0; j < MAX_LARGE_PRIMES; j++) {
		if (large_prime_r[j] == 1)
			continue;
		if (i == 0)
			tmp += sprintf(tmp, ":%x", large_prime_r[j]);
		else
			tmp += sprintf(tmp, ",%x", large_prime_r[j]);
		i++;
	}

	for (i = 0; i < num_factors_a; i++) {
		if (i == 0)
			tmp += sprintf(tmp, ":%x", factors_a[i]);
		else
			tmp += sprintf(tmp, ",%x", factors_a[i]);
	}
	for (j = 0; j < MAX_LARGE_PRIMES; j++) {
		if (large_prime_a[j] == 1)
			continue;
		if (i == 0)
			tmp += sprintf(tmp, ":%x", large_prime_a[j]);
		else
			tmp += sprintf(tmp, ",%x", large_prime_a[j]);
		i++;
	}
	sprintf(tmp, "\n");
}

/*------------------------------------------------------------------*/
void print_relation(savefile_t *savefile, int64 a, uint64 b, 
			uint32 *factors_r, uint32 num_factors_r, 
			uint32 large_prime_r[MAX_LARGE_PRIMES],
			uint32 *factors_a, uint32 num_factors_a, 
			uint32 large_prime_a[MAX_LARGE_PRIMES]) {
	
	char buf[LINE_BUF_SIZE];

	format_relation(buf, a, b, factors_r, num_factors_r, large_prime_r,
			factors_a, num_factors_a, large_prime_a);
	savefile_write_line(savefile, buf);
	if (savefile->relstore != NULL)	/* (the caller orders the file) */
		relstore_add_line((relstore_t *)savefile->relstore, buf);
}

/*------------------------------------------------------------------*/
uint32 fplog(uint32 k, double log_of_base) {

	/* express k in a different base */

	return (uint32)(log((double)k)/log_of_base + 0.5);
}

/*------------------------------------------------------------------*/
int32 fplog_eval_poly(int64 a, uint64 b, mpz_t scratch,
			mpz_poly_t *f, double log_base,
			uint32 *bits) { 

	/* Compute f(a,b) and take its logarithm in the
	   current base of sieve logarithms, rounding down */

	eval_poly(scratch, a, b, f);
	*bits = mpz_sizeinbase(scratch, 2);
	return (uint32)((*bits) * M_LN2 / log_base);
}

/*------------------------------------------------------------------*/
/* Bases will be chosen for logarithms with the goal of
   hitting this (base-2) target value at most */

#define LOG_TARGET 220

double get_log_base(mpz_poly_t *poly, 
			int64 a0, int64 a1, uint64 b) { 

	/* Decide on a base for the logs of one polynomial. 

	   The rational poly is assumed linear, its maximum value 
	   occurs at one of the endpoints of the sieve interval

	   Rigorously finding the extreme values of the algebraic poly
	   would require finding the minima and maxima, and comparing
	   the polynomial values there to the values at a0 and a1. However, 
	   experiments show that for small b the values at the endpoints
	   are much larger than those at the extreme values, and for large
	   b the values are close but the extreme values tend to be outside
	   the sieve interval. Hence we cheat and just do the same as with 
	   the rational poly */

	double t;
	mpz_t tmp1, tmp2;

	mpz_init(tmp1);
	mpz_init(tmp2);
	eval_poly(tmp1, a0, b, poly);
	eval_poly(tmp2, a1, b, poly);

	if (mpz_cmp(tmp2, tmp1) > 0) 
		t = mpz_sizeinbase(tmp1, 2);
	else
		t = mpz_sizeinbase(tmp2, 2);

	/* the base to use is a number x such that 
	   log_x (2^t) = LOG_TARGET. After much re-
	   arranging, x amounts to: */

	mpz_clear(tmp1);
	mpz_clear(tmp2);
	return pow(2.0, t / LOG_TARGET);
}

/*------------------------------------------------------------------*/
uint32 read_last_line(msieve_obj *obj, mpz_t n) {

	uint32 last_line = 0;
	char buf[LINE_BUF_SIZE];
	FILE *linefile;
	mpz_t read_n;

	sprintf(buf, "%s.line", obj->savefile.name);
	linefile = fopen(buf, "r");
	if (linefile == NULL)
		return last_line;

	fgets(buf, (int)sizeof(buf), linefile);
	mpz_init_set_ui(read_n, 0);
	if (buf[0] == 'N')
		gmp_sscanf(buf + 2, "%Zd", read_n);
	if (mpz_cmp(n, read_n) == 0) {
		fgets(buf, (int)sizeof(buf), linefile);
		last_line = atoi(buf);
	}

	fclose(linefile);
	mpz_clear(read_n);
	return last_line;
}

/*------------------------------------------------------------------*/
void write_last_line(msieve_obj *obj, mpz_t n, uint64 b) {

	char buf[LINE_BUF_SIZE];
	FILE *linefile;

	sprintf(buf, "%s.line", obj->savefile.name);
	linefile = fopen(buf, "w");
	if (linefile == NULL) {
		printf("error: cannot open linefile '%s'\n", buf);
		exit(-1);
	}

	gmp_fprintf(linefile, "N %Zd\n", n);
	fprintf(linefile, "%" PRIu64 "\n", b);
	fclose(linefile);
}

/*------------------------------------------------------------------*/
/* the lattice siever's resume point (<savefile>.lattice): the next
   special-q, or 0 */
uint64 read_last_q(msieve_obj *obj, mpz_t n) {

	uint64 q = 0;
	char buf[LINE_BUF_SIZE];
	FILE *f;
	mpz_t read_n;

	sprintf(buf, "%s.lattice", obj->savefile.name);
	f = fopen(buf, "r");
	if (f == NULL)
		return 0;
	mpz_init_set_ui(read_n, 0);
	if (fgets(buf, (int)sizeof(buf), f) != NULL && buf[0] == 'N')
		gmp_sscanf(buf + 2, "%Zd", read_n);
	if (mpz_cmp(n, read_n) == 0 && fgets(buf, (int)sizeof(buf), f) != NULL)
		q = strtoull(buf, NULL, 10);
	fclose(f);
	mpz_clear(read_n);
	return q;
}

/*------------------------------------------------------------------*/
void write_last_q(msieve_obj *obj, mpz_t n, uint64 q) {

	char buf[LINE_BUF_SIZE];
	FILE *f;

	sprintf(buf, "%s.lattice", obj->savefile.name);
	f = fopen(buf, "w");
	if (f == NULL) {
		printf("error: cannot open '%s'\n", buf);
		exit(-1);
	}
	gmp_fprintf(f, "N %Zd\n", n);
	fprintf(f, "%" PRIu64 "\n", q);
	fclose(f);
}

/*------------------------------------------------------------------*/
/* Whether p gives a free relation: both polynomials split into linear
   factors mod p. poly_get_zeros with count_only only reads the
   polynomials, so the candidates are tested on all CPUs (it was 6 s of
   a c120's filtering on one) and the relations then written in order,
   as before */

typedef struct {
	factor_base_t *fb;
	const uint32 *cand;
	uint8 *ok;
	uint32 n, k, step;
} free_job_t;

static uint32
free_rel_ok(factor_base_t *fb, uint32 p)
{
	uint32 roots[MAX_POLY_DEGREE];
	uint32 high_coeff;

	if (poly_get_zeros(roots, &fb->afb.poly, p, &high_coeff, 1) !=
			fb->afb.poly.degree || high_coeff == 0)
		return 0;
	return (poly_get_zeros(roots, &fb->rfb.poly, p, &high_coeff, 1) ==
			fb->rfb.poly.degree && high_coeff != 0);
}

#if !defined(WIN32) && !defined(_WIN64)
static void *
free_rel_worker(void *arg)
{
	free_job_t *w = (free_job_t *)arg;

	for (uint32 k = w->k; k < w->n; k += w->step)
		w->ok[k] = (uint8)free_rel_ok(w->fb, w->cand[k]);
	return NULL;
}
#endif

uint32 add_free_relations(msieve_obj *obj, 
			factor_base_t *fb, uint8 *free_bits) {

	uint32 i, j;
	uint32 num_relations = 0;
	uint32 free_bytes = (FREE_RELATION_LIMIT / 2 + 7) / 8;
	uint32 *cand, ncand = 0, nthreads = 1;
	uint8 *ok;

	/* the candidates: bit 8*i+j is p = 2 * (8*i+j) + 1 */
	for (i = 0; i < free_bytes; i++)
		for (j = 0; j < 8; j++)
			ncand += (free_bits[i] >> j) & 1;
	cand = (uint32 *)xmalloc((ncand + 1) * sizeof(uint32));
	ok = (uint8 *)xcalloc(ncand + 1, 1);
	ncand = 0;
	for (i = 0; i < free_bytes; i++) {
		if (free_bits[i] == 0)
			continue;
		for (j = 0; j < 8; j++)
			if (free_bits[i] & (1 << j))
				cand[ncand++] = 2 * (8 * i + j) + 1;
	}

#if !defined(WIN32) && !defined(_WIN64)
	{
		const char *e = getenv("MSIEVE_FILTER_THREADS");

		nthreads = (e != NULL) ? (uint32)atoi(e) :
				(uint32)sysconf(_SC_NPROCESSORS_ONLN);
		nthreads = MAX(1, MIN(256, nthreads));
	}
	if (nthreads > 1 && ncand >= 4096) {
		pthread_t *tid = (pthread_t *)xmalloc(nthreads *
						sizeof(pthread_t));
		free_job_t *jobs = (free_job_t *)xmalloc(nthreads *
						sizeof(free_job_t));

		for (uint32 t = 0; t < nthreads; t++) {
			jobs[t].fb = fb;
			jobs[t].cand = cand;
			jobs[t].ok = ok;
			jobs[t].n = ncand;
			jobs[t].k = t;
			jobs[t].step = nthreads;
			pthread_create(tid + t, NULL, free_rel_worker, jobs + t);
		}
		for (uint32 t = 0; t < nthreads; t++)
			pthread_join(tid[t], NULL);
		free(tid);
		free(jobs);
	}
	else
#endif
	{
		for (uint32 k = 0; k < ncand; k++)
			ok[k] = (uint8)free_rel_ok(fb, cand[k]);
	}

	savefile_open(&obj->savefile, SAVEFILE_APPEND);

	for (uint32 k = 0; k < ncand; k++) {
		char buf[LINE_BUF_SIZE];

		if (!ok[k])
			continue;
		sprintf(buf, "%u,0:\n", cand[k]);
		savefile_write_line(&obj->savefile, buf);
		relstore_add_line((relstore_t *)obj->savefile.relstore, buf);
		num_relations++;
	}
	free(cand);
	free(ok);

	if (num_relations)
		logprintf(obj, "added %u free relations\n", num_relations);
	savefile_flush(&obj->savefile);
	savefile_close(&obj->savefile);
	return num_relations;
}
