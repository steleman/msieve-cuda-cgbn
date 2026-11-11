/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

#include "relpar.h"

#if !defined(WIN32) && !defined(_WIN64)
#include <pthread.h>
#include <unistd.h>
#define RELPAR_THREADS 1
#endif

/*--------------------------------------------------------------------*/
static void copy_poly(mpz_poly_t *dst, mpz_poly_t *src) {

	uint32 i;

	mpz_poly_init(dst);
	dst->degree = src->degree;
	for (i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_set(dst->coeff[i], src->coeff[i]);
}

/*--------------------------------------------------------------------*/
void relpar_init(relpar_t *rp, factor_base_t *fb, uint32 compress,
		uint32 test_primality, uint32 want_ideals,
		uint32 filtmin_r, uint32 filtmin_a) {

	uint32 i;
	const char *e = getenv("MSIEVE_FILTER_THREADS");

	memset(rp, 0, sizeof(relpar_t));
	rp->fb = fb;
	rp->compress = compress;
	rp->test_primality = test_primality;
	rp->want_ideals = want_ideals;
	rp->filtmin_r = filtmin_r;
	rp->filtmin_a = filtmin_a;

	rp->nthreads = 1;
#ifdef RELPAR_THREADS
	rp->nthreads = (e != NULL) ? (uint32)atoi(e) :
				(uint32)sysconf(_SC_NPROCESSORS_ONLN);
#else
	(void)e;
#endif
	if (rp->nthreads < 1)
		rp->nthreads = 1;
	if (rp->nthreads > 256)
		rp->nthreads = 256;

	rp->items = (relpar_item_t *)xmalloc(RELPAR_BLOCK *
					sizeof(relpar_item_t));
	rp->offset = (size_t *)xmalloc(RELPAR_BLOCK * sizeof(size_t));
	rp->text_alloc = (size_t)RELPAR_BLOCK * 256;
	rp->text = (char *)xmalloc(rp->text_alloc);

	rp->fbs = (factor_base_t *)xmalloc(rp->nthreads *
					sizeof(factor_base_t));
	rp->scratch = (mpz_t *)xmalloc(rp->nthreads * sizeof(mpz_t));
	for (i = 0; i < rp->nthreads; i++) {
		rp->fbs[i] = *fb;	/* the entries are only read */
		copy_poly(&rp->fbs[i].rfb.poly, &fb->rfb.poly);
		copy_poly(&rp->fbs[i].afb.poly, &fb->afb.poly);
		mpz_init(rp->scratch[i]);
	}
}

/*--------------------------------------------------------------------*/
void relpar_free(relpar_t *rp) {

	uint32 i;

	for (i = 0; i < rp->nthreads; i++) {
		mpz_poly_free(&rp->fbs[i].rfb.poly);
		mpz_poly_free(&rp->fbs[i].afb.poly);
		mpz_clear(rp->scratch[i]);
	}
	free(rp->fbs);
	free(rp->scratch);
	free(rp->items);
	free(rp->offset);
	free(rp->text);
	memset(rp, 0, sizeof(relpar_t));
}

/*--------------------------------------------------------------------*/
uint32 relpar_add(relpar_t *rp, const char *line, uint32 rel_index) {

	size_t len = strlen(line) + 1;

	if (rp->ran) {
		rp->num = 0;
		rp->text_used = 0;
		rp->ran = 0;
	}
	if (rp->text_used + len > rp->text_alloc) {
		rp->text_alloc = 2 * rp->text_alloc + len;
		rp->text = (char *)xrealloc(rp->text, rp->text_alloc);
	}
	memcpy(rp->text + rp->text_used, line, len);
	rp->offset[rp->num] = rp->text_used;
	rp->items[rp->num].rel_index = rel_index;
	rp->items[rp->num].stored = 0;
	rp->text_used += len;
	rp->num++;
	return rp->num == RELPAR_BLOCK;
}

/*--------------------------------------------------------------------*/
uint32 relpar_add_stored(relpar_t *rp, relstore_t *rs, uint32 i,
			uint32 rel_index) {

	if (rp->ran) {
		rp->num = 0;
		rp->text_used = 0;
		rp->ran = 0;
	}
	rp->rs = rs;
	rp->items[rp->num].rel_index = rel_index;
	rp->items[rp->num].stored = i + 1;
	rp->num++;
	return rp->num == RELPAR_BLOCK;
}

/*--------------------------------------------------------------------*/
static void parse_one(relpar_t *rp, uint32 t, uint32 k) {

	relpar_item_t *it = rp->items + k;

	it->rel.factors = it->factors;
	if (it->stored) {
		it->status = relstore_get(rp->rs, it->stored - 1,
					rp->compress, &it->rel,
					&it->array_size);
	}
	else {
		it->status = nfs_read_relation(rp->text + rp->offset[k],
				rp->fbs + t, &it->rel, &it->array_size,
				rp->compress, rp->scratch[t],
				rp->test_primality);
	}
	if (it->status == 0 && rp->want_ideals)
		find_large_ideals(&it->rel, &it->lp,
				rp->filtmin_r, rp->filtmin_a);
}

#ifdef RELPAR_THREADS
typedef struct {
	relpar_t *rp;
	uint32 t;
} relpar_arg_t;

static pthread_mutex_t relpar_lock = PTHREAD_MUTEX_INITIALIZER;

static void *relpar_thread(void *arg) {

	relpar_arg_t *a = (relpar_arg_t *)arg;
	relpar_t *rp = a->rp;

	for (;;) {
		uint32 k0, k1;

		pthread_mutex_lock(&relpar_lock);
		k0 = rp->next;
		k1 = MIN(rp->num, k0 + 256);
		rp->next = k1;
		pthread_mutex_unlock(&relpar_lock);
		if (k0 >= k1)
			break;
		while (k0 < k1)
			parse_one(rp, a->t, k0++);
	}
	return NULL;
}
#endif

/*--------------------------------------------------------------------*/
void relpar_run(relpar_t *rp) {

	uint32 i;

	rp->next = 0;
#ifdef RELPAR_THREADS
	if (rp->nthreads > 1 && rp->num > 256) {
		pthread_t tid[256];
		relpar_arg_t arg[256];
		uint32 started = 0;

		for (i = 1; i < rp->nthreads; i++) {
			arg[i].rp = rp;
			arg[i].t = i;
			if (pthread_create(tid + i, NULL, relpar_thread,
					arg + i) != 0)
				break;
			started = i;
		}
		arg[0].rp = rp;
		arg[0].t = 0;
		relpar_thread(arg);
		for (i = 1; i <= started; i++)
			pthread_join(tid[i], NULL);
	}
	else
#endif
	{
		for (i = 0; i < rp->num; i++)
			parse_one(rp, 0, i);
	}
	rp->ran = 1;
}

/*--------------------------------------------------------------------*/
void relsrc_open(relsrc_t *s, savefile_t *sf) {

	memset(s, 0, sizeof(*s));
	s->sf = sf;
	if (relstore_prepare((relstore_t *)sf->relstore)) {
		s->rs = (relstore_t *)sf->relstore;
		return;
	}
	savefile_open(sf, SAVEFILE_READ);
	savefile_read_line(s->buf, sizeof(s->buf), sf);
}

/*--------------------------------------------------------------------*/
uint32 relsrc_next(relsrc_t *s) {

	if (s->rs != NULL) {
		if (s->started)
			s->cur++;
		s->started = 1;
		return s->cur < relstore_num(s->rs);
	}
	if (s->started)
		savefile_read_line(s->buf, sizeof(s->buf), s->sf);
	s->started = 1;
	while (!savefile_eof(s->sf) &&
	       s->buf[0] != '-' && !isdigit(s->buf[0]))
		savefile_read_line(s->buf, sizeof(s->buf), s->sf);
	return !savefile_eof(s->sf);
}

/*--------------------------------------------------------------------*/
uint32 relsrc_add(relsrc_t *s, relpar_t *rp, uint32 rel_index) {

	if (s->rs != NULL)
		return relpar_add_stored(rp, s->rs, s->cur, rel_index);
	return relpar_add(rp, s->buf, rel_index);
}

/*--------------------------------------------------------------------*/
void relsrc_ab(relsrc_t *s, int64 *a, uint64 *b) {

	char *next_field;

	if (s->rs != NULL) {
		relstore_get_ab(s->rs, s->cur, a, b);
		return;
	}
	*a = strtoll(s->buf, &next_field, 10);
	*b = strtoull(next_field + 1, NULL, 10);
}

/*--------------------------------------------------------------------*/
void relsrc_close(relsrc_t *s) {

	if (s->rs == NULL)
		savefile_close(s->sf);
}
