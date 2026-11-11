/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

#include "relstore.h"

#if !defined(WIN32) && !defined(_WIN64)
#include <pthread.h>
#include <unistd.h>
#define RELSTORE_TLS 1
#endif

struct relstore_t {
	msieve_obj *obj;
	factor_base_t fb;	/* only the polynomials are set */
	uint32 usable;		/* 0: dropped (limit) */
	uint32 num, alloc;
	int64 *a;
	uint64 *b;
	int8 *status;
	uint8 *num_r, *num_a;
	uint16 *len;
	uint64 *off;
	uint8 *bytes;
	uint64 bytes_used, bytes_alloc;
	uint64 limit;		/* bytes */
	uint32 gen;		/* tells thread-local copies apart */
	uint32 parsed;		/* relations [0, parsed) are parsed */
	char *text;		/* lines of relations [parsed, num) */
	uint64 text_used, text_alloc;
	uint64 *toff;		/* their offsets in text */
	uint32 toff_alloc;
};

typedef struct {
	int32 status;
	int64 a;
	uint64 b;
	uint8 num_r, num_a;
	uint32 size;
	uint8 factors[COMPRESSED_P_MAX_SIZE];
} relstore_rec_t;

/*--------------------------------------------------------------------*/
/* nfs_read_relation() keeps scratch values in the polynomials, so each
   thread parses with its own copies */

typedef struct {
	factor_base_t fb;
	mpz_t scratch;
	uint32 gen;
} relstore_tls_t;

static uint32 relstore_next_gen = 1;

static void tls_init(relstore_tls_t *t, relstore_t *rs) {

	uint32 i;

	memset(t, 0, sizeof(*t));
	mpz_poly_init(&t->fb.rfb.poly);
	mpz_poly_init(&t->fb.afb.poly);
	t->fb.rfb.poly.degree = rs->fb.rfb.poly.degree;
	t->fb.afb.poly.degree = rs->fb.afb.poly.degree;
	for (i = 0; i <= MAX_POLY_DEGREE; i++) {
		mpz_set(t->fb.rfb.poly.coeff[i], rs->fb.rfb.poly.coeff[i]);
		mpz_set(t->fb.afb.poly.coeff[i], rs->fb.afb.poly.coeff[i]);
	}
	mpz_init(t->scratch);
	t->gen = rs->gen;
}

static void tls_clear(relstore_tls_t *t) {

	mpz_poly_free(&t->fb.rfb.poly);
	mpz_poly_free(&t->fb.afb.poly);
	mpz_clear(t->scratch);
	t->gen = 0;
}

#ifdef RELSTORE_TLS
static pthread_key_t relstore_key;
static pthread_once_t relstore_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t relstore_gen_lock = PTHREAD_MUTEX_INITIALIZER;

static void tls_destroy(void *p) {

	relstore_tls_t *t = (relstore_tls_t *)p;

	if (t->gen)
		tls_clear(t);
	free(t);
}

static void key_make(void) {
	pthread_key_create(&relstore_key, tls_destroy);
}

static relstore_tls_t *tls_get(relstore_t *rs) {

	relstore_tls_t *t;

	pthread_once(&relstore_once, key_make);
	t = (relstore_tls_t *)pthread_getspecific(relstore_key);
	if (t == NULL) {
		t = (relstore_tls_t *)xcalloc(1, sizeof(relstore_tls_t));
		pthread_setspecific(relstore_key, t);
	}
	if (t->gen != rs->gen) {
		if (t->gen)
			tls_clear(t);
		tls_init(t, rs);
	}
	return t;
}
#else
static relstore_tls_t relstore_single;

static relstore_tls_t *tls_get(relstore_t *rs) {

	if (relstore_single.gen != rs->gen) {
		if (relstore_single.gen)
			tls_clear(&relstore_single);
		tls_init(&relstore_single, rs);
	}
	return &relstore_single;
}
#endif

/*--------------------------------------------------------------------*/
relstore_t *relstore_new(msieve_obj *obj, mpz_poly_t *rpoly,
			mpz_poly_t *apoly) {

	relstore_t *rs;
	const char *e = getenv("MSIEVE_RELSTORE");
	uint32 i;

	if (e != NULL && atoi(e) == 0)
		return NULL;

	rs = (relstore_t *)xcalloc(1, sizeof(relstore_t));
	rs->obj = obj;
	mpz_poly_init(&rs->fb.rfb.poly);
	mpz_poly_init(&rs->fb.afb.poly);
	rs->fb.rfb.poly.degree = rpoly->degree;
	rs->fb.afb.poly.degree = apoly->degree;
	for (i = 0; i <= MAX_POLY_DEGREE; i++) {
		mpz_set(rs->fb.rfb.poly.coeff[i], rpoly->coeff[i]);
		mpz_set(rs->fb.afb.poly.coeff[i], apoly->coeff[i]);
	}
#ifdef RELSTORE_TLS
	pthread_mutex_lock(&relstore_gen_lock);
	rs->gen = relstore_next_gen++;
	pthread_mutex_unlock(&relstore_gen_lock);
#else
	rs->gen = relstore_next_gen++;
#endif

	e = getenv("MSIEVE_RELSTORE_MB");
	if (e != NULL) {
		rs->limit = (uint64)strtoull(e, NULL, 10) << 20;
	}
	else {
		rs->limit = (uint64)1 << 32;
#ifdef RELSTORE_TLS
		{
			long pages = sysconf(_SC_PHYS_PAGES);
			long psize = sysconf(_SC_PAGESIZE);

			if (pages > 0 && psize > 0)
				rs->limit = (uint64)pages * psize / 4;
		}
#endif
	}
	rs->usable = 1;
	return rs;
}

/*--------------------------------------------------------------------*/
static void drop(relstore_t *rs) {

	free(rs->a); free(rs->b); free(rs->status);
	free(rs->num_r); free(rs->num_a); free(rs->len); free(rs->off);
	free(rs->bytes);
	free(rs->text);
	free(rs->toff);
	rs->text = NULL; rs->toff = NULL;
	rs->text_used = rs->text_alloc = 0;
	rs->toff_alloc = rs->parsed = 0;
	rs->a = NULL; rs->b = NULL; rs->status = NULL;
	rs->num_r = rs->num_a = NULL; rs->len = NULL; rs->off = NULL;
	rs->bytes = NULL;
	rs->num = rs->alloc = 0;
	rs->bytes_used = rs->bytes_alloc = 0;
	rs->usable = 0;
}

void relstore_free(relstore_t *rs) {

	if (rs == NULL)
		return;
	drop(rs);
	mpz_poly_free(&rs->fb.rfb.poly);
	mpz_poly_free(&rs->fb.afb.poly);
	free(rs);
}

/*--------------------------------------------------------------------*/
static void relstore_parse(relstore_t *rs, const char *line,
			relstore_rec_t *rec) {

	relstore_tls_t *t = tls_get(rs);
	relation_t r;
	uint32 size = 0;
	char buf[LINE_BUF_SIZE];

	strncpy(buf, line, sizeof(buf) - 1);
	buf[sizeof(buf) - 1] = 0;
	memset(&r, 0, sizeof(r));
	r.factors = rec->factors;
	rec->status = nfs_read_relation(buf, &t->fb, &r, &size, 0,
					t->scratch, 1);
	rec->a = r.a;
	rec->b = r.b;
	rec->num_r = r.num_factors_r;
	rec->num_a = r.num_factors_a;
	rec->size = (rec->status == 0) ? size : 0;
}

/*--------------------------------------------------------------------*/
/* the arrays for relation n (index n < rs->alloc after this) */
static uint32 grow_index(relstore_t *rs, uint32 n) {

	if (n < rs->alloc)
		return 1;
	{
		uint32 na = rs->alloc ? 2 * rs->alloc : 1u << 20;
		uint64 per = sizeof(int64) + sizeof(uint64) + 1 + 2 +
				sizeof(uint16) + 2 * sizeof(uint64);

		if ((uint64)na * per + rs->bytes_alloc + rs->text_alloc >
				rs->limit || na < rs->alloc) {
			logprintf(rs->obj, "relation store: over its limit, "
					"the savefile is read instead\n");
			drop(rs);
			return 0;
		}
		rs->a = (int64 *)xrealloc(rs->a, na * sizeof(int64));
		rs->b = (uint64 *)xrealloc(rs->b, na * sizeof(uint64));
		rs->status = (int8 *)xrealloc(rs->status, na);
		rs->num_r = (uint8 *)xrealloc(rs->num_r, na);
		rs->num_a = (uint8 *)xrealloc(rs->num_a, na);
		rs->len = (uint16 *)xrealloc(rs->len, na * sizeof(uint16));
		rs->off = (uint64 *)xrealloc(rs->off, na * sizeof(uint64));
		rs->toff = (uint64 *)xrealloc(rs->toff, na * sizeof(uint64));
		rs->alloc = na;
	}
	return 1;
}

/*--------------------------------------------------------------------*/
void relstore_add_line(relstore_t *rs, const char *line) {

	size_t len;

	if (rs == NULL || !rs->usable)
		return;
	if (line[0] != '-' && !isdigit(line[0]))
		return;
	if (!grow_index(rs, rs->num))
		return;
	len = strlen(line) + 1;
	if (rs->text_used + len > rs->text_alloc) {
		uint64 nt = rs->text_alloc ? 2 * rs->text_alloc : 1u << 26;

		if (nt + rs->bytes_alloc + (uint64)rs->alloc * 40 >
				rs->limit) {
			logprintf(rs->obj, "relation store: over its limit, "
					"the savefile is read instead\n");
			drop(rs);
			return;
		}
		rs->text = (char *)xrealloc(rs->text, nt);
		rs->text_alloc = nt;
	}
	memcpy(rs->text + rs->text_used, line, len);
	rs->toff[rs->num++] = rs->text_used;
	rs->text_used += len;
}

/*--------------------------------------------------------------------*/
/* relstore_prepare: the new lines parsed on all CPUs into records,
   then copied in order into the arrays */

typedef struct {
	relstore_t *rs;
	relstore_rec_t *recs;
	uint32 first, num;
	uint32 next;
} prep_work_t;

#ifdef RELSTORE_TLS
static pthread_mutex_t prep_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

static void *prep_thread(void *arg) {

	prep_work_t *w = (prep_work_t *)arg;

	for (;;) {
		uint32 k0, k1;

#ifdef RELSTORE_TLS
		pthread_mutex_lock(&prep_lock);
#endif
		k0 = w->next;
		k1 = MIN(w->num, k0 + 1024);
		w->next = k1;
#ifdef RELSTORE_TLS
		pthread_mutex_unlock(&prep_lock);
#endif
		if (k0 >= k1)
			break;
		for (; k0 < k1; k0++)
			relstore_parse(w->rs, w->rs->text +
					w->rs->toff[w->first + k0],
					w->recs + k0);
	}
	return NULL;
}

#define PREP_CHUNK 262144

uint32 relstore_prepare(relstore_t *rs) {

	relstore_rec_t *recs;
	uint32 nthreads = 1;

	if (rs == NULL || !rs->usable)
		return 0;
	if (rs->parsed == rs->num)
		return 1;

#ifdef RELSTORE_TLS
	{
		const char *e = getenv("MSIEVE_FILTER_THREADS");

		nthreads = (e != NULL) ? (uint32)atoi(e) :
				(uint32)sysconf(_SC_NPROCESSORS_ONLN);
		nthreads = MAX(1, MIN(256, nthreads));
	}
#endif
	recs = (relstore_rec_t *)xmalloc(PREP_CHUNK * sizeof(relstore_rec_t));
	while (rs->usable && rs->parsed < rs->num) {
		prep_work_t w;
		uint32 k;

		w.rs = rs;
		w.recs = recs;
		w.first = rs->parsed;
		w.num = MIN(PREP_CHUNK, rs->num - rs->parsed);
		w.next = 0;
#ifdef RELSTORE_TLS
		if (nthreads > 1) {
			pthread_t tid[256];
			uint32 started = 0;

			for (k = 1; k < nthreads; k++) {
				if (pthread_create(tid + k, NULL, prep_thread,
						&w) != 0)
					break;
				started = k;
			}
			prep_thread(&w);
			for (k = 1; k <= started; k++)
				pthread_join(tid[k], NULL);
		}
		else
#endif
			prep_thread(&w);

		for (k = 0; k < w.num && rs->usable; k++) {
			relstore_rec_t *rec = recs + k;
			uint32 n = w.first + k;

			/* too many factors with their powers counted, where
			   the filtering (each odd power once) might accept
			   the relation: the savefile then, so that nothing
			   differs */
			if (rec->status == -8 || rec->status == -10 ||
			    rec->status == -13 || rec->status == -14) {
				logprintf(rs->obj, "relation store: a relation "
					"with too many factors, the savefile "
					"is read instead\n");
				drop(rs);
				break;
			}
			if (rs->bytes_used + rec->size > rs->bytes_alloc) {
				uint64 nb = rs->bytes_alloc ?
					2 * rs->bytes_alloc : 1u << 26;

				if (nb + rs->text_alloc +
				    (uint64)rs->alloc * 40 > rs->limit) {
					logprintf(rs->obj, "relation store: "
						"over its limit, the savefile "
						"is read instead\n");
					drop(rs);
					break;
				}
				rs->bytes = (uint8 *)xrealloc(rs->bytes, nb);
				rs->bytes_alloc = nb;
			}
			rs->a[n] = rec->a;
			rs->b[n] = rec->b;
			rs->status[n] = (int8)(rec->status < -127 ? -127 :
						rec->status);
			rs->num_r[n] = rec->num_r;
			rs->num_a[n] = rec->num_a;
			rs->len[n] = (uint16)rec->size;
			rs->off[n] = rs->bytes_used;
			memcpy(rs->bytes + rs->bytes_used, rec->factors,
					rec->size);
			rs->bytes_used += rec->size;
		}
		if (rs->usable)
			rs->parsed = w.first + w.num;
	}
	free(recs);
	if (!rs->usable)
		return 0;

	/* the text is not needed any more */
	rs->text_used = 0;
	return 1;
}

/*--------------------------------------------------------------------*/
uint32 relstore_usable(relstore_t *rs) {
	return rs != NULL && rs->usable && rs->parsed == rs->num;
}

uint32 relstore_num(relstore_t *rs) {
	return rs->num;
}

void relstore_get_ab(relstore_t *rs, uint32 i, int64 *a, uint64 *b) {
	*a = rs->a[i];
	*b = rs->b[i];
}

/*--------------------------------------------------------------------*/
int32 relstore_get(relstore_t *rs, uint32 i, uint32 compress,
			relation_t *r, uint32 *array_size) {

	const uint8 *f = rs->bytes + rs->off[i];
	uint32 len = rs->len[i];
	int32 status = rs->status[i];

	r->a = rs->a[i];
	r->b = rs->b[i];
	r->num_factors_r = rs->num_r[i];
	r->num_factors_a = rs->num_a[i];
	if (status != 0) {
		*array_size = 0;
		return status;
	}
	if (!compress || r->b == 0) {	/* free relations: p and roots */
		memcpy(r->factors, f, len);
		*array_size = len;
		return 0;
	}

	/* nfs_read_relation(compress = 1) keeps each prime once if its
	   power is odd (and the rational sign always): the runs of equal
	   primes collapsed */
	{
		uint32 in = 0, out = 0, side, k;
		uint32 counts[2], kept[2] = { 0, 0 };

		counts[0] = rs->num_r[i];
		counts[1] = rs->num_a[i];
		for (side = 0; side < 2; side++) {
			k = 0;
			while (k < counts[side]) {
				uint64 p = decompress_p((uint8 *)f, &in);
				uint32 run = 1;

				k++;
				while (k < counts[side]) {
					uint32 save = in;
					uint64 q = decompress_p((uint8 *)f, &in);

					if (q != p) {
						in = save;
						break;
					}
					run++;
					k++;
				}
				if (p == 0 || (run & 1)) {
					out = compress_p(r->factors, p, out);
					kept[side]++;
				}
			}
		}
		r->num_factors_r = kept[0];
		r->num_factors_a = kept[1];
		*array_size = out;
	}
	return 0;
}
