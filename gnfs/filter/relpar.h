/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* Parsing relations on all CPUs for the filtering passes: a pass adds
   a block of savefile lines, relpar_run() parses (and, if asked,
   checks against the norms and finds the large ideals of) all of them
   on several threads, then the pass takes the results in order. The
   threads use their own copies of the polynomials, whose scratch
   values nfs_read_relation() writes. MSIEVE_FILTER_THREADS: the
   thread count (default: online CPUs) */

#ifndef _GNFS_FILTER_RELPAR_H_
#define _GNFS_FILTER_RELPAR_H_

#include <common.h>
#include "gnfs.h"
#include "relstore.h"

#define RELPAR_BLOCK 16384

typedef struct {
	uint32 rel_index;	/* the pass's relation number */
	uint32 stored;		/* 1 + its index in the relation store,
				   or 0 (parse the line) */
	int32 status;		/* nfs_read_relation's */
	uint32 array_size;
	relation_t rel;
	relation_lp_t lp;	/* if want_ideals and status == 0 */
	uint8 factors[COMPRESSED_P_MAX_SIZE];
} relpar_item_t;

typedef struct {
	factor_base_t *fb;
	uint32 compress, test_primality;
	uint32 want_ideals, filtmin_r, filtmin_a;
	uint32 nthreads;
	uint32 num;			/* items in the block */
	relpar_item_t *items;
	char *text;			/* the lines, one after another */
	size_t text_used, text_alloc;
	size_t *offset;
	factor_base_t *fbs;		/* per thread */
	mpz_t *scratch;
	uint32 next;			/* work distribution */
	uint32 ran;			/* the block was run */
	relstore_t *rs;			/* for the stored items */
} relpar_t;

void relpar_init(relpar_t *rp, factor_base_t *fb, uint32 compress,
		uint32 test_primality, uint32 want_ideals,
		uint32 filtmin_r, uint32 filtmin_a);

/* add a line; 1 if the block is now full (run it) */
uint32 relpar_add(relpar_t *rp, const char *line, uint32 rel_index);

/* add relation i of the relation store instead of a line */
uint32 relpar_add_stored(relpar_t *rp, relstore_t *rs, uint32 i,
			uint32 rel_index);

/* parse the block (rp->items[0, rp->num)); the next relpar_add starts
   a new block */
void relpar_run(relpar_t *rp);

void relpar_free(relpar_t *rp);

/* a pass's relations, from the relation store when it holds the whole
   savefile, else from the savefile: relsrc_next() moves to the next
   relation (0 at the end), relsrc_add() adds it to a block */
typedef struct {
	relstore_t *rs;
	savefile_t *sf;
	uint32 cur, started;
	char buf[LINE_BUF_SIZE];
} relsrc_t;

void relsrc_open(relsrc_t *s, savefile_t *sf);
uint32 relsrc_next(relsrc_t *s);
uint32 relsrc_add(relsrc_t *s, relpar_t *rp, uint32 rel_index);
void relsrc_ab(relsrc_t *s, int64 *a, uint64 *b);
void relsrc_close(relsrc_t *s);

#endif /* !_GNFS_FILTER_RELPAR_H_ */
