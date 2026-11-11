/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* The relations of this run, kept in memory so that the filtering
   passes, the matrix build and the square root need not read and parse
   the savefile again and again. The savefile is still written as
   before; the store is only used when it holds every relation of the
   savefile, in the same order (a savefile this process started:
   factor_gnfs creates the store then).

   While sieving, the store only copies each line as it is written (the
   caller holds the lock that orders the savefile), so the sievers do no
   more work. The first time filtering asks for the relations,
   relstore_prepare() parses the new lines once, on all CPUs, with
   nfs_read_relation() (as the first duplicate pass does: primality
   tested); everything after that reads the parsed relations.

   MSIEVE_RELSTORE=0: no store. MSIEVE_RELSTORE_MB: its memory limit
   (default: a quarter of physical memory); past it the store is
   dropped and everything reads the savefile */

#ifndef _GNFS_RELSTORE_H_
#define _GNFS_RELSTORE_H_

#include <common.h>
#include "gnfs.h"

typedef struct relstore_t relstore_t;

relstore_t *relstore_new(msieve_obj *obj, mpz_poly_t *rpoly,
			mpz_poly_t *apoly);
void relstore_free(relstore_t *rs);

/* a line just written to the savefile (non-relation lines are skipped;
   the caller holds the savefile's lock; rs may be NULL) */
void relstore_add_line(relstore_t *rs, const char *line);

/* parse the lines added since the last call; whether the store holds
   the whole savefile (then usable until more lines are added) */
uint32 relstore_prepare(relstore_t *rs);

uint32 relstore_usable(relstore_t *rs);
uint32 relstore_num(relstore_t *rs);

/* relation i as nfs_read_relation(compress) would give it: its status,
   r (r->factors must hold COMPRESSED_P_MAX_SIZE bytes) and the size of
   r->factors (after relstore_prepare) */
int32 relstore_get(relstore_t *rs, uint32 i, uint32 compress,
			relation_t *r, uint32 *array_size);
void relstore_get_ab(relstore_t *rs, uint32 i, int64 *a, uint64 *b);

#endif /* !_GNFS_RELSTORE_H_ */
