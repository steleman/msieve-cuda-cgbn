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

/* The root sieve's lattice choice on the GPU (built with RS_GPU_LATTICE;
   by default the CPU does it). Two requests, both reading the roots the
   setup left in the slot's part of the root area, so that no roots
   travel to the host:

   lt_xy: find_hits() of root_sieve_deg5_xy.c, the 2-D sieve of each
   prime of the xy lattice; the host combines the hits
   (compute_lattices).

   lt_x: the x-line choice of sieve_x_run_deg5/4 (root_sieve_deg45_x.c):
   for each xy lattice and y block, xdata_init(), find_hits(),
   root_sieve_x() and compute_lattices(), then the 300-entry heap of
   x-lines filled in the CPU's order. A y block's starting offsets are
   computed directly (start - b * stride) instead of one block after
   the other; y_base's residues come from CGBN. */

#ifndef _LATTICE_CORE_H_
#define _LATTICE_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LT_MAX_PRIMES 10	/* MAX_CRT_FACTORS */
#define LT_MAX_POWERS 32
#define LT_MAX_ROOTS 64		/* MAX_ROOTS */
#define LT_MAX_X_LATTICES 60	/* MAX_X_LATTICES */
#define LT_HEAP 300		/* XLINE_HEAP_SIZE */
#define LT_THREADS 128
#define LT_MAX_LATS 2048	/* xy lattices in one x-line choice (a GPU
				   batch holds this many); more: the CPU */
#define LT_MAX_SIEVE 256	/* x: the largest power of a lattice prime */

typedef struct {
	uint32_t power;
	uint32_t contrib;	/* sieve_contrib */
	uint32_t root0;		/* its roots, from the polynomial's first */
	uint32_t num_roots;
	uint32_t latmod;	/* x: the xy lattice size mod power */
	uint32_t pad;
} lt_power_t;

/* the lattice's primes and the powers of each that count */
typedef struct {
	uint32_t num_primes;
	uint32_t area_base;	/* the slot's first root (the engine's) */
	uint32_t num_powers[LT_MAX_PRIMES];
	uint32_t power0[LT_MAX_PRIMES];
	lt_power_t powers[LT_MAX_POWERS];
} lt_set_t;

/* hit_t of stage2.h */
typedef struct {
	uint32_t power;
	uint32_t num_roots;
	uint16_t score[LT_MAX_ROOTS];
	uint8_t roots[LT_MAX_ROOTS][2];
} lt_hit_t;

/* an x-line choice */
typedef struct {
	lt_set_t set;
	int64_t y_base;		/* xy y_base */
	uint64_t pad_y[3];
	uint64_t x_lattice;	/* x->lattice_size */
	uint32_t y_blocks;
	uint32_t num_lattices;	/* xy lattices */
	uint32_t lat0;		/* their lt_lat_t in the batch (engine) */
	uint32_t pad;
} lt_xreq_t;

typedef struct {
	uint64_t y;		/* the xy lattice's y and score */
	uint32_t score;
	uint32_t pad;
} lt_lat_t;

/* xline_t of root_sieve_deg45_x.c */
typedef struct {
	uint64_t resclass;
	uint32_t which_y_block;
	uint32_t which_lattice;
	uint16_t score;
	uint16_t pad;
	uint32_t pad2;
} lt_xline_t;

typedef struct {
	uint32_t n;
	uint32_t pad;
	lt_xline_t e[LT_HEAP];	/* the heap's array, in its order */
} lt_xout_t;

#ifdef __cplusplus
}
#endif

#endif /* !_LATTICE_CORE_H_ */
