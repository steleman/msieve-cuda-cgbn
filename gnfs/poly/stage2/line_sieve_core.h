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

/* The line sieve of the batched root sieve (root_sieve_batch.c), shared
   between the GPU kernel (ls_sieve in sizeopt_core.cu) and the host.

   A line is scored at positions 0 .. positions-1: position x scores
   the sum over the 25 primes below 100 of table_p[x mod size_p], where
   each prime's table holds the contributions of its powers' roots
   (prepare_sieve_line / prepare_sieve_lattice in root_sieve_line.c),
   and the LS_TOP best positions are kept: highest score first, and of
   equal scores the lowest position (the original's heap keeps an
   arbitrary one of a tie). The tables are built from the polynomial's
   roots and the line's coordinates, which must fit in 64 bits. */

#ifndef _LINE_SIEVE_CORE_H_
#define _LINE_SIEVE_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LS_TOP 25		/* LINE_HEAP_SIZE in root_sieve_line.c */
#define LS_NUM_PRIMES 25	/* the primes below 100 */
#define LS_MAX_POWERS 64
#define LS_MAX_TABLE 25600	/* entries in all of a polynomial's tables */
#define LS_MAX_DEG 6

typedef struct {
	uint16_t resclass;	/* sieve_root_t, each below its power */
	uint16_t step;
	uint16_t start;
	uint16_t pad;
} ls_root_t;

typedef struct {
	uint32_t table;		/* where its table starts, entries */
	uint32_t size;		/* contrib_array_size */
} ls_prime_t;

typedef struct {
	uint32_t power;
	uint32_t contrib;	/* sieve_contrib */
	uint32_t prime;		/* which of the polynomial's primes */
	uint32_t root0;		/* its roots in the batch */
	uint32_t num_roots;
} ls_power_t;

/* one polynomial's sieve: its primes and powers in the batch, and
   what root_sieve_line() needs to make a line's candidates */
typedef struct {
	uint32_t prime0;
	uint32_t power0;
	uint32_t num_powers;
	uint32_t table_size;	/* sum of its primes' table sizes */
	double apoly[LS_MAX_DEG + 1];	/* rs->apoly */
	double dbl_p, dbl_d;
	double random_root_score;
	double sieve_bias;
	uint32_t degree;
	uint32_t pad;
} ls_poly_t;

/* A line's coordinates come one of two ways. LS_INT64: as numbers
   (x_base, resclass, lattice, y). LS_PARTS: as the parts the x sieves
   (root_sieve_deg45_x.c) combine with GMP, with the constants of their
   run (ls_run_t); the GPU forms them with CGBN, the lattice CRT
   included:
     y        = y_base + ry + ly * k
     resclass = (r1 * crt0 + rx * crt1) mod lattice
     x_base   = trunc(trunc(line_min) / lattice) * lattice */
#define LS_INT64 0
#define LS_PARTS 1

typedef struct {
	int64_t x_base;		/* LS_INT64 */
	int64_t resclass;
	int64_t lattice;	/* 1: a line */
	int64_t y;
	int64_t z;		/* curr_z, both ways */
	uint64_t ry, r1, rx;	/* LS_PARTS */
	double line_min;
	uint32_t k;
	uint32_t run;		/* its ls_run_t */
	uint32_t poly;		/* its ls_poly_t */
	uint32_t positions;	/* whole blocks of DEFAULT_BLOCK_SIZE */
	uint32_t mode;
	uint32_t curr_score;	/* sieve_x_t curr_score */
	uint32_t lattice_one;	/* LS_PARTS: the lattice is 1 */
	uint32_t pad;
} ls_line_t;

#define LS_RUN_LIMBS 16		/* 512-bit constants */

typedef struct {
	int64_t y_base;		/* the run's numbers as words: the GPU */
	uint64_t ly;		/* forms the lattice size l1 * l2 in */
	uint64_t l1, l2;	/* CGBN */
	uint32_t crt0[LS_RUN_LIMBS];
	uint32_t crt1[LS_RUN_LIMBS];
} ls_run_t;

/* the best positions, best first */
#define LS_XLIMBS 4		/* x and y of candidates: 128 bits */

typedef struct {
	uint32_t n;
	uint32_t pos[LS_TOP];
	uint16_t score[LS_TOP];
	uint16_t pad;
	uint32_t y[LS_XLIMBS];	/* the line's y (GPU candidates) */
} ls_out_t;

/* a candidate made on the GPU: root_sieve_line()'s alpha and x; its
   optimize_basic() follows there */
typedef struct {
	double alpha;
	uint32_t x[LS_XLIMBS];	/* two's complement, low first */
	uint32_t valid;		/* alpha within ROOT_SCORE_COARSE_MIN */
	uint32_t pad;
} ls_cand_t;

/*------------------------------------------------------------------------*/
/* building the tables: the same code on the host and the GPU */

#if defined(__CUDACC__)
#define LS_FN __host__ __device__ static __forceinline__
#else
#define LS_FN static inline
#endif

/* a line's numbers modulo one power */
typedef struct {
	uint32_t xmin;		/* x_base mod power (floor) */
	uint32_t y;		/* curr_y mod power (floor) */
	uint32_t z;		/* curr_z mod power (floor) */
	uint32_t resclass;	/* lattice: |resclass| mod power */
	uint32_t cf;		/* lattice: gcd(lattice mod power, power) */
	uint32_t linv;		/* lattice: (lattice mod power / cf)^-1
				   mod (power / cf) */
	uint32_t skip;		/* lattice: power divides the lattice */
} ls_lp_t;

LS_FN uint32_t
ls_fmod(int64_t a, uint32_t m)
{
	int64_t r = a % (int64_t)m;

	return (uint32_t)(r < 0 ? r + (int64_t)m : r);
}

LS_FN uint32_t
ls_tmod(int64_t a, uint32_t m)
{
	int64_t r = a % (int64_t)m;

	return (uint32_t)(r < 0 ? -r : r);
}

LS_FN uint32_t
ls_gcd(uint32_t a, uint32_t b)
{
	while (b != 0) {
		uint32_t t = a % b;

		a = b;
		b = t;
	}
	return a;
}

/* a^-1 mod m, for a coprime to m >= 2 */
LS_FN uint32_t
ls_inv(uint32_t a, uint32_t m)
{
	int64_t t = 0, nt = 1, r = m, nr = a % m;

	while (nr != 0) {
		int64_t q = r / nr, tmp;

		tmp = t - q * nt; t = nt; nt = tmp;
		tmp = r - q * nr; r = nr; nr = tmp;
	}
	return (uint32_t)(t < 0 ? t + m : t);
}

LS_FN uint32_t
ls_modsub(uint32_t a, uint32_t b, uint32_t m)
{
	return (a >= b) ? a - b : a - b + m;
}

/* from the line's numbers modulo the power: x_base and y floor
   remainders, resclass and lattice absolute ones (lattice_one: the
   lattice is 1) */
LS_FN void
ls_line_power_res(uint32_t xmin, uint32_t y, uint32_t z, uint32_t resclass,
		uint32_t lat, uint32_t lattice_one, uint32_t power, ls_lp_t *lp)
{
	lp->xmin = xmin;
	lp->y = y;
	lp->z = z;
	lp->resclass = 0;
	lp->cf = 1;
	lp->linv = 0;
	lp->skip = 0;

	if (!lattice_one) {
		if (lat == 0) {
			lp->skip = 1;
			return;
		}
		lp->cf = ls_gcd(lat, power);
		lp->resclass = resclass;
		lp->linv = ls_inv(lat / lp->cf, power / lp->cf);
	}
}

LS_FN void
ls_line_power(const ls_line_t *l, uint32_t power, ls_lp_t *lp)
{
	ls_line_power_res(ls_fmod(l->x_base, power), ls_fmod(l->y, power),
			ls_fmod(l->z, power), ls_tmod(l->resclass, power),
			ls_tmod(l->lattice, power), l->lattice == 1, power,
			lp);
}

/* where root r starts in its prime's table, and its step; 0 if it
   adds nothing (prepare_sieve_line, fill_contrib_array and
   fill_contrib_array_special in root_sieve_line.c) */
LS_FN uint32_t
ls_root_start(const ls_root_t *r, uint32_t power, uint32_t lattice,
		const ls_lp_t *lp, uint32_t *start_out, uint32_t *step_out)
{
	uint32_t start, a;

	a = (uint32_t)r->resclass * lp->z % power;
	start = lp->y + a;
	if (start >= power)
		start -= power;
	start = ls_modsub(r->start, (uint32_t)r->resclass * start % power,
				power);
	start = ls_modsub(start, lp->xmin, power);

	if (!lattice) {
		*step_out = r->step;
		*start_out = (r->step != power) ? start % r->step : start;
		return 1;
	}

	start = ls_modsub(start, lp->resclass, power);
	if (lp->cf == 1) {
		*start_out = start * lp->linv % power;
		*step_out = power;
		return 1;
	}
	if (start % lp->cf != 0)
		return 0;
	*step_out = power / lp->cf;
	*start_out = (start / lp->cf) * lp->linv % *step_out;
	return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* !_LINE_SIEVE_CORE_H_ */
