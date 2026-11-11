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

Part of a fork of msieve (public domain, by Jason Papadopoulos); the
routines ported here from msieve keep their public-domain origin.
--------------------------------------------------------------------*/

/* Size optimization on the GPU (sizeopt_core.cu), shared between the
   kernels and the host driver (stage1_sizeopt_gpu.c).

   Each stage 1 collision stays on the GPU: it is rebuilt into a
   polynomial (handle_collision, pol_expand), size optimized
   (optimize_initial) and scored for projective roots, and only the
   polynomials that pass the stage 2 norm bound are copied back.
   The integer work uses CGBN; the norm minimization is msieve's own
   double precision code, compiled without fused multiply-adds so it
   rounds exactly as the CPU code does. Degrees 4 and 5 only. */

#ifndef _SIZEOPT_CORE_H_
#define _SIZEOPT_CORE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define SO_MAX_DEG 6
#define SO_NUM_PRIMES 25	/* the primes below 100 */
#define SO_MAX_VARS 5		/* = MAX_VARS in common.h */

/* big integers per candidate, in this order, each a CGBN value of
   the launch width (two's complement) */
#define SO_BIG_A 0		/* a[0..SO_MAX_DEG] */
#define SO_BIG_LIN0 (SO_MAX_DEG + 1)	/* rational coefficient 0 */
#define SO_BIG_P (SO_MAX_DEG + 2)	/* rational coefficient 1 */
#define SO_BIG_D (SO_MAX_DEG + 3)	/* -lin0 */
#define SO_NUM_BIG (SO_MAX_DEG + 4)

/* per-coefficient big constants */
#define SO_CONST_N 0
#define SO_CONST_TRANS_N 1
#define SO_CONST_TRANS_M0 2
#define SO_CONST_AD 3		/* leading coefficient a_d */
#define SO_CONST_MOD 4		/* degree * a_d */
#define SO_NUM_CONST 5

/* candidate status */
#define SO_ACTIVE 0
#define SO_DONE 1

/* counters (so_stats_t) */
#define SO_STAT_CANDS 0		/* collisions that expanded to a poly */
#define SO_STAT_SURVIVORS 1
#define SO_STAT_CRAP 2		/* failed Kleinjung's modular condition */
#define SO_STAT_EXPAND_FAILED 3
#define SO_STAT_LINE_ITERS 4	/* "too many line iterations" */
#define SO_STAT_OVERFLOW 5	/* candidates dropped: arrays full */
#define SO_STAT_NEXT 6		/* work counter of so_minimize_kernel */
#define SO_NUM_STATS 7

typedef struct {
	uint32 degree;
	uint32 sizeof_q;		/* 4 or 8 */
	uint32 cap;			/* candidate capacity */
	uint32 found_max;		/* FOUND_ARRAY_SIZE - 1 */
	uint32 slot_base;		/* slot of this batch's collision 0 */
	int64 half_sieve;		/* sieve_size / 2 */
	double cmax_over_m0;		/* coeff_max / m0 */
	double stage1_norm;		/* pol_expand's coefficient bound */
	double sizeopt_bound;		/* stage 2 norm bound, loosened */
	uint32 primes[SO_NUM_PRIMES];
	double log_primes[SO_NUM_PRIMES];	/* log() from the host */
} so_params_t;

typedef struct {
	double apoly[SO_MAX_DEG + 1];
	double r0, r1;
	double best[SO_MAX_VARS];
	double score, last_score;
	double rot[3];			/* floor(best[ROTATE0+j] + 0.5) */
	int64 xlate;			/* (int64)(best[TRANSLATE] + 0.5) */
	double pol_norm, alpha;
	uint32 pass, nmin, status, pad;
} so_state_t;

/* optimize_basic() (optimize.c) for root optimization: one request
   per call, in any order; so_basic_<degree> fills one result each.
   Counters: next request to take, and "too many line iterations" */

#define SO_BASIC_NEXT 0
#define SO_BASIC_LINE_ITERS 1
#define SO_BASIC_NUM_STATS 2

typedef struct {
	double apoly[SO_MAX_DEG + 1];	/* coefficients, unused ones 0 */
} so_basic_in_t;

typedef struct {
	double norm;			/* sqrt of the minimized norm */
	double skewness;
	double translation;
} so_basic_out_t;

#ifdef __cplusplus
}
#endif

#endif /* !_SIZEOPT_CORE_H_ */
