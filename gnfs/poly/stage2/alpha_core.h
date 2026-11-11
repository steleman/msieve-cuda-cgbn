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

/* Root properties (Murphy's alpha) and Bernstein's size score of the
   rotations in a root heap, on the GPU: alpha_<bits> and bs_size_<deg>
   in sizeopt_core.cu, used by optimize_final_batch() (optimize.c)
   through a hook.

   Per rotation (x, y, z) of a polynomial a with linear part P x - d,
   the coefficients b = a + (x + y X + z X^2)(P X - d) are formed
   exactly in CGBN and reduced modulo the 303 primes below 2000 (for
   the primes below 100, modulo their largest power below 65000); the
   root counts mod p, special roots lifted mod p^k, are then
   analyze_poly_roots() (root_score.c) with poly_get_zeros() and
   get_zeros_rec() (ffpoly.c) ported. The terms are added in the same
   order with the host's log(p), so alpha is the CPU's, bit for bit.

   The size score is analyze_poly_size() (size_score.c): b converted to
   double-double as dd_gmp2dd() does, its roots found by ports of
   find_poly_roots() (common/polyroot.c), and the double-exponential
   integral of |R(x) A(x)|^(-2/(deg+1)). The GPU's exp, log, sin, cos and
   pow differ from the C library's in the last bit, so the score agrees
   to about the integral's accuracy, not bit for bit; the factor from
   alpha, pow(exp(alpha), -2/(deg+1)), is applied on the host. */

#ifndef _ALPHA_CORE_H_
#define _ALPHA_CORE_H_

#include <stdint.h>
#include "line_sieve_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RA_NUM_PRIMES 303	/* the primes below PRIME_BOUND = 2000 */
#define RA_NUM_SMALL 25		/* below SMALL_PRIME_BOUND = 100 */
#define RA_MAX_DEG 6
#define RA_MAX_LIMBS 32		/* 1024 bits */

typedef struct {
	uint32_t prime[RA_NUM_PRIMES];
	uint32_t modulus[RA_NUM_PRIMES];	/* p, or for small p its
						   largest power <= 65000 */
	double log_prime[RA_NUM_PRIMES];	/* the host's log(p) */
} ra_tables_t;

/* one polynomial: coefficients, P and d as two's complement 32-bit
   limbs of the launch width, low limb first */
typedef struct {
	uint32_t a[RA_MAX_DEG + 1][RA_MAX_LIMBS];
	uint32_t p[RA_MAX_LIMBS];
	uint32_t d[RA_MAX_LIMBS];
	double r_hi[2];		/* the rational polynomial, double-double */
	double r_lo[2];
	double min_e_bernstein;	/* the size score bound of optimize_final() */
	double pad3;
	uint32_t degree;
	uint32_t pad;		/* the launch width, 512 or 1024 */
	uint32_t pad2[2];	/* a multiple of 16 bytes: CGBN loads the
				   limbs with vector loads */
} ra_poly_t;

/* per rotation */
typedef struct {
	double alpha;
	double size;		/* the size score; 0 if it failed (as
				   analyze_poly_size()), -1 if not computed
				   (alpha > -4.5, or no GPU size scores) */
	double score;		/* Murphy E, if RA_MURPHY */
	double skewness;
	double translation;
	uint32_t flags;
	uint32_t num_real_roots;
	uint32_t keep;		/* its Murphy E input in the polynomial's
				   kept ones, or RA_NO_MF */
	uint32_t pad;
} ra_res_t;

#define RA_KEEP 32		/* Murphy E inputs kept per polynomial */

/* a polynomial to keep: its rotation and translation applied exactly
   (rf_final): b = a + (x + yX + zX^2)(PX - d), then translated by k
   as translate_gmp() does; the linear part becomes (P, -d - P k) */
typedef struct {
	int64_t x, y, z;
	int64_t k;
	uint32_t poly;
	uint32_t pad;
} ra_final_t;

#define RF_LIMBS 32		/* rf_final works at 1024 bits */
#define RF_MAX_JOBS 64		/* per request (a GPU batch holds this
				   many): the host sends more in parts */

typedef struct {
	uint32_t b[RA_MAX_DEG + 1][RF_LIMBS];	/* two's complement */
	uint32_t lin0[RF_LIMBS];
} ra_final_out_t;

/* the coefficients of a rotated polynomial as double-double, passed
   from alpha_<bits> to bs_size_<deg> on the GPU */
typedef struct {
	double hi[RA_MAX_DEG + 1];
	double lo[RA_MAX_DEG + 1];
} ra_dd_t;

typedef struct {
	int64_t x, y, z;	/* the rotation */
	uint32_t poly;		/* its ra_poly_t */
	uint32_t mf_index;	/* where bs_size_<deg> puts its Murphy E
				   input if it passes, or RA_NO_MF */
} ra_rot_t;

#define RA_NO_MF 0xffffffffu
#define RA_PASSED 1		/* ra_res_t flags: passed alpha and size */
#define RA_MURPHY 2		/* Murphy E done on the GPU */

/* The root sieve's setup (init_sieve in root_sieve.c) on the GPU,
   rs_setup_<bits>: for each prime power p^k < 1000 of the primes below
   100, the coefficients, P and d reduced mod p^k in CGBN, then for each
   residue class i the root of the rotated polynomial, in the CPU's order
   (by power, then by i). The roots stay on the GPU for the line sieve;
   the host gets a copy, and the number of classes that add to the
   sieve bias, which it adds up in the CPU's order. */

#define RS_MAX_POWERS 64
#define RS_MAX_ROOTS 8192	/* roots per polynomial: below the sum of
				   the prime powers, about 8000 */
#define RS_THREADS 256

typedef struct {
	uint32_t num_powers;
	uint32_t pad;
	uint32_t prime[RS_MAX_POWERS];
	uint32_t power[RS_MAX_POWERS];
} rs_powers_t;

typedef struct {
	uint32_t poly;		/* its ra_poly_t */
	uint32_t base;		/* where its roots go in the area */
} rs_job_t;

typedef struct {
	uint32_t num_roots[RS_MAX_POWERS];
	uint32_t proj[RS_MAX_POWERS];	/* classes adding to the bias */
	uint32_t total;
	uint32_t pad;
} rs_setup_out_t;

#ifdef __cplusplus
}
#endif

#endif /* !_ALPHA_CORE_H_ */
