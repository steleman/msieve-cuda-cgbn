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

/* Murphy E optimization of the batched root sieve, shared between the
   GPU kernel (mf_opt_<degree> in sizeopt_core.cu) and the host.

   Each candidate is a polynomial pair that passed the root-property
   and Bernstein bounds in optimize_final() (optimize.c); its Murphy E is
   maximized over skewness and translation as optimize_final_core()
   does: minimize() of -E, E from analyze_poly_murphy() (size_score.c),
   a double-exponential integral (common/integrate.c) of Dickman rho
   terms (common/dickman.c). The polynomial's roots, which split the
   integral, are found once on the host: a translation by t moves them
   by t. The GPU's cos, sin and log round differently in the last bit
   from the C library's, so its results agree with the CPU's to about
   the integral's tolerance, not bit for bit. */

#ifndef _MURPHY_CORE_H_
#define _MURPHY_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MF_MAX_DEG 6
#define MF_MAX_ROOTS (MF_MAX_DEG + 1)	/* of the pair, imaginary part >= 0 */
#define MF_TRAP_LEVELS 7		/* MAX_TRAP_LEVELS in integrate.c */
#define MF_MAX_POINTS 16		/* de_t num_points, at most */
#define MF_DICKMAN_LINES 31		/* MAX_LINE + 1 in dickman.c */
#define MF_DICKMAN_COEFFS 2048

/* the integrator's rule and Dickman's function: the same for every
   candidate, filled by the host from its own integrate_t and dickman_t */
typedef struct {
	double abscissa;
	double weight;
	double residual_weight;
} mf_point_t;

typedef struct {
	uint32_t num_points;
	uint32_t pad;
	double relative_error;
	double center_weight;
	double center_residual_weight;
	double log_rfb_limit;		/* log(5e6), from the host's libm */
	double log_afb_limit;		/* log(1e7) */
	double jt_cos94;		/* cos and sin of 94 degrees, for the
					   root finder (the host's libm) */
	double jt_sin94;
	mf_point_t rule[MF_MAX_POINTS << MF_TRAP_LEVELS];	/* de_t layout:
				   initial_rule, then refined_rules */
	uint32_t dk_num_coeffs[MF_DICKMAN_LINES];
	uint32_t dk_offset[MF_DICKMAN_LINES];
	double dk_coeffs[MF_DICKMAN_COEFFS];
} mf_tables_t;

typedef struct {
	double a_hi[MF_MAX_DEG + 1];	/* algebraic polynomial, double-double */
	double a_lo[MF_MAX_DEG + 1];
	double r_hi[2];			/* rational polynomial */
	double r_lo[2];
	double root_score_r;		/* alpha of each */
	double root_score_a;
	double roots[MF_MAX_ROOTS];	/* real parts of the roots with
					   imaginary part >= 0 */
	uint32_t degree;
	uint32_t num_roots;
	uint32_t num_real_roots;	/* of the algebraic polynomial */
	uint32_t pad;
} mf_in_t;

typedef struct {
	double score;			/* Murphy E */
	double skewness;
	double translation;		/* to apply, rounded as (int64)(t+0.5) */
	uint32_t num_real_roots;
	uint32_t pad;
} mf_out_t;

#ifdef __cplusplus
}
#endif

#endif /* !_MURPHY_CORE_H_ */
