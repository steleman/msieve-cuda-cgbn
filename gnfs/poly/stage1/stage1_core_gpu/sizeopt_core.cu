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

/* Stage 1 collisions to size-optimized polynomials, on the GPU.

   The integer steps follow handle_collision() (stage1.c), pol_expand()
   and check_poly() (stage2.c) and the rotation / translation updates of
   optimize_initial() (optimize.c), in CGBN, two's complement at the
   launch width. Results are the same integers GMP computes; where GMP
   needs more bits than the width (check_poly's P^d, pol_expand's
   unreduced powers of 1/d mod P) the same test is done modulo N or P.

   The norm minimization is minimize() (common/minimize.c) with
   poly_rotate_callback() and the ifs_* norms (optimize.c), copied
   operation for operation. Built with -fmad=false, it rounds exactly
   like the CPU code compiled without floating-point contraction.

   A collision's candidate slot is slot_base + its index in the found
   array (the host collects several batches before optimizing); slots
   that survive rebuilding are listed (compacted) for the later
   kernels. Every lane of a CGBN group runs the scalar code too, and
   only lane 0 writes scalar results. */

#include "cuda_intrinsics.h"
#include <cgbn/cgbn.h>
#include "stage1_core.h"
#include "sizeopt_core.h"
#include "../../stage2/line_sieve_core.h"
#include "../../stage2/murphy_core.h"
#include "../../stage2/alpha_core.h"
#include "../../stage2/lattice_core.h"

#define SO_SKEWNESS 0
#define SO_TRANSLATE_SIZE 1
#define SO_ROTATE0 2

#define SO_PHI 1.618033988749
#define SO_COMP_PHI (2.0 - SO_PHI)

#define SO_MIN(a,b) ((a) < (b)? (a) : (b))
#define SO_MAX(a,b) ((a) > (b)? (a) : (b))

#define SO_ROOT_BUF_SIZE 200	/* ROOT_BUF_SIZE in root_score.c */

/*------------------------------------------------------------------------*/
/* double precision part: optimize.c and minimize.c */


__device__ __forceinline__ static double
so_ifs_rectangular(const double *a, uint32 degree, double s)
{
	double a0, a1, a2, a3, a4, a5, a6;
	double s2, s3, s4, s5, s6;
	double norm;

	if (s < 1)
		return 1e200;

	a0 = a[0];
	a1 = a[1] * s;
	s2 = s * s;
	a2 = a[2] * s2;
	s3 = s2 * s;
	a3 = a[3] * s3;
	s4 = s3 * s;
	a4 = a[4] * s4;

	switch (degree) {
	case 4:
		norm = 1.0 / 9.0 * (a4 * a4 + a0 * a0) +
		       2.0 / 21.0 * (a4 * a2 + a2 * a0) +
		       1.0 / 21.0 * (a3 * a3 + a1 * a1) +
		       2.0 / 25.0 * (a4 * a0 + a3 * a1) +
		       1.0 / 25.0 * a2 * a2;
		return norm / s4;

	case 5:
		s5 = s4 * s;
		a5 = a[5] * s5;
		norm = 1.0 / 11.0 * (a5 * a5 + a0 * a0) +
		       2.0 / 27.0 * (a5 * a3 + a2 * a0) +
		       1.0 / 27.0 * (a4 * a4 + a1 * a1) +
		       1.0 / 35.0 * (a3 * a3 + a2 * a2) +
		       2.0 / 35.0 * (a5 * a1 + a4 * a2 + a4 * a0 + a3 * a1);
		return norm / s5;

	case 6:
		s5 = s4 * s;
		a5 = a[5] * s5;
		s6 = s5 * s;
		a6 = a[6] * s6;
		norm = 1.0 / 13.0 * (a6 * a6 + a0 * a0) +
		       2.0 / 33.0 * (a6 * a4 + a2 * a0) +
		       1.0 / 33.0 * (a5 * a5 + a1 * a1) +
		       1.0 / 45.0 * (a4 * a4 + a2 * a2) +
		       2.0 / 45.0 * (a6 * a2 + a5 * a3 + a4 * a0 + a3 * a1) +
		       2.0 / 49.0 * (a6 * a0 + a5 * a1 + a4 * a2) +
		       1.0 / 49.0 * a3 * a3;
		return norm / s6;
	}

	return 1e200;
}

__device__ __forceinline__ static double
so_ifs_radial(const double *a, uint32 degree, double s)
{
	double a0, a1, a2, a3, a4, a5, a6;
	double s2, s3, s4, s5, s6;
	double norm;

	if (s < 1)
		return 1e200;

	a0 = a[0];
	a1 = a[1] * s;
	s2 = s * s;
	a2 = a[2] * s2;
	s3 = s2 * s;
	a3 = a[3] * s3;
	s4 = s3 * s;
	a4 = a[4] * s4;

	switch (degree) {
	case 4:
		norm = 35.0 * (a4 * a4 + a0 * a0) +
		       10.0 * (a4 * a2 + a2 * a0) +
		        5.0 * (a3 * a3 + a1 * a1) +
		        6.0 * (a4 * a0 + a3 * a1) +
		        3.0 * a2 * a2;
		return norm / s4;

	case 5:
		s5 = s4 * s;
		a5 = a[5] * s5;
		norm = 63.0 * (a5 * a5 + a0 * a0) +
		       14.0 * (a5 * a3 + a2 * a0) +
		        7.0 * (a4 * a4 + a1 * a1) +
		        3.0 * (a3 * a3 + a2 * a2) +
		        6.0 * (a5 * a1 + a4 * a2 + a4 * a0 + a3 * a1);
		return norm / s5;

	case 6:
		s5 = s4 * s;
		a5 = a[5] * s5;
		s6 = s5 * s;
		a6 = a[6] * s6;
		norm = 231.0 * (a6 * a6 + a0 * a0) +
		        42.0 * (a6 * a4 + a2 * a0) +
		        21.0 * (a5 * a5 + a1 * a1) +
		         7.0 * (a4 * a4 + a2 * a2) +
		        14.0 * (a6 * a2 + a5 * a3 + a4 * a0 + a3 * a1) +
		        10.0 * (a6 * a0 + a5 * a1 + a4 * a2) +
		         5.0 * a3 * a3;
		return norm / s6;
	}

	return 1e200;
}

__device__ static const double so_xlate_weights[6+1][6+1] = {
	{  1.,  1.,  1.,  1.,  1.,  1.,  1.},
	{  0.,  1.,  2.,  3.,  4.,  5.,  6.},
	{  0.,  0.,  1.,  3.,  6., 10., 15.},
	{  0.,  0.,  0.,  1.,  4., 10., 20.},
	{  0.,  0.,  0.,  0.,  1.,  5., 15.},
	{  0.,  0.,  0.,  0.,  0.,  1.,  6.},
	{  0.,  0.,  0.,  0.,  0.,  0.,  1.},
};

__device__ __forceinline__ static void
so_translate_d(double *c, const double *d, uint32 deg, double x)
{
	uint32 i, j;

	for (i = 0, x = -x; i < deg; i++) {

		const double *weights = so_xlate_weights[i];
		double accum = d[deg] * weights[deg];

		for (j = deg; j > i; j--) {
			accum = accum * x + d[j-1] * weights[j-1];
		}
		c[i] = accum;
	}
	c[i] = d[i];
}

/* minimize() of common/minimize.c (with minimize_line, bracket_min and
   minimize_line_core), ndim > 1, as a resumable state machine.

   so_mz_step() runs one search until it needs the objective at mz->x,
   and returns 1; the caller evaluates it and calls again with the
   value. It returns 0 when minimize() would have returned (result in
   mz->result, the point in mz->p). This way the threads of a warp
   evaluate the objective together, at one call site, whatever part
   of the search each is in: run as nested calls, the bracketing, the
   Brent steps and the Powell steps of different threads took turns.

   Every arithmetic operation is the original's, in the original order,
   so the results are bit for bit those of the CPU code. */

enum {
	MZ_START, MZ_GOT_BEST0, MZ_ITER, MZ_DIR, MZ_AFTER_DIR_LINE,
	MZ_DIRS_DONE, MZ_GOT_EXTRAP, MZ_AFTER_EXTRAP_LINE,
	MZ_L_START, MZ_B_FA, MZ_B_FB, MZ_B_WHILE1, MZ_B_FB2, MZ_B_AFTER1,
	MZ_B_FC, MZ_B_LOOP, MZ_B_CASE1, MZ_B_CASE2, MZ_B_SHIFT_EVAL,
	MZ_B_SHIFT, MZ_B_END, MZ_C_START, MZ_C_LOOP, MZ_C_GOT, MZ_L_END
};

template<uint32 NDIM>
struct so_mz_t {
	uint32 pc;
	uint32 max_iter;
	double ftol;		/* minimize()'s tolerance */
	double ltol;		/* MAX(ftol, 1e-7), the line searches' */
	double x[NDIM];	/* point to evaluate */

	/* minimize */
	double p[NDIM];
	double p_old[NDIM], p_extrap[NDIM];
	double dir_extrap[NDIM];
	double directions[NDIM][NDIM];
	double best_f, curr_f, start_f, del, result;
	uint32 it, j, ibig;
	int32 status;

	/* minimize_line: along directions[line_dir], or dir_extrap
	   when line_dir < 0; line_extrap says where to resume */
	int32 line_dir;
	uint32 line_extrap;
	double normalized[NDIM];
	double backup[NDIM];
	double line_result, min_dist;

	/* bracket_min */
	double a, b, c, fa, fb, fc, u, fu;
	double ba, bb, bc, bfb;

	/* minimize_line_core */
	double ca, cb, cx, cw, cv, cu;
	double fx, fw, fv, cfu, e;
	uint32 ci;
};

/* c ? b : a, as a blend of the bit patterns: exact for every value,
   and it keeps the compiler from turning a select over all rows into
   a computed index (which would put the whole state in memory) */
__device__ __forceinline__ static double
so_sel(double a, double b, bool c)
{
	long long m = -(long long)c;
	return __longlong_as_double((__double_as_longlong(a) & ~m) |
				(__double_as_longlong(b) & m));
}

/* evaluate(base = p, dist, normalized) */
template<uint32 NDIM>
__device__ __forceinline__ static void
so_mz_point(so_mz_t<NDIM> *mz, double dist)
{
	uint32 i;

#pragma unroll
	for (i = 0; i < NDIM; i++)
		mz->x[i] = mz->p[i] + dist * mz->normalized[i];
}

template<uint32 NDIM>
__device__ __forceinline__ static uint32
so_mz_step(so_mz_t<NDIM> *mz, double f, uint32 *line_iters)
{
	uint32 i, j;

	for (;;) {
	switch (mz->pc) {

	/*--------------------------------------- minimize() */
	case MZ_START:
#pragma unroll
		for (i = 0; i < NDIM; i++) {
#pragma unroll
			for (j = 0; j < NDIM; j++)
				mz->directions[i][j] = (i == j) ? 1.0 : 0.0;
			mz->p_old[i] = mz->p[i];
		}
		mz->status = 0;
#pragma unroll
		for (i = 0; i < NDIM; i++)
			mz->x[i] = mz->p[i];
		mz->pc = MZ_GOT_BEST0;
		return 1;

	case MZ_GOT_BEST0:
		mz->best_f = f;
		mz->it = 0;
		mz->pc = MZ_ITER;
		break;

	case MZ_ITER:
		if (mz->it >= mz->max_iter) {
			mz->result = mz->best_f;
			return 0;
		}
		mz->ibig = 0;
		mz->del = 0.0;
		mz->start_f = mz->best_f;
		mz->j = 0;
		mz->pc = MZ_DIR;
		break;

	case MZ_DIR:
		if (mz->j >= NDIM) {
			mz->pc = MZ_DIRS_DONE;
			break;
		}
		mz->line_dir = (int32)mz->j;
		mz->line_extrap = 0;
		mz->pc = MZ_L_START;
		break;

	case MZ_AFTER_DIR_LINE:
		mz->curr_f = mz->line_result;
		if (mz->status) {
			mz->result = mz->best_f;
			return 0;
		}
		if (mz->best_f - mz->curr_f > mz->del) {
			mz->ibig = mz->j;
			mz->del = mz->best_f - mz->curr_f;
		}
		mz->best_f = mz->curr_f;
		mz->j++;
		mz->pc = MZ_DIR;
		break;

	case MZ_DIRS_DONE:
		if (2.0 * fabs(mz->start_f - mz->best_f) <= 
				mz->ftol * (fabs(mz->start_f) + 
					fabs(mz->best_f)) + 1e-20) {
			mz->result = mz->best_f;
			return 0;
		}
#pragma unroll
		for (j = 0; j < NDIM; j++) {
			mz->p_extrap[j] = 2.0 * mz->p[j] - mz->p_old[j];
			mz->dir_extrap[j] = mz->p[j] - mz->p_old[j];
			mz->p_old[j] = mz->p[j];
		}
#pragma unroll
		for (i = 0; i < NDIM; i++)
			mz->x[i] = mz->p_extrap[i];
		mz->pc = MZ_GOT_EXTRAP;
		return 1;

	case MZ_GOT_EXTRAP:
		mz->curr_f = f;
		if (mz->curr_f < mz->start_f) {

			double t1 = mz->start_f - mz->best_f - mz->del;
			double t2 = mz->start_f - mz->curr_f;
			double t = 2.0 * (mz->start_f - 2.0 * mz->best_f + 
					mz->curr_f) * t1 * t1 - 
					mz->del * t2 * t2;

			if (t < 0.0) {
				mz->line_dir = -1;
				mz->line_extrap = 1;
				mz->pc = MZ_L_START;
				break;
			}
		}
		mz->it++;
		mz->pc = MZ_ITER;
		break;

	case MZ_AFTER_EXTRAP_LINE:
		mz->best_f = mz->line_result;
		if (mz->status) {
			mz->result = mz->best_f;
			return 0;
		}
#pragma unroll
		for (j = 0; j < NDIM; j++) {
#pragma unroll
			for (i = 0; i < NDIM; i++)
				mz->directions[i][j] = so_sel(
						mz->directions[i][j],
						mz->directions[NDIM-1][j],
						i == mz->ibig);
			mz->directions[NDIM-1][j] = mz->dir_extrap[j];
		}
		mz->it++;
		mz->pc = MZ_ITER;
		break;

	/*--------------------------------------- minimize_line() */
	case MZ_L_START:
	{
		double dir[NDIM];
		double max_dir;

		/* the search direction, selected without a computed
		   index so that the state can stay in registers */
#pragma unroll
		for (j = 0; j < NDIM; j++) {
			dir[j] = mz->dir_extrap[j];
#pragma unroll
			for (i = 0; i < NDIM; i++)
				dir[j] = so_sel(dir[j], mz->directions[i][j],
						(int32)i == mz->line_dir);
		}

#pragma unroll
		for (i = 0; i < NDIM; i++)
			mz->backup[i] = mz->p[i];

		max_dir = fabs(dir[0]);
#pragma unroll
		for (i = 1; i < NDIM; i++)
			max_dir = SO_MAX(max_dir, fabs(dir[i]));
#pragma unroll
		for (i = 0; i < NDIM; i++)
			mz->normalized[i] = dir[i] / max_dir;

		/* bracket_min() */
		mz->a = 0;
		so_mz_point(mz, mz->a);
		mz->pc = MZ_B_FA;
		return 1;
	}

	case MZ_B_FA:
		mz->fa = f;
		mz->b = 1;
		so_mz_point(mz, mz->b);
		mz->pc = MZ_B_FB;
		return 1;

	case MZ_B_FB:
		mz->fb = f;
		mz->pc = MZ_B_WHILE1;
		break;

	case MZ_B_WHILE1:
		if (fabs(mz->fb - mz->fa) < 10 * mz->ltol * fabs(mz->fa)) {
			mz->b *= 2;
			so_mz_point(mz, mz->b);
			mz->pc = MZ_B_FB2;
			return 1;
		}
		mz->pc = MZ_B_AFTER1;
		break;

	case MZ_B_FB2:
		mz->fb = f;
		mz->pc = MZ_B_WHILE1;
		break;

	case MZ_B_AFTER1:
		if (mz->fb > mz->fa) {
			double t;
			t = mz->a; mz->a = mz->b; mz->b = t;
			t = mz->fa; mz->fa = mz->fb; mz->fb = t;
		}
		mz->c = mz->b + SO_PHI * (mz->b - mz->a);
		so_mz_point(mz, mz->c);
		mz->pc = MZ_B_FC;
		return 1;

	case MZ_B_FC:
		mz->fc = f;
		mz->pc = MZ_B_LOOP;
		break;

	case MZ_B_LOOP:
	{
		double a = mz->a, b = mz->b, c = mz->c;
		double fa = mz->fa, fb = mz->fb, fc = mz->fc;
		double r, q, t, u, max_u;

		if (!(fb > fc)) {
			mz->pc = MZ_B_END;
			break;
		}

		r = (b - a) * (fb - fc);
		q = (b - c) * (fb - fa);

		t = fabs(q - r);
		t = SO_MAX(t, 1e-20);
		if (q - r < 0.0)
			t = -t;

		u = b - ((b - c) * q - (b - a) * r) / (2.0 * t);
		max_u = b + 100.0 * (c - b);

		if ((b - u) * (u - c) > 0.0) {
			mz->u = u;
			so_mz_point(mz, u);
			mz->pc = MZ_B_CASE1;
			return 1;
		}
		else if ((c - u) * (u - max_u) > 0.0) {
			mz->u = u;
			so_mz_point(mz, u);
			mz->pc = MZ_B_CASE2;
			return 1;
		}
		else if ((u - max_u) * (max_u - c) >= 0.0) {
			u = max_u;
		}
		else {
			u = c + SO_PHI * (c - b);
		}
		mz->u = u;
		so_mz_point(mz, u);
		mz->pc = MZ_B_SHIFT_EVAL;
		return 1;
	}

	case MZ_B_CASE1:
		mz->fu = f;
		if (mz->fu < mz->fc) {
			mz->ba = mz->b; mz->bb = mz->u; mz->bc = mz->c;
			mz->bfb = mz->fu;
			mz->pc = MZ_C_START;
			break;
		}
		else if (mz->fu > mz->fb) {
			mz->ba = mz->a; mz->bb = mz->b; mz->bc = mz->u;
			mz->bfb = mz->fb;
			mz->pc = MZ_C_START;
			break;
		}
		mz->u = mz->c + SO_PHI * (mz->c - mz->b);
		so_mz_point(mz, mz->u);
		mz->pc = MZ_B_SHIFT_EVAL;
		return 1;

	case MZ_B_CASE2:
		mz->fu = f;
		if (mz->fu < mz->fc) {
			mz->b = mz->c; mz->c = mz->u;
			mz->u = mz->c + SO_PHI * (mz->c - mz->b);
			mz->fb = mz->fc; 
			mz->fc = mz->fu; 
			so_mz_point(mz, mz->u);
			mz->pc = MZ_B_SHIFT_EVAL;
			return 1;
		}
		mz->pc = MZ_B_SHIFT;
		break;

	case MZ_B_SHIFT_EVAL:
		mz->fu = f;
		mz->pc = MZ_B_SHIFT;
		break;

	case MZ_B_SHIFT:
		mz->a = mz->b; mz->b = mz->c; mz->c = mz->u;
		mz->fa = mz->fb; mz->fb = mz->fc; mz->fc = mz->fu;
		mz->pc = MZ_B_LOOP;
		break;

	case MZ_B_END:
		mz->ba = mz->a; mz->bb = mz->b; mz->bc = mz->c;
		mz->bfb = mz->fb;
		mz->pc = MZ_C_START;
		break;

	/*--------------------------------------- minimize_line_core() */
	case MZ_C_START:
		mz->ca = SO_MIN(mz->ba, mz->bc);
		mz->cb = SO_MAX(mz->ba, mz->bc);
		mz->cx = mz->cw = mz->cv = mz->bb;
		mz->fx = mz->fw = mz->fv = mz->bfb;
		mz->e = 0.0;
		mz->ci = 0;
		mz->pc = MZ_C_LOOP;
		break;

	case MZ_C_LOOP:
	{
		double a = mz->ca, b = mz->cb, x = mz->cx;
		double w = mz->cw, v = mz->cv;
		double fx = mz->fx, fw = mz->fw, fv = mz->fv;
		double e = mz->e, u;
		double xm, tol1, tol2;
		double d = 0, etemp;

		if (mz->ci >= 300) {
			/* the CPU code prints ':' here; the host prints
			   the count */
			atomicAdd(line_iters, 1);
			mz->min_dist = x;
			mz->status = 1;
			mz->line_result = fx;
			mz->pc = MZ_L_END;
			break;
		}

		xm = 0.5 * (a + b);
		tol1 = mz->ltol * fabs(x) + 1e-20;
		tol2 = 2.0 * tol1;

		if (fabs(x - xm) <= tol2 - 0.5 * (b - a)) {
			mz->min_dist = x;
			mz->line_result = fx;
			mz->pc = MZ_L_END;
			break;
		}

		if (fabs(e) > tol1) {
			double r = (x - w) * (fx - fv);
			double q = (x - v) * (fx - fw);
			double p = (x - v) * q - (x - w) * r;
			
			q = 2.0 * (q - r);
			if (q > 0.0)
				p = -p;
			q = fabs(q);
			etemp = e;
			e = d;
			if (fabs(p) >= fabs(0.5 * q * etemp) ||
			    p <= q * (a - x) || p >= q * (b - x)) {
	
				e = b - x;
				if (x >= xm)
					e = a - x;				
				d = SO_COMP_PHI * e;
			}
			else {
				d = p / q;
				u = x + d;
				if (u - a < tol2 || b - u < tol2)
					d = (xm >= x) ? tol1 : -tol1;
			}
		}
		else {
			e = (x >= xm) ? a - x : b - x;
			d = SO_COMP_PHI * e;
		}

		if (fabs(d) < tol1)
			u = x + ((d >= 0) ? tol1 : -tol1);
		else
			u = x + d;

		mz->e = e;
		mz->cu = u;
		so_mz_point(mz, u);
		mz->pc = MZ_C_GOT;
		return 1;
	}

	case MZ_C_GOT:
	{
		double a = mz->ca, b = mz->cb, x = mz->cx;
		double w = mz->cw, v = mz->cv, u = mz->cu;
		double fx = mz->fx, fw = mz->fw, fv = mz->fv;
		double fu = f;

		if (fu <= fx) {
			if (u >= x)
				a = x;
			else
				b = x;

			v = w; w = x; x = u;
			fv = fw; fw = fx; fx = fu;
		}
		else {
			if (u < x)
				a = u;
			else
				b = u;
			if (fu <= fw || w == x) {
				v = w; w = u;
				fv = fw; fw = fu;
			}
			else if (fu <= fv || v == x || v == w) {
				v = u;
				fv = fu;
			}
		}
		mz->ca = a; mz->cb = b; mz->cx = x;
		mz->cw = w; mz->cv = v;
		mz->fx = fx; mz->fw = fw; mz->fv = fv;
		mz->ci++;
		mz->pc = MZ_C_LOOP;
		break;
	}

	/*--------------------------------------- end of minimize_line() */
	case MZ_L_END:
		if (mz->status) {
#pragma unroll
			for (i = 0; i < NDIM; i++)
				mz->p[i] = mz->backup[i];
		}
		else {
#pragma unroll
			for (i = 0; i < NDIM; i++) {
				double v = mz->normalized[i] * mz->min_dist;

				mz->dir_extrap[i] = so_sel(mz->dir_extrap[i], v,
						mz->line_dir < 0);
#pragma unroll
				for (j = 0; j < NDIM; j++)
					mz->directions[j][i] = so_sel(
						mz->directions[j][i], v,
						(int32)j == mz->line_dir);
				mz->p[i] += v;
			}
		}
		mz->pc = mz->line_extrap ? MZ_AFTER_EXTRAP_LINE :
					   MZ_AFTER_DIR_LINE;
		break;
	}
	}
}

/*------------------------------------------------------------------------*/
/* poly_rotate_callback for a fixed degree, so that its arrays are
   registers */

template<uint32 DEG>
__device__ __forceinline__ static double
so_objective_deg(const double *v, const double *ad, double r0, double r1,
		uint32 radial)
{
	uint32 i;
	double apoly[DEG + 1];
	double translated[DEG + 1];
	double s = floor(v[SO_SKEWNESS] + 0.5);
	double t = floor(v[SO_TRANSLATE_SIZE] + 0.5);

	if (s < 1.0)
		return 1e200;

#pragma unroll
	for (i = 0; i <= DEG; i++)
		apoly[i] = ad[i];

#pragma unroll
	for (i = 0; i <= DEG - 4; i++) {
		double c = floor(v[SO_ROTATE0 + i] + 0.5);
		apoly[i] += r0 * c;
		apoly[i+1] += r1 * c;
	}
	
	so_translate_d(translated, apoly, DEG, t);
	if (radial)
		return so_ifs_radial(translated, DEG, s);
	return so_ifs_rectangular(translated, DEG, s);
}

/*------------------------------------------------------------------------*/
/* one minimize() call of optimize_initial's loop for every listed
   active candidate. Each thread runs one search at a time and takes
   the next candidate when it finishes; the objective is evaluated by
   the whole warp at once */

template<uint32 DEG>
__device__ static void
so_minimize_deg(so_state_t *state, const uint32 *list,
		uint32 *counters, const so_params_t &prm)
{
	const uint32 NDIM = DEG - 1;
	uint32 *line_iters = counters + SO_STAT_LINE_ITERS;
	uint32 n = counters[SO_STAT_CANDS];
	so_mz_t<NDIM> mz;
	double ad[DEG + 1], r0 = 0, r1 = 0;
	uint32 radial = 0;
	uint32 h = 0, i, j;
	uint32 active = 1;

#pragma unroll
	for (j = 0; j <= DEG; j++)
		ad[j] = 0;

	/* fetch the next active candidate and start its search */
#define SO_FETCH()							\
	for (;;) {							\
		so_state_t *s;						\
		if ((i = atomicAdd(counters + SO_STAT_NEXT, 1)) >= n) {	\
			active = 0;					\
			break;						\
		}							\
		h = list[i];						\
		s = state + h;						\
		if (s->status != SO_ACTIVE)				\
			continue;					\
		_Pragma("unroll")					\
		for (j = 0; j <= DEG; j++)				\
			ad[j] = s->apoly[j];				\
		r0 = s->r0;						\
		r1 = s->r1;						\
		radial = (s->pass == 0);				\
		mz.pc = MZ_START;					\
		mz.max_iter = 40;					\
		mz.ftol = 1e-5;						\
		mz.ltol = SO_MAX(mz.ftol, 1e-7);			\
		_Pragma("unroll")					\
		for (j = 0; j < NDIM; j++)				\
			mz.p[j] = s->best[j];				\
		s->last_score = s->score;				\
		so_mz_step<NDIM>(&mz, 0, line_iters);			\
		break;							\
	}

	SO_FETCH()

	while (__any_sync(0xffffffff, active)) {
		double f = 0;

		__syncwarp();
		if (active)
			f = so_objective_deg<DEG>(mz.x, ad, r0, r1, radial);

		if (active && !so_mz_step<NDIM>(&mz, f, line_iters)) {

			so_state_t *s = state + h;

			s->score = mz.result;
#pragma unroll
			for (j = 0; j <= DEG - 4; j++)
				s->rot[j] = floor(mz.p[SO_ROTATE0 + j] + 0.5);
			s->xlate = (int64)(mz.p[SO_TRANSLATE_SIZE] + 0.5);
#pragma unroll
			for (j = 0; j < NDIM; j++)
				s->best[j] = mz.p[j];

			SO_FETCH()
		}
	}
#undef SO_FETCH
}

extern "C" __global__ void
so_minimize_4(so_state_t *state, const uint32 *list,
		uint32 *counters, so_params_t prm)
{
	so_minimize_deg<4>(state, list, counters, prm);
}

extern "C" __global__ void
so_minimize_5(so_state_t *state, const uint32 *list,
		uint32 *counters, so_params_t prm)
{
	so_minimize_deg<5>(state, list, counters, prm);
}

/*------------------------------------------------------------------------*/
/* optimize_basic() for root optimization: minimize() over skewness and
   translation with poly_xlate_callback (the skewness is not rounded,
   the translation is). One request per call; threads take requests
   from a counter and the warp evaluates the objective together, as in
   so_minimize_deg */

template<uint32 DEG>
__device__ __forceinline__ static double
so_xlate_objective(const double *v, const double *ad)
{
	double translated[DEG + 1];
	double t = floor(v[SO_TRANSLATE_SIZE] + 0.5);
	double s = v[SO_SKEWNESS];

	so_translate_d(translated, ad, DEG, t);
	return so_ifs_rectangular(translated, DEG, s);
}

template<uint32 DEG>
__device__ static void
so_basic_deg(const so_basic_in_t *in, so_basic_out_t *out, uint32 n,
		uint32 *counters)
{
	uint32 *line_iters = counters + SO_BASIC_LINE_ITERS;
	so_mz_t<2> mz;
	double ad[DEG + 1];
	uint32 r = 0, i, j;
	uint32 active = 1;

#pragma unroll
	for (j = 0; j <= DEG; j++)
		ad[j] = 0;

/* an entry whose apoly[0] is NaN is no call (unused candidate slots of
   the line sieve's) */
#define SO_BASIC_FETCH()						\
	while ((i = atomicAdd(counters + SO_BASIC_NEXT, 1)) < n &&	\
		isnan(in[i].apoly[0]))					\
		;							\
	if (i >= n) {							\
		active = 0;						\
	}								\
	else {								\
		r = i;							\
		_Pragma("unroll")					\
		for (j = 0; j <= DEG; j++)				\
			ad[j] = in[r].apoly[j];				\
		mz.pc = MZ_START;					\
		mz.max_iter = 40;					\
		mz.ftol = 1e-5;						\
		mz.ltol = SO_MAX(mz.ftol, 1e-7);			\
		mz.p[SO_SKEWNESS] = 1000;				\
		mz.p[SO_TRANSLATE_SIZE] = 0;				\
		so_mz_step<2>(&mz, 0, line_iters);			\
	}

	SO_BASIC_FETCH()

	while (__any_sync(0xffffffff, active)) {
		double f = 0;

		__syncwarp();
		if (active)
			f = so_xlate_objective<DEG>(mz.x, ad);

		if (active && !so_mz_step<2>(&mz, f, line_iters)) {
			out[r].norm = sqrt(mz.result);
			out[r].skewness = mz.p[SO_SKEWNESS];
			out[r].translation = 
				floor(mz.p[SO_TRANSLATE_SIZE] + 0.5);
			SO_BASIC_FETCH()
		}
	}
#undef SO_BASIC_FETCH
}

#define SO_BASIC_KERNEL(DEG)						\
extern "C" __global__ void						\
so_basic_##DEG(const so_basic_in_t *in, so_basic_out_t *out,		\
		uint32 n, uint32 *counters)				\
{									\
	so_basic_deg<DEG>(in, out, n, counters);			\
}

SO_BASIC_KERNEL(4)
SO_BASIC_KERNEL(5)
SO_BASIC_KERNEL(6)

/*------------------------------------------------------------------------*/
/* CGBN helpers */

/* mpz_get_d: truncate toward zero to 53 bits */
template<class env_t>
__device__ static double
so_get_d(env_t env, const typename env_t::cgbn_t &a)
{
	typename env_t::cgbn_t m;
	bool neg = cgbn_signed_abs(env, m, a);
	uint32 nb = env_t::BITS - cgbn_clz(env, m);
	uint32 sh = (nb > 53) ? nb - 53 : 0;
	uint64 top = (uint64)cgbn_extract_bits_ui32(env, m, sh, 32) |
		((uint64)cgbn_extract_bits_ui32(env, m, sh + 32, 21) << 32);
	double r = ldexp((double)top, (int)sh);

	return neg ? -r : r;
}

/* sign of mpz_cmpabs_d(a, x): compare |a| with |x| exactly */
template<class env_t>
__device__ static int32
so_cmpabs_d(env_t env, const typename env_t::cgbn_t &a, double x)
{
	typename env_t::cgbn_t m;
	uint32 nb, sh;
	double t;

	x = fabs(x);
	cgbn_signed_abs(env, m, a);
	nb = env_t::BITS - cgbn_clz(env, m);
	sh = (nb > 53) ? nb - 53 : 0;
	t = ldexp((double)((uint64)cgbn_extract_bits_ui32(env, m, sh, 32) |
		((uint64)cgbn_extract_bits_ui32(env, m, sh + 32, 21) << 32)),
		(int)sh);
	if (t > x)
		return 1;
	if (t < x)
		return -1;
	/* t == x: |a| > x exactly when bits below the top 53 are set */
	if (sh > 0 && cgbn_ctz(env, m) < sh)
		return 1;
	return 0;
}

template<class env_t>
__device__ static void
so_set_u64(env_t env, typename env_t::cgbn_t &r, uint64 v)
{
	cgbn_set_ui32(env, r, (uint32)v);
	cgbn_insert_bits_ui32(env, r, r, 32, 32, (uint32)(v >> 32));
}

template<class env_t>
__device__ static void
so_set_i64(env_t env, typename env_t::cgbn_t &r, int64 v)
{
	so_set_u64(env, r, (uint64)v);
	if (v < 0)
		cgbn_bitwise_mask_ior(env, r, r, -(int32)(env_t::BITS - 64));
}

/* a mod m in [0, m), for m > 0 */
template<class env_t>
__device__ static void
so_mod(env_t env, typename env_t::cgbn_t &r,
	const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &m)
{
	cgbn_signed_rem(env, r, a, m);
	if (cgbn_signed_is_negative(env, r))
		cgbn_add(env, r, r, m);
}

/* mpz_fdiv_ui */
template<class env_t>
__device__ static uint32
so_fdiv_ui(env_t env, const typename env_t::cgbn_t &a, uint32 m)
{
	typename env_t::cgbn_t t;
	bool neg = cgbn_signed_abs(env, t, a);
	uint32 r = cgbn_rem_ui32(env, t, m);

	if (neg && r != 0)
		r = m - r;
	return r;
}

/* r = a * b mod n, for 0 <= a, b < n */
template<class env_t>
__device__ static void
so_mulmod(env_t env, typename env_t::cgbn_t &r,
	const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b,
	const typename env_t::cgbn_t &n)
{
	typename env_t::cgbn_wide_t w;
	cgbn_mul_wide(env, w, a, b);
	cgbn_rem_wide(env, r, w, n);
}

__device__ static uint64
so_gcd64(uint64 a, uint64 b)
{
	while (b != 0) {
		uint64 t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/*------------------------------------------------------------------------*/
/* handle_collision + pol_expand + check_poly, one CGBN group per
   collision. Returns 2 for a polynomial to optimize (as pol_expand),
   1 for a valid one over the coefficient bound, 0 to drop it */

template<class env_t>
__device__ static uint32
so_rebuild(env_t env, const typename env_t::cgbn_t *k, uint32 deg,
	uint64 p, uint64 q, const uint32 *qroot, int64 res,
	const so_params_t &prm, uint32 *counters,
	typename env_t::cgbn_t *a, typename env_t::cgbn_t &P,
	typename env_t::cgbn_t &d)
{
	typedef typename env_t::cgbn_t bn_t;
	bn_t t1, t2, t3, m, mod;
	bn_t b[SO_MAX_DEG + 1], c[SO_MAX_DEG + 1];
	uint32 i, j;

	/* handle_collision: p and special_q must be coprime */

	if (so_gcd64(p, q) != 1)
		return 0;

	so_set_u64(env, t1, p);
	so_set_u64(env, t2, q);
	cgbn_mul(env, P, t1, t2);

	/* correction special_q_root + res * special_q^2 */

	cgbn_set_ui32(env, t3, qroot[0]);
	cgbn_insert_bits_ui32(env, t3, t3, 32, 32, qroot[1]);
	cgbn_insert_bits_ui32(env, t3, t3, 64, 32, qroot[2]);
	cgbn_insert_bits_ui32(env, t3, t3, 96, 32, qroot[3]);
	so_set_i64(env, t2, res);
	so_set_u64(env, t1, q);
	cgbn_mul(env, t1, t1, t1);		/* q^2 */
	cgbn_mul(env, t2, t2, t1);
	cgbn_add(env, t3, t3, t2);

	{
		double dp = so_get_d(env, P);
		if (fabs(so_get_d(env, t3)) > prm.cmax_over_m0 * dp * dp)
			return 0;
	}
	cgbn_add(env, m, k[SO_CONST_TRANS_M0], t3);

	/* Kleinjung's modular condition */

	cgbn_set(env, t1, m);
	for (i = 1; i < deg; i++)
		cgbn_mul(env, t1, t1, m);
	cgbn_mul(env, t2, P, P);
	cgbn_sub(env, t1, k[SO_CONST_TRANS_N], t1);
	cgbn_signed_rem(env, t3, t1, t2);
	if (!cgbn_equals_ui32(env, t3, 0)) {
		if (threadIdx.x % env_t::TPI == 0)
			atomicAdd(counters + SO_STAT_CRAP, 1);
		return 0;
	}

	/* translate m back to the original polynomial */

	cgbn_set(env, mod, k[SO_CONST_MOD]);
	cgbn_signed_rem(env, t2, m, mod);
	cgbn_rem(env, t1, P, mod);
	if (!cgbn_modular_inverse(env, t3, t1, mod))
		cgbn_set_ui32(env, t3, 0);	/* GMP leaves the old 0 */
	cgbn_mul(env, t2, t3, t2);
	cgbn_signed_rem(env, t2, t2, mod);
	cgbn_shift_right(env, t3, mod, 1);
	if (cgbn_signed_compare(env, t2, t3) > 0)
		cgbn_sub(env, t2, t2, mod);
	cgbn_mul(env, t1, t2, P);
	cgbn_sub(env, m, m, t1);
	cgbn_signed_div(env, d, m, mod);

	/* pol_expand(N, a_d, P, d) */

	if (cgbn_equals_ui32(env, P, 1)) {
		cgbn_set_ui32(env, t1, 1);
	}
	else {
		so_mod(env, t2, d, P);
		if (!cgbn_modular_inverse(env, t1, t2, P))
			goto expand_failed;
	}

	/* powers of 1/d, reduced mod P (only their residues are used) */
	cgbn_set(env, b[1], t1);
	for (i = 2; i < deg; i++) {
		cgbn_mul(env, b[i], b[i-1], t1);
		cgbn_rem(env, b[i], b[i], P);
	}

	cgbn_set(env, c[1], d);
	for (i = 2; i <= deg; i++)
		cgbn_mul(env, c[i], c[i-1], d);

	cgbn_set(env, a[deg], k[SO_CONST_AD]);
	cgbn_set(env, t2, k[SO_CONST_N]);	/* help2 */

	for (i = deg - 1; (int32)i >= 0; i--) {

		cgbn_mul(env, t3, a[i+1], c[i+1]);
		cgbn_sub(env, t3, t2, t3);
		cgbn_signed_div(env, t2, t3, P);

		if (i > 0) {
			cgbn_signed_div(env, a[i], t2, c[i]);

			/* (help2 * b[i] - a[i]) mod P, in [0, P) */
			so_mod(env, t3, t2, P);
			cgbn_mul(env, t3, t3, b[i]);
			cgbn_sub(env, t3, t3, a[i]);
			so_mod(env, t1, t3, P);
			cgbn_add(env, a[i], a[i], t1);
		}
	}
	cgbn_set(env, a[0], t2);

	/* bring each coefficient into (-d/2, d/2] */

	cgbn_signed_abs(env, t1, d);
	cgbn_shift_right(env, t1, t1, 1);	/* |help1| */
	for (i = 0; i < deg; i++) {
		for (j = 0; j < 10; j++) {
			cgbn_signed_abs(env, t3, a[i]);
			if (cgbn_compare(env, t3, t1) <= 0)
				break;

			if (cgbn_signed_is_negative(env, a[i])) {
				cgbn_add(env, a[i], a[i], d);
				cgbn_sub(env, a[i+1], a[i+1], P);
			}
			else {
				cgbn_sub(env, a[i], a[i], d);
				cgbn_add(env, a[i+1], a[i+1], P);
			}
		}
		if (j == 10)
			goto expand_failed;
	}

	/* check_poly: a(x) at x = d/P, homogenized, vanishes mod N.
	   Computed mod N (P^d alone can exceed the width) */
	{
		const bn_t &N = k[SO_CONST_N];
		bn_t h1, h2, L, ai, Pm;

		so_mod(env, h1, a[deg], N);
		so_mod(env, Pm, P, N);
		cgbn_set(env, h2, Pm);
		cgbn_negate(env, t1, d);	/* lin0 = -d */
		so_mod(env, L, t1, N);

		for (i = deg; i; i--) {
			so_mulmod(env, h1, h1, L, N);
			if (!cgbn_equals_ui32(env, h1, 0))
				cgbn_sub(env, h1, N, h1);
			so_mod(env, ai, a[i-1], N);
			so_mulmod(env, ai, ai, h2, N);
			cgbn_add(env, h1, h1, ai);
			if (cgbn_compare(env, h1, N) >= 0)
				cgbn_sub(env, h1, h1, N);
			so_mulmod(env, h2, h2, Pm, N);
		}
		if (!cgbn_equals_ui32(env, h1, 0))
			goto expand_failed;
	}

	if (so_cmpabs_d(env, a[deg - 2], prm.stage1_norm) > 0)
		return 1;
	return 2;

expand_failed:
	if (threadIdx.x % env_t::TPI == 0)
		atomicAdd(counters + SO_STAT_EXPAND_FAILED, 1);
	return 0;
}

/* the doubles of optimize_initial, from the integers */
template<class env_t>
__device__ static void
so_load_doubles(env_t env, so_state_t *s, uint32 deg,
		const typename env_t::cgbn_t *a,
		const typename env_t::cgbn_t &lin0,
		const typename env_t::cgbn_t &P,
		bool write)
{
	uint32 j;
	double r0 = so_get_d(env, lin0);
	double r1 = so_get_d(env, P);
	double ad[SO_MAX_DEG + 1];

	for (j = 0; j <= deg; j++)
		ad[j] = so_get_d(env, a[j]);
	if (write) {
		s->r0 = r0;
		s->r1 = r1;
		for (j = 0; j <= deg; j++)
			s->apoly[j] = ad[j];
		for (; j <= SO_MAX_DEG; j++)
			s->apoly[j] = 0;
	}
}

/*------------------------------------------------------------------------*/
template<uint32 BITS, uint32 TPI>
__device__ static void
so_collide(const found_t *found, const void *q_array,
	const uint32 *which_q, const void *roots,
	const cgbn_mem_t<BITS> *consts, cgbn_mem_t<BITS> *big,
	so_state_t *state, uint32 *list, uint32 *counters,
	so_params_t prm)
{
	typedef cgbn_context_t<TPI> context_t;
	typedef cgbn_env_t<context_t, BITS> env_t;
	typedef typename env_t::cgbn_t bn_t;

	context_t ctx(cgbn_no_checks);
	env_t env(ctx.template env<env_t>());
	uint32 deg = prm.degree;
	uint32 num_found = SO_MIN(*(const uint32 *)found, prm.found_max);
	uint32 stride = (gridDim.x * blockDim.x) / TPI;
	uint32 h, i;
	bool lead = (threadIdx.x % TPI == 0);
	bn_t k[SO_NUM_CONST];

	for (i = 0; i < SO_NUM_CONST; i++)
		cgbn_load(env, k[i], (cgbn_mem_t<BITS> *)consts + i);

	for (h = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
			h < num_found; h += stride) {

		const found_t *f = found + h + 1;
		uint32 slot = prm.slot_base + h;
		int64 res = (int64)(f->offset) - prm.half_sieve;
		uint64 p = (uint64)f->p1 * f->p2;
		uint32 which_qroot = f->which_qroot;
		uint32 wq = which_q[which_qroot];
		uint64 q;
		uint32 qroot[4];
		bn_t a[SO_MAX_DEG + 1], P, d, lin0;
		uint32 status;

		if (prm.sizeof_q == 4) {
			uint64 r = ((const uint64 *)roots)[which_qroot];
			q = ((const uint32 *)q_array)[wq];
			qroot[0] = (uint32)r;
			qroot[1] = (uint32)(r >> 32);
			qroot[2] = qroot[3] = 0;
		}
		else {
			const uint32 *r = (const uint32 *)roots + 4 * which_qroot;
			q = ((const uint64 *)q_array)[wq];
			qroot[0] = r[0];
			qroot[1] = r[1];
			qroot[2] = r[2];
			qroot[3] = r[3];
		}

		status = so_rebuild<env_t>(env, k, deg, p, q, qroot, res,
					prm, counters, a, P, d);
		if (status != 2)
			continue;

		/* optimize_initial's starting point */

		cgbn_negate(env, lin0, d);
		for (i = 0; i <= deg; i++)
			cgbn_store(env, big + slot * SO_NUM_BIG + SO_BIG_A + i,
					a[i]);
		cgbn_store(env, big + slot * SO_NUM_BIG + SO_BIG_LIN0, lin0);
		cgbn_store(env, big + slot * SO_NUM_BIG + SO_BIG_P, P);
		cgbn_store(env, big + slot * SO_NUM_BIG + SO_BIG_D, d);

		{
			so_state_t *s = state + slot;

			so_load_doubles<env_t>(env, s, deg, a, lin0, P, lead);
			if (lead) {
				s->best[SO_SKEWNESS] = 1000;
				for (i = 1; i < SO_MAX_VARS; i++)
					s->best[i] = 0;
				s->score = 1e200;
				s->last_score = 0;
				s->pass = 0;
				s->nmin = 0;
				s->status = SO_ACTIVE;
				list[atomicAdd(counters + SO_STAT_CANDS, 1)] = slot;
			}
		}
	}
}

/*------------------------------------------------------------------------*/
/* after one minimize(): apply the rotation and translation to the
   integers (optimize_initial's loop body), reload the doubles, and
   decide whether the loop goes on */

template<uint32 BITS, uint32 TPI>
__device__ static void
so_update(cgbn_mem_t<BITS> *big, so_state_t *state,
	const uint32 *list, uint32 *counters, so_params_t prm)
{
	typedef cgbn_context_t<TPI> context_t;
	typedef cgbn_env_t<context_t, BITS> env_t;
	typedef typename env_t::cgbn_t bn_t;

	context_t ctx(cgbn_no_checks);
	env_t env(ctx.template env<env_t>());
	uint32 deg = prm.degree;
	uint32 rotate_dim = deg - 4;
	uint32 n = counters[SO_STAT_CANDS];
	uint32 stride = (gridDim.x * blockDim.x) / TPI;
	bool lead = (threadIdx.x % TPI == 0);
	uint32 c, i, j;

	for (c = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
			c < n; c += stride) {

		uint32 h = list[c];
		so_state_t *s = state + h;
		cgbn_mem_t<BITS> *bm = big + h * SO_NUM_BIG;
		bn_t a[SO_MAX_DEG + 1], lin0, P, d, h1, h2, h3;
		double score, last_score;

		if (s->status != SO_ACTIVE)
			continue;

		for (i = 0; i <= deg; i++)
			cgbn_load(env, a[i], bm + SO_BIG_A + i);
		cgbn_load(env, lin0, bm + SO_BIG_LIN0);
		cgbn_load(env, P, bm + SO_BIG_P);
		cgbn_load(env, d, bm + SO_BIG_D);

		/* rotation */

		for (j = 0; j <= rotate_dim; j++) {
			so_set_i64(env, h1, (int64)s->rot[j]);
			cgbn_mul(env, h2, h1, P);
			cgbn_add(env, a[j+1], a[j+1], h2);
			cgbn_mul(env, h2, h1, d);
			cgbn_sub(env, a[j], a[j], h2);
		}

		/* translate_gmp(c, gmp_a, deg, gmp_lina, xlate) */

		so_set_i64(env, h1, -s->xlate);
		for (i = 0; i < deg; i++) {
			const double *weights = so_xlate_weights[i];

			cgbn_mul_ui32(env, h2, a[deg], (uint32)weights[deg]);
			for (j = deg; j > i; j--) {
				cgbn_mul_ui32(env, h3, a[j-1],
						(uint32)weights[j-1]);
				cgbn_mul(env, d, h2, h1);	/* d is free */
				cgbn_add(env, h3, h3, d);
				cgbn_set(env, h2, h3);
			}
			cgbn_set(env, a[i], h2);
		}
		cgbn_mul(env, h2, P, h1);
		cgbn_add(env, lin0, lin0, h2);
		cgbn_negate(env, d, lin0);

		for (i = 0; i <= deg; i++)
			cgbn_store(env, bm + SO_BIG_A + i, a[i]);
		cgbn_store(env, bm + SO_BIG_LIN0, lin0);
		cgbn_store(env, bm + SO_BIG_D, d);

		so_load_doubles<env_t>(env, s, deg, a, lin0, P, lead);

		if (!lead)
			continue;

		for (i = 1; i < SO_MAX_VARS; i++)
			s->best[i] = 0;

		score = s->score;
		last_score = s->last_score;
		if (++s->nmin >= 5 ||
		    !(fabs(score - last_score) > .001 * fabs(score))) {

			if (s->pass == 0) {
				s->pass = 1;
				s->nmin = 0;
				s->score = so_ifs_rectangular(s->apoly, deg,
						s->best[SO_SKEWNESS]);
			}
			else {
				s->pol_norm = sqrt(fabs(score));
				s->status = SO_DONE;
			}
		}
	}
}

/*------------------------------------------------------------------------*/
/* get_root_freq (root_score.c), on coefficients already reduced
   mod the largest power of p that is at most 65000 */

__device__ static double
so_root_freq(const uint32 *coeffs_in, uint32 degree,
		uint32 initial_root, uint32 p, uint32 max_power)
{
	uint32 i, j;
	uint32 roots[SO_ROOT_BUF_SIZE];
	uint32 new_roots[SO_ROOT_BUF_SIZE];
	uint32 coeffs[SO_MAX_DEG + 1];
	uint32 power;
	uint32 num_roots, num_new_roots;
	double contrib;

	for (i = 0; i <= degree; i++)
		coeffs[i] = coeffs_in[i];
	while (degree > 0 && coeffs[degree] == 0)
		degree--;

	num_roots = 1;
	roots[0] = initial_root;
	power = p;
	contrib = 1.0 / power;

	for (i = 2; i <= max_power; i++) {

		uint32 next_power = power * p;
		uint32 x;

		for (j = num_new_roots = 0; j < num_roots; j++) {

			for (x = roots[j]; x < next_power; x += power) {

				uint32 k;
				uint32 polyval = coeffs[degree] % next_power;
				for (k = degree; k; k--) {
					polyval = (polyval * x +
						coeffs[k-1]) % next_power;
				}
				if (polyval == 0 &&
				    num_new_roots < SO_ROOT_BUF_SIZE) {
					new_roots[num_new_roots++] = x;
				}
			}
		}

		if (num_new_roots == 0)
			break;

		contrib += (double)num_new_roots / next_power;
		for (j = 0; j < num_new_roots; j++)
			roots[j] = new_roots[j];
		power = next_power;
		num_roots = num_new_roots;
	}

	return contrib * p / (p + 1);
}

/* analyze_poly_roots_projective(poly, 100), then the stage 2 bound */

template<uint32 BITS, uint32 TPI>
__device__ static void
so_finish(cgbn_mem_t<BITS> *big, so_state_t *state,
	const uint32 *list, uint32 *surv, uint32 *counters,
	so_params_t prm)
{
	typedef cgbn_context_t<TPI> context_t;
	typedef cgbn_env_t<context_t, BITS> env_t;
	typedef typename env_t::cgbn_t bn_t;

	context_t ctx(cgbn_no_checks);
	env_t env(ctx.template env<env_t>());
	uint32 deg = prm.degree;
	uint32 n = counters[SO_STAT_CANDS];
	uint32 stride = (gridDim.x * blockDim.x) / TPI;
	bool lead = (threadIdx.x % TPI == 0);
	uint32 c, i, j;

	for (c = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
			c < n; c += stride) {

		uint32 h = list[c];
		so_state_t *s = state + h;
		cgbn_mem_t<BITS> *bm = big + h * SO_NUM_BIG;
		bn_t a[SO_MAX_DEG + 1];
		double root_score = 0;
		uint32 rdeg;

		if (s->status != SO_DONE)
			continue;

		for (i = 0; i <= deg; i++)
			cgbn_load(env, a[i], bm + SO_BIG_A + i);

		/* reversed polynomial, trailing zeros dropped */
		rdeg = deg;
		while (rdeg && cgbn_equals_ui32(env, a[deg - rdeg], 0))
			rdeg--;

		for (i = 0; i < SO_NUM_PRIMES; i++) {
			uint32 p = prm.primes[i];
			uint32 power, next_power, max_power;
			uint32 coeffs[SO_MAX_DEG + 1];

			if (so_fdiv_ui(env, a[deg], p) != 0)
				continue;

			max_power = 1;
			power = p;
			while (1) {
				next_power = power * p;
				if (next_power > 65000)
					break;
				power = next_power;
				max_power++;
			}
			for (j = 0; j <= rdeg; j++)
				coeffs[j] = so_fdiv_ui(env, a[deg - j], power);

			root_score -= prm.log_primes[i] *
				so_root_freq(coeffs, rdeg, 0, p, max_power);
		}

		if (lead) {
			s->alpha = root_score;
			if (s->pol_norm * exp(root_score) <= prm.sizeopt_bound)
				surv[atomicAdd(counters + SO_STAT_SURVIVORS, 1)] = h;
		}
	}
}

/*------------------------------------------------------------------------*/
/* copy the survivors' integers and state into contiguous arrays */

template<uint32 BITS>
__device__ static void
so_gather(const cgbn_mem_t<BITS> *big, const so_state_t *state,
	const uint32 *surv, const uint32 *counters,
	cgbn_mem_t<BITS> *out_big, so_state_t *out_state)
{
	const uint32 words = SO_NUM_BIG * (BITS / 32);
	uint32 n = counters[SO_STAT_SURVIVORS];
	uint32 i;

	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n * words;
			i += gridDim.x * blockDim.x) {
		uint32 c = i / words;
		uint32 w = i % words;
		const uint32 *src = (const uint32 *)(big +
					(size_t)surv[c] * SO_NUM_BIG);

		((uint32 *)(out_big + (size_t)c * SO_NUM_BIG))[w] = src[w];
		if (w == 0)
			out_state[c] = state[surv[c]];
	}
}

/*------------------------------------------------------------------------*/
#define SO_KERNELS(BITS, TPI)						\
extern "C" __global__ void						\
so_collide_##BITS(const found_t *found, const void *q_array,		\
	const uint32 *which_q, const void *roots,			\
	const cgbn_mem_t<BITS> *consts, cgbn_mem_t<BITS> *big,		\
	so_state_t *state, uint32 *list, uint32 *counters,		\
	so_params_t prm)						\
{									\
	so_collide<BITS, TPI>(found, q_array, which_q, roots,		\
			consts, big, state, list, counters, prm);	\
}									\
extern "C" __global__ void						\
so_update_##BITS(cgbn_mem_t<BITS> *big, so_state_t *state,		\
	const uint32 *list, uint32 *counters, so_params_t prm)		\
{									\
	so_update<BITS, TPI>(big, state, list, counters, prm);		\
}									\
extern "C" __global__ void						\
so_finish_##BITS(cgbn_mem_t<BITS> *big, so_state_t *state,		\
	const uint32 *list, uint32 *surv, uint32 *counters,		\
	so_params_t prm)						\
{									\
	so_finish<BITS, TPI>(big, state, list, surv, counters, prm);	\
}									\
extern "C" __global__ void						\
so_gather_##BITS(const cgbn_mem_t<BITS> *big,				\
	const so_state_t *state, const uint32 *surv,			\
	const uint32 *counters, cgbn_mem_t<BITS> *out_big,		\
	so_state_t *out_state)						\
{									\
	so_gather<BITS>(big, state, surv, counters, out_big, out_state); \
}

SO_KERNELS(512, 8)
SO_KERNELS(1024, 16)

/*------------------------------------------------------------------------*/
/* The line sieve of the batched root sieve (line_sieve_core.h): one
   block per line. The block builds the polynomial's 25 tables for the
   line in shared memory, then scores the positions, thread t taking
   t, t + LS_THREADS, ...; a position is kept if it beats the current
   threshold (the LS_TOP-th best kept so far), and when the kept list
   fills it is cut back to the LS_TOP best. Keys order by score, then
   by lower position, so the result does not depend on the order the
   threads find positions in. */

#define LS_THREADS 512

/* the primes' tables are laid out the same for every polynomial (their
   sizes depend only on the prime): their ends, sizes and the step of a
   thread's index, set by the host (ls_tables_set), so that the sieve
   loop keeps only its indices in registers */
__constant__ uint32 ls_c_end[LS_NUM_PRIMES];
__constant__ uint32 ls_c_size[LS_NUM_PRIMES];
__constant__ uint32 ls_c_step[LS_NUM_PRIMES];

#define LS_TPI 8			/* CGBN: 256 bits, ample for a line's
				   numbers (the host checks their sizes) */
#define LS_BITS 256

typedef cgbn_context_t<LS_TPI> ls_context_t;
typedef cgbn_env_t<ls_context_t, LS_BITS> ls_env_t;
typedef ls_env_t::cgbn_t ls_bn_t;
typedef cgbn_mem_t<LS_BITS> ls_mem_t;

__device__ static void
ls_bn_u64(ls_env_t env, ls_bn_t &r, uint64 v)
{
	cgbn_set_ui32(env, r, (uint32)(v >> 32));
	cgbn_shift_left(env, r, r, 32);
	cgbn_add_ui32(env, r, r, (uint32)v);
}

__device__ static void
ls_bn_i64(ls_env_t env, ls_bn_t &r, int64 v)
{
	ls_bn_u64(env, r, (uint64)v);
	if (v < 0) {
		ls_bn_t u;

		cgbn_set_ui32(env, u, 1);
		cgbn_shift_left(env, u, u, 64);
		cgbn_sub(env, r, r, u);
	}
}

/* an integer-valued double, as mpz_set_d makes it */
__device__ static void
ls_bn_double(ls_env_t env, ls_bn_t &r, double v)
{
	double a = fabs(v);

	if (a < 18446744073709551616.0) {
		ls_bn_u64(env, r, (uint64)a);
	}
	else {
		int e = ilogb(a);

		ls_bn_u64(env, r, (uint64)ldexp(a, 63 - e));
		cgbn_shift_left(env, r, r, e - 63);
	}
	if (v < 0)
		cgbn_negate(env, r, r);
}

/* floor (mpz_fdiv_ui) and absolute (mpz_tdiv_ui) remainders */
__device__ static uint32
ls_bn_fmod(ls_env_t env, const ls_bn_t &a, uint32 m)
{
	ls_bn_t t;
	bool neg = cgbn_extract_bits_ui32(env, a, LS_BITS - 1, 1) != 0;
	uint32 r;

	if (neg)
		cgbn_negate(env, t, a);
	else
		cgbn_set(env, t, a);
	r = cgbn_rem_ui32(env, t, m);
	return (neg && r != 0) ? m - r : r;
}

__device__ static uint32
ls_bn_tmod(ls_env_t env, const ls_bn_t &a, uint32 m)
{
	ls_bn_t t;

	if (cgbn_extract_bits_ui32(env, a, LS_BITS - 1, 1))
		cgbn_negate(env, t, a);
	else
		cgbn_set(env, t, a);
	return cgbn_rem_ui32(env, t, m);
}

/* mpz_get_d(): the value truncated to a double */
__device__ static double
ls_bn_get_d(ls_env_t env, const ls_bn_t &a)
{
	ls_bn_t t;
	bool neg = cgbn_extract_bits_ui32(env, a, LS_BITS - 1, 1) != 0;
	uint32 bits;
	double d;

	if (neg)
		cgbn_negate(env, t, a);
	else
		cgbn_set(env, t, a);
	bits = LS_BITS - cgbn_clz(env, t);
	if (bits <= 53) {
		uint64 u = ((uint64)cgbn_extract_bits_ui32(env, t, 32, 32) << 32) |
				cgbn_extract_bits_ui32(env, t, 0, 32);
		d = (double)u;
	}
	else {
		uint64 u;

		cgbn_shift_right(env, t, t, bits - 53);
		u = ((uint64)cgbn_extract_bits_ui32(env, t, 32, 32) << 32) |
				cgbn_extract_bits_ui32(env, t, 0, 32);
		d = ldexp((double)u, bits - 53);
	}
	return neg ? -d : d;
}
#define LS_CAP 1536		/* kept positions before a cut */
#define LS_SMEM (LS_MAX_TABLE * 2 + LS_CAP * 8)	/* dynamic shared */

/* cut keys[0..n) back to the LS_TOP largest, in order */
__device__ static void
ls_cut(uint64 *keys, uint32 n, uint64 *best, uint32 *nkeys, uint64 *theta)
{
	uint32 m = SO_MIN(n, LS_TOP);

	for (uint32 i = threadIdx.x; i < n; i += LS_THREADS) {
		uint64 k = keys[i];
		uint32 r = 0;

		for (uint32 j = 0; j < n; j++)
			r += (keys[j] > k);
		if (r < LS_TOP)
			best[r] = k;
	}
	__syncthreads();
	if (threadIdx.x < m)
		keys[threadIdx.x] = best[threadIdx.x];
	if (threadIdx.x == 0) {
		*nkeys = m;
		if (m == LS_TOP)
			*theta = best[LS_TOP - 1];
	}
	__syncthreads();
}

/* the line's numbers: one group forms them, all take residues */
__device__ static __noinline__ void
ls_prologue(const ls_line_t *l, const ls_poly_t *pl, const ls_power_t *powers,
	const ls_run_t *runs, ls_lp_t *lps, ls_mem_t *xb_mem, ls_mem_t *rc_mem,
	ls_mem_t *lat_mem, ls_mem_t *y_mem, ls_mem_t *xbr_mem, double *y_d,
	uint32 *y_limbs)
{
		const uint32 tid = threadIdx.x;
		ls_context_t ctx(cgbn_no_checks);
		ls_env_t env(ctx.env<ls_env_t>());
		const uint32 group = tid / LS_TPI;
		ls_bn_t XB, RC, L, Y, t, u;
		const uint32 one = (l->mode == LS_PARTS) ? l->lattice_one :
						(l->lattice == 1);

		if (group == 0) {
			if (l->mode == LS_PARTS) {
				const ls_run_t *run = runs + l->run;
				ls_bn_t c0, c1;

				ls_bn_u64(env, L, run->l1);
				ls_bn_u64(env, t, run->l2);
				cgbn_mul(env, L, L, t);
				ls_bn_i64(env, Y, run->y_base);
				ls_bn_u64(env, t, run->ly);
				cgbn_mul_ui32(env, t, t, l->k);
				cgbn_add(env, Y, Y, t);
				ls_bn_u64(env, t, l->ry);
				cgbn_add(env, Y, Y, t);

				cgbn_load(env, c0, (ls_mem_t *)run->crt0);
				cgbn_load(env, c1, (ls_mem_t *)run->crt1);
				ls_bn_u64(env, t, l->r1);
				cgbn_mul(env, t, t, c0);
				ls_bn_u64(env, u, l->rx);
				cgbn_mul(env, u, u, c1);
				cgbn_add(env, t, t, u);
				cgbn_rem(env, RC, t, L);

				ls_bn_double(env, t, trunc(l->line_min));
				cgbn_signed_div(env, XB, t, L);
				cgbn_mul(env, XB, XB, L);
			}
			else {
				ls_bn_i64(env, XB, l->x_base);
				ls_bn_i64(env, RC, l->resclass);
				ls_bn_i64(env, L, l->lattice);
				ls_bn_i64(env, Y, l->y);
			}
			cgbn_store(env, xb_mem, XB);
			cgbn_store(env, rc_mem, RC);
			cgbn_store(env, lat_mem, L);
			cgbn_store(env, y_mem, Y);
			cgbn_add(env, t, XB, RC);
			cgbn_store(env, xbr_mem, t);
			{
				double yd = ls_bn_get_d(env, Y);
				uint32 yl[LS_XLIMBS];

				for (uint32 i = 0; i < LS_XLIMBS; i++)
					yl[i] = cgbn_extract_bits_ui32(env, Y,
								32 * i, 32);
				if (tid == 0) {
					*y_d = yd;
					for (uint32 i = 0; i < LS_XLIMBS; i++)
						y_limbs[i] = yl[i];
				}
			}
		}
		__syncthreads();

		cgbn_load(env, XB, xb_mem);
		cgbn_load(env, RC, rc_mem);
		cgbn_load(env, L, lat_mem);
		cgbn_load(env, Y, y_mem);
		for (uint32 j = group; j < pl->num_powers;
		     j += LS_THREADS / LS_TPI) {
			uint32 power = powers[pl->power0 + j].power;
			uint32 xmin = ls_bn_fmod(env, XB, power);
			uint32 ym = ls_bn_fmod(env, Y, power);
			uint32 rc = ls_bn_tmod(env, RC, power);
			uint32 lat = ls_bn_tmod(env, L, power);

			if (tid % LS_TPI == 0)
				ls_line_power_res(xmin, ym, ls_fmod(l->z, power),
					rc, lat, one, power, lps + j);
		}
}

/* the line's candidates: group k makes candidate k */
__device__ static __noinline__ void
ls_epilogue(const ls_line_t *l, const ls_poly_t *pl, const uint64 *best,
	uint32 nkeys, ls_mem_t *xbr_mem, ls_mem_t *lat_mem, double y_d,
	so_basic_in_t *cand_in, ls_cand_t *cand)
{
	const uint32 tid = threadIdx.x;

		ls_context_t ctx(cgbn_no_checks);
		ls_env_t env(ctx.env<ls_env_t>());
		const uint32 k = tid / LS_TPI;
		const uint32 slot = blockIdx.x * LS_TOP + k;
		ls_bn_t X, L;
		double alpha = 0, xd;
		uint32 pos = 0, valid = 0, xl[LS_XLIMBS];

		if (k < nkeys) {
			uint32 score = (uint32)(best[k] >> 32);

			pos = 0xffffffffu - (uint32)best[k];
			alpha = pl->random_root_score - (pl->sieve_bias +
				(double)(score + l->curr_score) / 1000);
			valid = !(alpha > -4.0);  /* ROOT_SCORE_COARSE_MIN */
		}

		cgbn_load(env, X, xbr_mem);
		cgbn_load(env, L, lat_mem);
		cgbn_mul_ui32(env, L, L, pos);
		cgbn_add(env, X, X, L);
		xd = ls_bn_get_d(env, X);
		for (uint32 i = 0; i < LS_XLIMBS; i++)
			xl[i] = cgbn_extract_bits_ui32(env, X, 32 * i, 32);

		if (tid % LS_TPI == 0) {
			double a[LS_MAX_DEG + 1];

			for (uint32 i = 0; i <= LS_MAX_DEG; i++)
				a[i] = pl->apoly[i];
			a[3] += pl->dbl_p * (double)l->z;
			a[2] -= pl->dbl_d * (double)l->z;
			a[2] += pl->dbl_p * y_d;
			a[1] -= pl->dbl_d * y_d;
			a[1] += pl->dbl_p * xd;
			a[0] -= pl->dbl_d * xd;

			for (uint32 i = 0; i <= SO_MAX_DEG; i++)
				cand_in[slot].apoly[i] = (i <= pl->degree &&
						i <= LS_MAX_DEG) ? a[i] : 0;
			if (!valid)
				cand_in[slot].apoly[0] = __longlong_as_double(
						0x7ff8000000000000ll);	/* NaN */
			cand[slot].alpha = alpha;
			for (uint32 i = 0; i < LS_XLIMBS; i++)
				cand[slot].x[i] = xl[i];
			cand[slot].valid = valid;
			cand[slot].pad = 0;
		}
}

extern "C" __global__ void __launch_bounds__(LS_THREADS)
ls_sieve(const ls_poly_t *polys, const ls_prime_t *primes,
	const ls_power_t *powers, const ls_root_t *roots,
	const ls_line_t *lines, ls_out_t *out, const ls_run_t *runs,
	so_basic_in_t *cand_in, ls_cand_t *cand)
{
	extern __shared__ uint32 ls_smem[];
	uint32 *tab = ls_smem;
	uint64 *keys = (uint64 *)(ls_smem + LS_MAX_TABLE / 2);
	const uint16 *tab16 = (const uint16 *)tab;
	__shared__ ls_lp_t lps[LS_MAX_POWERS];
	__shared__ uint64 best[LS_TOP];
	__shared__ uint64 theta;
	__shared__ uint32 nkeys;
	/* the line's x_base, resclass, lattice, y, x_base + resclass */
	__shared__ ls_mem_t xb_mem, rc_mem, lat_mem, y_mem, xbr_mem;
	__shared__ double y_d;
	__shared__ uint32 y_limbs[LS_XLIMBS];

	const ls_line_t &l = lines[blockIdx.x];
	const ls_poly_t &pl = polys[l.poly];
	/* the fields the loops use, in registers (the structs are read
	   in place: copied, they would not leave the sieve loop its
	   registers) */
	const uint32 positions = l.positions, prime0 = pl.prime0;
	const uint32 power0 = pl.power0, num_powers = pl.num_powers;
	const uint32 tid = threadIdx.x;
	const uint32 lattice = (l.mode == LS_PARTS) ? !l.lattice_one :
					(l.lattice != 1);
	uint32 a[LS_NUM_PRIMES];
	uint64 th;

	for (uint32 i = tid; i < (pl.table_size + 1) / 2; i += LS_THREADS)
		tab[i] = 0;
	/* the line's numbers, from its parts in CGBN (the lattice CRT),
	   and their residues modulo the powers (out of line: the sieve
	   loop below needs the registers) */
	ls_prologue(&l, &pl, powers, runs, lps, &xb_mem, &rc_mem, &lat_mem,
			&y_mem, &xbr_mem, &y_d, y_limbs);
	if (tid == 0) {
		nkeys = 0;
		theta = 0;
	}
	__syncthreads();

	/* the tables; entries are 16 bits, two to a word */
	for (uint32 j = 0; j < num_powers; j++) {
		const ls_power_t pw = powers[power0 + j];
		const ls_prime_t pr = primes[prime0 + pw.prime];
		const ls_lp_t lp = lps[j];

		if (lattice && lp.skip)
			continue;
		for (uint32 k = tid; k < pw.num_roots; k += LS_THREADS) {
			const ls_root_t r = roots[pw.root0 + k];
			uint32 st, step;

			if (!ls_root_start(&r, pw.power, lattice, &lp,
						&st, &step))
				continue;
			for (; st < pr.size; st += step) {
				uint32 ent = pr.table + st;

				atomicAdd(tab + (ent >> 1),
					pw.contrib << ((ent & 1) * 16));
			}
		}
	}

#pragma unroll
	for (uint32 i = 0; i < LS_NUM_PRIMES; i++)
		a[i] = ls_c_end[i] - ls_c_size[i] + tid % ls_c_size[i];
	__syncthreads();

	/* positions is a multiple of DEFAULT_BLOCK_SIZE, so of LS_THREADS */
	th = 0;
	for (uint32 x = tid; x < positions; x += LS_THREADS) {
		uint32 sc = 0;
		uint64 key;

#pragma unroll
		for (uint32 i = 0; i < LS_NUM_PRIMES; i++) {
			sc += tab16[a[i]];
			a[i] += ls_c_step[i];
			if (a[i] >= ls_c_end[i])
				a[i] -= ls_c_size[i];
		}

		key = ((uint64)(sc & 0xffff) << 32) | (0xffffffffu - x);
		if (key > th)
			keys[atomicAdd(&nkeys, 1)] = key;
		__syncthreads();
		if (nkeys > LS_CAP - LS_THREADS)
			ls_cut(keys, nkeys, best, &nkeys, &theta);
		th = theta;
	}

	__syncthreads();
	ls_cut(keys, nkeys, best, &nkeys, &theta);
	if (tid < nkeys) {
		out[blockIdx.x].pos[tid] = 0xffffffffu - (uint32)best[tid];
		out[blockIdx.x].score[tid] = (uint16)(best[tid] >> 32);
	}
	if (tid < LS_XLIMBS)
		out[blockIdx.x].y[tid] = y_limbs[tid];
	if (tid == 0)
		out[blockIdx.x].n = nkeys;

	/* the line's candidates (root_sieve_line()): one CGBN group
	   each; their optimize_basic() follows */
	if (cand != NULL && tid / LS_TPI < LS_TOP)
		ls_epilogue(&l, &pl, best, nkeys, &xbr_mem, &lat_mem, y_d,
				cand_in, cand);
}

/*------------------------------------------------------------------------*/
/* Murphy E optimization (murphy_core.h): one warp per candidate. All 32
   lanes step the same minimize() state machine (so_mz_step) and so
   take the same decisions; each Murphy E is a double-exponential
   integral (de_run / de_run_core of common/integrate.c) whose integrand
   points, level by level, are spread over the lanes. The integrand is
   murphy_integrand() of size_score.c. */

#define MF_WARPS 4			/* per block */
#define MF_HEAP 64			/* subintervals per integral */
#define MF_NEXT 0			/* counters */
#define MF_LINE_ITERS 1
#define MF_FULL 0xffffffffu

__device__ static const double mf_recip_factorial[16] = {
  5.00000000000000000e-01, 1.66666666666666657e-01,
  4.16666666666666644e-02, 8.33333333333333322e-03,
  1.38888888888888894e-03, 1.98412698412698413e-04,
  2.48015873015873016e-05, 2.75573192239858925e-06,
  2.75573192239858883e-07, 2.50521083854417202e-08,
  2.08767569878681002e-09, 1.60590438368216133e-10,
  1.14707455977297245e-11, 7.64716373181981641e-13,
  4.77947733238738525e-14, 2.81145725434552060e-15,
};

typedef struct {
	double left, right, result, error;
	uint32 level;
} mf_sub_t;

/* one point of the search: the translated pair and its skew */
template<uint32 DEG>
struct mf_poly_t {
	double r[2];
	double a[DEG + 1];
	double rs_r, rs_a;
	double skew_x, skew_y;
};

/* double-double operations of include/dd.h (no fused multiply-adds) */
typedef struct { double hi, lo; } mf_dd_t;

__device__ __forceinline__ static mf_dd_t
mf_dd(double hi, double lo)
{
	mf_dd_t r;
	r.hi = hi;
	r.lo = lo;
	return r;
}

__device__ __forceinline__ static mf_dd_t
mf_add_dd(mf_dd_t a, mf_dd_t b)
{
	double s1, s2, t1, t2, u, bb;

	u = a.hi + b.hi; bb = u - a.hi;
	s1 = u; s2 = (a.hi - (u - bb)) + (b.hi - bb);
	u = a.lo + b.lo; bb = u - a.lo;
	t1 = u; t2 = (a.lo - (u - bb)) + (b.lo - bb);
	s2 += t1;
	u = s1 + s2; s2 = s2 - (u - s1); s1 = u;
	s2 += t2;
	u = s1 + s2; s2 = s2 - (u - s1); s1 = u;
	return mf_dd(s1, s2);
}

__device__ __forceinline__ static mf_dd_t
mf_sub_dd(mf_dd_t a, mf_dd_t b)
{
	double s1, s2, t1, t2, u, bb;

	u = a.hi - b.hi; bb = u - a.hi;
	s1 = u; s2 = (a.hi - (u - bb)) - (b.hi + bb);
	u = a.lo - b.lo; bb = u - a.lo;
	t1 = u; t2 = (a.lo - (u - bb)) - (b.lo + bb);
	s2 += t1;
	u = s1 + s2; s2 = s2 - (u - s1); s1 = u;
	s2 += t2;
	u = s1 + s2; s2 = s2 - (u - s1); s1 = u;
	return mf_dd(s1, s2);
}

__device__ __forceinline__ static mf_dd_t
mf_mul_d(mf_dd_t a, double b)
{
	const double split = 134217729.0;
	double u = a.hi * b, p2, ah, al, bh, bl, t;

	t = split * a.hi; ah = t - (t - a.hi); al = a.hi - ah;
	t = split * b; bh = t - (t - b); bl = b - bh;
	p2 = ((ah * bh - u) + ah * bl + al * bh) + al * bl;
	p2 += (a.lo * b);
	t = u + p2; p2 = p2 - (t - u);
	return mf_dd(t, p2);
}

/* more of include/dd.h, with no fused multiply-adds */
__device__ __forceinline__ static mf_dd_t
mf_add_d(mf_dd_t a, double b)
{
	double u = a.hi + b, bb = u - a.hi, s1 = u;
	double s2 = (a.hi - (u - bb)) + (b - bb);

	s2 += a.lo;
	u = s1 + s2; s2 = s2 - (u - s1); s1 = u;
	return mf_dd(s1, s2);
}

__device__ __forceinline__ static mf_dd_t
mf_mul_dd(mf_dd_t a, mf_dd_t b)
{
	const double split = 134217729.0;
	double u = a.hi * b.hi, p2, ah, al, bh, bl, t;

	t = split * a.hi; ah = t - (t - a.hi); al = a.hi - ah;
	t = split * b.hi; bh = t - (t - b.hi); bl = b.hi - bh;
	p2 = ((ah * bh - u) + ah * bl + al * bh) + al * bl;
	p2 += (a.hi * b.lo);
	p2 += (a.lo * b.hi);
	t = u + p2; p2 = p2 - (t - u);
	return mf_dd(t, p2);
}

__device__ __forceinline__ static mf_dd_t
mf_div_dd(mf_dd_t a, mf_dd_t b)
{
	double q1, q2, q3, u;
	mf_dd_t r;

	q1 = a.hi / b.hi;
	r = mf_sub_dd(a, mf_mul_d(b, q1));
	q2 = r.hi / b.hi;
	r = mf_sub_dd(r, mf_mul_d(b, q2));
	q3 = r.hi / b.hi;
	u = q1 + q2; q2 = q2 - (u - q1); q1 = u;
	return mf_add_d(mf_dd(q1, q2), q3);
}

__device__ __forceinline__ static mf_dd_t
mf_neg(mf_dd_t a)
{
	return mf_dd(-a.hi, -a.lo);
}

__device__ __forceinline__ static mf_dd_t
mf_fabs(mf_dd_t a)
{
	if (a.hi < 0 || (a.hi == 0 && a.lo < 0))
		return mf_dd(-a.hi, -a.lo);
	return a;
}

__device__ __forceinline__ static int32
mf_cmp_dd(mf_dd_t a, mf_dd_t b)
{
	if (a.hi < b.hi) return -1;
	if (a.hi > b.hi) return 1;
	if (a.lo < b.lo) return -1;
	if (a.lo > b.lo) return 1;
	return 0;
}

/* eval_dpoly() of size_score.c */
template<uint32 DEG>
__device__ __forceinline__ static double
mf_eval_dpoly(const double *coeff, double x0, double xh, double y0, double yh)
{
	double xb[DEG + 1][DEG + 1], yb[DEG + 1][DEG + 1], sum[DEG + 1];
	double res;

	xb[1][0] = xh;
	xb[1][1] = x0;
	yb[1][0] = yh;
	yb[1][1] = y0;
#pragma unroll
	for (uint32 i = 2; i <= DEG; i++) {
#pragma unroll
		for (uint32 j = 0; j <= i; j++) {
			xb[i][j] = 0;
			yb[i][j] = 0;
		}
#pragma unroll
		for (uint32 j = 0; j < i; j++) {
			xb[i][j + 0] += xb[i - 1][j] * xb[1][0];
			xb[i][j + 1] += xb[i - 1][j] * xb[1][1];
		}
#pragma unroll
		for (uint32 j = 0; j < i; j++) {
			yb[i][j + 0] += yb[i - 1][j] * yb[1][0];
			yb[i][j + 1] += yb[i - 1][j] * yb[1][1];
		}
	}

#pragma unroll
	for (uint32 i = 0; i <= DEG; i++)
		sum[i] = xb[DEG][i] * coeff[DEG] + yb[DEG][i] * coeff[0];

#pragma unroll
	for (uint32 i = 1; i < DEG; i++) {
#pragma unroll
		for (uint32 j = 0; j <= i; j++) {
#pragma unroll
			for (uint32 k = 0; k <= DEG - i; k++)
				sum[j + k] += coeff[i] * xb[i][j] * yb[DEG - i][k];
		}
	}

	res = sum[0];
#pragma unroll
	for (uint32 i = 1; i <= DEG; i++)
		res += sum[i];
	return res;
}

/* dickman() of common/dickman.c */
__device__ static double
mf_dickman(const mf_tables_t *tab, double arg)
{
	double int_arg, frac_arg, sum, term;
	const double *coeffs;
	uint32 num_coeffs, line;

	if (arg <= 1.0)
		return 1.0;
	if (arg > MF_DICKMAN_LINES - 1)
		return 0.0;

	int_arg = ceil(arg);
	frac_arg = int_arg - arg;
	line = (uint32)int_arg;
	num_coeffs = tab->dk_num_coeffs[line];
	coeffs = tab->dk_coeffs + tab->dk_offset[line];

	sum = coeffs[0];
	term = frac_arg;
	for (uint32 i = 1; i < num_coeffs; i++) {
		double iterm = term * coeffs[i];

		sum += iterm;
		if (iterm < sum * 0.5 * 1e-8)	/* DICKMAN_ACCURACY */
			break;
		term *= frac_arg;
	}
	return sum;
}

/* murphy_integrand() of size_score.c */
template<uint32 DEG>
__device__ static double
mf_integrand(const mf_poly_t<DEG> &P, const mf_tables_t *tab,
		double r, double h)
{
	double x0, xh, y0, yh, pr, pa;

	if (fabs(h) > 0.1) {
		sincos(r + h, &y0, &x0);
		xh = yh = 0;
	}
	else {
		double term, hpow;

		sincos(r, &y0, &x0);
		hpow = h;
		xh = -y0 * h;
		yh = x0 * h;

		for (uint32 i = 0; i < 16; i += 2) {
			hpow *= -h;
			term = hpow * mf_recip_factorial[i];
			xh += x0 * term;
			yh += y0 * term;

			hpow *= h;
			term = hpow * mf_recip_factorial[i + 1];
			xh -= y0 * term;
			yh += x0 * term;

			if (fabs(term) <= 1e-16 * (fabs(xh) + fabs(yh)))
				break;
		}
	}

	x0 *= P.skew_x;
	xh *= P.skew_x;
	y0 *= P.skew_y;
	yh *= P.skew_y;

	pr = mf_eval_dpoly<1>(P.r, x0, xh, y0, yh);
	pa = mf_eval_dpoly<DEG>(P.a, x0, xh, y0, yh);

	return mf_dickman(tab, (log(fabs(pr)) + P.rs_r) /
				tab->log_rfb_limit) *
	       mf_dickman(tab, (log(fabs(pa)) + P.rs_a) /
				tab->log_afb_limit);
}

/* the sum over the warp, the same value in every lane */
__device__ __forceinline__ static double
mf_wsum(double v)
{
#pragma unroll
	for (uint32 o = 16; o; o >>= 1)
		v += __shfl_xor_sync(MF_FULL, v, o);
	return __shfl_sync(MF_FULL, v, 0);
}

/* the Murphy E integrand as a function object, for the integrator */
template<uint32 DEG>
struct mf_murphy_f {
	const mf_poly_t<DEG> &P;
	const mf_tables_t *tab;

	__device__ mf_murphy_f(const mf_poly_t<DEG> &P_,
			const mf_tables_t *tab_) : P(P_), tab(tab_) {}
	__device__ double operator()(double r, double h) const {
		return mf_integrand<DEG>(P, tab, r, h);
	}
};

/* de_run_core() of integrate.c, the points of each level on the lanes;
   F(base, offset) is the integrand */
template<class F>
__device__ static void
mf_de_core(const F &f, const mf_tables_t *tab, mf_sub_t *s, uint32 lane)
{
	const uint32 np = tab->num_points;
	const double lower = s->left, upper = s->right;
	const double interval = upper - lower;
	const mf_point_t *pts = tab->rule + np;
	double result, residual, curr_error, step_size = 1.0;
	double rw = 0, rr = 0;
	uint32 i = 0, groups = 1;

	if (lane < np) {
		mf_point_t p = tab->rule[lane];
		double ab = interval * p.abscissa;
		double v = f(lower, ab) + f(upper, -ab);

		rw = v * p.weight;
		rr = v * p.residual_weight;
	}
	else if (lane == np) {
		double v = f(0.5 * (lower + upper), 0.0);

		rw = v * tab->center_weight;
		rr = v * tab->center_residual_weight;
	}
	result = mf_wsum(rw);
	residual = mf_wsum(rr);

	do {
		double old_result = result, old_residual = residual;
		uint32 n = groups * np;

		rw = rr = 0;
		for (uint32 k = lane; k < n; k += 32) {
			mf_point_t p = pts[k];
			double ab = interval * p.abscissa;
			double v = f(lower, ab) + f(upper, -ab);

			rw += v * p.weight;
			rr += v * p.residual_weight;
		}
		result += mf_wsum(rw);
		residual += mf_wsum(rr);
		pts += n;

		curr_error = fabs(result - 2.0 * old_result) +
			     fabs(residual - 2.0 * old_residual);
		step_size *= 0.5;
		groups *= 2;
	} while (++i < MF_TRAP_LEVELS &&
		curr_error > 0.1 * tab->relative_error * fabs(result));

	__syncwarp();
	if (lane == 0) {
		s->result = result * interval * step_size;
		s->error = 2 * curr_error * interval;
	}
	__syncwarp();
}

/* de_run() of integrate.c: the subinterval with the largest error is
   halved until the error is small enough. Returns the integral, and
   its error estimate in *error_out if not NULL */
template<class F>
__device__ static double
mf_integrate(const F &f, const mf_tables_t *tab,
		mf_sub_t *heap, const double *angles, uint32 num_angles,
		uint32 lane, double *error_out = NULL)
{
	const double rel = tab->relative_error;
	double result = 0, error = 0;
	uint32 nh = num_angles - 1;

	for (uint32 i = 0; i < nh; i++) {
		if (lane == 0) {
			heap[i].left = angles[i];
			heap[i].right = angles[i + 1];
			heap[i].level = 0;
		}
		__syncwarp();
		mf_de_core(f, tab, heap + i, lane);
		result += heap[i].result;
		error += heap[i].error;
	}

	while (error > rel * fabs(result)) {
		uint32 m = 0;
		double left, right, mid;
		uint32 level;

		for (uint32 j = 1; j < nh; j++) {
			if (heap[j].error > heap[m].error)
				m = j;
		}
		left = heap[m].left;
		right = heap[m].right;
		level = heap[m].level;
		if (level > 30 || nh == MF_HEAP)	/* MAX_LEVELS */
			break;

		mid = 0.5 * (left + right);
		result -= heap[m].result;
		error -= heap[m].error;
		__syncwarp();
		if (lane == 0) {
			heap[m].right = mid;
			heap[m].level = level + 1;
			heap[nh].left = mid;
			heap[nh].right = right;
			heap[nh].level = level + 1;
		}
		__syncwarp();
		mf_de_core(f, tab, heap + m, lane);
		result += heap[m].result;
		error += heap[m].error;
		mf_de_core(f, tab, heap + nh, lane);
		result += heap[nh].result;
		error += heap[nh].error;
		nh++;
	}
	if (error_out != NULL)
		*error_out = error;
	return result;
}

/* poly_murphy_callback() of optimize.c: -E at v = (skewness,
   translation), with analyze_poly_murphy() */
template<uint32 DEG>
__device__ static double
mf_objective(const mf_in_t *c, const mf_tables_t *tab, mf_sub_t *heap,
		double *angles, uint32 lane, const double *v)
{
	const double t = floor(v[SO_TRANSLATE_SIZE] + 0.5);
	const double s = v[SO_SKEWNESS];
	const double x = -t;
	mf_poly_t<DEG> P;
	mf_dd_t A[DEG + 1];
	uint32 na;

	if (s < 0.5)
		return 1e50;

	/* the pair translated by t (translate_dd), double-double */
	P.r[1] = c->r_hi[1];
	P.r[0] = mf_sub_dd(mf_dd(c->r_hi[0], c->r_lo[0]),
			mf_mul_d(mf_dd(c->r_hi[1], c->r_lo[1]), t)).hi;
#pragma unroll
	for (uint32 i = 0; i <= DEG; i++)
		A[i] = mf_dd(c->a_hi[i], c->a_lo[i]);
#pragma unroll
	for (uint32 i = 0; i < DEG; i++) {
		mf_dd_t accum = mf_mul_d(A[DEG], so_xlate_weights[i][DEG]);

#pragma unroll
		for (uint32 j = DEG; j > i; j--) {
			accum = mf_add_dd(mf_mul_d(accum, x),
					mf_mul_d(A[j - 1],
						so_xlate_weights[i][j - 1]));
		}
		P.a[i] = accum.hi;
	}
	P.a[DEG] = A[DEG].hi;

	P.rs_r = c->root_score_r;
	P.rs_a = c->root_score_a;
	P.skew_x = sqrt(1e16 * s);	/* sieve_area */
	P.skew_y = P.skew_x / s;

	/* the integral's breakpoints: the roots' angles, skewed */
	na = c->num_roots + 2;
	if (lane == 0) {
		for (uint32 i = 0; i < c->num_roots; i++)
			angles[i] = atan2(1.0, (c->roots[i] + t) / s);
		angles[c->num_roots] = 0.0;
		angles[c->num_roots + 1] = M_PI;
		for (uint32 i = 1; i < na; i++) {
			double a = angles[i];
			uint32 j = i;

			for (; j > 0 && angles[j - 1] > a; j--)
				angles[j] = angles[j - 1];
			angles[j] = a;
		}
	}
	__syncwarp();

	return -(mf_integrate(mf_murphy_f<DEG>(P, tab), tab, heap, angles,
				na, lane) / M_PI);
}

template<uint32 DEG>
__device__ static void
mf_opt_deg(const mf_tables_t *tab, const mf_in_t *in, const uint32 *idx_list,
		mf_out_t *out, uint32 n, uint32 *counters)
{
	__shared__ mf_sub_t heaps[MF_WARPS][MF_HEAP];
	__shared__ double angles[MF_WARPS][MF_MAX_ROOTS + 2];
	__shared__ uint32 dummy[MF_WARPS];
	const uint32 warp = threadIdx.x / 32, lane = threadIdx.x & 31;
	uint32 *line_iters = (lane == 0) ? counters + MF_LINE_ITERS :
						dummy + warp;

	for (;;) {
		so_mz_t<2> mz;
		uint32 idx = 0;

		if (lane == 0)
			idx = atomicAdd(counters + MF_NEXT, 1);
		idx = __shfl_sync(MF_FULL, idx, 0);
		if (idx >= n)
			break;

		/* optimize_final_core() */
		mz.pc = MZ_START;
		mz.max_iter = 40;
		mz.ftol = 1e-5;
		mz.ltol = SO_MAX(mz.ftol, 1e-7);
		mz.p[SO_SKEWNESS] = 1000;
		mz.p[SO_TRANSLATE_SIZE] = 0;
		so_mz_step<2>(&mz, 0, line_iters);
		for (;;) {
			double f = mf_objective<DEG>(in + (idx_list ?
						idx_list[idx] : idx), tab,
					heaps[warp], angles[warp], lane, mz.x);

			if (!so_mz_step<2>(&mz, f, line_iters))
				break;
		}
		if (lane == 0) {
			out[idx].score = fabs(mz.result);
			out[idx].skewness = mz.p[SO_SKEWNESS];
			out[idx].translation = mz.p[SO_TRANSLATE_SIZE];
		}
	}
}

#define MF_KERNEL(DEG)							\
extern "C" __global__ void __launch_bounds__(MF_WARPS * 32)		\
mf_opt_##DEG(const mf_tables_t *tab, const mf_in_t *in,		\
		const uint32 *idx_list, mf_out_t *out, uint32 n,	\
		uint32 *counters)					\
{									\
	mf_opt_deg<DEG>(tab, in, idx_list, out, n, counters);		\
}

MF_KERNEL(4)
MF_KERNEL(5)
MF_KERNEL(6)

/*------------------------------------------------------------------------*/
/* Root properties (alpha_core.h): one block per rotation. CGBN forms
   the rotated coefficients exactly and reduces them modulo each prime
   (or prime power); then one thread per prime counts roots with ports
   of ffpoly.c and root_score.c; then thread 0 adds the terms in the
   CPU's order. Integer arithmetic throughout, so the result is the
   CPU's bit for bit. */

#define RA_THREADS 128
#define RA_PMAX 13		/* coefficients of a product of two
				   polynomials of degree <= 6 */
#define RA_ROOT_BUF 200		/* ROOT_BUF_SIZE in root_score.c */

typedef struct {
	uint32 coef[RA_PMAX];
	uint32 degree;
} ra_pl_t;

/* mp.h */
__device__ __forceinline__ static uint32
ra_mulm(uint32 a, uint32 b, uint32 n)
{
	return (uint32)(((uint64)a * (uint64)b) % n);
}

__device__ __forceinline__ static uint32
ra_subm(uint32 a, uint32 b, uint32 p)
{
	uint32 t = a - b;

	return (a >= b) ? t : t + p;
}

__device__ __forceinline__ static uint32
ra_addm(uint32 a, uint32 b, uint32 p)
{
	return ra_subm(a, p - b, p);
}

__device__ static uint32
ra_expo(uint32 a, uint32 b, uint32 n)
{
	uint32 res = 1;

	while (b) {
		if (b & 1)
			res = ra_mulm(res, a, n);
		a = ra_mulm(a, a, n);
		b = b >> 1;
	}
	return res;
}

__device__ static uint32
ra_modinv(uint32 a, uint32 p)
{
	uint32 ps1, ps2, dividend, divisor, rem, q, parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1) {
		q = dividend / divisor;
		rem = dividend % divisor;
		q = q * ps1 + ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}
	return (parity == 0) ? ps1 : p - ps1;
}

__device__ static int32
ra_legendre(uint32 a, uint32 p)
{
	uint32 x = a, y = p, tmp;
	int32 out = 1;

	while (x) {
		while ((x & 1) == 0) {
			x = x / 2;
			if ((y & 7) == 3 || (y & 7) == 5)
				out = -out;
		}
		tmp = x; x = y; y = tmp;
		if ((x & 3) == 3 && (y & 3) == 3)
			out = -out;
		x = x % y;
	}
	return (y == 1) ? out : 0;
}

/* mp_modsqrt_1() of common/mp.c: the same root of the two */
__device__ static uint32
ra_modsqrt(uint32 a, uint32 p)
{
	uint32 a0 = a;

	if ((p & 7) == 3 || (p & 7) == 7)
		return ra_expo(a0, (p + 1) / 4, p);

	if ((p & 7) == 5) {
		uint32 x, y;

		if (a0 >= p)
			a0 = a0 % p;
		x = ra_expo(a0, (p + 3) / 8, p);
		if (ra_mulm(x, x, p) == a0)
			return x;
		y = ra_expo(2, (p - 1) / 4, p);
		return ra_mulm(x, y, p);
	}

	if (a0 == 1)
		return 1;
	for (uint32 d0 = 2; d0 < p; d0++) {
		uint32 d1, a1, s, t, m;

		if (ra_legendre(d0, p) != -1)
			continue;
		t = p - 1;
		s = 0;
		while (!(t & 1)) {
			s++;
			t = t / 2;
		}
		a1 = ra_expo(a0, t, p);
		d1 = ra_expo(d0, t, p);
		m = 0;
		for (uint32 i = 0; i < s; i++) {
			uint32 ad = ra_expo(d1, m, p);

			ad = ra_mulm(ad, a1, p);
			ad = ra_expo(ad, (uint32)1 << (s - 1 - i), p);
			if (ad == p - 1)
				m += (1 << i);
		}
		a1 = ra_expo(a0, (t + 1) / 2, p);
		d1 = ra_expo(d1, m / 2, p);
		return ra_mulm(a1, d1, p);
	}
	return 0;
}

/* ffpoly.c */
__device__ static void
ra_fix_degree(ra_pl_t *op)
{
	int32 i = op->degree;

	while (i > 0 && op->coef[i] == 0)
		i--;
	op->degree = i;
}

__device__ static void
ra_make_monic(ra_pl_t *res, const ra_pl_t *a, uint32 p)
{
	uint32 d = a->degree;
	uint32 msw = a->coef[d];

	if (msw != 1) {
		msw = ra_modinv(msw, p);
		res->degree = d;
		res->coef[d] = 1;
		for (uint32 i = 0; i < d; i++)
			res->coef[i] = ra_mulm(msw, a->coef[i], p);
	}
	else {
		*res = *a;
	}
}

__device__ static void
ra_poly_mod(ra_pl_t *res, const ra_pl_t *op, const ra_pl_t *_mod, uint32 p)
{
	ra_pl_t tmp, mod;

	if (_mod->degree == 0) {
		for (uint32 i = 0; i < RA_PMAX; i++)
			res->coef[i] = 0;
		res->degree = 0;
		return;
	}
	tmp = *op;
	ra_make_monic(&mod, _mod, p);

	while (tmp.degree >= mod.degree) {
		uint32 msw = tmp.coef[tmp.degree];

		tmp.coef[tmp.degree] = 0;
		for (int32 i = mod.degree - 1; i >= 0; i--) {
			uint32 c = ra_mulm(msw, mod.coef[i], p);
			uint32 j = tmp.degree - (mod.degree - i);

			tmp.coef[j] = ra_subm(tmp.coef[j], c, p);
		}
		ra_fix_degree(&tmp);
	}
	*res = tmp;
}

__device__ static void
ra_poly_gcd(ra_pl_t *g_in, const ra_pl_t *h_in, uint32 p)
{
	ra_pl_t g, h;

	if (g_in->degree > h_in->degree) {
		g = *g_in;
		h = *h_in;
	}
	else {
		h = *g_in;
		g = *h_in;
	}
	while (h.degree > 0 || h.coef[h.degree]) {
		ra_pl_t r;

		ra_poly_mod(&r, &g, &h, p);
		g = h;
		h = r;
	}
	if (g.degree == 0)
		g.coef[0] = 1;
	*g_in = g;
}

/* (x + shift)^n mod f (poly_xpow): the remainder is unique, so any
   exact method gives the original's result */
__device__ static void
ra_poly_xpow(ra_pl_t *res, uint32 shift, uint32 n, const ra_pl_t *f,
		uint32 p)
{
	ra_pl_t m;
	uint32 r[RA_PMAX], prod[RA_PMAX];
	uint32 d, msw;

	ra_make_monic(&m, f, p);
	d = m.degree;

	for (uint32 i = 0; i < RA_PMAX; i++)
		r[i] = 0;
	r[0] = shift;
	r[1] = 1;

	msw = 0x80000000;
	while (!(n & msw))
		msw >>= 1;
	msw >>= 1;

	while (msw) {
		/* r = r^2 mod m */
		for (uint32 i = 0; i < 2 * d - 1; i++)
			prod[i] = 0;
		for (uint32 i = 0; i < d; i++) {
			for (uint32 j = 0; j < d; j++) {
				prod[i + j] = ra_addm(prod[i + j],
						ra_mulm(r[i], r[j], p), p);
			}
		}
		for (int32 k = 2 * d - 2; k >= (int32)d; k--) {
			uint32 c = prod[k];

			prod[k] = 0;
			for (uint32 j = 0; j < d; j++) {
				prod[k - d + j] = ra_subm(prod[k - d + j],
						ra_mulm(c, m.coef[j], p), p);
			}
		}
		for (uint32 i = 0; i < d; i++)
			r[i] = prod[i];

		if (n & msw) {
			/* r = r * (x + shift) mod m */
			uint32 c = r[d - 1];

			for (int32 i = d - 1; i >= 1; i--) {
				r[i] = ra_addm(ra_mulm(r[i], shift, p),
						r[i - 1], p);
			}
			r[0] = ra_mulm(r[0], shift, p);
			for (uint32 j = 0; j < d; j++)
				r[j] = ra_subm(r[j], ra_mulm(c, m.coef[j], p),
						p);
		}
		msw >>= 1;
	}

	res->degree = d;
	for (uint32 i = 0; i < d; i++)
		res->coef[i] = r[i];
	res->coef[d] = 0;
	ra_fix_degree(res);
}

__device__ static void
ra_zeros_rec(uint32 *zeros, uint32 shift, uint32 *num_zeros,
		const ra_pl_t *f, uint32 p)
{
	ra_pl_t g, xpow;
	uint32 degree1, degree2;

	if (f->degree == 1) {
		uint32 w = f->coef[1];

		if (w != 1) {
			w = ra_modinv(w, p);
			zeros[(*num_zeros)++] = ra_mulm(p - f->coef[0], w, p);
		}
		else {
			zeros[(*num_zeros)++] = (f->coef[0] == 0 ? 0 :
							p - f->coef[0]);
		}
		return;
	}
	else if (f->degree == 2) {
		uint32 d = ra_mulm(f->coef[0], f->coef[2], p);
		uint32 root1 = p - f->coef[1];
		uint32 root2 = root1;
		uint32 ainv = ra_modinv(ra_addm(f->coef[2], f->coef[2], p),
					p);

		d = ra_subm(ra_mulm(f->coef[1], f->coef[1], p),
				ra_mulm(4, d, p), p);
		d = ra_modsqrt(d, p);

		root1 = ra_addm(root1, d, p);
		root2 = ra_subm(root2, d, p);
		zeros[(*num_zeros)++] = ra_mulm(root1, ainv, p);
		zeros[(*num_zeros)++] = ra_mulm(root2, ainv, p);
		return;
	}

	if (shift >= p)
		return;

	while (shift < p) {
		ra_poly_xpow(&xpow, shift, (p - 1) / 2, f, p);
		g = xpow;
		g.coef[0] = ra_subm(g.coef[0], 1, p);
		ra_fix_degree(&g);
		ra_poly_gcd(&g, f, p);
		if (g.degree > 0 && g.degree < f->degree)
			break;
		shift++;
	}

	degree1 = g.degree;
	ra_zeros_rec(zeros, shift + 1, num_zeros, &g, p);

	g = xpow;
	g.coef[0] = ra_addm(g.coef[0], 1, p);
	ra_fix_degree(&g);
	ra_poly_gcd(&g, f, p);
	degree2 = g.degree;

	if (degree2 > 0)
		ra_zeros_rec(zeros, shift + 1, num_zeros, &g, p);

	if (degree1 + degree2 < f->degree)
		zeros[(*num_zeros)++] = (shift == 0 ? 0 : p - shift);
}

/* poly_get_zeros(), f already reduced mod p (degree not yet fixed) */
__device__ static uint32
ra_get_zeros(uint32 *zeros, ra_pl_t f, uint32 p, uint32 count_only)
{
	ra_pl_t g;
	uint32 i, j, num_zeros = 0;

	ra_fix_degree(&f);
	if (f.degree == 0)
		return 0;

	if (f.coef[0] == 0) {
		for (i = 1; i <= f.degree; i++) {
			if (f.coef[i])
				break;
		}
		for (j = i; i <= f.degree; i++)
			f.coef[i - j] = f.coef[i];
		f.degree = i - j - 1;
		zeros[num_zeros++] = 0;
	}

	if (f.degree == 0)
		return num_zeros;
	if (f.degree == 1) {
		uint32 w = f.coef[1];

		if (count_only)
			return num_zeros + 1;
		if (w != 1) {
			w = ra_modinv(w, p);
			zeros[num_zeros++] = ra_mulm(p - f.coef[0], w, p);
		}
		else {
			zeros[num_zeros++] = (f.coef[0] == 0 ? 0 :
						p - f.coef[0]);
		}
		return num_zeros;
	}

	if (p == 2) {
		uint32 parity = 0;

		for (i = 0; i <= f.degree; i++)
			parity ^= f.coef[i];
		if (parity == 0)
			zeros[num_zeros++] = 1;
		return num_zeros;
	}

	ra_poly_xpow(&g, 0, p - 1, &f, p);
	g.coef[0] = ra_subm(g.coef[0], 1, p);
	ra_fix_degree(&g);
	ra_poly_gcd(&g, &f, p);

	if (g.degree < 1 || count_only)
		return num_zeros + g.degree;

	ra_zeros_rec(zeros, 0, &num_zeros, &g, p);
	return num_zeros;
}

/* get_root_freq() of root_score.c: coeffs are the polynomial mod the
   largest power of p not above 65000 */
__device__ static double
ra_root_freq(const uint32 *coeffs, uint32 degree, uint32 initial_root,
		uint32 p)
{
	uint32 roots[RA_ROOT_BUF], new_roots[RA_ROOT_BUF];
	uint32 max_power = 1, power = p, num_roots, num_new_roots;
	double contrib;

	for (;;) {
		uint32 next_power = power * p;

		if (next_power > 65000)
			break;
		power = next_power;
		max_power++;
	}

	while (degree > 0 && coeffs[degree] == 0)
		degree--;

	num_roots = 1;
	roots[0] = initial_root;
	power = p;
	contrib = 1.0 / power;

	for (uint32 i = 2; i <= max_power; i++) {
		uint32 next_power = power * p;

		num_new_roots = 0;
		for (uint32 j = 0; j < num_roots; j++) {
			for (uint32 x = roots[j]; x < next_power; x += power) {
				uint32 polyval = coeffs[degree] % next_power;

				for (uint32 k = degree; k; k--) {
					polyval = (polyval * x +
						coeffs[k - 1]) % next_power;
				}
				if (polyval == 0 &&
				    num_new_roots < RA_ROOT_BUF)
					new_roots[num_new_roots++] = x;
			}
		}
		if (num_new_roots == 0)
			break;

		contrib += (double)num_new_roots / next_power;
		for (uint32 j = 0; j < num_new_roots; j++)
			roots[j] = new_roots[j];
		power = next_power;
		num_roots = num_new_roots;
	}
	return contrib * p / (p + 1);
}

/* the term of prime number j in analyze_poly_roots() */
__device__ static double
ra_term(const ra_tables_t *tab, uint32 j, uint32 deg,
	uint32 (*res)[RA_NUM_PRIMES], const uint32 *zero)
{
	const uint32 prime = tab->prime[j];
	uint32 zeros[RA_MAX_DEG];
	ra_pl_t f;
	uint32 high_coeff, num_roots;

	f.degree = deg;
	for (uint32 i = 0; i <= deg; i++)
		f.coef[i] = res[i][j] % prime;
	for (uint32 i = deg + 1; i < RA_PMAX; i++)
		f.coef[i] = 0;
	high_coeff = f.coef[deg];

	if (j >= RA_NUM_SMALL) {
		num_roots = ra_get_zeros(zeros, f, prime, 1);
		if (high_coeff == 0)
			num_roots++;
		return (1.0 - (double)num_roots * prime / (prime + 1)) *
			tab->log_prime[j] / (prime - 1);
	}
	else {
		uint32 coeffs[RA_MAX_DEG + 1];
		uint32 num_ordinary = 0;
		double contrib;

		/* poly_get_zeros_and_mult() */
		num_roots = ra_get_zeros(zeros, f, prime, 0);
		contrib = 1.0 / (prime - 1);

		if (num_roots > 0) {
			ra_pl_t fr = f;
			bool all_simple;

			ra_fix_degree(&fr);
			all_simple = (fr.degree == num_roots);
			for (uint32 i = 0; i <= deg; i++)
				coeffs[i] = res[i][j];

			for (uint32 r = 0; r < num_roots; r++) {
				uint32 mult = 0;

				if (!all_simple) {
					ra_pl_t g, rem;
					uint32 root = zeros[r];

					g.degree = 2;
					g.coef[0] = ra_mulm(root, root, prime);
					g.coef[1] = prime -
						ra_addm(root, root, prime);
					g.coef[2] = 1;
					ra_poly_mod(&rem, &fr, &g, prime);
					if (rem.degree == 0)
						mult = 1;
				}
				if (mult == 0)
					num_ordinary++;
				else
					contrib -= ra_root_freq(coeffs, deg,
							zeros[r], prime);
			}
		}

		if (high_coeff == 0) {
			/* the reversed polynomial, without the exact zeros
			   at its top */
			uint32 rdeg = deg;

			for (uint32 i = 0; i <= deg; i++)
				coeffs[i] = res[deg - i][j];
			while (rdeg && zero[deg - rdeg])
				rdeg--;
			contrib -= ra_root_freq(coeffs, rdeg, 0, prime);
		}

		return tab->log_prime[j] * (contrib -
				(double)num_ordinary * prime /
				(prime * prime - 1));
	}
}

template<uint32 BITS, uint32 TPI>
__device__ static void
ra_alpha(const ra_tables_t *tab, const ra_poly_t *polys,
	const ra_rot_t *rots, ra_res_t *out, ra_dd_t *bdd)
{
	typedef cgbn_context_t<TPI> context_t;
	typedef cgbn_env_t<context_t, BITS> env_t;
	typedef typename env_t::cgbn_t bn_t;

	__shared__ uint32 res[RA_MAX_DEG + 1][RA_NUM_PRIMES];
	__shared__ uint32 zero[RA_MAX_DEG + 1];
	__shared__ double terms[RA_NUM_PRIMES];
	__shared__ cgbn_mem_t<BITS> mag[RA_MAX_DEG + 1];
	__shared__ uint32 sign[RA_MAX_DEG + 1];

	const ra_rot_t rot = rots[blockIdx.x];
	const ra_poly_t *pl = polys + rot.poly;
	const uint32 deg = pl->degree;
	const uint32 group = threadIdx.x / TPI;
	const uint32 num_groups = RA_THREADS / TPI;

	{
		context_t ctx(cgbn_no_checks);
		env_t env(ctx.template env<env_t>());
		bn_t b[RA_MAX_DEG + 1], P, D, t, u;
		int64 v[3] = { rot.x, rot.y, rot.z };
		bn_t r[3];

		for (uint32 i = 0; i <= deg; i++)
			cgbn_load(env, b[i], (cgbn_mem_t<BITS> *)pl->a[i]);
		cgbn_load(env, P, (cgbn_mem_t<BITS> *)pl->p);
		cgbn_load(env, D, (cgbn_mem_t<BITS> *)pl->d);

		/* x, y, z as two's complement numbers of the width */
		for (uint32 i = 0; i < 3; i++) {
			uint64 w = (uint64)v[i];

			cgbn_set_ui32(env, r[i], (uint32)(w >> 32));
			cgbn_shift_left(env, r[i], r[i], 32);
			cgbn_add_ui32(env, r[i], r[i], (uint32)w);
			if (v[i] < 0) {
				cgbn_set_ui32(env, u, 1);
				cgbn_shift_left(env, u, u, 64);
				cgbn_sub(env, r[i], r[i], u);
			}
		}

		/* optimize_final(): b = a + (x + y X + z X^2)(P X - d) */
		cgbn_mul(env, t, P, r[2]);
		cgbn_add(env, b[3], b[3], t);
		cgbn_mul(env, t, D, r[2]);
		cgbn_sub(env, b[2], b[2], t);
		cgbn_mul(env, t, P, r[1]);
		cgbn_add(env, b[2], b[2], t);
		cgbn_mul(env, t, D, r[1]);
		cgbn_sub(env, b[1], b[1], t);
		cgbn_mul(env, t, P, r[0]);
		cgbn_add(env, b[1], b[1], t);
		cgbn_mul(env, t, D, r[0]);
		cgbn_sub(env, b[0], b[0], t);

		/* residues, floor mod (mpz_fdiv_ui) */
		for (uint32 k = group; k < (deg + 1) * RA_NUM_PRIMES;
		     k += num_groups) {
			uint32 i = k / RA_NUM_PRIMES;
			uint32 j = k % RA_NUM_PRIMES;
			uint32 m = tab->modulus[j];
			bool neg = (cgbn_extract_bits_ui32(env, b[i],
						BITS - 1, 1) != 0);
			uint32 rm;

			if (neg)
				cgbn_negate(env, t, b[i]);
			else
				cgbn_set(env, t, b[i]);
			rm = cgbn_rem_ui32(env, t, m);
			if (neg && rm != 0)
				rm = m - rm;
			if (threadIdx.x % TPI == 0)
				res[i][j] = rm;
		}
		if (group == 0) {
			for (uint32 i = 0; i <= deg; i++) {
				bool z = cgbn_equals_ui32(env, b[i], 0);
				bool neg = (cgbn_extract_bits_ui32(env, b[i],
						BITS - 1, 1) != 0);

				if (neg)
					cgbn_negate(env, t, b[i]);
				else
					cgbn_set(env, t, b[i]);
				cgbn_store(env, mag + i, t);
				if (threadIdx.x == 0) {
					zero[i] = z;
					sign[i] = neg;
				}
			}
		}
	}
	__syncthreads();

	for (uint32 j = threadIdx.x; j < RA_NUM_PRIMES; j += RA_THREADS)
		terms[j] = ra_term(tab, j, deg, res, zero);

	/* the coefficients as double-double, as dd_gmp2dd() makes them:
	   from the top 32-bit word down, d = d * 2^32 + word */
	if (threadIdx.x >= RA_THREADS - (RA_MAX_DEG + 1)) {
		uint32 i = threadIdx.x - (RA_THREADS - (RA_MAX_DEG + 1));
		mf_dd_t d = mf_dd(0, 0);

		if (i <= deg) {
			for (int32 w = BITS / 32 - 1; w >= 0; w--) {
				d = mf_dd(d.hi * 4294967296.0,
					  d.lo * 4294967296.0);
				d = mf_add_d(d, (double)mag[i]._limbs[w]);
			}
			if (sign[i])
				d = mf_neg(d);
		}
		bdd[blockIdx.x].hi[i] = d.hi;
		bdd[blockIdx.x].lo[i] = d.lo;
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		double root_score = 0;

		for (uint32 j = 0; j < RA_NUM_PRIMES; j++)
			root_score += terms[j];
		out[blockIdx.x].alpha = root_score;
		out[blockIdx.x].size = -1;
		out[blockIdx.x].flags = 0;
		out[blockIdx.x].keep = RA_NO_MF;
	}
}

#define RA_KERNEL(BITS, TPI)						\
extern "C" __global__ void __launch_bounds__(RA_THREADS)		\
alpha_##BITS(const ra_tables_t *tab, const ra_poly_t *polys,		\
		const ra_rot_t *rots, ra_res_t *out, ra_dd_t *bdd)	\
{									\
	ra_alpha<BITS, TPI>(tab, polys, rots, out, bdd);		\
}

RA_KERNEL(512, 8)
RA_KERNEL(1024, 32)

/*------------------------------------------------------------------------*/
/* Bernstein's size score (alpha_core.h): bs_size_<degree>, one warp per
   rotation, after alpha_<bits> in the same stream. Lane 0 finds the
   roots (find_poly_roots of common/polyroot.c: Jenkins-Traub in double
   precision, then Newton polishing in double-double complex); the
   integral (analyze_poly_size of size_score.c) is spread over the
   lanes like Murphy E's. */

#define BS_WARPS 4
#define BS_LIMIT 1e12		/* INTEGRATE_LIMIT */
#define BS_MAX_ROOTS 10		/* MAX_ROOTFINDER_DEGREE */
#define BS_DBL_EPSILON 2.2204460492503131e-16

/* include/ddcomplex.h */
typedef struct { mf_dd_t r, i; } bs_ddc_t;

__device__ __forceinline__ static bs_ddc_t
bs_ddc(mf_dd_t r, mf_dd_t i)
{
	bs_ddc_t c;

	c.r = r;
	c.i = i;
	return c;
}

__device__ __forceinline__ static bs_ddc_t
bs_cadd(bs_ddc_t a, bs_ddc_t b)
{
	return bs_ddc(mf_add_dd(a.r, b.r), mf_add_dd(a.i, b.i));
}

__device__ __forceinline__ static bs_ddc_t
bs_csub(bs_ddc_t a, bs_ddc_t b)
{
	return bs_ddc(mf_sub_dd(a.r, b.r), mf_sub_dd(a.i, b.i));
}

__device__ __forceinline__ static bs_ddc_t
bs_cmul(bs_ddc_t a, bs_ddc_t b)
{
	return bs_ddc(mf_sub_dd(mf_mul_dd(a.r, b.r), mf_mul_dd(a.i, b.i)),
		      mf_add_dd(mf_mul_dd(a.r, b.i), mf_mul_dd(a.i, b.r)));
}

__device__ static bs_ddc_t
bs_cdiv(bs_ddc_t a, bs_ddc_t b)
{
	if (mf_cmp_dd(mf_fabs(b.r), mf_fabs(b.i)) >= 0) {
		mf_dd_t q = mf_div_dd(b.i, b.r);
		mf_dd_t den = mf_add_dd(b.r, mf_mul_dd(q, b.i));

		return bs_ddc(mf_div_dd(mf_add_dd(a.r, mf_mul_dd(q, a.i)), den),
			mf_div_dd(mf_sub_dd(a.i, mf_mul_dd(q, a.r)), den));
	}
	else {
		mf_dd_t q = mf_div_dd(b.r, b.i);
		mf_dd_t den = mf_add_dd(b.i, mf_mul_dd(q, b.r));

		return bs_ddc(mf_div_dd(mf_add_dd(mf_mul_dd(q, a.r), a.i), den),
			mf_div_dd(mf_sub_dd(mf_mul_dd(q, a.i), a.r), den));
	}
}

/* common/polyroot.c: Jenkins-Traub */
typedef struct { double r, i; } bs_c_t;

__device__ __forceinline__ static bs_c_t
bs_c(double r, double i)
{
	bs_c_t c;

	c.r = r;
	c.i = i;
	return c;
}

__device__ __forceinline__ static bs_c_t
bs_add(bs_c_t x, bs_c_t y) { return bs_c(x.r + y.r, x.i + y.i); }

__device__ __forceinline__ static bs_c_t
bs_neg(bs_c_t x) { return bs_c(-x.r, -x.i); }

__device__ __forceinline__ static bs_c_t
bs_scale(double a, bs_c_t b) { return bs_c(a * b.r, a * b.i); }

__device__ __forceinline__ static bs_c_t
bs_mul(bs_c_t a, bs_c_t b)
{
	return bs_c(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

__device__ __forceinline__ static bs_c_t
bs_mac(bs_c_t a, bs_c_t x, bs_c_t b)
{
	return bs_c(a.r * x.r - a.i * x.i + b.r, a.r * x.i + a.i * x.r + b.i);
}

__device__ static double
bs_mod(bs_c_t x)
{
	double re = fabs(x.r), im = fabs(x.i);

	if (re == im)
		return re * M_SQRT2;
	if (re < im) {
		double t = re / im;
		return im * sqrt(1.0 + t * t);
	}
	else {
		double t = im / re;
		return re * sqrt(1.0 + t * t);
	}
}

__device__ static bs_c_t
bs_div(bs_c_t x, bs_c_t y)
{
	double zr, zi;

	if (y.r == 0 && y.i == 0) {
		zr = zi = 1.7976931348623157e308;	/* DBL_MAX */
	}
	else if (fabs(y.r) < fabs(y.i)) {
		double u = y.r / y.i, v = y.i + u * y.r;

		zr = (x.r * u + x.i) / v;
		zi = (x.i * u - x.r) / v;
	}
	else {
		double u = y.i / y.r, v = y.r + u * y.i;

		zr = (x.r + x.i * u) / v;
		zi = (x.i - x.r * u) / v;
	}
	return bs_c(zr, zi);
}

typedef struct {
	bs_c_t poly[BS_MAX_ROOTS + 1];
	bs_c_t poly_aux[BS_MAX_ROOTS + 1];
	bs_c_t hpoly[BS_MAX_ROOTS + 1];
	bs_c_t hpoly_aux[BS_MAX_ROOTS + 1];
	bs_c_t angle, angle_inc;
	uint32 hpoly_root_found;
	uint32 degree;
} bs_jt_t;

__device__ static double
bs_cauchy_bound(uint32 n, const bs_c_t *p)
{
	double x, xmax, f, dx, df;
	double norms[BS_MAX_ROOTS + 1];
	uint32 i;

	for (i = 0; i < n; i++)
		norms[i] = bs_mod(p[i]);
	norms[i] = -bs_mod(p[i]);

	xmax = exp((log(-norms[n]) - log(norms[0])) / (double)n);
	if (norms[n - 1] != 0.0) {
		x = -norms[n] / norms[n - 1];
		xmax = SO_MIN(x, xmax);
	}
	do {
		x = xmax;
		xmax = 0.1 * x;
		f = norms[0];
		for (i = 1; i <= n; i++)
			f = f * xmax + norms[i];
	} while (f > 0.0);

	dx = x;
	while (fabs(dx / x) > 0.005) {
		df = 0;
		f = norms[0];
		for (i = 1; i <= n; i++) {
			df = df * x + f;
			f = f * x + norms[i];
		}
		dx = f / df;
		x -= dx;
	}
	return x;
}

__device__ static bs_c_t
bs_poly_val(uint32 n, bs_c_t s, const bs_c_t *p, bs_c_t *q)
{
	bs_c_t pv;

	pv = q[0] = p[0];
	for (uint32 i = 1; i <= n; i++)
		pv = q[i] = bs_mac(pv, s, p[i]);
	return pv;
}

__device__ static void
bs_next_hpoly(bs_c_t correction, bs_jt_t *w)
{
	if (w->hpoly_root_found == 0) {
		w->hpoly[0] = w->poly_aux[0];
		for (uint32 i = 1; i < w->degree; i++)
			w->hpoly[i] = bs_mac(w->hpoly_aux[i - 1], correction,
						w->poly_aux[i]);
	}
	else {
		w->hpoly[0] = bs_c(0, 0);
		for (uint32 i = 1; i < w->degree; i++)
			w->hpoly[i] = w->hpoly_aux[i - 1];
	}
}

__device__ static bs_c_t
bs_next_correction(bs_c_t pval, bs_c_t curr_root, bs_jt_t *w)
{
	bs_c_t hval = bs_poly_val(w->degree - 1, curr_root, w->hpoly,
				w->hpoly_aux);

	if (bs_mod(hval) <= 10.0 * BS_DBL_EPSILON *
				bs_mod(w->hpoly[w->degree - 1])) {
		w->hpoly_root_found = 1;
		return bs_c(0, 0);
	}
	w->hpoly_root_found = 0;
	return bs_div(bs_neg(pval), hval);
}

__device__ static uint32
bs_stage3(bs_c_t *root, bs_jt_t *w)
{
	const double mre = 2.0 * M_SQRT2 * BS_DBL_EPSILON;
	bs_c_t curr_root = *root;

	for (uint32 i = 0; i < 10; i++) {	/* STAGE3_ITER */
		bs_c_t pval, correction;
		double mp, ms, tp;

		pval = bs_poly_val(w->degree, curr_root, w->poly, w->poly_aux);
		mp = bs_mod(pval);
		ms = bs_mod(curr_root);
		tp = bs_mod(w->poly_aux[0]) * mre / (BS_DBL_EPSILON + mre);
		for (uint32 j = 0; j <= w->degree; j++)
			tp = tp * ms + bs_mod(w->poly_aux[j]);
		tp = tp * (BS_DBL_EPSILON + mre) - mp * mre;

		if (mp <= 20.0 * tp) {
			*root = curr_root;
			return 1;
		}
		correction = bs_next_correction(pval, curr_root, w);
		bs_next_hpoly(correction, w);
		correction = bs_next_correction(pval, curr_root, w);
		curr_root = bs_add(curr_root, correction);
	}
	return 0;
}

__device__ static uint32
bs_stage2(uint32 stage2_iter, bs_c_t *root, bs_jt_t *w)
{
	bs_c_t curr_root = *root, correction, pval;

	pval = bs_poly_val(w->degree, curr_root, w->poly, w->poly_aux);
	correction = bs_next_correction(pval, curr_root, w);
	for (uint32 i = 0; i < stage2_iter; i++) {
		bs_next_hpoly(correction, w);
		correction = bs_next_correction(pval, curr_root, w);
		if (w->hpoly_root_found == 1)
			break;
	}
	*root = bs_add(curr_root, correction);
	return bs_stage3(root, w);
}

__device__ static void
bs_stage1(uint32 n, const bs_c_t *p, bs_c_t *h)
{
	uint32 i, j;

	for (i = 0; i < n; i++)
		h[i] = bs_scale((double)(n - i) / n, p[i]);
	for (i = 0; i < 5; i++) {		/* STAGE1_ITER */
		if (bs_mod(h[n - 1]) <= 10.0 * BS_DBL_EPSILON *
					bs_mod(p[n - 1])) {
			for (j = n - 1; j; j--)
				h[j] = h[j - 1];
			h[j] = bs_c(0, 0);
		}
		else {
			bs_c_t tmp = bs_div(bs_neg(p[n]), h[n - 1]);

			for (j = n - 1; j; j--)
				h[j] = bs_mac(h[j - 1], tmp, p[j]);
			h[j] = p[0];
		}
	}
}

__device__ static uint32
bs_find_one_root(bs_c_t *root, bs_jt_t *w)
{
	bs_c_t hpoly_start[BS_MAX_ROOTS + 1];
	double bound;

	if (w->degree <= 1) {
		*root = bs_div(bs_neg(w->poly[1]), w->poly[0]);
		return 1;
	}
	bound = bs_cauchy_bound(w->degree, w->poly);
	bs_stage1(w->degree, w->poly, hpoly_start);

	for (uint32 i = 0; i < 2; i++) {
		for (uint32 j = 1; j < 10; j++) {
			w->angle = bs_mul(w->angle, w->angle_inc);
			*root = bs_scale(bound, w->angle);
			for (uint32 k = 0; k < w->degree; k++)
				w->hpoly[k] = hpoly_start[k];
			if (bs_stage2(10 * j, root, w) == 1)
				return 1;
		}
	}
	return 0;
}

/* find_poly_roots(): 0 on success */
__device__ static uint32
bs_find_roots(const mf_dd_t *poly, uint32 degree, bs_ddc_t *roots,
		const mf_tables_t *tab)
{
	bs_jt_t w;
	bs_c_t droots[BS_MAX_ROOTS + 1];
	uint32 i, found;

	if (degree == 1) {
		roots[0].r = mf_div_dd(mf_neg(poly[0]), poly[1]);
		roots[0].i = mf_dd(0, 0);
		return 0;
	}

	/* jenkins_traub(), on the reversed double coefficients */
	for (i = degree, found = 0; i; i--, found++) {
		if (poly[degree - i].hi != 0.0)
			break;
		droots[found] = bs_c(0, 0);
	}
	w.degree = i;
	for (i = 0; i <= w.degree; i++)
		w.poly[i] = bs_c(poly[degree - i].hi, 0.0);
	w.angle = bs_c(M_SQRT1_2, -M_SQRT1_2);
	w.angle_inc = bs_c(tab->jt_cos94, tab->jt_sin94);

	for (; found < degree; found++) {
		if (bs_find_one_root(droots + found, &w) == 0)
			break;
		(w.degree)--;
		for (i = 0; i <= w.degree; i++)
			w.poly[i] = w.poly_aux[i];
	}
	if (found != degree)
		return 1;

	/* polish_root() each */
	for (i = 0; i < degree; i++) {
		bs_ddc_t x = bs_ddc(mf_dd(droots[i].r, 0), mf_dd(droots[i].i, 0));
		const double eps2 = 1e-30 * 1e-30;

		for (uint32 it = 0; it < 10; it++) {	/* NEWTON_ITER */
			bs_ddc_t f = bs_ddc(poly[degree], mf_dd(0, 0));
			bs_ddc_t df = bs_ddc(mf_dd(0, 0), mf_dd(0, 0));
			bs_ddc_t dx;
			mf_dd_t abs_x, abs_dx;

			for (int32 j = degree - 1; j >= 0; j--) {
				df = bs_cadd(bs_cmul(df, x), f);
				f = bs_cadd(bs_cmul(f, x),
					bs_ddc(poly[j], mf_dd(0, 0)));
			}
			dx = bs_cdiv(f, df);
			x = bs_csub(x, dx);

			abs_x = mf_add_dd(mf_mul_dd(x.r, x.r),
					mf_mul_dd(x.i, x.i));
			abs_dx = mf_add_dd(mf_mul_dd(dx.r, dx.r),
					mf_mul_dd(dx.i, dx.i));
			if (mf_cmp_dd(abs_dx, mf_mul_d(abs_x, eps2)) <= 0)
				break;
		}
		roots[i] = x;
		if (mf_cmp_dd(mf_fabs(roots[i].i),
			      mf_mul_d(mf_fabs(roots[i].r), 1e-30)) <= 0)
			roots[i].i = mf_dd(0, 0);
	}
	return 0;
}

/* get_polyval() and integrand() of size_score.c */
__device__ static double
bs_polyval(const double *p, uint32 degree, double x, double h)
{
	double base, off, hpow;

	switch (degree) {
	case 1:
		base = p[1]*x+p[0];
		off = p[1]*h;
		break;
	case 4:
		base = (((p[4]*x+p[3])*x+p[2])*x+p[1])*x+p[0];
		hpow = h;
		off = hpow * (((4*p[4]*x+3*p[3])*x+2*p[2])*x+p[1]);
		hpow *= h;
		off += hpow * ((6*p[4]*x+3*p[3])*x+p[2]);
		hpow *= h;
		off += hpow * (4*p[4]*x+p[3]);
		hpow *= h;
		off += hpow * p[4];
		break;
	case 5:
		base = ((((p[5]*x+p[4])*x+p[3])*x+p[2])*x+p[1])*x+p[0];
		hpow = h;
		off = hpow * ((((5*p[5]*x+4*p[4])*x+3*p[3])*x+2*p[2])*x+p[1]);
		hpow *= h;
		off += hpow * (((10*p[5]*x+6*p[4])*x+3*p[3])*x+p[2]);
		hpow *= h;
		off += hpow * ((10*p[5]*x+4*p[4])*x+p[3]);
		hpow *= h;
		off += hpow * (5*p[5]*x+p[4]);
		hpow *= h;
		off += hpow * p[5];
		break;
	case 6:
		base = (((((p[6]*x+p[5])*x+p[4])*x+p[3])*x+p[2])*x+p[1])*x+p[0];
		hpow = h;
		off = hpow * (((((6*p[6]*x+5*p[5])*x+4*p[4])*x+3*p[3])*x+2*p[2])*x+p[1]);
		hpow *= h;
		off += hpow * ((((15*p[6]*x+10*p[5])*x+6*p[4])*x+3*p[3])*x+p[2]);
		hpow *= h;
		off += hpow * (((20*p[6]*x+10*p[5])*x+4*p[4])*x+p[3]);
		hpow *= h;
		off += hpow * ((15*p[6]*x+5*p[5])*x+p[4]);
		hpow *= h;
		off += hpow * (6*p[6]*x+p[5]);
		hpow *= h;
		off += hpow * p[6];
		break;
	default:
		base = off = 0;
		break;
	}
	return base + off;
}

struct bs_size_f {
	const double *r, *a;
	uint32 deg;
	double power;

	__device__ double operator()(double x, double h) const {
		double v = bs_polyval(r, 1, x, h) * bs_polyval(a, deg, x, h);

		return pow(fabs(v), power);
	}
};

template<uint32 DEG>
__device__ static void
bs_size_deg(const mf_tables_t *tab, const ra_poly_t *polys,
	const ra_rot_t *rots, const ra_dd_t *bdd, ra_res_t *res, uint32 n,
	uint32 *counter, mf_in_t *mf_area, mf_in_t *keep, uint32 *keep_cnt)
{
	__shared__ mf_sub_t heaps[BS_WARPS][MF_HEAP];
	__shared__ double ends[BS_WARPS][2 * BS_MAX_ROOTS + 2];
	__shared__ uint32 num_ends[BS_WARPS];
	__shared__ double mf_roots[BS_WARPS][MF_MAX_ROOTS];
	__shared__ uint32 mf_num_roots[BS_WARPS], mf_real[BS_WARPS];
	const uint32 warp = threadIdx.x / 32, lane = threadIdx.x & 31;

	for (;;) {
		uint32 idx = 0;
		double r[2], a[DEG + 1];
		bs_size_f f;
		double result, error;

		__syncwarp();
		if (lane == 0)
			idx = atomicAdd(counter, 1);
		idx = __shfl_sync(MF_FULL, idx, 0);
		if (idx >= n)
			break;
		if (res[idx].alpha > -4.5) {	/* optimize_final() stops */
			if (lane == 0) {
				res[idx].size = -1;
				res[idx].flags = 0;
			}
			continue;
		}

		/* the roots of R(x) A(x), and the integral's breakpoints */
		if (lane == 0) {
			const ra_poly_t *pl = polys + rots[idx].poly;
			mf_dd_t rp[2], ap[DEG + 1];
			bs_ddc_t roots[2 * BS_MAX_ROOTS];
			uint32 j = 0, fail;

			for (uint32 i = 0; i <= 1; i++)
				rp[i] = mf_dd(pl->r_hi[i], pl->r_lo[i]);
			for (uint32 i = 0; i <= DEG; i++)
				ap[i] = mf_dd(bdd[idx].hi[i], bdd[idx].lo[i]);
			fail = bs_find_roots(rp, 1, roots, tab) ||
				bs_find_roots(ap, DEG, roots + 1, tab);
			if (!fail) {
				/* for Murphy E: the real parts of the roots
				   with imaginary part >= 0, and the real
				   roots of A */
				uint32 nr = 0, real = 0;

				for (uint32 i = 0; i < DEG; i++) {
					if (roots[1 + i].i.hi == 0.0 &&
					    roots[1 + i].i.lo == 0.0)
						real++;
				}
				for (uint32 i = 0; i < DEG + 1; i++) {
					if (roots[i].i.hi >= 0.0)
						mf_roots[warp][nr++] =
							roots[i].r.hi;
				}
				mf_num_roots[warp] = nr;
				mf_real[warp] = real;

				for (uint32 i = 0; i < DEG + 1; i++) {
					if (roots[i].i.hi < 0.0 ||
					    roots[i].r.hi <= -BS_LIMIT ||
					    roots[i].r.hi >= BS_LIMIT)
						continue;
					ends[warp][j++] = roots[i].r.hi;
				}
				ends[warp][j++] = -BS_LIMIT;
				ends[warp][j++] = BS_LIMIT;
				for (uint32 i = 1; i < j; i++) {
					double e = ends[warp][i];
					uint32 k = i;

					for (; k > 0 && ends[warp][k - 1] > e;
					     k--)
						ends[warp][k] = ends[warp][k - 1];
					ends[warp][k] = e;
				}
			}
			num_ends[warp] = fail ? 0 : j;
		}
		__syncwarp();
		if (num_ends[warp] == 0) {
			if (lane == 0) {
				res[idx].size = 0;
				res[idx].flags = 0;
			}
			continue;
		}

		{
			const ra_poly_t *pl = polys + rots[idx].poly;

			r[0] = pl->r_hi[0];
			r[1] = pl->r_hi[1];
			for (uint32 i = 0; i <= DEG; i++)
				a[i] = bdd[idx].hi[i];
		}
		f.r = r;
		f.a = a;
		f.deg = DEG;
		f.power = -2.0 / (1 + DEG);

		result = mf_integrate(f, tab, heaps[warp], ends[warp],
				num_ends[warp], lane, &error);

		/* analyze_poly_size()'s test for convergence; then
		   get_bernstein_score() and optimize_final()'s bound */
		if (lane == 0) {
			const ra_poly_t *pl = polys + rots[idx].poly;
			double size = (result > 1 ||
				error > 1e-6 * fabs(result)) ? 0 : result;
			double bscore = size * pow(exp(res[idx].alpha),
						-2. / (DEG + 1));

			res[idx].size = size;
			res[idx].flags = 0;
			res[idx].keep = RA_NO_MF;
			if (bscore > pl->min_e_bernstein) {
				uint32 m = rots[idx].mf_index;
				mf_in_t *f = NULL;

				res[idx].flags = RA_PASSED;
				res[idx].num_real_roots = mf_real[warp];
				if (mf_area != NULL && m != RA_NO_MF) {
					f = mf_area + m;
				}
				else if (keep != NULL) {
					/* for the host's Murphy E */
					uint32 q = rots[idx].poly;
					uint32 slot = atomicAdd(keep_cnt + q, 1);

					if (slot < RA_KEEP) {
						f = keep + q * RA_KEEP + slot;
						res[idx].keep = slot;
					}
				}
				if (f != NULL) {
					/* the Murphy E input, as
					   optimize_final_batch() makes it */

					for (uint32 i = 0; i <= MF_MAX_DEG; i++) {
						f->a_hi[i] = (i <= DEG) ?
							bdd[idx].hi[i] : 0;
						f->a_lo[i] = (i <= DEG) ?
							bdd[idx].lo[i] : 0;
					}
					for (uint32 i = 0; i < 2; i++) {
						f->r_hi[i] = pl->r_hi[i];
						f->r_lo[i] = pl->r_lo[i];
					}
					/* analyze_poly_roots() of a
					   linear polynomial */
					f->root_score_r = 0.569959993064325;
					f->root_score_a = res[idx].alpha;
					for (uint32 i = 0; i < MF_MAX_ROOTS; i++)
						f->roots[i] = (i < mf_num_roots[warp]) ?
							mf_roots[warp][i] : 0;
					f->degree = DEG;
					f->num_roots = mf_num_roots[warp];
					f->num_real_roots = mf_real[warp];
					f->pad = 0;
				}
			}
		}
		__syncwarp();
	}
}

#define BS_KERNEL(DEG)							\
extern "C" __global__ void __launch_bounds__(BS_WARPS * 32)		\
bs_size_##DEG(const mf_tables_t *tab, const ra_poly_t *polys,		\
		const ra_rot_t *rots, const ra_dd_t *bdd,		\
		ra_res_t *res, uint32 n, uint32 *counter,		\
		mf_in_t *mf_area, mf_in_t *keep, uint32 *keep_cnt)	\
{									\
	bs_size_deg<DEG>(tab, polys, rots, bdd, res, n, counter,	\
			mf_area, keep, keep_cnt);			\
}

BS_KERNEL(4)
BS_KERNEL(5)
BS_KERNEL(6)

/*------------------------------------------------------------------------*/
/* The root sieve's setup (alpha_core.h): init_sieve() and
   init_sieve_core() of root_sieve.c, one block per polynomial. CGBN
   reduces the coefficients, d and P modulo each prime power; then, for
   each power in turn, the threads take the residue classes i and the
   roots are compacted in order of i, as the CPU lists them. */

template<uint32 BITS, uint32 TPI>
__device__ static void
rs_setup(const rs_powers_t *pw, const ra_poly_t *polys,
	const rs_job_t *jobs, ls_root_t *area, rs_setup_out_t *out)
{
	typedef cgbn_context_t<TPI> context_t;
	typedef cgbn_env_t<context_t, BITS> env_t;
	typedef typename env_t::cgbn_t bn_t;

	/* per power: a_0 .. a_deg, then d, then P, mod the power */
	__shared__ uint32 res[RS_MAX_POWERS][RA_MAX_DEG + 3];
	__shared__ uint32 warp_count[RS_THREADS / 32];

	const rs_job_t job = jobs[blockIdx.x];
	const ra_poly_t *pl = polys + job.poly;
	const uint32 deg = pl->degree;
	const uint32 np = pw->num_powers;
	const uint32 tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
	uint32 offset = 0;

	{
		context_t ctx(cgbn_no_checks);
		env_t env(ctx.template env<env_t>());
		bn_t v[RA_MAX_DEG + 3], t;
		const uint32 nv = deg + 3;
		const uint32 group = tid / TPI, num_groups = RS_THREADS / TPI;

		for (uint32 i = 0; i <= deg; i++)
			cgbn_load(env, v[i], (cgbn_mem_t<BITS> *)pl->a[i]);
		cgbn_load(env, v[deg + 1], (cgbn_mem_t<BITS> *)pl->d);
		cgbn_load(env, v[deg + 2], (cgbn_mem_t<BITS> *)pl->p);

		/* floor remainders, as mpz_fdiv_ui */
		for (uint32 k = group; k < np * nv; k += num_groups) {
			uint32 j = k / nv, q = k % nv;
			uint32 m = pw->power[j];
			bool neg = false;
			uint32 rm;

			for (uint32 i = 0; i < nv; i++) {
				if (i == q) {
					neg = (cgbn_extract_bits_ui32(env,
						v[i], BITS - 1, 1) != 0);
					if (neg)
						cgbn_negate(env, t, v[i]);
					else
						cgbn_set(env, t, v[i]);
				}
			}
			rm = cgbn_rem_ui32(env, t, m);
			if (neg && rm != 0)
				rm = m - rm;
			if (tid % TPI == 0)
				res[j][q] = rm;
		}
	}
	__syncthreads();

	for (uint32 j = 0; j < np; j++) {
		const uint32 p = pw->prime[j], pk = pw->power[j];
		const uint32 md = res[j][deg + 1], mp = res[j][deg + 2];
		uint32 num_roots = 0, proj = 0;

		for (uint32 i0 = 0; i0 < pk; i0 += RS_THREADS) {
			uint32 i = i0 + tid;
			bool is_root = false, is_proj = false;
			ls_root_t r;
			uint32 ballot, rank, before = 0, chunk = 0;

			if (i < pk) {
				uint32 mi = ra_subm(md, ra_mulm(i, mp, pk), pk);
				uint32 fi = res[j][deg];

				for (uint32 k = deg; k; k--)
					fi = (fi * i + res[j][k - 1]) % pk;

				if (mi != 0) {
					uint32 step = pk;

					while (mi % p == 0 && fi % p == 0) {
						mi /= p;
						fi /= p;
						step /= p;
					}
					if (mi % p) {
						is_root = true;
						r.resclass = (uint16)i;
						r.step = (uint16)step;
						r.start = (uint16)ra_mulm(fi,
							ra_modinv(mi, step),
							step);
						r.pad = 0;
					}
				}
				else if (fi == 0) {
					is_proj = true;
				}
			}

			/* the roots of this chunk, in order of i */
			ballot = __ballot_sync(0xffffffff, is_root);
			rank = __popc(ballot & ((1u << lane) - 1));
			if (lane == 0)
				warp_count[warp] = __popc(ballot);
			proj += __syncthreads_count(is_proj);
			for (uint32 w = 0; w < RS_THREADS / 32; w++) {
				if (w < warp)
					before += warp_count[w];
				chunk += warp_count[w];
			}
			if (is_root && offset + num_roots + before + rank <
						RS_MAX_ROOTS) {
				area[job.base + offset + num_roots + before +
							rank] = r;
			}
			num_roots += chunk;
			__syncthreads();
		}

		if (tid == 0) {
			out[blockIdx.x].num_roots[j] = num_roots;
			out[blockIdx.x].proj[j] = proj;
		}
		offset += num_roots;
	}
	if (tid == 0)
		out[blockIdx.x].total = offset;
}

#define RS_KERNEL(BITS, TPI)						\
extern "C" __global__ void __launch_bounds__(RS_THREADS)		\
rs_setup_##BITS(const rs_powers_t *pw, const ra_poly_t *polys,		\
		const rs_job_t *jobs, ls_root_t *area,			\
		rs_setup_out_t *out)					\
{									\
	rs_setup<BITS, TPI>(pw, polys, jobs, area, out);		\
}

RS_KERNEL(512, 8)
RS_KERNEL(1024, 32)

/*------------------------------------------------------------------------*/
/* The lattice choice (lattice_core.h): lt_xy and lt_x. Ports of
   root_sieve_deg5_xy.c and root_sieve_deg45_x.c, the same integer
   operations in the same order. */

/* find_hits() of root_sieve_deg5_xy.c: one thread per lattice prime */
extern "C" __global__ void
lt_xy(const lt_set_t *sets, const ls_root_t *area, lt_hit_t *hits)
{
	const lt_set_t *st = sets + blockIdx.x;
	const uint32 j = threadIdx.x;
	uint16 sieve[16 * 16];		/* MAX_DIM */
	lt_hit_t *h;
	uint32 np, dim, size, max_val = 0, k = 0;
	const lt_power_t *pw;

	if (j >= st->num_primes)
		return;
	h = hits + blockIdx.x * LT_MAX_PRIMES + j;
	np = st->num_powers[j];
	pw = st->powers + st->power0[j];
	dim = pw[np - 1].power;
	size = dim * dim;
	if (size > 16 * 16) {
		h->power = 0;
		h->num_roots = 0xffffffffu;	/* tell the host */
		return;
	}
	for (uint32 i = 0; i < size; i++)
		sieve[i] = 0;

	for (uint32 q = 0; q < np; q++) {
		const uint16 contrib = (uint16)pw[q].contrib;

		for (uint32 r = 0; r < pw[q].num_roots; r++) {
			const ls_root_t rt = area[st->area_base + pw[q].root0 + r];
			uint32 start = rt.start, step = rt.step;
			uint32 resclass = rt.resclass;

			if (resclass >= step)
				resclass %= step;
			for (uint32 i = 0; i < dim; i++) {
				uint32 ri = start;

				do {
					sieve[i * dim + ri] += contrib;
					ri += step;
				} while (ri < dim);
				start = ra_subm(start, resclass, step);
			}
		}
	}

	for (uint32 i = 0; i < size; i++) {
		uint32 s = sieve[i];

		if (s >= max_val) {
			if (s > max_val) {
				max_val = s;
				k = 0;
			}
			if (k == LT_MAX_ROOTS)
				break;
			h->score[k] = (uint16)s;
			h->roots[k][0] = (uint8)(i % dim);
			h->roots[k][1] = (uint8)((i / dim) % dim);
			k++;
		}
	}
	h->power = dim;
	h->num_roots = k;
}

/* compute_lattices() of root_sieve_util.c, dim 1: the combinations in
   the original's order (the last prime's index outermost, every index
   counting down) */
__device__ static uint32
lt_lattices(uint32 n, const uint32 *power, const uint32 *nr,
		const uint16 (*score)[LT_MAX_ROOTS],
		const uint8 (*root)[LT_MAX_ROOTS], uint64 lattice_size,
		uint32 num_lattices, uint64 *out_x, uint16 *out_score)
{
	uint64 prod[LT_MAX_PRIMES], acc[LT_MAX_PRIMES + 1];
	uint16 sc[LT_MAX_PRIMES + 1];
	uint32 d[LT_MAX_PRIMES], cnt = 0;

	for (uint32 j = 0; j < n; j++) {
		uint32 p = power[j];

		if (nr[j] == 0)
			return 0;
		prod[j] = lattice_size / p;
		prod[j] *= ra_modinv((uint32)(prod[j] % p), p);
		d[j] = nr[j] - 1;
	}
	acc[n] = 0;
	sc[n] = 0;
	for (int32 j = n - 1; j >= 0; j--) {
		sc[j] = sc[j + 1] + score[j][d[j]];
		acc[j] = acc[j + 1] + prod[j] * root[j][d[j]];
	}

	for (;;) {
		uint32 j = 0;

		if (cnt == num_lattices)
			break;
		out_x[cnt] = acc[0] % lattice_size;
		out_score[cnt] = sc[0];
		cnt++;

		while (j < n && d[j] == 0) {
			d[j] = nr[j] - 1;
			j++;
		}
		if (j == n)
			break;
		d[j]--;
		for (int32 q = j; q >= 0; q--) {
			sc[q] = sc[q + 1] + score[q][d[q]];
			acc[q] = acc[q + 1] + prod[q] * root[q][d[q]];
		}
	}
	return cnt;
}

/* one xy lattice and y block of sieve_x_run_deg5's loop: xdata_init,
   find_hits' do_sieving and root_sieve_x */
__device__ static uint32
lt_item(const lt_xreq_t *rq, const lt_lat_t *lat, uint32 b,
	const uint32 *ybmod, const ls_root_t *area,
	uint64 *out_x, uint16 *out_score)
{
	const lt_set_t *st = &rq->set;
	uint16 sieve[LT_MAX_SIEVE];
	uint32 power[LT_MAX_PRIMES], nr[LT_MAX_PRIMES];
	uint16 score[LT_MAX_PRIMES][LT_MAX_ROOTS];
	uint8 root[LT_MAX_PRIMES][LT_MAX_ROOTS];
	uint32 num_lattices = 1;

	for (uint32 j = 0; j < st->num_primes; j++) {
		const lt_power_t *pw = st->powers + st->power0[j];
		const uint32 np = st->num_powers[j];
		const uint32 sieve_size = pw[np - 1].power;
		uint32 max_val = 0, k = 0, thr;

		for (uint32 i = 0; i < sieve_size; i++)
			sieve[i] = 0;
		for (uint32 q = 0; q < np; q++) {
			const uint32 p = pw[q].power;
			const uint16 contrib = (uint16)pw[q].contrib;
			const uint32 qi = st->power0[j] + q;
			uint32 y_mod = ra_addm((uint32)(lat->y % p), ybmod[qi], p);

			for (uint32 r = 0; r < pw[q].num_roots; r++) {
				const ls_root_t rt = area[st->area_base +
							pw[q].root0 + r];
				uint32 stride = ra_mulm(rt.resclass, pw[q].latmod, p);
				uint32 start = ra_subm(rt.start,
						ra_mulm(rt.resclass, y_mod, p), p);

				start = ra_subm(start, (uint32)((uint64)b *
							stride % p), p);
				for (uint32 x = start; x < sieve_size; x += p)
					sieve[x] += contrib;
			}
		}

		for (uint32 i = 0; i < sieve_size; i++) {
			if (sieve[i] > max_val)
				max_val = sieve[i];
		}
		thr = (uint16)(0.7 * max_val);
		for (uint32 i = 0; i < sieve_size; i++) {
			if (sieve[i] >= thr) {
				if (k == LT_MAX_ROOTS)
					break;
				score[j][k] = sieve[i];
				root[j][k] = (uint8)i;
				k++;
			}
		}
		power[j] = sieve_size;
		nr[j] = k;
		num_lattices *= k;
	}

	num_lattices = SO_MIN(num_lattices, LT_MAX_X_LATTICES);
	return lt_lattices(st->num_primes, power, nr, score, root,
			rq->x_lattice, num_lattices, out_x, out_score);
}

/* save_xline() and heapify() of root_sieve_deg45_x.c */
__device__ static void
lt_heapify(lt_xline_t *h, uint32 index, uint32 size)
{
	uint32 c;
	lt_xline_t tmp;

	for (c = 2 * index + 1; c < size - 1; index = c, c = 2 * index + 1) {
		if (h[c].score > h[c + 1].score)
			c++;
		if (h[index].score > h[c].score) {
			tmp = h[index]; h[index] = h[c]; h[c] = tmp;
		}
		else {
			return;
		}
	}
	if (c == size - 1 && h[index].score > h[c].score) {
		tmp = h[index]; h[index] = h[c]; h[c] = tmp;
	}
}

__device__ static void
lt_save(lt_xline_t *h, uint32 *n, uint32 score, uint32 block,
	uint32 lattice, uint64 resclass)
{
	if (*n <= LT_HEAP - 1) {
		lt_xline_t *s = h + (*n)++;

		s->score = (uint16)score;
		s->which_y_block = block;
		s->which_lattice = lattice;
		s->resclass = resclass;
		if (*n == LT_HEAP) {
			for (int32 i = (LT_HEAP - 1) >> 1; i >= 0; i--)
				lt_heapify(h, (uint32)i, LT_HEAP);
		}
	}
	else if (h->score < score) {
		h->score = (uint16)score;
		h->which_y_block = block;
		h->which_lattice = lattice;
		h->resclass = resclass;
		lt_heapify(h, 0, LT_HEAP);
	}
}

/* the x-line choice: one block per request; the threads take the
   (lattice, y block) items in turn, thread 0 fills the heap in the
   CPU's order */
extern "C" __global__ void __launch_bounds__(LT_THREADS)
lt_x(const lt_xreq_t *reqs, const lt_lat_t *lats, const ls_root_t *area,
	uint64 *scratch_x, uint16 *scratch_score, lt_xout_t *out)
{
	__shared__ lt_xline_t heap[LT_HEAP];
	__shared__ uint32 ybmod[LT_MAX_POWERS];
	__shared__ uint32 cnt[LT_THREADS];
	__shared__ uint32 hn;
	const lt_xreq_t *rq = reqs + blockIdx.x;
	const uint32 tid = threadIdx.x;
	const uint32 items = rq->num_lattices * rq->y_blocks;
	uint64 *sx = scratch_x + ((size_t)blockIdx.x * LT_THREADS + tid) *
					LT_MAX_X_LATTICES;
	uint16 *ss = scratch_score + ((size_t)blockIdx.x * LT_THREADS + tid) *
					LT_MAX_X_LATTICES;

	/* y_base modulo each power (mpz_fdiv_ui), in CGBN */
	if (tid < LS_TPI) {
		ls_context_t ctx(cgbn_no_checks);
		ls_env_t env(ctx.env<ls_env_t>());
		ls_bn_t yb;
		uint32 np = 0;

		for (uint32 j = 0; j < rq->set.num_primes; j++)
			np += rq->set.num_powers[j];
		ls_bn_i64(env, yb, rq->y_base);
		for (uint32 q = 0; q < np; q++) {
			uint32 r = ls_bn_fmod(env, yb, rq->set.powers[q].power);

			if (tid == 0)
				ybmod[q] = r;
		}
	}
	if (tid == 0)
		hn = 0;
	__syncthreads();

	for (uint32 base = 0; base < items; base += LT_THREADS) {
		uint32 it = base + tid;

		cnt[tid] = 0;
		if (it < items) {
			uint32 i = it / rq->y_blocks, b = it % rq->y_blocks;

			cnt[tid] = lt_item(rq, lats + rq->lat0 + i, b, ybmod,
						area, sx, ss);
		}
		__syncthreads();

		if (tid == 0) {
			for (uint32 t = 0; t < LT_THREADS && base + t < items;
			     t++) {
				uint32 i = (base + t) / rq->y_blocks;
				uint32 b = (base + t) % rq->y_blocks;
				uint32 xy_score = lats[rq->lat0 + i].score;
				const uint64 *tx = scratch_x +
					((size_t)blockIdx.x * LT_THREADS + t) *
					LT_MAX_X_LATTICES;
				const uint16 *ts = scratch_score +
					((size_t)blockIdx.x * LT_THREADS + t) *
					LT_MAX_X_LATTICES;

				for (uint32 e = 0; e < cnt[t]; e++) {
					uint32 score = xy_score + ts[e];

					if (hn < LT_HEAP || score > heap[0].score)
						lt_save(heap, &hn, score, b, i,
							tx[e]);
				}
			}
		}
		__syncthreads();
	}

	for (uint32 e = tid; e < hn; e += LT_THREADS)
		out[blockIdx.x].e[e] = heap[e];
	if (tid == 0)
		out[blockIdx.x].n = hn;
}

/*------------------------------------------------------------------------*/
/* The polynomials optimize_final() keeps (alpha_core.h): rotation and
   translation applied exactly, in CGBN, one 32-thread group each.
   translate_gmp() of optimize.c: c_i = sum_j w_ij b_j (-k)^(j-i), by
   Horner's rule with the binomial weights */

__device__ static const uint32 rf_weights[6 + 1][6 + 1] = {
	{  1,  1,  1,  1,  1,  1,  1},
	{  0,  1,  2,  3,  4,  5,  6},
	{  0,  0,  1,  3,  6, 10, 15},
	{  0,  0,  0,  1,  4, 10, 20},
	{  0,  0,  0,  0,  1,  5, 15},
	{  0,  0,  0,  0,  0,  1,  6},
	{  0,  0,  0,  0,  0,  0,  1},
};

extern "C" __global__ void __launch_bounds__(32)
rf_final(const ra_poly_t *polys, const ra_final_t *jobs, ra_final_out_t *out)
{
	typedef cgbn_context_t<32> context_t;
	typedef cgbn_env_t<context_t, 1024> env_t;
	typedef env_t::cgbn_t bn_t;

	context_t ctx(cgbn_no_checks);
	env_t env(ctx.env<env_t>());
	const ra_final_t job = jobs[blockIdx.x];
	const ra_poly_t *pl = polys + job.poly;
	const uint32 deg = pl->degree;
	bn_t b[RA_MAX_DEG + 1], P, D, t, mk, acc;
	int64 v[3] = { job.x, job.y, job.z };
	bn_t r[3];

	for (uint32 i = 0; i <= deg; i++)
		cgbn_load(env, b[i], (cgbn_mem_t<1024> *)pl->a[i]);
	cgbn_load(env, P, (cgbn_mem_t<1024> *)pl->p);
	cgbn_load(env, D, (cgbn_mem_t<1024> *)pl->d);
	for (uint32 i = 0; i < 3; i++) {
		uint64 w = (uint64)v[i];

		cgbn_set_ui32(env, r[i], (uint32)(w >> 32));
		cgbn_shift_left(env, r[i], r[i], 32);
		cgbn_add_ui32(env, r[i], r[i], (uint32)w);
		if (v[i] < 0) {
			cgbn_set_ui32(env, t, 1);
			cgbn_shift_left(env, t, t, 64);
			cgbn_sub(env, r[i], r[i], t);
		}
	}
	{
		uint64 w = (uint64)(-job.k);

		cgbn_set_ui32(env, mk, (uint32)(w >> 32));
		cgbn_shift_left(env, mk, mk, 32);
		cgbn_add_ui32(env, mk, mk, (uint32)w);
		if (-job.k < 0) {
			cgbn_set_ui32(env, t, 1);
			cgbn_shift_left(env, t, t, 64);
			cgbn_sub(env, mk, mk, t);
		}
	}

	/* optimize_final(): the rotation */
	cgbn_mul(env, t, P, r[2]);
	cgbn_add(env, b[3], b[3], t);
	cgbn_mul(env, t, D, r[2]);
	cgbn_sub(env, b[2], b[2], t);
	cgbn_mul(env, t, P, r[1]);
	cgbn_add(env, b[2], b[2], t);
	cgbn_mul(env, t, D, r[1]);
	cgbn_sub(env, b[1], b[1], t);
	cgbn_mul(env, t, P, r[0]);
	cgbn_add(env, b[1], b[1], t);
	cgbn_mul(env, t, D, r[0]);
	cgbn_sub(env, b[0], b[0], t);

	/* translate_gmp() */
	for (uint32 i = 0; i < deg; i++) {
		cgbn_mul_ui32(env, acc, b[deg], rf_weights[i][deg]);
		for (uint32 j = deg; j > i; j--) {
			cgbn_mul(env, acc, acc, mk);
			cgbn_mul_ui32(env, t, b[j - 1], rf_weights[i][j - 1]);
			cgbn_add(env, acc, acc, t);
		}
		cgbn_store(env, (cgbn_mem_t<1024> *)out[blockIdx.x].b[i], acc);
	}
	cgbn_store(env, (cgbn_mem_t<1024> *)out[blockIdx.x].b[deg], b[deg]);

	/* the linear part: lin0 = -d, plus lin1 (= P) times -k */
	cgbn_mul(env, t, P, mk);
	cgbn_sub(env, t, t, D);
	cgbn_store(env, (cgbn_mem_t<1024> *)out[blockIdx.x].lin0, t);
}
