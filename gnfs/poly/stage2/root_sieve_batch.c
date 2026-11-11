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


/* The batched root sieve (see stage2.h): msieve's root sieve with its
   optimize_basic() calls made many at a time, so that a GPU can take
   them (root_sieve_gpu.c). The searches are the original ones; what
   changes is where some of them start (see the notes below), which
   moves a line's ends by at most the original search tolerance. */

#include "stage2.h"
#include "line_sieve_core.h"

#if defined(_MSC_VER)
__declspec(thread) optimize_basic_batch_hook_t optimize_basic_batch_hook;
__declspec(thread) uint32 root_sieve_batched;
#else
_Thread_local optimize_basic_batch_hook_t optimize_basic_batch_hook;
_Thread_local uint32 root_sieve_batched;
#endif

/*------------------------------------------------------------------------*/
void
optimize_basic_batch(dpoly_t *in, uint32 n, basic_result_t *out)
{
	if (n == 0)
		return;

	if (optimize_basic_batch_hook != NULL) {
		optimize_basic_batch_hook(in, n, out);
		return;
	}

	for (uint32 i = 0; i < n; i++)
		out[i].norm = optimize_basic(in + i, &out[i].skewness,
					&out[i].translation);
}

/*------------------------------------------------------------------------*/
uint32
rs_batch_wanted(void)
{
	static int env = -1;

	if (root_sieve_batched)
		return 1;
	if (env < 0) {
		const char *e = getenv("MSIEVE_ROOTOPT_BATCHED");
		env = (e != NULL && strcmp(e, "1") == 0);
	}
	return (uint32)env;
}

/*------------------------------------------------------------------------*/
/* compute_line_size() (root_sieve_util.c) as a state machine: one per
   end of the line. Each step takes the norm at the point it asked
   for and either asks for the next point or finishes. The sequence of
   points is the original's. */

enum { LS_FIRST, LS_TOWARD, LS_AWAY, LS_HALVE, LS_DONE };

typedef struct {
	line_job_t *job;
	uint32 is_max;		/* which end: line_max, or line_min */
	uint32 phase;
	double offset, d0, d1;
	double pt;		/* the point asked for */
} ls_end_t;

/* the original's bisection, until the interval is small enough */
static void
ls_halve_or_finish(ls_end_t *e)
{
	if (e->d1 - e->d0 > 500 && e->d1 - e->d0 > 1e-6 * fabs(e->d0)) {
		e->pt = (e->d0 + e->d1) / 2;
		e->phase = LS_HALVE;
		return;
	}
	if (e->is_max)
		e->job->line_max = e->d1;
	else
		e->job->line_min = e->d0;
	e->phase = LS_DONE;
}

/* v: the norm at e->pt; max: the line's bound on it */
static void
ls_step(ls_end_t *e, double v)
{
	line_job_t *j = e->job;
	double max = j->max_norm;

	if (!e->is_max) {
		double last = j->last_min;

		switch (e->phase) {
		case LS_FIRST:
			e->offset = MAX(1e-6 * fabs(last), 10.0);
			if (v > max) {
				/* outside: move in until inside */
				e->d0 = last;
				e->d1 = last + e->offset;
				e->phase = LS_TOWARD;
				e->pt = last + e->offset;
			}
			else {
				/* inside: move out until outside */
				e->d0 = last - e->offset;
				e->d1 = last;
				e->phase = LS_AWAY;
				e->pt = last - e->offset;
			}
			return;
		case LS_TOWARD:
			if (v <= max || e->pt >= j->last_max) {
				e->d1 = e->pt;
				ls_halve_or_finish(e);
				return;
			}
			e->d0 = e->pt;
			e->offset *= 2;
			e->pt = last + e->offset;
			return;
		case LS_AWAY:
			if (v > max) {
				e->d0 = e->pt;
				ls_halve_or_finish(e);
				return;
			}
			e->d1 = e->pt;
			e->offset *= 2;
			e->pt = last - e->offset;
			return;
		case LS_HALVE:
			if (v > max)
				e->d0 = e->pt;
			else
				e->d1 = e->pt;
			ls_halve_or_finish(e);
			return;
		}
	}
	else {
		double last = j->last_max;

		switch (e->phase) {
		case LS_FIRST:
			e->offset = MAX(1e-6 * fabs(last), 10.0);
			if (v > max) {
				e->d0 = last - e->offset;
				e->d1 = last;
				e->phase = LS_TOWARD;
				e->pt = last - e->offset;
			}
			else {
				e->d0 = last;
				e->d1 = last + e->offset;
				e->phase = LS_AWAY;
				e->pt = last + e->offset;
			}
			return;
		case LS_TOWARD:
			if (v <= max || e->pt <= j->last_min) {
				e->d0 = e->pt;
				ls_halve_or_finish(e);
				return;
			}
			e->d1 = e->pt;
			e->offset *= 2;
			e->pt = last - e->offset;
			return;
		case LS_AWAY:
			if (v > max) {
				e->d1 = e->pt;
				ls_halve_or_finish(e);
				return;
			}
			e->d0 = e->pt;
			e->offset *= 2;
			e->pt = last + e->offset;
			return;
		case LS_HALVE:
			if (v > max)
				e->d1 = e->pt;
			else
				e->d0 = e->pt;
			ls_halve_or_finish(e);
			return;
		}
	}
}

/* the rotated polynomial at distance dist along the job's direction
   (APPLY_ROTATION in root_sieve_util.c) */
static void
ls_rotate(dpoly_t *b, const line_job_t *j, double dist,
		double dbl_p, double dbl_d)
{
	const dpoly_t *a = &j->apoly;
	double x_dist = floor(dist * j->direction[0]);
	double y_dist = floor(dist * j->direction[1]);
	double z_dist = floor(dist * j->direction[2]);

	*b = *a;
	b->coeff[0] = a->coeff[0] - dbl_d * x_dist;
	b->coeff[1] = a->coeff[1] + dbl_p * x_dist - dbl_d * y_dist;
	b->coeff[2] = a->coeff[2] + dbl_p * y_dist - dbl_d * z_dist;
	b->coeff[3] = a->coeff[3] + dbl_p * z_dist;
}

void
compute_line_sizes(line_job_t *jobs, uint32 n, double dbl_p, double dbl_d)
{
	uint32 num_ends = 2 * n;
	ls_end_t *ends;
	dpoly_t *polys;
	basic_result_t *res;
	uint32 *which;

	if (n == 0)
		return;

	ends = (ls_end_t *)xmalloc(num_ends * sizeof(ls_end_t));
	polys = (dpoly_t *)xmalloc(num_ends * sizeof(dpoly_t));
	res = (basic_result_t *)xmalloc(num_ends * sizeof(basic_result_t));
	which = (uint32 *)xmalloc(num_ends * sizeof(uint32));

	for (uint32 i = 0; i < n; i++) {
		for (uint32 k = 0; k < 2; k++) {
			ls_end_t *e = ends + 2 * i + k;

			e->job = jobs + i;
			e->is_max = k;
			e->phase = LS_FIRST;
			e->pt = k ? jobs[i].last_max : jobs[i].last_min;
		}
	}

	for (;;) {
		uint32 m = 0;

		for (uint32 i = 0; i < num_ends; i++) {
			ls_end_t *e = ends + i;

			if (e->phase == LS_DONE)
				continue;
			ls_rotate(polys + m, e->job, e->pt, dbl_p, dbl_d);
			which[m++] = i;
		}
		if (m == 0)
			break;

		optimize_basic_batch(polys, m, res);

		for (uint32 i = 0; i < m; i++) {
			ls_end_t *e = ends + which[i];

			ls_step(e, res[i].norm);

			/* the original has no limit here; this one is
			   never reached by a sensible polynomial */
			if (e->phase == LS_AWAY && e->offset > 1e30) {
				e->d0 = e->d1 = e->pt;
				ls_halve_or_finish(e);
			}
		}
	}

	free(ends);
	free(polys);
	free(res);
	free(which);
}

/*------------------------------------------------------------------------*/
/* Degree 5: the x-line size of every y-block (sieve_xy_run_deg5). The
   original searches the blocks in order, each starting from the
   previous block's size. Here: every 64th block (and the last) starts
   from the original's first interval, then every 8th and then the
   rest start from the sizes of their nearest searched neighbours,
   interpolated; each round's searches run together. */

#define XL_DEFAULT_MIN -10000.0
#define XL_DEFAULT_MAX 10000.0

/* the same for degree 6's y-lines of the z-blocks (sieve_xyz_run_deg6):
   block i at v = v0 + i * step moves coefficients c and c - 1, the
   search goes along direction dir */
static void
rs_lines_strided(root_sieve_t *rs, uint32 y_blocks, int64 y0,
		uint64 lattice_size, uint32 c, uint32 dir,
		double *line_min_out, double *line_max_out);

void
rs_xlines_deg5(root_sieve_t *rs, uint32 y_blocks, int64 y0,
		uint64 lattice_size)
{
	rs_lines_strided(rs, y_blocks, y0, lattice_size, 2, 0,
			rs->xydata.x_line_min, rs->xydata.x_line_max);
}

void
rs_ylines_deg6(root_sieve_t *rs, uint32 z_blocks, int64 z0,
		uint64 lattice_size)
{
	rs_lines_strided(rs, z_blocks, z0, lattice_size, 3, 1,
			rs->xyzdata.y_line_min, rs->xyzdata.y_line_max);
}

static void
rs_lines_strided(root_sieve_t *rs, uint32 y_blocks, int64 y0,
		uint64 lattice_size, uint32 c, uint32 dir,
		double *line_min_out, double *line_max_out)
{
	static const uint32 strides[] = { 64, 8, 1 };
	uint8 *done;
	line_job_t *jobs;
	uint32 *idx;

	if (y_blocks == 0)
		return;

	done = (uint8 *)xcalloc(y_blocks, 1);
	jobs = (line_job_t *)xmalloc(y_blocks * sizeof(line_job_t));
	idx = (uint32 *)xmalloc(y_blocks * sizeof(uint32));

	for (uint32 r = 0; r < sizeof(strides) / sizeof(strides[0]); r++) {
		uint32 stride = strides[r];
		uint32 n = 0;

		for (uint32 i = 0; i < y_blocks; i++) {
			line_job_t *j;
			double y;

			if (done[i])
				continue;
			if (i % stride != 0 && !(r == 0 && i == y_blocks - 1))
				continue;

			j = jobs + n;
			idx[n++] = i;
			y = (double)(y0 + (int64)i * (int64)lattice_size);

			j->apoly = rs->apoly;
			j->apoly.coeff[c] += rs->dbl_p * y;
			j->apoly.coeff[c - 1] -= rs->dbl_d * y;
			j->direction[0] = (dir == 0);
			j->direction[1] = (dir == 1);
			j->direction[2] = (dir == 2);
			j->max_norm = rs->max_norm;
			j->last_min = XL_DEFAULT_MIN;
			j->last_max = XL_DEFAULT_MAX;

			if (r > 0) {
				/* nearest searched blocks below and above */
				int64 a = (int64)i - 1, b = (int64)i + 1;
				uint32 ea, eb;

				while (a >= 0 && !done[a])
					a--;
				while (b < (int64)y_blocks && !done[b])
					b++;

				ea = (a >= 0 && line_min_out[a] <
						line_max_out[a]);
				eb = (b < (int64)y_blocks &&
						line_min_out[b] <
						line_max_out[b]);

				if (ea && eb) {
					double f = (double)(i - a) / (b - a);

					j->last_min = line_min_out[a] + f *
						(line_min_out[b] -
						 line_min_out[a]);
					j->last_max = line_max_out[a] + f *
						(line_max_out[b] -
						 line_max_out[a]);
				}
				else if (ea) {
					j->last_min = line_min_out[a];
					j->last_max = line_max_out[a];
				}
				else if (eb) {
					j->last_min = line_min_out[b];
					j->last_max = line_max_out[b];
				}
			}
		}

		compute_line_sizes(jobs, n, rs->dbl_p, rs->dbl_d);

		/* mark them after the round: the interpolation above
		   uses the previous rounds only */
		for (uint32 k = 0; k < n; k++) {
			uint32 i = idx[k];
			line_job_t *j = jobs + k;

			if (j->line_min >= j->line_max) {
				line_min_out[i] = 0;
				line_max_out[i] = 0;
			}
			else {
				line_min_out[i] = j->line_min;
				line_max_out[i] = j->line_max;
			}
			done[i] = 1;
		}
	}

	free(done);
	free(jobs);
	free(idx);
}

/*------------------------------------------------------------------------*/
/* candidates from root_sieve_line(), scored together */

#define CAND_MAX 2048	/* scored when this many are waiting */
#define RS_LINES_MAX 128	/* lines sieved together */
#define RS_RUNS_MAX 16		/* runs of them */

typedef struct {
	uint32 n, cap;
	uint32 defer;		/* no scoring until the end of a loop that
				   reads the engine's buffers */
	dpoly_t *apoly;
	basic_result_t *res;
	double *alpha;
	int64 *xs, *ys;		/* the rotations, when they fit */
	uint8 *big;		/* else in x, y */
	mpz_t *x, *y;
	int64 *z;

	/* lines waiting to be sieved */
	uint32 num_lines;
	ls_line_t lines[RS_LINES_MAX];
	uint32 line_score[RS_LINES_MAX];	/* sieve_x_t curr_score */
	ls_out_t line_out[RS_LINES_MAX];

	/* GPU: the runs of the lines waiting, the current run, and the
	   parts of the next line */
	ls_run_t runs[RS_RUNS_MAX];
	uint32 num_runs;
	ls_run_t run;
	uint32 run_ok, run_in_table, run_index, run_lattice_one;
	double run_lattice;
	ls_line_t pend;
	uint32 pend_ok;

	/* the polynomial's sieve for them, and work space */
	ls_poly_t poly;
	ls_prime_t primes[LS_NUM_PRIMES];
	ls_power_t powers[LS_MAX_POWERS];
	ls_root_t *roots;
	uint32 roots_cap;
	uint16 table[LS_MAX_TABLE];
	uint16 block[DEFAULT_BLOCK_SIZE];
} rs_batch_t;

static rs_batch_t *
rs_batch_get(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;

	if (b == NULL) {
		b = (rs_batch_t *)xcalloc(1, sizeof(rs_batch_t));
		rs->batch = b;
	}
	if (b->n == b->cap) {
		uint32 cap = b->defer ? MAX(64, 2 * b->cap) :
				MIN(CAND_MAX, MAX(64, 2 * b->cap));

		b->apoly = (dpoly_t *)xrealloc(b->apoly,
					cap * sizeof(dpoly_t));
		b->res = (basic_result_t *)xrealloc(b->res,
					cap * sizeof(basic_result_t));
		b->alpha = (double *)xrealloc(b->alpha, cap * sizeof(double));
		b->xs = (int64 *)xrealloc(b->xs, cap * sizeof(int64));
		b->ys = (int64 *)xrealloc(b->ys, cap * sizeof(int64));
		b->big = (uint8 *)xrealloc(b->big, cap * sizeof(uint8));
		b->x = (mpz_t *)xrealloc(b->x, cap * sizeof(mpz_t));
		b->y = (mpz_t *)xrealloc(b->y, cap * sizeof(mpz_t));
		b->z = (int64 *)xrealloc(b->z, cap * sizeof(int64));
		for (uint32 i = b->cap; i < cap; i++) {
			mpz_init(b->x[i]);
			mpz_init(b->y[i]);
		}
		b->cap = cap;
	}
	return b;
}

void
rs_batch_free(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;

	if (b == NULL)
		return;
	for (uint32 i = 0; i < b->cap; i++) {
		mpz_clear(b->x[i]);
		mpz_clear(b->y[i]);
	}
	free(b->apoly);
	free(b->res);
	free(b->alpha);
	free(b->xs);
	free(b->ys);
	free(b->big);
	free(b->x);
	free(b->y);
	free(b->z);
	free(b->roots);
	free(b);
	rs->batch = NULL;
}

/* the rotation is rs->curr_x, curr_y, curr_z */
void
rs_cand_add(root_sieve_t *rs, dpoly_t *apoly, double alpha)
{
	rs_batch_t *b = rs_batch_get(rs);
	uint32 i = b->n++;

	b->apoly[i] = *apoly;
	b->alpha[i] = alpha;
	b->big[i] = !(sizeof(long) >= 8 && mpz_fits_slong_p(rs->curr_x) &&
			mpz_fits_slong_p(rs->curr_y));
	if (b->big[i]) {
		mpz_set(b->x[i], rs->curr_x);
		mpz_set(b->y[i], rs->curr_y);
	}
	else {
		b->xs[i] = mpz_get_si(rs->curr_x);
		b->ys[i] = mpz_get_si(rs->curr_y);
	}
	b->z[i] = rs->curr_z;

	if (b->n >= CAND_MAX && !b->defer)
		rs_cand_flush(rs);
}

/* the same, for a rotation the GPU gave: no GMP */
static void
rs_cand_add_si(root_sieve_t *rs, dpoly_t *apoly, double alpha,
		int64 x, int64 y, int64 z)
{
	rs_batch_t *b = rs_batch_get(rs);
	uint32 i = b->n++;

	b->apoly[i] = *apoly;
	b->alpha[i] = alpha;
	b->big[i] = 0;
	b->xs[i] = x;
	b->ys[i] = y;
	b->z[i] = z;

	if (b->n >= CAND_MAX && !b->defer)
		rs_cand_flush(rs);
}

void
rs_cand_flush(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;

	if (b == NULL || b->n == 0)
		return;

	optimize_basic_batch(b->apoly, b->n, b->res);

	for (uint32 i = 0; i < b->n; i++) {
		float score = log(b->res[i].norm) + b->alpha[i];

		if (b->big[i])
			save_rotation(&rs->root_heap, b->x[i], b->y[i],
					b->z[i], score);
		else
			save_rotation_si(&rs->root_heap, b->xs[i], b->ys[i],
					b->z[i], score);
	}
	b->n = 0;
}

/* drop what a stopped run left */
void
rs_cand_reset(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;

	if (b != NULL) {
		b->n = 0;
		b->num_lines = 0;
		b->num_runs = 0;
		b->run_in_table = 0;
		b->run_ok = 0;
		b->pend_ok = 0;
	}
}

/*------------------------------------------------------------------------*/
/* lines from root_sieve_line(), sieved together (see line_sieve_core.h) */

#if defined(_MSC_VER)
__declspec(thread) line_sieve_batch_hook_t line_sieve_batch_hook;
#else
_Thread_local line_sieve_batch_hook_t line_sieve_batch_hook;
#endif

static uint32
rs_get_int64(mpz_t a, int64_t *out)
{
	if (sizeof(long) < 8 || !mpz_fits_slong_p(a))
		return 0;
	*out = (int64_t)mpz_get_si(a);
	return 1;
}

/* a 128-bit two's complement number (low limb first) as an int64, if
   it fits */
static uint32
rs_limbs_to_si(const uint32_t *limbs, int64 *out)
{
	uint32_t ext = (limbs[1] >> 31) ? 0xffffffff : 0;

	for (uint32 i = 2; i < LS_XLIMBS; i++) {
		if (limbs[i] != ext)
			return 0;
	}
	*out = (int64)(((uint64)limbs[1] << 32) | limbs[0]);
	return 1;
}

/* a 128-bit two's complement number (low limb first) */
static void
rs_limbs_to_mpz(mpz_t r, const uint32_t *limbs)
{
	uint32_t a[LS_XLIMBS];
	uint32 neg = limbs[LS_XLIMBS - 1] >> 31;
	uint32 carry = 1;

	for (uint32 i = 0; i < LS_XLIMBS; i++) {
		a[i] = limbs[i];
		if (neg) {
			uint64 w = (uint64)(uint32_t)~a[i] + carry;

			a[i] = (uint32_t)w;
			carry = (uint32)(w >> 32);
		}
	}
	mpz_import(r, LS_XLIMBS, -1, sizeof(uint32_t), 0, 0, a);
	if (neg)
		mpz_neg(r, r);
}

/* the constants of a run of lines whose coordinates the GPU forms */
void
rs_line_run(root_sieve_t *rs, uint32 ok, int64 y_base, uint64 ly,
		uint64 l1, uint64 l2,
		const uint64 *crt0, const uint64 *crt1)
{
	static const uint64 zero2[2] = { 0, 0 }, one2[2] = { 1, 0 };
	rs_batch_t *b;

	/* also without a GPU (MSIEVE_ROOTOPT_BATCHED=1): ls_sieve_cpu()
	   then forms the numbers with GMP, so the CPU and GPU paths sieve
	   the same lines in the same order */
	if (!rs->batched)
		return;
	b = rs_batch_get(rs);
	b->run_ok = 0;
	b->run_in_table = 0;
	if (!ok || l1 == 0 || l2 == 0)
		return;
	if (crt0 == NULL) crt0 = zero2;
	if (crt1 == NULL) crt1 = one2;

	/* words only: the GPU forms the lattice size in CGBN; y and the
	   lines' x fit in LS_XLIMBS words (y < 2^96) */
	memset(&b->run, 0, sizeof(b->run));
	b->run.y_base = y_base;
	b->run.ly = ly;
	b->run.l1 = l1;
	b->run.l2 = l2;
	for (uint32 i = 0; i < 2; i++) {
		b->run.crt0[2 * i] = (uint32_t)crt0[i];
		b->run.crt0[2 * i + 1] = (uint32_t)(crt0[i] >> 32);
		b->run.crt1[2 * i] = (uint32_t)crt1[i];
		b->run.crt1[2 * i + 1] = (uint32_t)(crt1[i] >> 32);
	}
	b->run_ok = 1;
	b->run_lattice = rs_lattice_d(l1, l2);
	b->run_lattice_one = (l1 == 1 && l2 == 1);
}

/* the CRT constants of sieve_x_run_deg5() (its mpz_invert calls), in
   128-bit words: crt0 = (lxy^-1 mod lx) lxy, crt1 = (lx^-1 mod lxy) lx;
   an inverse mod 1 is 0, as mpz_invert gives it */
__extension__ typedef unsigned __int128 rs_u128;

/* a * b as a double, truncated as mpz_get_d() does */
double
rs_lattice_d(uint64 a, uint64 b)
{
	rs_u128 v = (rs_u128)a * b;
	uint32 n = 0;

	if ((v >> 53) == 0)
		return (double)(uint64)v;
	while ((v >> n) >> 53)
		n++;
	return ldexp((double)(uint64)(v >> n), (int)n);
}

/* a + b + c * d (exactly, in 128 bits) as a double, truncated as
   mpz_get_d() does */
double
rs_i128_d(int64 a, uint64 b, uint64 c, uint64 d)
{
	__extension__ __int128 v = (__int128)a + (__int128)b +
				(__int128)((rs_u128)c * d);
	rs_u128 m = (v < 0) ? (rs_u128)(-v) : (rs_u128)v;
	uint32 n = 0;
	double r;

	if ((m >> 53) == 0) {
		r = (double)(uint64)m;
	}
	else {
		while ((m >> n) >> 53)
			n++;
		r = ldexp((double)(uint64)(m >> n), (int)n);
	}
	return (v < 0) ? -r : r;
}

/* the mpz_t numbers of the xy and x lattices, for the CPU code (the
   GPU paths use the words) */
void
rs_xy_mpz(sieve_xy_t *xy)
{
	if (!xy->mp_stale)
		return;
	uint64_2gmp(xy->lattice_size, xy->mp_lattice_size);
	int64_2gmp(xy->y_base_si, xy->y_base);
	xy->mp_stale = 0;
}

void
rs_x_mpz(root_sieve_t *rs)
{
	sieve_x_t *x = &rs->xdata;

	if (!x->mp_stale)
		return;
	uint64_2gmp(rs->xydata.lattice_size, x->mp_lattice_size);
	uint64_2gmp(x->lattice_size, x->tmp1);
	mpz_mul(x->mp_lattice_size, x->mp_lattice_size, x->tmp1);
	x->mp_stale = 0;
}

static uint64
rs_inv64(uint64 a, uint64 m)
{
	__extension__ __int128 t = 0, nt = 1, r = m, nr = a % m, q, tmp;

	if (m == 1)
		return 0;
	while (nr != 0) {
		q = r / nr;
		tmp = t - q * nt; t = nt; nt = tmp;
		tmp = r - q * nr; r = nr; nr = tmp;
	}
	return (uint64)(t < 0 ? t + m : t);
}

void
rs_lattice_crt(uint64 lxy, uint64 lx, uint64 *crt0, uint64 *crt1)
{
	rs_u128 c0 = (rs_u128)rs_inv64(lxy % lx, lx) * lxy;
	rs_u128 c1 = (rs_u128)rs_inv64(lx % lxy, lxy) * lx;

	crt0[0] = (uint64)c0;
	crt0[1] = (uint64)(c0 >> 64);
	crt1[0] = (uint64)c1;
	crt1[1] = (uint64)(c1 >> 64);
}

/* the parts of the next line of the run; rs->xdata.x_blocks set */
uint32
rs_line_parts(root_sieve_t *rs, uint64 ry, uint32 k, double line_min,
		uint64 r1, uint64 rx)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;
	double positions;

	if (b == NULL || !b->run_ok || rs->num_primes != LS_NUM_PRIMES)
		return 0;
	positions = (double)(rs->xdata.x_blocks / DEFAULT_BLOCK_SIZE + 2) *
				DEFAULT_BLOCK_SIZE;
	if (fabs(line_min) + b->run_lattice * (positions + 2) > 0x1p118)
		return 0;

	memset(&b->pend, 0, sizeof(b->pend));
	b->pend.ry = ry;
	b->pend.k = k;
	b->pend.line_min = line_min;
	b->pend.r1 = r1;
	b->pend.rx = rx;
	b->pend_ok = 1;
	return 1;
}

/* the line root_sieve_line() would sieve now: keep it for later.
   Returns 0 if its numbers do not fit (then sieve it as before) */
uint32
rs_line_add(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;
	sieve_x_t *x = &rs->xdata;
	ls_line_t l;

	memset(&l, 0, sizeof(l));
	if (b != NULL && b->pend_ok) {
		/* its parts, for the GPU */
		b->pend_ok = 0;
		if (!b->run_in_table) {
			if (b->num_runs == RS_RUNS_MAX)
				rs_line_flush(rs);
			b->runs[b->num_runs] = b->run;
			b->run_index = b->num_runs++;
			b->run_in_table = 1;
		}
		l = b->pend;
		l.mode = LS_PARTS;
		l.run = b->run_index;
		l.lattice_one = b->run_lattice_one;
		l.z = rs->curr_z;
		l.positions = (x->x_blocks / DEFAULT_BLOCK_SIZE + 1) *
					DEFAULT_BLOCK_SIZE;
		l.curr_score = x->curr_score;
		b->lines[b->num_lines] = l;
		b->line_score[b->num_lines] = x->curr_score;
		if (++b->num_lines == RS_LINES_MAX)
			rs_line_flush(rs);
		return 1;
	}

	rs_x_mpz(rs);
	if (rs->num_primes != LS_NUM_PRIMES ||
	    !rs_get_int64(x->x_base, &l.x_base) ||
	    !rs_get_int64(x->resclass, &l.resclass) ||
	    !rs_get_int64(x->mp_lattice_size, &l.lattice) ||
	    !rs_get_int64(rs->curr_y, &l.y))
		return 0;
	l.z = rs->curr_z;
	l.poly = 0;
	l.mode = LS_INT64;
	l.curr_score = x->curr_score;
	l.positions = (x->x_blocks / DEFAULT_BLOCK_SIZE + 1) *
				DEFAULT_BLOCK_SIZE;

	b = rs_batch_get(rs);
	b->lines[b->num_lines] = l;
	b->line_score[b->num_lines] = x->curr_score;
	if (++b->num_lines == RS_LINES_MAX)
		rs_line_flush(rs);
	return 1;
}

/* the polynomial's primes, powers and roots, in the batch's form */
static void
rs_line_poly(root_sieve_t *rs, rs_batch_t *b)
{
	uint32 num_roots = 0, table = 0, np = 0;

	for (uint32 i = 0; i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		for (uint32 j = 0; j < sp->num_powers; j++)
			num_roots += sp->powers[j].num_roots;
	}
	if (num_roots > b->roots_cap) {
		b->roots_cap = num_roots;
		b->roots = (ls_root_t *)xrealloc(b->roots,
					num_roots * sizeof(ls_root_t));
	}

	num_roots = 0;
	for (uint32 i = 0; i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		b->primes[i].table = table;
		b->primes[i].size = sp->contrib_array_size;
		table += sp->contrib_array_size;

		for (uint32 j = 0; j < sp->num_powers; j++) {
			sieve_power_t *pw = sp->powers + j;
			ls_power_t *lp = b->powers + np++;

			lp->power = pw->power;
			lp->contrib = pw->sieve_contrib;
			lp->prime = i;
			lp->root0 = num_roots;
			lp->num_roots = pw->num_roots;
			for (uint32 k = 0; k < pw->num_roots; k++) {
				ls_root_t *r = b->roots + num_roots++;

				if (!rs->roots_on_host)
					continue;	/* the GPU has them */
				r->resclass = (uint16_t)pw->roots[k].resclass;
				r->step = (uint16_t)pw->roots[k].step;
				r->start = (uint16_t)pw->roots[k].start;
				r->pad = 0;
			}
		}
	}
	b->poly.prime0 = 0;
	b->poly.power0 = 0;
	b->poly.num_powers = np;
	b->poly.table_size = table;
	for (uint32 i = 0; i <= LS_MAX_DEG; i++)
		b->poly.apoly[i] = (i <= rs->apoly.degree) ?
					rs->apoly.coeff[i] : 0;
	b->poly.dbl_p = rs->dbl_p;
	b->poly.dbl_d = rs->dbl_d;
	b->poly.random_root_score = rs->random_root_score;
	b->poly.sieve_bias = rs->sieve_bias;
	b->poly.degree = rs->apoly.degree;
}

/* the line sieve on the CPU; the GPU kernel does the same */

static void
ls_heap_down(uint64_t *h, uint32 n, uint32 i)
{
	for (;;) {
		uint32 c = 2 * i + 1;
		uint64_t t;

		if (c >= n)
			return;
		if (c + 1 < n && h[c + 1] < h[c])
			c++;
		if (h[i] <= h[c])
			return;
		t = h[i]; h[i] = h[c]; h[c] = t;
		i = c;
	}
}

static int
ls_key_cmp(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

	return (x < y) ? 1 : (x > y) ? -1 : 0;	/* largest first */
}

static void
ls_sieve_cpu(rs_batch_t *b, const ls_line_t *l, ls_out_t *out,
		mpz_t *num)
{
	uint16 *tab = b->table, *block = b->block;
	uint32 lattice = (l->mode == LS_PARTS) ? !l->lattice_one :
						(l->lattice != 1);
	uint64_t heap[LS_TOP];
	uint32 n = 0;

	memset(tab, 0, b->poly.table_size * sizeof(uint16));
	for (uint32 j = 0; j < b->poly.num_powers; j++) {
		const ls_power_t *pw = b->powers + j;
		const ls_prime_t *pr = b->primes + pw->prime;
		ls_lp_t lp;

		if (l->mode == LS_PARTS) {
			/* num: x_base, resclass, lattice, y (rs_parts_mpz) */
			uint32 q = pw->power;

			ls_line_power_res(mpz_fdiv_ui(num[0], q),
				mpz_fdiv_ui(num[3], q), ls_fmod(l->z, q),
				mpz_tdiv_ui(num[1], q), mpz_tdiv_ui(num[2], q),
				l->lattice_one, q, &lp);
		}
		else {
			ls_line_power(l, pw->power, &lp);
		}
		if (lattice && lp.skip)
			continue;
		for (uint32 k = 0; k < pw->num_roots; k++) {
			uint32 st, step;

			if (!ls_root_start(b->roots + pw->root0 + k, pw->power,
					lattice, &lp, &st, &step))
				continue;
			for (; st < pr->size; st += step)
				tab[pr->table + st] += (uint16)pw->contrib;
		}
	}

	for (uint32 x0 = 0; x0 < l->positions; x0 += DEFAULT_BLOCK_SIZE) {
		memset(block, 0, sizeof(b->block));
		for (uint32 i = 0; i < LS_NUM_PRIMES; i++) {
			const uint16 *t = tab + b->primes[i].table;
			uint32 size = b->primes[i].size;
			uint32 pos = x0 % size, j = 0;

			while (j < DEFAULT_BLOCK_SIZE) {
				uint32 len = MIN(size - pos,
						DEFAULT_BLOCK_SIZE - j);

				for (uint32 k = 0; k < len; k++)
					block[j + k] += t[pos + k];
				j += len;
				pos = 0;
			}
		}
		for (uint32 j = 0; j < DEFAULT_BLOCK_SIZE; j++) {
			uint64_t key = ((uint64_t)block[j] << 32) |
					(0xffffffffu - (x0 + j));
			if (n < LS_TOP) {
				heap[n++] = key;
				if (n == LS_TOP) {
					for (int32 i = LS_TOP / 2; i >= 0; i--)
						ls_heap_down(heap, n, (uint32)i);
				}
			}
			else if (key > heap[0]) {
				heap[0] = key;
				ls_heap_down(heap, n, 0);
			}
		}
	}

	qsort(heap, n, sizeof(uint64_t), ls_key_cmp);
	out->n = n;
	for (uint32 i = 0; i < n; i++) {
		out->score[i] = (uint16_t)(heap[i] >> 32);
		out->pos[i] = 0xffffffffu - (uint32_t)heap[i];
	}
}

/* sieve the waiting lines, then make their candidates as
   root_sieve_line() does */
/* a parts line's x_base, resclass, lattice and y with GMP, as ls_sieve
   forms them in CGBN (line_sieve_core.h) */
static void
rs_parts_mpz(const ls_run_t *run, const ls_line_t *l, mpz_t *num)
{
	mpz_t c, t;

	mpz_init(c);
	mpz_init(t);

	/* lattice = l1 l2 */
	uint64_2gmp(run->l1, num[2]);
	uint64_2gmp(run->l2, t);
	mpz_mul(num[2], num[2], t);

	/* y = y_base + ly k + ry */
	int64_2gmp(run->y_base, num[3]);
	uint64_2gmp(run->ly, t);
	mpz_addmul_ui(num[3], t, l->k);
	uint64_2gmp(l->ry, t);
	mpz_add(num[3], num[3], t);

	/* resclass = (r1 crt0 + rx crt1) mod lattice */
	mpz_import(c, LS_RUN_LIMBS, -1, sizeof(uint32_t), 0, 0, run->crt0);
	uint64_2gmp(l->r1, t);
	mpz_mul(num[1], c, t);
	mpz_import(c, LS_RUN_LIMBS, -1, sizeof(uint32_t), 0, 0, run->crt1);
	uint64_2gmp(l->rx, t);
	mpz_addmul(num[1], c, t);
	mpz_tdiv_r(num[1], num[1], num[2]);

	/* x_base = trunc(trunc(line_min) / lattice) lattice */
	mpz_set_d(t, l->line_min);
	mpz_tdiv_q(num[0], t, num[2]);
	mpz_mul(num[0], num[0], num[2]);

	mpz_clear(c);
	mpz_clear(t);
}

void
rs_line_flush(root_sieve_t *rs)
{
	rs_batch_t *b = (rs_batch_t *)rs->batch;
	mpz_t lat, num[4];

	if (b == NULL || b->num_lines == 0)
		return;

	rs_line_poly(rs, b);
	if (line_sieve_batch_hook != NULL) {
		/* the GPU makes the candidates and runs their
		   optimize_basic(): only their places come back */
		const ls_cand_t *cand;
		const basic_result_t *norm;
		const so_basic_in_t *cpoly;
		mpz_t x, y;

		line_sieve_batch_hook(&b->poly, b->primes, b->powers,
				b->roots, b->lines, b->num_lines, b->line_out,
				b->runs, b->num_runs, &cand, &norm, &cpoly);
		mpz_init(x);
		mpz_init(y);
		b->defer = 1;
		for (uint32 i = 0; i < b->num_lines; i++) {
			int64 ys = 0;
			uint32 y_ok = rs_limbs_to_si(b->line_out[i].y, &ys);

			if (!y_ok)
				rs_limbs_to_mpz(y, b->line_out[i].y);
			for (uint32 k = 0; k < LS_TOP; k++) {
				const ls_cand_t *c = cand + i * LS_TOP + k;
				int64 xs;

				if (!c->valid)
					continue;
				if (y_ok && rs_limbs_to_si(c->x, &xs)) {
					/* the usual case: no GMP */
					if (norm != NULL) {
						save_rotation_si(&rs->root_heap,
							xs, ys, b->lines[i].z,
							log(norm[i * LS_TOP +
								k].norm) +
							c->alpha);
					}
					else {
						dpoly_t apoly;

						apoly.degree = b->poly.degree;
						for (uint32 j = 0;
						     j <= MAX_POLY_DEGREE; j++) {
							apoly.coeff[j] =
							  (j <= SO_MAX_DEG) ?
							  cpoly[i * LS_TOP +
							     k].apoly[j] : 0;
						}
						rs_cand_add_si(rs, &apoly,
							c->alpha, xs, ys,
							b->lines[i].z);
					}
					continue;
				}
				if (y_ok)
					int64_2gmp(ys, y);
				rs_limbs_to_mpz(x, c->x);
				if (norm != NULL) {
					/* optimize_basic() done there */
					save_rotation(&rs->root_heap, x, y,
						b->lines[i].z,
						log(norm[i * LS_TOP + k].norm) +
							c->alpha);
				}
				else {
					/* scored with the others' batches,
					   which the CPU can share */
					dpoly_t apoly;

					apoly.degree = b->poly.degree;
					for (uint32 j = 0; j <= MAX_POLY_DEGREE;
					     j++) {
						apoly.coeff[j] = (j <= SO_MAX_DEG) ?
						  cpoly[i * LS_TOP + k].apoly[j] :
						  0;
					}
					mpz_set(rs->curr_x, x);
					mpz_set(rs->curr_y, y);
					rs->curr_z = b->lines[i].z;
					rs_cand_add(rs, &apoly, c->alpha);
				}
			}
		}
		mpz_clear(x);
		mpz_clear(y);
		b->defer = 0;
		if (b->n >= CAND_MAX)
			rs_cand_flush(rs);
		b->num_lines = 0;
		b->num_runs = 0;
		b->run_in_table = 0;
		return;
	}

	mpz_init(lat);
	for (uint32 i = 0; i < 4; i++)
		mpz_init(num[i]);
	for (uint32 i = 0; i < b->num_lines; i++) {
		const ls_line_t *l = b->lines + i;
		const ls_out_t *o = b->line_out + i;

		/* the line's numbers: from its parts as ls_sieve forms
		   them in CGBN, or its words */
		if (l->mode == LS_PARTS) {
			rs_parts_mpz(b->runs + l->run, l, num);
		}
		else {
			int64_2gmp(l->x_base, num[0]);
			int64_2gmp(l->resclass, num[1]);
			int64_2gmp(l->lattice, num[2]);
			int64_2gmp(l->y, num[3]);
		}
		ls_sieve_cpu(b, l, b->line_out + i, num);

		mpz_set(lat, num[2]);
		mpz_set(rs->curr_y, num[3]);
		rs->curr_z = l->z;

		for (uint32 k = 0; k < o->n; k++) {
			double alpha = rs->random_root_score -
					(rs->sieve_bias +
					(double)(o->score[k] + b->line_score[i]) /
							LOG_SCALE_FACTOR);
			dpoly_t apoly = rs->apoly;

			if (alpha > ROOT_SCORE_COARSE_MIN)
				continue;

			mpz_add(rs->curr_x, num[0], num[1]);
			mpz_addmul_ui(rs->curr_x, lat, o->pos[k]);

			apoly.coeff[3] += rs->dbl_p * (double)(rs->curr_z);
			apoly.coeff[2] -= rs->dbl_d * (double)(rs->curr_z);

			apoly.coeff[2] += rs->dbl_p * mpz_get_d(rs->curr_y);
			apoly.coeff[1] -= rs->dbl_d * mpz_get_d(rs->curr_y);

			apoly.coeff[1] += rs->dbl_p * mpz_get_d(rs->curr_x);
			apoly.coeff[0] -= rs->dbl_d * mpz_get_d(rs->curr_x);

			rs_cand_add(rs, &apoly, alpha);
		}
	}
	mpz_clear(lat);
	for (uint32 i = 0; i < 4; i++)
		mpz_clear(num[i]);
	b->num_lines = 0;
	b->num_runs = 0;
	b->run_in_table = 0;
}

/*------------------------------------------------------------------------*/
/* init_sieve() on the GPU: the roots of every prime power (in CGBN
   there, alpha_core.h), stored where init_sieve() stores them, and the
   sieve bias added up as init_sieve_core() adds it. 0 if there is no
   GPU or the numbers do not fit; then init_sieve() does it */

#if defined(_MSC_VER)
__declspec(thread) rs_setup_hook_t rs_setup_hook;
#else
_Thread_local rs_setup_hook_t rs_setup_hook;
#endif

uint32
rs_setup_gpu(curr_poly_t *c, root_sieve_t *rs, uint32 deg, double sieve_bias)
{
	ra_poly_t *pl;
	ls_root_t *roots;
	rs_setup_out_t out;
	uint32 bits, need = 0, k = 0, off = 0;

	if (rs_setup_hook == NULL || deg > RA_MAX_DEG)
		return 0;
	for (uint32 i = 0; i <= deg; i++)
		need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_a[i], 2));
	need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_p, 2));
	need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_d, 2));
	bits = (need + 8 <= 512) ? 512 : (need + 8 <= 1024) ? 1024 : 0;
	if (bits == 0)
		return 0;

	pl = (ra_poly_t *)xcalloc(1, sizeof(ra_poly_t));
	roots = (ls_root_t *)xmalloc(RS_MAX_ROOTS * sizeof(ls_root_t));
	pl->degree = deg;
	pl->pad = bits;
	for (uint32 i = 0; i <= deg; i++)
		ra_limbs(pl->a[i], c->gmp_a[i], bits);
	ra_limbs(pl->p, c->gmp_p, bits);
	ra_limbs(pl->d, c->gmp_d, bits);

#if defined(RS_GPU_LATTICE)
	/* the lattice choice is on the GPU too: only the CPU's fallbacks
	   want the roots here (rs_roots_need), and degree 6 */
	pl->pad2[0] = (deg == 4 || deg == 5) && lattice_x_hook != NULL;
#endif
	rs_setup_hook(pl, roots, &out);
	free(pl);

	if (out.total > RS_MAX_ROOTS) {
		free(roots);
		return 0;
	}
	for (uint32 i = 0; i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		for (uint32 j = 0; j < sp->num_powers; j++, k++) {
			sieve_power_t *pw = sp->powers + j;
			double contrib = 0;

			pw->num_roots = out.num_roots[k];
			for (uint32 r = 0; !out.pad &&
					r < pw->num_roots; r++) {
				const ls_root_t *g = roots + off + r;

				pw->roots[r].resclass = g->resclass;
				pw->roots[r].step = g->step;
				pw->roots[r].start = g->start;
			}
			off += pw->num_roots;
			for (uint32 r = 0; r < out.proj[k]; r++)
				contrib += pw->root_contrib;
			sieve_bias += contrib;
		}
	}
	free(roots);
	rs->sieve_bias = sieve_bias;
	rs->roots_on_host = !out.pad;	/* pad: the roots stayed there */
	return 1;
}

/*------------------------------------------------------------------------*/
/* the lattice choice on the GPU (lattice_core.h) */

#if defined(_MSC_VER)
__declspec(thread) lattice_xy_hook_t lattice_xy_hook;
__declspec(thread) lattice_x_hook_t lattice_x_hook;
__declspec(thread) roots_fetch_hook_t roots_fetch_hook;
#else
_Thread_local lattice_xy_hook_t lattice_xy_hook;
_Thread_local lattice_x_hook_t lattice_x_hook;
_Thread_local roots_fetch_hook_t roots_fetch_hook;
#endif

/* where a power's roots start in the polynomial's root list (the
   setup's layout: by prime, then by power) */
static uint32
rs_root_offset(root_sieve_t *rs, uint32 power)
{
	uint32 off = 0;

	for (uint32 i = 0; i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		for (uint32 j = 0; j < sp->num_powers; j++) {
			if (sp->powers[j].power == power)
				return off;
			off += sp->powers[j].num_roots;
		}
	}
	return 0xffffffffu;
}

/* the lattice primes and their powers, as the GPU takes them; x: with
   the xy lattice size modulo each power */
static uint32
rs_lattice_set(root_sieve_t *rs, sieve_prime_t *lp, uint32 n,
		uint64 xy_lattice_size, lt_set_t *set)
{
	uint32 q = 0;

	if (n > LT_MAX_PRIMES)
		return 0;
	memset(set, 0, sizeof(*set));
	set->num_primes = n;
	for (uint32 j = 0; j < n; j++) {
		set->power0[j] = q;
		set->num_powers[j] = lp[j].num_powers;
		for (uint32 k = 0; k < lp[j].num_powers; k++, q++) {
			sieve_power_t *pw = lp[j].powers + k;
			lt_power_t *w = set->powers + q;

			if (q == LT_MAX_POWERS)
				return 0;
			w->power = pw->power;
			w->contrib = pw->sieve_contrib;
			w->root0 = rs_root_offset(rs, pw->power);
			w->num_roots = pw->num_roots;
			w->latmod = (uint32)(xy_lattice_size % pw->power);
			if (w->root0 == 0xffffffffu)
				return 0;
		}
	}
	return 1;
}

uint32
rs_lattice_xy(root_sieve_t *rs, sieve_prime_t *lattice_primes,
		uint32 num_lattice_primes, hit_t *hitlist)
{
	lt_set_t set;
	lt_hit_t hits[LT_MAX_PRIMES];

	if (lattice_xy_hook == NULL || rs->roots_on_host ||
	    !rs_lattice_set(rs, lattice_primes, num_lattice_primes, 1, &set))
		return 0;

	lattice_xy_hook(&set, hits);
	for (uint32 j = 0; j < num_lattice_primes; j++) {
		if (hits[j].num_roots > LT_MAX_ROOTS)
			return 0;
	}
	for (uint32 j = 0; j < num_lattice_primes; j++) {
		hit_t *h = hitlist + j;

		h->power = (uint8)hits[j].power;
		h->num_roots = (uint8)hits[j].num_roots;
		for (uint32 k = 0; k < hits[j].num_roots; k++) {
			h->score[k] = hits[j].score[k];
			h->roots[k][0] = hits[j].roots[k][0];
			h->roots[k][1] = hits[j].roots[k][1];
			h->roots[k][2] = 0;
		}
	}
	return 1;
}

uint32
rs_lattice_x(root_sieve_t *rs, sieve_prime_t *lattice_primes,
		uint32 num_lattice_primes, uint32 y_ok, int64 y_base,
		uint64 xy_lattice_size, const lattice_t *lattices,
		uint32 num_lattices, uint32 y_blocks,
		uint64 x_lattice_size, lt_xout_t *out)
{
	lt_xreq_t req;
	lt_lat_t *lats;

	memset(&req, 0, sizeof(req));
	if (lattice_x_hook == NULL || rs->roots_on_host || !y_ok ||
	    num_lattices == 0 || num_lattices > LT_MAX_LATS || y_blocks == 0 ||
	    !rs_lattice_set(rs, lattice_primes, num_lattice_primes,
			xy_lattice_size, &req.set))
		return 0;
	req.y_base = y_base;
	for (uint32 j = 0; j < num_lattice_primes; j++) {
		const lt_power_t *pw = req.set.powers + req.set.power0[j];

		if (pw[req.set.num_powers[j] - 1].power > LT_MAX_SIEVE)
			return 0;
	}

	req.x_lattice = x_lattice_size;
	req.y_blocks = y_blocks;
	req.num_lattices = num_lattices;
	req.lat0 = 0;
	req.pad = 0;
	lats = (lt_lat_t *)xmalloc(num_lattices * sizeof(lt_lat_t));
	for (uint32 i = 0; i < num_lattices; i++) {
		lats[i].y = lattices[i].y;
		lats[i].score = lattices[i].score;
		lats[i].pad = 0;
	}
	lattice_x_hook(&req, lats, out);
	free(lats);
	return 1;
}

/* the roots on the host, if only the GPU has them */
void
rs_roots_need(root_sieve_t *rs)
{
	ls_root_t *roots;
	uint32 total = 0, off = 0;

	if (rs->roots_on_host || roots_fetch_hook == NULL)
		return;
	for (uint32 i = 0; i < rs->num_primes; i++) {
		for (uint32 j = 0; j < rs->primes[i].num_powers; j++)
			total += rs->primes[i].powers[j].num_roots;
	}
	roots = (ls_root_t *)xmalloc(MAX(total, 1) * sizeof(ls_root_t));
	roots_fetch_hook(roots, total);
	for (uint32 i = 0; i < rs->num_primes; i++) {
		sieve_prime_t *sp = rs->primes + i;

		for (uint32 j = 0; j < sp->num_powers; j++) {
			sieve_power_t *pw = sp->powers + j;

			for (uint32 r = 0; r < pw->num_roots; r++, off++) {
				pw->roots[r].resclass = roots[off].resclass;
				pw->roots[r].step = roots[off].step;
				pw->roots[r].start = roots[off].start;
			}
		}
	}
	free(roots);
	rs->roots_on_host = 1;
}
