/*--------------------------------------------------------------------
MIT License

Copyright (c) 2026 Stefan Teleman

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
--------------------------------------------------------------------*/

/* GPU kernels of the NFS line siever (sieve_line_gpu.c). They compute
   exactly what msieve's line siever (sieve_line.c) computes: the same
   byte sieve values (the sums of logarithms, modulo 256), the same
   cutoffs (from the exact bit sizes of the norms at the same sample
   points) and so the same candidates, plus the factor base primes of
   1000 and up that divide each candidate (its resieving). A line b has
   ltot positions (whole 65536-position blocks); odd b: position i is
   a = min_a + i, even b: a = min_a + 2 i + 1 (odd a only). Lines are
   done in batches, each in chunks of LS_CHUNK positions; a block of
   the segment kernel takes LS_SEG positions of one line. */

#include "sieve_core.h"

typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef long long int64;

/*------------------------------------------------------------------*/
/* where entry (p, r) first hits line b, or LS_SKIP: init_one_sieve()
   of sieve_line.c. rem0 = p - min_a % p (reduced) */

__device__ static uint32
ls_start(uint32 p, uint32 r, uint32 rem0, uint32 orig_p, uint64 b,
	 uint32 power)
{
	uint32 bp = (uint32)(b % p);
	uint32 br = (uint32)((uint64)r * bp % p);
	uint64 rem;

	if (power) {
		if (br == 0 && b % orig_p == 0)
			return LS_SKIP;
	}
	else if (br == 0 && (r == p || bp == 0)) {
		return LS_SKIP;
	}
	rem = (uint64)rem0 + br;
	if (rem >= p)
		rem -= p;
	if ((b & 1) == 0) {
		if ((rem & 1) == 0)
			rem += p;
		rem /= 2;
	}
	return (uint32)rem;
}

/* the start of every entry (and prime power) of one side on every
   line of the batch: start[line * n + i] */
extern "C" __global__ void
ls_starts(const ls_side_t *sd, const uint64 *lines, uint32 num_lines,
	  uint32 *start)
{
	uint32 n = sd->num_entries + sd->num_powers;

	for (uint32 t = blockIdx.x * blockDim.x + threadIdx.x;
	     t < n * num_lines; t += gridDim.x * blockDim.x) {
		uint32 l = t / n, i = t % n;
		uint64 b = lines[l];

		if (i < sd->num_entries)
			start[t] = ls_start(sd->p[i], sd->r[i], sd->rem0[i],
					0, b, 0);
		else {
			uint32 k = i - sd->num_entries;

			start[t] = ls_start(sd->pw_p[k], sd->pw_r[k],
					sd->pw_rem0[k], sd->pw_orig[k], b, 1);
		}
	}
}

/*------------------------------------------------------------------*/
/* the large entries (p > LS_SEG) of one side: their hits in chunk
   [c0, c0 + LS_CHUNK) of each line, into per-segment buckets of
   (p << 24 | logp << 16 | offset) */
extern "C" __global__ void
ls_bucket(const ls_side_t *sd, const ls_batch_t *bt, const uint32 *start,
	  uint64 c0, uint64 *bucket, uint32 *count, uint32 *overflow)
{
	uint32 n = sd->num_entries + sd->num_powers;
	uint32 nl = sd->num_entries - sd->num_direct;

	for (uint32 t = blockIdx.x * blockDim.x + threadIdx.x;
	     t < nl * bt->num_lines; t += gridDim.x * blockDim.x) {
		uint32 l = t / nl, i = sd->num_direct + t % nl;
		uint32 st = start[(uint64)l * n + i];
		uint64 p = sd->p[i], ltot = bt->ltot[l];
		uint64 end = c0 + LS_CHUNK, pos;

		if (st == LS_SKIP)
			continue;
		if (end > ltot)
			end = ltot;
		pos = st;
		if (pos < c0)
			pos += (c0 - pos + p - 1) / p * p;
		for (; pos < end; pos += p) {
			uint32 off = (uint32)(pos - c0);
			uint32 seg = off / LS_SEG;
			uint32 q = l * LS_SEGS + seg;
			uint32 slot = atomicAdd(count + q, 1);

			if (slot < LS_BUCKET_CAP)
				bucket[(uint64)q * LS_BUCKET_CAP + slot] =
					(p << 24) |
					((uint64)sd->logp[i] << 16) |
					(off % LS_SEG);
			else
				*overflow = 1;
		}
	}
}

/*------------------------------------------------------------------*/
/* the exact bit size of |f(a, b)| (mpz_sizeinbase, 1 for 0), f with
   coefficients of LS_POLY_WORDS 32-bit words (two's complement):
   Horner's rule in LS_EVAL_WORDS words */

#define EW LS_EVAL_WORDS

__device__ static void
ev_mul_small(uint32 *x, uint64 m)	/* x *= m, m < 2^63, x signed */
{
	uint32 neg = x[EW - 1] >> 31, carry = 1;
	uint64 c = 0;

	if (neg) {		/* |x| */
		for (int i = 0; i < EW; i++) {
			uint64 w = (uint64)(uint32)~x[i] + carry;
			x[i] = (uint32)w;
			carry = (uint32)(w >> 32);
		}
	}
	uint32 m0 = (uint32)m, m1 = (uint32)(m >> 32);
	uint32 y[EW];
	for (int i = 0; i < EW; i++)
		y[i] = 0;
	for (int i = 0; i < EW; i++) {		/* y = |x| * m0 */
		uint64 w = (uint64)x[i] * m0 + c;
		y[i] = (uint32)w;
		c = w >> 32;
	}
	c = 0;
	for (int i = 0; i + 1 < EW; i++) {	/* y += |x| * m1 << 32 */
		uint64 w = (uint64)x[i] * m1 + y[i + 1] + c;
		y[i + 1] = (uint32)w;
		c = w >> 32;
	}
	carry = 1;
	for (int i = 0; i < EW; i++) {
		if (neg) {
			uint64 w = (uint64)(uint32)~y[i] + carry;
			x[i] = (uint32)w;
			carry = (uint32)(w >> 32);
		}
		else {
			x[i] = y[i];
		}
	}
}

__device__ static void
ev_add(uint32 *x, const uint32 *y)
{
	uint64 c = 0;

	for (int i = 0; i < EW; i++) {
		uint64 w = (uint64)x[i] + y[i] + c;
		x[i] = (uint32)w;
		c = w >> 32;
	}
}

/* f(a, b) = sum c_i a^i b^(d-i), as Horner in a with the b powers */
__device__ static uint32
ls_bits(const ls_poly_t *f, int64 a, uint64 b)
{
	uint32 acc[EW], t[EW];
	int d = (int)f->degree;
	uint64 ua = (a < 0) ? (uint64)(-a) : (uint64)a;

	for (int i = 0; i < EW; i++)
		acc[i] = (i < LS_POLY_WORDS) ? f->c[d][i] :
				((f->c[d][LS_POLY_WORDS - 1] >> 31) ?
					0xffffffff : 0);
	for (int k = d - 1; k >= 0; k--) {
		ev_mul_small(acc, ua);
		if (a < 0) {			/* acc = -acc */
			uint32 carry = 1;
			for (int i = 0; i < EW; i++) {
				uint64 w = (uint64)(uint32)~acc[i] + carry;
				acc[i] = (uint32)w;
				carry = (uint32)(w >> 32);
			}
		}
		for (int i = 0; i < EW; i++)
			t[i] = (i < LS_POLY_WORDS) ? f->c[k][i] :
				((f->c[k][LS_POLY_WORDS - 1] >> 31) ?
					0xffffffff : 0);
		for (int j = 0; j < d - k; j++)
			ev_mul_small(t, b);
		ev_add(acc, t);
	}
	if (acc[EW - 1] >> 31) {		/* |acc| */
		uint32 carry = 1;
		for (int i = 0; i < EW; i++) {
			uint64 w = (uint64)(uint32)~acc[i] + carry;
			acc[i] = (uint32)w;
			carry = (uint32)(w >> 32);
		}
	}
	for (int i = EW - 1; i >= 0; i--) {
		if (acc[i])
			return 32 * i + 32 - __clz(acc[i]);
	}
	return 1;
}

__device__ static uint64
ls_gcd64(uint64 u, uint64 v)
{
	if (u == 0)
		return v;
	if (v == 0)
		return u;
	int sh = __ffsll(u | v) - 1;
	u >>= __ffsll(u) - 1;
	do {
		v >>= __ffsll(v) - 1;
		if (u > v) {
			uint64 t = u; u = v; v = t;
		}
		v -= u;
	} while (v);
	return u << sh;
}

/*------------------------------------------------------------------*/
/* one LS_SEG-position segment of one line: both sieves (rational in
   the high 16 bits, algebraic in the low 16; the low byte of each is
   msieve's byte), the cutoffs, the candidates, and their resieving */

__device__ static void
seg_add(uint32 *sv, uint32 off, uint32 v)
{
	atomicAdd(sv + off, v);
}

/* sieve the direct entries of one side with index in [i0, i1) and
   p >= LS_TINY (entry-parallel); with resieve set, record hits on
   candidates instead (sv holds their global index + 1) */
__device__ static void
seg_direct(const ls_side_t *sd, const uint32 *st_line, uint32 i0,
	   uint32 i1, uint64 seg_base, uint32 *sv, uint32 shift,
	   uint32 resieve, ls_cand_t *cand, uint32 side)
{
	for (uint32 i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
		uint32 st = st_line[i];
		uint32 p = sd->p[i];
		uint64 first;

		if (st == LS_SKIP || p < LS_TINY)
			continue;
		first = st;
		if (first < seg_base)
			first += (seg_base - first + p - 1) / p * p;
		for (uint32 off = (uint32)(first - seg_base); off < LS_SEG;
		     off += p) {
			if (!resieve) {
				seg_add(sv, off, (uint32)sd->logp[i] << shift);
			}
			else if (sv[off]) {
				ls_cand_t *c = cand + (sv[off] - 1);
				uint32 k = atomicAdd(&c->nf[side], 1);

				if (k < LS_CAND_FACTORS)
					c->f[side][k] = p;
			}
		}
	}
}

/* the tiny entries (p < LS_TINY) and the prime powers of one side,
   position-parallel: each thread its LS_SEG / blockDim positions */
__device__ static void
seg_tiny(const ls_side_t *sd, const uint32 *st_line, uint64 seg_base,
	 uint32 *sv, uint32 shift)
{
	uint32 per = LS_SEG / blockDim.x;
	uint64 lo = seg_base + threadIdx.x * per, hi = lo + per;
	uint32 n = sd->num_entries + sd->num_powers;

	for (uint32 i = 0; i < n; i++) {
		uint32 p, lp, st;
		uint64 pos;

		if (i < sd->num_entries) {
			p = sd->p[i];
			if (p >= LS_TINY) {
				i = sd->num_entries - 1;	/* to the powers */
				continue;
			}
			lp = sd->logp[i];
		}
		else {
			p = sd->pw_p[i - sd->num_entries];
			lp = sd->pw_logp[i - sd->num_entries];
		}
		st = st_line[i];
		if (st == LS_SKIP)
			continue;
		pos = st;
		if (pos < lo)
			pos += (lo - pos + p - 1) / p * p;
		for (; pos < hi; pos += p)
			sv[pos - seg_base] += lp << shift;
	}
}

extern "C" __global__ void __launch_bounds__(LS_THREADS)
ls_segment(const ls_side_t *rat, const ls_side_t *alg, const ls_batch_t *bt,
	   uint32 line0, const uint32 *start_r, const uint32 *start_a, uint64 c0,
	   const uint64 *bucket_r, const uint32 *count_r,
	   const uint64 *bucket_a, const uint32 *count_a,
	   ls_cand_t *cand, uint32 *num_cand, uint32 *overflow)
{
	extern __shared__ uint32 sv[];			/* LS_SEG */
	__shared__ int32_t cut_a[LS_SEG / 256], cut_r[LS_SEG / 256];
	__shared__ uint32 nlp[LS_SEG / 256];
	__shared__ uint32 s_count, s_base;

	const uint32 l = line0 + blockIdx.x / LS_SEGS, sg = blockIdx.x % LS_SEGS;
	const uint64 seg_base = c0 + (uint64)sg * LS_SEG;
	const uint64 b = bt->lines[l];
	const uint32 even = ((b & 1) == 0);
	const uint32 nr = rat->num_entries + rat->num_powers;
	const uint32 na = alg->num_entries + alg->num_powers;
	const uint32 *st_r = start_r + (uint64)l * nr;
	const uint32 *st_a = start_a + (uint64)l * na;
	const uint32 q = l * LS_SEGS + sg;

	if (seg_base >= bt->ltot[l])
		return;

	/* the projective biases */
	for (uint32 i = threadIdx.x; i < LS_SEG; i += blockDim.x)
		sv[i] = (uint32)bt->proj_a[l] | ((uint32)bt->proj_r[l] << 16);
	if (threadIdx.x == 0)
		s_count = 0;
	__syncthreads();

	seg_tiny(alg, st_a, seg_base, sv, 0);
	seg_tiny(rat, st_r, seg_base, sv, 16);
	__syncthreads();
	seg_direct(alg, st_a, 0, alg->num_direct, seg_base, sv, 0,
			0, NULL, 1);
	seg_direct(rat, st_r, 0, rat->num_direct, seg_base, sv, 16,
			0, NULL, 0);
	{
		uint32 n = min(count_a[q], (uint32)LS_BUCKET_CAP);
		const uint64 *u = bucket_a + (uint64)q * LS_BUCKET_CAP;

		for (uint32 i = threadIdx.x; i < n; i += blockDim.x)
			seg_add(sv, (uint32)(u[i] & 0xffff),
				(uint32)((u[i] >> 16) & 0xff));
		n = min(count_r[q], (uint32)LS_BUCKET_CAP);
		u = bucket_r + (uint64)q * LS_BUCKET_CAP;
		for (uint32 i = threadIdx.x; i < n; i += blockDim.x)
			seg_add(sv, (uint32)(u[i] & 0xffff),
				(uint32)((u[i] >> 16) & 0xff) << 16);
	}

	/* the cutoffs of the segment's windows (do_factoring) */
	const uint32 win = even ? 256 : 512;	/* positions per window */
	const uint32 nwin = LS_SEG / win;
	const uint64 step = even ? 2 * LS_BLOCK : LS_BLOCK;
	const uint64 blk = seg_base / LS_BLOCK;
	const int64 block_start = bt->min_a + (int64)(blk * step);
	const uint32 k0 = (uint32)(seg_base % LS_BLOCK) / win;

	if (threadIdx.x < nwin) {
		uint32 k = k0 + threadIdx.x;
		uint32 rb = ls_bits(&bt->poly_r, block_start + (int64)step, b);
		uint32 rl = ls_bits(&bt->poly_r, block_start, b);
		uint32 al = ls_bits(&bt->poly_a, block_start + (int64)k * 512, b);
		uint32 ab = ls_bits(&bt->poly_a,
				block_start + (int64)(k + 1) * 512, b);
		int32_t cr = (bt->tab_r[min(rl, (uint32)LS_TAB - 1)] +
			      bt->tab_r[min(rb, (uint32)LS_TAB - 1)]) / 2;
		int32_t ca = (bt->tab_a[min(al, (uint32)LS_TAB - 1)] +
			      bt->tab_a[min(ab, (uint32)LS_TAB - 1)]) / 2;
		uint32 lr = 2, la = 2;

		if (ab > 150) {
			la = 3;
			ca -= bt->sc3_a;
			if (ab > 185) {
				lr = 3;
				cr -= bt->sc3_r;
			}
			else {
				cr -= bt->sc2_r;
			}
		}
		else if (rb > 150) {
			lr = 3;
			cr -= bt->sc3_r;
			if (rb > 185) {
				la = 3;
				ca -= bt->sc3_a;
			}
			else {
				ca -= bt->sc2_a;
			}
		}
		else {
			cr -= bt->sc2_r;
			ca -= bt->sc2_a;
		}
		cut_r[threadIdx.x] = max(cr, 0);
		cut_a[threadIdx.x] = max(ca, 0);
		nlp[threadIdx.x] = lr | (la << 4);
	}
	__syncthreads();

	/* the candidates: each thread LS_SEG / blockDim consecutive
	   positions, flagged in a register */
	const uint32 per = LS_SEG / blockDim.x;
	const uint32 my0 = threadIdx.x * per;
	uint32 mask = 0, mine, local0;

	for (uint32 j = 0; j < per; j++) {
		uint32 off = my0 + j, w = off / win, v = sv[off];

		if ((int32_t)(v & 0xff) <= cut_a[w] ||
		    (int32_t)((v >> 16) & 0xff) <= cut_r[w])
			continue;
		{
			uint64 pos = seg_base + off;
			int64 a = even ? bt->min_a + 2 * (int64)pos + 1 :
					 bt->min_a + (int64)pos;
			int64 g = a % (int64)b;

			if (g < 0)
				g += (int64)b;
			if (ls_gcd64((uint64)g, b) != 1)
				continue;
		}
		mask |= 1u << j;
	}
	mine = __popc(mask);
	local0 = mine ? atomicAdd(&s_count, mine) : 0;
	__syncthreads();
	if (threadIdx.x == 0) {
		s_base = atomicAdd(num_cand, s_count);
		if (s_base + s_count > bt->max_cands)
			*overflow = 3;	/* the host redoes it line by line */
	}
	__syncthreads();

	/* the candidates' (global index + 1), in the sieve array */
	for (uint32 j = 0, k = local0; j < per; j++) {
		uint32 off = my0 + j;

		if (((mask >> j) & 1) && s_base + k < bt->max_cands) {
			uint64 pos = seg_base + off;
			uint32 g = s_base + k++;
			ls_cand_t *c = cand + g;

			c->line = l;
			c->pos = (uint32)pos;
			c->nlp = nlp[off / win];
			c->nf[0] = c->nf[1] = 0;
			sv[off] = g + 1;
		}
		else {
			sv[off] = 0;
		}
	}
	__syncthreads();
	if (s_count == 0)
		return;

	/* resieving: the primes of 1000 and up that hit a candidate */
	seg_direct(alg, st_a, alg->small_size, alg->num_direct, seg_base,
			sv, 0, 1, cand, 1);
	seg_direct(rat, st_r, rat->small_size, rat->num_direct, seg_base,
			sv, 16, 1, cand, 0);
	for (uint32 side = 0; side < 2; side++) {
		uint32 n = min(side ? count_a[q] : count_r[q],
				(uint32)LS_BUCKET_CAP);
		const uint64 *u = (side ? bucket_a : bucket_r) +
				(uint64)q * LS_BUCKET_CAP;

		for (uint32 i = threadIdx.x; i < n; i += blockDim.x) {
			uint32 off = (uint32)(u[i] & 0xffff);

			if (sv[off]) {
				ls_cand_t *c = cand + (sv[off] - 1);
				uint32 k = atomicAdd(&c->nf[side], 1);

				if (k < LS_CAND_FACTORS)
					c->f[side][k] = (uint32)(u[i] >> 24);
			}
		}
	}
}
