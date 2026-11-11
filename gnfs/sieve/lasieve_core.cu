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

/* GPU kernels of the NFS lattice siever (sieve_lattice_gpu.c). For a
   batch of special-q lattices (lasieve_core.h): la_roots finds each
   factor base entry's root rho in every lattice; la_large adds the
   entries with p >= I to the sieve in global memory, walking their
   hits with Franke and Kleinjung's reduced basis (one hit per step),
   and with resieve set reports the hits on candidates instead;
   la_tile takes a tile of rows into shared memory, adds the entries
   with p < I row by row, picks the candidates against thresholds from
   the norms' sizes, and resieves its primes >= LA_RESIEVE_MIN for
   them. */

#include <cgbn/cgbn.h>
#include "lasieve_core.h"

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned long long uint64;
typedef long long int64;


extern "C" __device__ const unsigned int la_abi[2] =
		{ LA_ABI, (unsigned int)sizeof(la_batch_t) };	/* positions per tile: rows = this >> logI */

__device__ static uint32
la_mod(int64 v, uint32 p)
{
	int64 m = v % (int64)p;

	return (uint32)(m < 0 ? m + p : m);
}

/* u^-1 mod p, 0 < u < p < 2^31: 32-bit divisions (the 64-bit ones
   are emulated) */
__device__ static uint32
la_inv32(uint32 u, uint32 p)
{
	int32_t t0 = 0, t1 = 1;
	uint32 r0 = p, r1 = u;

	while (r1) {
		uint32 q = r0 / r1, t;
		int32_t tt;

		t = r0 - q * r1; r0 = r1; r1 = t;
		tt = t0 - (int32_t)q * t1; t0 = t1; t1 = tt;
	}
	return (uint32)(t0 < 0 ? t0 + (int32_t)p : t0);
}

/* u^-1 mod p, 0 < u < p */
__device__ static uint32
la_inv(uint32 u, uint32 p)
{
	int64 t0 = 0, t1 = 1, r0 = p, r1 = u;

	while (r1) {
		int64 q = r0 / r1, t;

		t = r0 - q * r1; r0 = r1; r1 = t;
		t = t0 - q * t1; t0 = t1; t1 = t;
	}
	return (uint32)(t0 < 0 ? t0 + p : t0);
}

__device__ static uint64
la_gcd(uint64 u, uint64 v)
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

/* log2 |F(a, b)|, F homogeneous of degree d */
__device__ static double
la_log2norm(const double *c, uint32 d, double a, double b)
{
	double res = c[d], bp = b;

	for (int k = (int)d - 1; k >= 0; k--) {
		res = res * a + c[k] * bp;
		bp *= b;
	}
	return log2(fabs(res) + 1.0);
}

__device__ static uint32
la_hash(uint64 key, uint32 mask)
{
	key *= 0x9e3779b97f4a7c15ull;
	return (uint32)(key >> 32) & mask;
}

/* a resieved prime of the candidate at key, into its list */
__device__ static void
la_emit_hit(const la_batch_t *bt, uint64 key, uint32 p_side, uint32 live)
{
	uint32 h = la_hash(key, bt->hmask);

	/* live: la_tile looks keys up while other blocks still insert
	   theirs, so it reads the slots from L2 (__ldcg). A plain load can
	   return a line this SM cached before another SM filled a slot of
	   it; the probe then stopped at a stale empty slot and lost the hit
	   (a few relations in some runs). la_large's resieve runs after
	   la_tile, so its plain loads (faster, through L1) are safe */
	for (;;) {
		uint64 k = live ? __ldcg((const unsigned long long *)bt->hkey +
					 h) : bt->hkey[h];

		if (k == key)
			break;
		if (k == LA_EMPTY)
			return;		/* candidate not stored (overflow) */
		h = (h + 1) & bt->hmask;
	}
	{
		la_cand_t *c = (la_cand_t *)bt->cands + bt->hval[h];
		uint32 n = atomicAdd(&c->nf, 1);

		if (n < LA_CF)
			c->f[n] = p_side;
	}
}

/*------------------------------------------------------------------*/
/* rho of every entry in every lattice: a = R b mod p becomes
   (a0 - R b0) i = -(a1 - R b1) j, so rho = -v / u with u = a0 - R b0,
   v = a1 - R b1 (projective, b = 0: u = b0, v = b1). u = 0: the entry
   hits whole rows j = 0 mod p (la_rows), or every position (p = q) */

extern "C" __global__ void
la_roots(const la_side_t *side, const la_batch_t *bt)
{
	const uint32 n0 = side[0].n, per = n0 + side[1].n;
	const uint64 total = (uint64)bt->num_lat * per;

	for (uint64 t = blockIdx.x * (uint64)blockDim.x + threadIdx.x;
	     t < total; t += (uint64)gridDim.x * blockDim.x) {
		uint32 l = (uint32)(t / per), e = (uint32)(t % per);
		uint32 s = (e >= n0);
		const la_side_t *sd = side + s;
		const la_lat_t *L = bt->lat + l;
		uint32 p, r, b0, b1, u, v, rho = LA_SKIP;

		if (s)
			e -= n0;
		p = sd->p[e];
		r = sd->r[e];
		b0 = la_mod(L->b0, p);
		b1 = la_mod(L->b1, p);
		if (r == p) {
			u = b0;
			v = b1;
		}
		else {
			uint64 pp = (uint64)p * p;

			u = (uint32)((la_mod(L->a0, p) + pp -
					(uint64)r * b0) % p);
			v = (uint32)((la_mod(L->a1, p) + pp -
					(uint64)r * b1) % p);
		}
		if (u)
			rho = v ? (uint32)((uint64)(p - v) *
				(p < (1u << 31) ? la_inv32(u, p) :
				 la_inv(u, p)) % p) : 0;
		bt->rho[t] = rho;
	}
}

/*------------------------------------------------------------------*/
/* Franke-Kleinjung: a basis (alpha, beta), (gamma, delta) of the
   lattice of (x, j) with x = rho j mod p, -I < alpha <= 0 <= gamma < I,
   gamma - alpha >= I, beta, delta > 0. From x in [0, I) the next
   lattice point in the strip (smallest larger j) is x + v1 if
   x >= -alpha, x + v2 if x < I - gamma, else x + v1 + v2. Returns 0 if
   the reduction degenerates (rare; the entry is not sieved) */

__device__ static uint32
la_fk_reduce(uint32 p, uint32 rho, int64 I, int64 *alpha, int64 *beta,
		int64 *gamma, int64 *delta)
{
	int64 al = -(int64)p, be = 0, ga = rho, de = 1;

	for (;;) {
		if (al == 0 || ga == 0)
			return 0;
		if (-al < I) {
			if (ga >= I) {
				int64 k = (ga - (I + al)) / (-al);

				ga += k * al;
				de += k * be;
			}
			break;
		}
		if (ga < I) {
			int64 k = (-al - (I - ga)) / ga;

			al += k * ga;
			be += k * de;
			break;
		}
		if (-al > ga) {
			int64 k = (-al) / ga;

			al += k * ga;
			be += k * de;
		}
		else {
			int64 k = ga / (-al);

			ga += k * al;
			de += k * be;
		}
	}
	if (al == 0 || ga == 0 || ga - al < I || be <= 0 || de <= 0)
		return 0;
	*alpha = al;
	*beta = be;
	*gamma = ga;
	*delta = de;
	return 1;
}

/* la_fk_reduce in 32-bit arithmetic, for p < 2^30: the basis entries
   are bounded by p (al, ga in [-p, p]; be, de the denominators of a
   continued fraction of rho/p, at most p), and so is every product
   k * al, k * be, k * ga, k * de below (each is the change of an entry
   that stays within those bounds). The same steps and results as
   la_fk_reduce, without 64-bit divisions, which the GPU emulates */
__device__ static uint32
la_fk_reduce32(uint32 p, uint32 rho, int32_t I, int32_t *alpha,
		int32_t *beta, int32_t *gamma, int32_t *delta)
{
	int32_t al = -(int32_t)p, be = 0, ga = (int32_t)rho, de = 1;

	for (;;) {
		if (al == 0 || ga == 0)
			return 0;
		if (-al < I) {
			if (ga >= I) {
				int32_t k = (ga - (I + al)) / (-al);

				ga += k * al;
				de += k * be;
			}
			break;
		}
		if (ga < I) {
			int32_t k = (-al - (I - ga)) / ga;

			al += k * ga;
			be += k * de;
			break;
		}
		if (-al > ga) {
			int32_t k = (-al) / ga;

			al += k * ga;
			be += k * de;
		}
		else {
			int32_t k = ga / (-al);

			ga += k * al;
			de += k * be;
		}
	}
	if (al == 0 || ga == 0 || ga - al < I || be <= 0 || de <= 0)
		return 0;
	*alpha = al;
	*beta = be;
	*gamma = ga;
	*delta = de;
	return 1;
}

/* the entries with p >= I of lattices [lat0, lat0 + nlat): add their
   logs (resieve = 0) or report their hits on candidates (resieve = 1).
   The host sieves one lattice per launch, so that the atomics stay in
   the L2 cache (one lattice's sieve is 32 MB at I = 4096) */
extern "C" __global__ void
la_large(const la_side_t *side, const la_batch_t *bt, uint32 resieve,
	 uint32 lat0, uint32 nlat)
{
	const uint32 logI = bt->logI, J = bt->J;
	const int64 I = (int64)1 << logI;
	const uint32 nall = side[0].n + side[1].n;
	const uint32 l0 = side[0].n - side[0].n_small;
	const uint32 per = l0 + side[1].n - side[1].n_small;
	const uint64 total = (uint64)nlat * per;

	for (uint64 t = blockIdx.x * (uint64)blockDim.x + threadIdx.x;
	     t < total; t += (uint64)gridDim.x * blockDim.x) {
		uint32 l = lat0 + (uint32)(t / per), e = (uint32)(t % per);
		uint32 s = (e >= l0);
		const la_side_t *sd = side + s;
		uint32 rho, p;
		int64 al, be, ga, de, x, j;
		uint32 *sv;
		const uint32 *bm;
		uint32 lg;

		e = (s ? e - l0 : e) + sd->n_small;
		rho = bt->rho[(uint64)l * nall + (s ? side[0].n : 0) + e];
		if (rho == LA_SKIP || rho == 0)
			continue;
		p = sd->p[e];
		sv = bt->sieve + (uint64)(bt->one_sieve ? 0 : l) *
					((uint64)J << logI);
		bm = bt->bitmap + (uint64)l * (((uint64)J << logI) / 32);
		lg = (uint32)sd->logp[e] << (16 * s);
		if (p < (1u << 30)) {
			/* 32-bit reduction and walk. x stays in [0, I); a step
			   of j that reaches J ends the walk, so the steps are
			   clamped to J (no overflow, the same hits) */
			int32_t a32, b32, g32, d32, x32 = (int32_t)(I / 2);
			uint32 j32 = 0, s1, s2, s12;

			if (!la_fk_reduce32(p, rho, (int32_t)I, &a32, &b32,
						&g32, &d32))
				continue;
			s1 = min((uint32)b32, J);
			s2 = min((uint32)d32, J);
			s12 = min((uint32)b32 + (uint32)d32, J);
			for (;;) {
				uint32 pos;

				if (x32 >= -a32) {
					x32 += a32;
					j32 += s1;
				}
				else if (x32 < (int32_t)I - g32) {
					x32 += g32;
					j32 += s2;
				}
				else {
					x32 += a32 + g32;
					j32 += s12;
				}
				if (j32 >= J)
					break;
				pos = (j32 << logI) | (uint32)x32;
				if (!resieve)
					atomicAdd(sv + pos, lg);
				else if ((bm[pos >> 5] >> (pos & 31)) & 1)
					la_emit_hit(bt, ((uint64)l << 32) | pos,
							p | (s << 31), 0);
			}
			continue;
		}
		if (!la_fk_reduce(p, rho, I, &al, &be, &ga, &de))
			continue;
		x = I / 2;
		j = 0;
		for (;;) {
			uint32 pos;

			if (x >= -al) {
				x += al;
				j += be;
			}
			else if (x < I - ga) {
				x += ga;
				j += de;
			}
			else {
				x += al + ga;
				j += be + de;
			}
			if (j >= J)
				break;
			pos = ((uint32)j << logI) | (uint32)x;
			if (!resieve)
				atomicAdd(sv + pos, lg);
			else if ((bm[pos >> 5] >> (pos & 31)) & 1)
				la_emit_hit(bt, ((uint64)l << 32) | pos,
						p | (s << 31), 0);
		}
	}
}

/*------------------------------------------------------------------*/
/* one tile of rows of one lattice: the small entries, the candidates
   and their resieving. Dynamic shared memory: bt->tile_pos words */

__device__ static void
la_small(const la_side_t *sd, const uint32 *rho, uint32 i0, uint32 i1,
	 uint32 rows, uint32 row0, uint32 logI, uint32 *sv, uint32 shift,
	 uint32 resieve, const la_batch_t *bt, uint32 l, uint32 s)
{
	const uint32 I = 1u << logI;
	const uint32 items = (i1 - i0) * rows;

	for (uint32 it = threadIdx.x; it < items; it += blockDim.x) {
		uint32 e = i0 + it / rows, row = it % rows;
		uint32 j = row0 + row, r = rho[e], p, x, lg;
		uint32 *rp = sv + (row << logI);

		if (j == 0 || r == LA_SKIP)
			continue;
		p = sd->p[e];
		/* r < p < I and j < J: r * j + I / 2 < I * J, which the
		   32-bit positions already need to fit: no 64-bit modulo */
		x = (r * j + I / 2) % p;
		if (!resieve) {
			lg = (uint32)sd->logp[e] << shift;
			for (; x < I; x += p)
				atomicAdd(rp + x, lg);
		}
		else {
			for (; x < I; x += p) {
				if (rp[x])
					la_emit_hit(bt, ((uint64)l << 32) |
						((j << logI) | x),
						p | (s << 31), 1);
			}
		}
	}
}

/* the logs of the entries with u = 0 of one side on row j: each prime
   p | j sieved here (p >= the first sieved prime), those of its entries
   whose rho is LA_SKIP */
__device__ static uint32
la_rows(const la_side_t *sd, const uint32 *rho, uint32 j)
{
	uint32 sum = 0, m = j;

	for (uint32 p = 2; m > 1; p++) {
		uint32 lo, hi;

		if (p * p > m)
			p = m;		/* what is left is prime */
		if (m % p)
			continue;
		while (m % p == 0)
			m /= p;
		if (sd->n_tiny >= sd->n_small || p < sd->p[sd->n_tiny])
			continue;
		lo = sd->n_tiny;	/* first entry with prime >= p */
		hi = sd->n_small;
		while (lo < hi) {
			uint32 mid = (lo + hi) / 2;

			if (sd->p[mid] < p)
				lo = mid + 1;
			else
				hi = mid;
		}
		for (; lo < sd->n_small && sd->p[lo] == p; lo++) {
			if (rho[lo] == LA_SKIP)
				sum += sd->logp[lo];
		}
	}
	return sum;
}

/* every row of every lattice: its la_rows sums */
extern "C" __global__ void
la_rowlogs(const la_side_t *side, const la_batch_t *bt)
{
	const uint32 J = bt->J, per = side[0].n + side[1].n;
	const uint32 total = bt->num_lat * J;

	for (uint32 t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
	     t += gridDim.x * blockDim.x) {
		uint32 l = t / J, j = t % J;
		const uint32 *rho = bt->rho + (uint64)l * per;

		bt->rowlog[t] = (j == 0) ? 0 :
			la_rows(side, rho, j) |
			(la_rows(side + 1, rho + side[0].n, j) << 16);
	}
}

extern "C" __global__ void __launch_bounds__(LA_THREADS)
la_tile(const la_side_t *side, const la_batch_t *bt, uint32 lat0)
{
	extern __shared__ uint32 sv[];
	__shared__ uint32 rowadd[LA_TILE_POS_MAX >> 9];	/* logI >= 9 */
	const uint32 logI = bt->logI, I = 1u << logI, J = bt->J;
	const uint32 rows = bt->tile_pos >> logI, tiles = J / rows;
	const uint32 l = lat0 + blockIdx.x / tiles;
	const uint32 row0 = (blockIdx.x % tiles) * rows;
	const uint32 npos = rows << logI, per = npos / LA_THREADS;
	const la_lat_t *L = bt->lat + l;
	const uint32 *rho = bt->rho + (uint64)l * (side[0].n + side[1].n);
	uint32 *g = bt->sieve + ((uint64)(bt->one_sieve ? 0 : l) * J + row0) *
			I;
	const uint32 my0 = threadIdx.x * per;
	uint32 mask = 0;

	/* the tile into shared memory, 16 bytes at a time (one word at a
	   time, each store waited for its load: a fifth of la_tile's stall
	   samples); with one_sieve the array is the next lattice's, so it
	   is cleared as it is read. g is 16-byte aligned (row0 * I words) */
	{
		uint4 *g4 = (uint4 *)g, *s4 = (uint4 *)sv;
		const uint4 z = make_uint4(0, 0, 0, 0);

		if (bt->one_sieve) {
			for (uint32 k = threadIdx.x; k < npos / 4;
			     k += blockDim.x) {
				s4[k] = g4[k];
				g4[k] = z;
			}
		}
		else {
			for (uint32 k = threadIdx.x; k < npos / 4;
			     k += blockDim.x)
				s4[k] = g4[k];
		}
	}
	__syncthreads();
	la_small(side, rho, side[0].n_tiny, side[0].n_small, rows, row0,
			logI, sv, 0, 0, bt, l, 0);
	la_small(side + 1, rho + side[0].n, side[1].n_tiny, side[1].n_small,
			rows, row0, logI, sv, 16, 0, bt, l, 1);
	if (threadIdx.x < rows)
		rowadd[threadIdx.x] = bt->rowlog[(uint64)l * J + row0 +
						threadIdx.x];
	__syncthreads();
	if (bt->debug_sums) {	/* the full sums, for la_check */
		for (uint32 k = threadIdx.x; k < npos; k += blockDim.x)
			g[k] = sv[k] + rowadd[k >> logI];
	}

	/* candidates: this thread's run of per positions, the thresholds
	   from the norms at its middle */
	{
		uint32 row = my0 >> logI, j = row0 + row;
		uint32 x0 = my0 & (I - 1);
		int64 ic = (int64)(x0 + per / 2) - I / 2;
		double ad = (double)L->a0 * ic + (double)L->a1 * j;
		double bd = (double)L->b0 * ic + (double)L->b1 * j;
		int32_t thr[2];
		uint32 nlp = 0;

		for (uint32 s = 0; s < 2; s++) {
			double nb = la_log2norm(bt->coef[s], bt->deg[s], ad, bd);
			uint32 three;

			if (s)
				nb -= L->log2q;
			three = (nb > bt->bits3[s]);
			thr[s] = (int32_t)((nb - (three ? bt->cut3[s] :
					bt->cut2[s])) * LA_SCALE) -
					side[s].tiny_bias -
					(int32_t)((rowadd[row] >> (16 * s)) &
						  0xffff);
			nlp |= (three ? 3 : 2) << (4 * s);
		}
		/* each lane starts its run at its own offset (k = kk + lane
		   mod per): the runs are per words apart, so in the same
		   order every lane of a warp read the same bank (Nsight
		   Compute: 18 M bank conflicts a lattice). The same
		   positions; only the order of the candidates in their list
		   changes, which the atomics already left open */
		for (uint32 kk = 0; kk < per && j > 0; kk++) {
			uint32 k = (kk + threadIdx.x) & (per - 1);
			uint32 w = sv[my0 + k];
			int64 i, a, b;

			if ((int32_t)(w & 0xffff) < thr[0] ||
			    (int32_t)(w >> 16) < thr[1])
				continue;
			i = (int64)(x0 + k) - I / 2;
			a = L->a0 * i + L->a1 * (int64)j;
			b = L->b0 * i + L->b1 * (int64)j;
			if (b == 0)
				continue;
			if (b < 0) {
				a = -a;
				b = -b;
			}
			if (la_gcd((uint64)(a < 0 ? -a : a), (uint64)b) != 1)
				continue;
			{
				uint32 pos = (j << logI) | (x0 + k);
				uint32 c = atomicAdd(&bt->count[0], 1);
				uint64 bw = (uint64)l * (((uint64)J << logI) / 32)
						+ (pos >> 5);

				if (c < bt->max_cands) {
					la_cand_t *cd = (la_cand_t *)bt->cands + c;
					uint64 key = ((uint64)l << 32) | pos;
					uint32 h = la_hash(key, bt->hmask);

					cd->key = key;
					cd->a = a;
					cd->b = b;
					cd->nlp = nlp;
					cd->nf = 0;
					while (atomicCAS((unsigned long long *)
						bt->hkey + h, LA_EMPTY, key) !=
						LA_EMPTY)
						h = (h + 1) & bt->hmask;
					bt->hval[h] = c;
				}
				else {
					bt->count[2] = 1;
				}
				atomicOr(bt->bitmap + bw, 1u << (pos & 31));
				mask |= 1u << k;
			}
		}
	}
	__syncthreads();
	for (uint32 kk = 0; kk < per; kk++) {
		uint32 k = (kk + threadIdx.x) & (per - 1);

		sv[my0 + k] = (mask >> k) & 1;
	}
	__syncthreads();
	la_small(side, rho, side[0].n_noresieve, side[0].n_small, rows, row0,
			logI, sv, 0, 1, bt, l, 0);
	la_small(side + 1, rho + side[0].n, side[1].n_noresieve,
			side[1].n_small, rows, row0, logI, sv, 16, 1, bt, l, 1);
}

/*------------------------------------------------------------------*/
/* the rational cofactor test of the host (la_lp_ok of
   sieve_lattice_gpu.c) on the exact norm, and the survivors compacted
   into bt->out. A candidate whose primes did not all fit (nf > LA_CF)
   goes to the host untested */

typedef unsigned __int128 u128;

__device__ static u128
la_u128(const uint64_t *w)
{
	return ((u128)w[1] << 64) | w[0];
}

/* n / p if p divides n (n changed), else n unchanged; 1 if divided */
__device__ static uint32
la_div128(u128 *n, uint32 p)
{
	uint64 hi = (uint64)(*n >> 64), lo = (uint64)*n;
	uint64 qh = hi / p, r = hi % p;
	uint64 m1 = (r << 32) | (lo >> 32);
	uint64 q1 = m1 / p;
	uint64 m0;

	r = m1 % p;
	m0 = (r << 32) | (lo & 0xffffffffull);
	if (m0 % p)
		return 0;
	*n = ((u128)qh << 64) | (q1 << 32) | (m0 / p);
	return 1;
}

/* the primes below LA_RESIEVE_MIN (trial-divided, never resieved), in
   groups whose products fit 32 bits: one multiword remainder per group,
   then a 32-bit test per prime. Written out, so that every divisor is
   a compile-time constant (the divisions become multiplications) */

__device__ static uint32
la_rem128(u128 n, uint32 d)
{
	uint64 hi = (uint64)(n >> 64), lo = (uint64)n;
	uint64 r = hi % d;

	r = ((r << 32) | (lo >> 32)) % d;
	return (uint32)(((r << 32) | (lo & 0xffffffffull)) % d);
}

/* N without its prime factors below LA_RESIEVE_MIN */
__device__ static void
la_tdiv128(u128 *N)
{
	uint32 r;

	r = la_rem128(*N, 223092870u);
	if (r % 2 == 0)
		while (la_div128(N, 2))
			;
	if (r % 3 == 0)
		while (la_div128(N, 3))
			;
	if (r % 5 == 0)
		while (la_div128(N, 5))
			;
	if (r % 7 == 0)
		while (la_div128(N, 7))
			;
	if (r % 11 == 0)
		while (la_div128(N, 11))
			;
	if (r % 13 == 0)
		while (la_div128(N, 13))
			;
	if (r % 17 == 0)
		while (la_div128(N, 17))
			;
	if (r % 19 == 0)
		while (la_div128(N, 19))
			;
	if (r % 23 == 0)
		while (la_div128(N, 23))
			;
	r = la_rem128(*N, 2756205443u);
	if (r % 29 == 0)
		while (la_div128(N, 29))
			;
	if (r % 31 == 0)
		while (la_div128(N, 31))
			;
	if (r % 37 == 0)
		while (la_div128(N, 37))
			;
	if (r % 41 == 0)
		while (la_div128(N, 41))
			;
	if (r % 43 == 0)
		while (la_div128(N, 43))
			;
	if (r % 47 == 0)
		while (la_div128(N, 47))
			;
	r = la_rem128(*N, 907383479u);
	if (r % 53 == 0)
		while (la_div128(N, 53))
			;
	if (r % 59 == 0)
		while (la_div128(N, 59))
			;
	if (r % 61 == 0)
		while (la_div128(N, 61))
			;
	if (r % 67 == 0)
		while (la_div128(N, 67))
			;
	if (r % 71 == 0)
		while (la_div128(N, 71))
			;
	r = la_rem128(*N, 4132280413u);
	if (r % 73 == 0)
		while (la_div128(N, 73))
			;
	if (r % 79 == 0)
		while (la_div128(N, 79))
			;
	if (r % 83 == 0)
		while (la_div128(N, 83))
			;
	if (r % 89 == 0)
		while (la_div128(N, 89))
			;
	if (r % 97 == 0)
		while (la_div128(N, 97))
			;
	r = la_rem128(*N, 121330189u);
	if (r % 101 == 0)
		while (la_div128(N, 101))
			;
	if (r % 103 == 0)
		while (la_div128(N, 103))
			;
	if (r % 107 == 0)
		while (la_div128(N, 107))
			;
	if (r % 109 == 0)
		while (la_div128(N, 109))
			;
	r = la_rem128(*N, 257557397u);
	if (r % 113 == 0)
		while (la_div128(N, 113))
			;
	if (r % 127 == 0)
		while (la_div128(N, 127))
			;
	if (r % 131 == 0)
		while (la_div128(N, 131))
			;
	if (r % 137 == 0)
		while (la_div128(N, 137))
			;
	r = la_rem128(*N, 490995677u);
	if (r % 139 == 0)
		while (la_div128(N, 139))
			;
	if (r % 149 == 0)
		while (la_div128(N, 149))
			;
	if (r % 151 == 0)
		while (la_div128(N, 151))
			;
	if (r % 157 == 0)
		while (la_div128(N, 157))
			;
	r = la_rem128(*N, 842952707u);
	if (r % 163 == 0)
		while (la_div128(N, 163))
			;
	if (r % 167 == 0)
		while (la_div128(N, 167))
			;
	if (r % 173 == 0)
		while (la_div128(N, 173))
			;
	if (r % 179 == 0)
		while (la_div128(N, 179))
			;
	r = la_rem128(*N, 1314423991u);
	if (r % 181 == 0)
		while (la_div128(N, 181))
			;
	if (r % 191 == 0)
		while (la_div128(N, 191))
			;
	if (r % 193 == 0)
		while (la_div128(N, 193))
			;
	if (r % 197 == 0)
		while (la_div128(N, 197))
			;
	r = la_rem128(*N, 2125525169u);
	if (r % 199 == 0)
		while (la_div128(N, 199))
			;
	if (r % 211 == 0)
		while (la_div128(N, 211))
			;
	if (r % 223 == 0)
		while (la_div128(N, 223))
			;
	if (r % 227 == 0)
		while (la_div128(N, 227))
			;
	r = la_rem128(*N, 3073309843u);
	if (r % 229 == 0)
		while (la_div128(N, 229))
			;
	if (r % 233 == 0)
		while (la_div128(N, 233))
			;
	if (r % 239 == 0)
		while (la_div128(N, 239))
			;
	if (r % 241 == 0)
		while (la_div128(N, 241))
			;
	r = la_rem128(*N, 251u);
	if (r % 251 == 0)
		while (la_div128(N, 251))
			;
}

/* whether a candidate's primes are all known (in f, or its row's
   la_jprime): not if they did not fit (nf > LA_CF), nor at i = 0
   (x = I/2), where the large primes with rho = 0 hit the whole column
   and are not walked (la_fk_reduce; the candidate there, j = 1 only by
   the gcd, goes to the host untested, which scans the factor base) */
__device__ static uint32
la_testable(const la_batch_t *bt, const la_cand_t *c)
{
	return c->nf <= LA_CF &&
	       ((uint32)c->key & ((1u << bt->logI) - 1)) !=
	       (1u << bt->logI) / 2;
}

/* msieve's large prime bounds on one side's cofactor N (la_lp_ok) */
__device__ static uint32
la_lp_ok128(u128 N, uint32 nlp, uint32 lp1max, const uint64_t *lp2min,
	const uint64_t *lp2max, const uint64_t *lp3min,
	const uint64_t *lp3max)
{
	if (nlp == 3) {
		if (N > la_u128(lp3max))
			return 0;
		if (N < la_u128(lp3min) && N > la_u128(lp2max))
			return 0;
	}
	else if (N > la_u128(lp2max)) {
		return 0;
	}
	if (N > (u128)lp1max && N < la_u128(lp2min))
		return 0;
	return 1;
}

/* the rational cofactor of a testable candidate: |r1 a + r0 b| without
   its listed rational primes, its row's prime and the small primes;
   0 if the norm is 0 */
__device__ static u128
la_rat_cofactor(const la_batch_t *bt, const la_cand_t *c)
{
	__int128 v = (__int128)la_u128(bt->r1) * c->a +
		     (__int128)la_u128(bt->r0) * c->b;
	u128 N = (u128)(v < 0 ? -v : v);
	uint32 pj = la_jprime((uint32)c->key >> bt->logI);

	if (N == 0)
		return 0;
	for (uint32 i = 0; i < c->nf; i++) {
		if ((c->f[i] >> 31) == 0)
			while (la_div128(&N, c->f[i]))
				;
	}
	la_tdiv128(&N);
	if (pj > 1)
		while (la_div128(&N, pj))
			;
	return N;
}

extern "C" __global__ void
la_filter(const la_batch_t *bt)
{
	const uint32 n = min(bt->count[LA_N_CANDS], bt->max_cands);
	const la_cand_t *cand = (const la_cand_t *)bt->cands;

	for (uint32 k = blockIdx.x * blockDim.x + threadIdx.x; k < n;
	     k += gridDim.x * blockDim.x) {
		const la_cand_t *c = cand + k;
		uint32 pass = 1;

		if (bt->rat_filter && la_testable(bt, c)) {
			u128 N = la_rat_cofactor(bt, c);

			pass = (N != 0 && la_lp_ok128(N, c->nlp & 15,
					bt->lp1max, bt->lp2min, bt->lp2max,
					bt->lp3min, bt->lp3max));
		}
		if (pass) {
			uint32 o = atomicAdd(&bt->count[LA_N_OUT], 1);

			((la_cand_t *)bt->out)[o] = *c;
		}
	}
}

/*------------------------------------------------------------------*/
/* la_cofactor: the host's remaining tests on the GPU, so that only
   the candidates that can become relations are copied back. One CGBN
   instance (LA_COF_TPI threads, LA_COF_BITS) per candidate of out:
   the algebraic norm F(a, b) by Horner's rule in two's complement mod
   2^LA_COF_BITS (exact, as |F| < 2^(LA_COF_BITS - 3) by the host's
   per-batch bound), divided by q, the listed algebraic primes, the
   row's prime and the small primes; msieve's bounds on it; and the
   host's Fermat test (la_composite_ok: a base-2 probable prime above
   2^32 - 1 is a large prime too large) on both cofactors */

typedef cgbn_context_t<LA_COF_TPI> la_ctx_t;
typedef cgbn_env_t<la_ctx_t, LA_COF_BITS> la_env_t;
typedef la_env_t::cgbn_t la_bn_t;
typedef cgbn_env_t<la_ctx_t, 128> la_env128_t;	/* the Fermat tests */
typedef la_env128_t::cgbn_t la_bn128_t;
#define LA_COF_WORDS (LA_COF_BITS / 32)

__device__ static void
la_bn_set(la_env_t env, la_bn_t &r, uint64 lo, uint64 hi, uint32 ext)
{
	cgbn_mem_t<LA_COF_BITS> m;

	m._limbs[0] = (uint32)lo;
	m._limbs[1] = (uint32)(lo >> 32);
	m._limbs[2] = (uint32)hi;
	m._limbs[3] = (uint32)(hi >> 32);
	for (uint32 k = 4; k < LA_COF_WORDS; k++)
		m._limbs[k] = ext;
	cgbn_load(env, r, &m);
}

/* the value of r if it is below 2^127, else 0 */
__device__ static u128
la_bn_u128(la_env_t env, const la_bn_t &r)
{
	u128 v = 0;

	if (LA_COF_BITS - cgbn_clz(env, r) > 127)
		return 0;
	for (int k = 3; k >= 0; k--)
		v = (v << 32) | cgbn_extract_bits_ui32(env, r, 32 * k, 32);
	return v;
}

__device__ static void
la_bn_divout(la_env_t env, la_bn_t &v, uint32 p)
{
	while (cgbn_rem_ui32(env, v, p) == 0)
		cgbn_div_ui32(env, v, v, p);
}

__device__ static void
la_bn128_set(la_env128_t env, la_bn128_t &r, u128 v)
{
	cgbn_mem_t<128> m;

	m._limbs[0] = (uint32)v;
	m._limbs[1] = (uint32)(v >> 32);
	m._limbs[2] = (uint32)(v >> 64);
	m._limbs[3] = (uint32)(v >> 96);
	cgbn_load(env, r, &m);
}

/* v without its prime factors below LA_RESIEVE_MIN (as la_tdiv128) */
__device__ static void
la_bn_tdiv(la_env_t env, la_bn_t &v)
{
	uint32 r;

	r = cgbn_rem_ui32(env, v, 223092870u);
	if (r % 2 == 0)
		la_bn_divout(env, v, 2);
	if (r % 3 == 0)
		la_bn_divout(env, v, 3);
	if (r % 5 == 0)
		la_bn_divout(env, v, 5);
	if (r % 7 == 0)
		la_bn_divout(env, v, 7);
	if (r % 11 == 0)
		la_bn_divout(env, v, 11);
	if (r % 13 == 0)
		la_bn_divout(env, v, 13);
	if (r % 17 == 0)
		la_bn_divout(env, v, 17);
	if (r % 19 == 0)
		la_bn_divout(env, v, 19);
	if (r % 23 == 0)
		la_bn_divout(env, v, 23);
	r = cgbn_rem_ui32(env, v, 2756205443u);
	if (r % 29 == 0)
		la_bn_divout(env, v, 29);
	if (r % 31 == 0)
		la_bn_divout(env, v, 31);
	if (r % 37 == 0)
		la_bn_divout(env, v, 37);
	if (r % 41 == 0)
		la_bn_divout(env, v, 41);
	if (r % 43 == 0)
		la_bn_divout(env, v, 43);
	if (r % 47 == 0)
		la_bn_divout(env, v, 47);
	r = cgbn_rem_ui32(env, v, 907383479u);
	if (r % 53 == 0)
		la_bn_divout(env, v, 53);
	if (r % 59 == 0)
		la_bn_divout(env, v, 59);
	if (r % 61 == 0)
		la_bn_divout(env, v, 61);
	if (r % 67 == 0)
		la_bn_divout(env, v, 67);
	if (r % 71 == 0)
		la_bn_divout(env, v, 71);
	r = cgbn_rem_ui32(env, v, 4132280413u);
	if (r % 73 == 0)
		la_bn_divout(env, v, 73);
	if (r % 79 == 0)
		la_bn_divout(env, v, 79);
	if (r % 83 == 0)
		la_bn_divout(env, v, 83);
	if (r % 89 == 0)
		la_bn_divout(env, v, 89);
	if (r % 97 == 0)
		la_bn_divout(env, v, 97);
	r = cgbn_rem_ui32(env, v, 121330189u);
	if (r % 101 == 0)
		la_bn_divout(env, v, 101);
	if (r % 103 == 0)
		la_bn_divout(env, v, 103);
	if (r % 107 == 0)
		la_bn_divout(env, v, 107);
	if (r % 109 == 0)
		la_bn_divout(env, v, 109);
	r = cgbn_rem_ui32(env, v, 257557397u);
	if (r % 113 == 0)
		la_bn_divout(env, v, 113);
	if (r % 127 == 0)
		la_bn_divout(env, v, 127);
	if (r % 131 == 0)
		la_bn_divout(env, v, 131);
	if (r % 137 == 0)
		la_bn_divout(env, v, 137);
	r = cgbn_rem_ui32(env, v, 490995677u);
	if (r % 139 == 0)
		la_bn_divout(env, v, 139);
	if (r % 149 == 0)
		la_bn_divout(env, v, 149);
	if (r % 151 == 0)
		la_bn_divout(env, v, 151);
	if (r % 157 == 0)
		la_bn_divout(env, v, 157);
	r = cgbn_rem_ui32(env, v, 842952707u);
	if (r % 163 == 0)
		la_bn_divout(env, v, 163);
	if (r % 167 == 0)
		la_bn_divout(env, v, 167);
	if (r % 173 == 0)
		la_bn_divout(env, v, 173);
	if (r % 179 == 0)
		la_bn_divout(env, v, 179);
	r = cgbn_rem_ui32(env, v, 1314423991u);
	if (r % 181 == 0)
		la_bn_divout(env, v, 181);
	if (r % 191 == 0)
		la_bn_divout(env, v, 191);
	if (r % 193 == 0)
		la_bn_divout(env, v, 193);
	if (r % 197 == 0)
		la_bn_divout(env, v, 197);
	r = cgbn_rem_ui32(env, v, 2125525169u);
	if (r % 199 == 0)
		la_bn_divout(env, v, 199);
	if (r % 211 == 0)
		la_bn_divout(env, v, 211);
	if (r % 223 == 0)
		la_bn_divout(env, v, 223);
	if (r % 227 == 0)
		la_bn_divout(env, v, 227);
	r = cgbn_rem_ui32(env, v, 3073309843u);
	if (r % 229 == 0)
		la_bn_divout(env, v, 229);
	if (r % 233 == 0)
		la_bn_divout(env, v, 233);
	if (r % 239 == 0)
		la_bn_divout(env, v, 239);
	if (r % 241 == 0)
		la_bn_divout(env, v, 241);
	r = cgbn_rem_ui32(env, v, 251u);
	if (r % 251 == 0)
		la_bn_divout(env, v, 251);
}

/* N > 2^32 - 1 (N < 2^127, odd) and a base-2 probable prime: no use */
__device__ static uint32
la_fermat_bad(la_env128_t env, u128 N)
{
	la_bn128_t x, e, m, r;

	if (N <= 0xffffffffu)
		return 0;
	la_bn128_set(env, m, N);
	la_bn128_set(env, e, N - 1);
	cgbn_set_ui32(env, x, 2);
	cgbn_modular_power(env, r, x, e, m);
	return cgbn_equals_ui32(env, r, 1);
}

extern "C" __global__ void
la_cofactor(const la_batch_t *bt)
{
	la_ctx_t ctx(cgbn_no_checks);
	la_env_t env(ctx.env<la_env_t>());
	la_env128_t env128(ctx.env<la_env128_t>());
	const uint32 n = bt->count[LA_N_OUT];
	const uint32 stride = gridDim.x * blockDim.x / LA_COF_TPI;
	const uint32 lead = (threadIdx.x % LA_COF_TPI == 0);
	const uint32 d = bt->deg[1];

	for (uint32 k = (blockIdx.x * blockDim.x + threadIdx.x) /
			LA_COF_TPI; k < n; k += stride) {
		const la_cand_t *c = (const la_cand_t *)bt->out + k;
		uint32 why = 0;		/* 0 pass, else the counter */

		if (la_testable(bt, c)) {
			const la_lat_t *L = bt->lat + (uint32)(c->key >> 32);
			la_bn_t v, t, aa, bb, bp;
			u128 Na, Nr = la_rat_cofactor(bt, c);
			uint32 pj = la_jprime((uint32)c->key >> bt->logI);

			la_bn_set(env, aa, (uint64)c->a, c->a < 0 ? ~0ull : 0,
					c->a < 0 ? ~0u : 0);
			la_bn_set(env, bb, (uint64)c->b, 0, 0);
			cgbn_set_ui32(env, bp, 1);
			cgbn_load(env, v, (cgbn_mem_t<LA_COF_BITS> *)
					bt->acoef[d]);
			for (int i = (int)d - 1; i >= 0; i--) {
				cgbn_mul(env, bp, bp, bb);
				cgbn_mul(env, v, v, aa);
				cgbn_load(env, t, (cgbn_mem_t<LA_COF_BITS> *)
						bt->acoef[i]);
				cgbn_mul(env, t, t, bp);
				cgbn_add(env, v, v, t);
			}
			if (cgbn_extract_bits_ui32(env, v, LA_COF_BITS - 1, 1))
				cgbn_negate(env, v, v);

			la_bn_divout(env, v, L->q);
			for (uint32 i = 0; i < c->nf; i++) {
				if (c->f[i] >> 31)
					la_bn_divout(env, v,
						c->f[i] & 0x7fffffff);
			}
			if (pj > 1)
				la_bn_divout(env, v, pj);
			la_bn_tdiv(env, v);

			Na = la_bn_u128(env, v);
			if (Na == 0 ||
			    !la_lp_ok128(Na, (c->nlp >> 4) & 15, bt->lpa1max,
					bt->lpa2min, bt->lpa2max, bt->lpa3min,
					bt->lpa3max))
				why = LA_N_ALG;
			else if (la_fermat_bad(env128, Nr) ||
				 la_fermat_bad(env128, Na))
				why = LA_N_FERMAT;
		}
		if (lead) {
			if (why == 0) {
				uint32 o = atomicAdd(&bt->count[LA_N_OUT2], 1);

				((la_cand_t *)bt->out2)[o] = *c;
			}
			else {
				atomicAdd(&bt->count[why], 1);
			}
		}
	}
}
