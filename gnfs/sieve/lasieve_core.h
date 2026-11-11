/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* shared by the GPU lattice siever's kernels (lasieve_core.cu) and its
   host driver (sieve_lattice_gpu.c).

   A lattice is one special-q (q, r) on the algebraic side: the (a, b)
   with a = a0 i + a1 j, b = b0 i + b1 j, -I/2 <= i < I/2, 0 <= j < J.
   Position x = i + I/2, row j; the sieve value of a lattice is one
   32-bit word per position, rational log sum in the low 16 bits,
   algebraic in the high 16 (logs are LA_SCALE units per bit, so a sum
   cannot reach 2^16). A factor base entry (p, r) (a = r b mod p, r = p
   projective) hits the positions with i = rho j mod p */

#ifndef _GNFS_SIEVE_LASIEVE_CORE_H_
#define _GNFS_SIEVE_LASIEVE_CORE_H_

#include <stdint.h>

#define LA_SCALE 16		/* log units per bit */
#define LA_MAX_DEG 8
#define LA_ROWS 4		/* rows per tile (shared memory block) */
#define LA_THREADS 512
#define LA_SKIP 0xffffffffu	/* entry does not hit this lattice */
#define LA_RESIEVE_MIN 256	/* primes resieved (smaller: host) */
#define LA_CF 24		/* resieved primes kept per candidate */
#define LA_EMPTY 0xffffffffffffffffull	/* free hash slot */
#define LA_COF_BITS 256		/* la_cofactor's CGBN width */
#define LA_COF_TPI 4		/* threads per candidate there */

/* bt->count: */
#define LA_N_CANDS 0		/* candidates found */
#define LA_N_OUT 1		/* past la_filter (in out) */
#define LA_N_OVERFLOW 2		/* the batch did not fit */
#define LA_N_OUT2 3		/* past la_cofactor (in out2) */
#define LA_N_ALG 4		/* failed la_cofactor's algebraic bounds */
#define LA_N_FERMAT 5		/* failed its Fermat tests */
#define LA_N_COUNTS 8

/* the interface between sieve_lattice_gpu.c and the kernels (their
   arguments, la_batch_t): msieve loads the PTX from the current
   directory, and a PTX file older than the binary made illegal memory
   accesses. The PTX holds la_abi = { LA_ABI, sizeof(la_batch_t) }, which
   the host compares with its own before sieving. Raise LA_ABI when a
   kernel's arguments change */
#define LA_ABI 3

/* la_tile's tile: positions in shared memory (4 bytes each), whole rows,
   at most LA_TILE_POS_MAX (its 32-bit candidate mask per thread); the
   batch's tile_pos is max(LA_TILE_POS, I) */
#define LA_TILE_POS_MAX 16384
#ifndef LA_TILE_POS
#define LA_TILE_POS 8192
#endif

#ifdef __CUDACC__
#define LA_PTR(t) t *
#else
#define LA_PTR(t) uint64_t
#endif

#ifdef __CUDACC__
#define LA_HD __host__ __device__
#else
#define LA_HD
#endif

/* The entries with u = 0 (la_roots; rho = LA_SKIP) divide the norm at
   every position of the rows j = 0 mod p, so they are neither walked
   nor resieved: la_tile adds their logs to those rows, and la_filter
   and the host divide their primes out of the norms, found from j.
   j < J <= 2^13 has at most one prime factor >= LA_RESIEVE_MIN; this
   returns it, or 1 */
static inline LA_HD uint32_t
la_jprime(uint32_t j)
{
	uint32_t p;

	for (p = 2; p < LA_RESIEVE_MIN && p * p <= j; p++) {
		while (j % p == 0)
			j /= p;
	}
	/* j is now 1, a prime, or (p reached LA_RESIEVE_MIN) a product of
	   primes >= LA_RESIEVE_MIN, which j < 2^16 makes one prime */
	return (j >= LA_RESIEVE_MIN) ? j : 1;
}

/* one factor base (side), sorted by p */
typedef struct {
	LA_PTR(const uint32_t) p;
	LA_PTR(const uint32_t) r;
	LA_PTR(const uint16_t) logp;	/* floor(LA_SCALE log2 p) */
	uint32_t n;			/* entries */
	uint32_t n_small;		/* entries with p < I */
	uint32_t n_noresieve;		/* entries with p < LA_RESIEVE_MIN */
	uint32_t n_tiny;		/* entries not sieved (small p; their
					   expected logs lower the threshold) */
	int32_t tiny_bias;		/* that expectation, log units */
	uint32_t pad;
} la_side_t;

/* one special-q lattice */
typedef struct {
	int64_t a0, b0, a1, b1;
	uint32_t q, r;
	double log2q;
} la_lat_t;

/* one batch of lattices */
typedef struct {
	uint32_t logI, J, num_lat;
	uint32_t max_cands, max_hits;
	uint32_t deg[2];
	uint32_t pad;
	double coef[2][LA_MAX_DEG + 1];	/* the polynomials, as doubles */
	double cut2[2], cut3[2];	/* unsieved bits allowed (2/3 LP) */
	double bits3[2];		/* norm bits above which 3 LP */
	LA_PTR(const la_lat_t) lat;
	LA_PTR(uint32_t) rho;		/* [lat][side][entry] */
	LA_PTR(uint32_t) sieve;		/* [lat][j][x] */
	LA_PTR(uint32_t) bitmap;	/* [lat] candidates, 1 bit/position */
	LA_PTR(uint32_t) count;		/* [0] cands, [1] survivors,
					   [2] overflow */
	LA_PTR(void) cands;		/* la_cand_t */
	LA_PTR(void) out;		/* la_cand_t, those passing la_filter */
	LA_PTR(uint64_t) hkey;		/* hash: candidate key -> index */
	LA_PTR(uint32_t) hval;
	LA_PTR(uint32_t) rowlog;	/* [lat][j]: the logs of the u = 0
					   entries on row j (la_rowlogs),
					   rational low 16 bits */
	uint32_t hmask;			/* hash slots - 1 */
	uint32_t debug_sums;		/* write the tile sums back (test) */
	uint32_t one_sieve;		/* 1: one sieve array for the batch,
					   each lattice sieved by la_large and
					   read (and cleared) by la_tile before
					   the next, so that it stays in L2;
					   0: an array per lattice (check mode) */
	uint32_t tile_pos;		/* la_tile: positions per tile */

	/* la_filter: the rational side's cofactor test of the host
	   (la_lp_ok), on the exact norm |r1 a + r0 b| < 2^127 */
	uint32_t rat_filter;		/* 0: every candidate to the host */
	uint32_t lp1max;
	uint64_t r0[2], r1[2];		/* two's complement, lo, hi */
	uint64_t lp2min[2], lp2max[2], lp3min[2], lp3max[2];	/* lo, hi */

	/* la_cofactor: the rest of the host's tests (the algebraic side's
	   la_lp_ok, and la_composite_ok on both sides) in CGBN, on the
	   candidates past la_filter; survivors into out2. Needs rat_filter
	   and norms below 2^(LA_COF_BITS - 3) (checked per batch) */
	uint32_t cof_filter;
	uint32_t lpa1max;
	uint64_t lpa2min[2], lpa2max[2], lpa3min[2], lpa3max[2];
	uint32_t acoef[LA_MAX_DEG + 1][LA_COF_BITS / 32];	/* algebraic
					   polynomial, two's complement */
	LA_PTR(void) out2;
} la_batch_t;

/* a candidate; key = lattice << 32 | (j << logI | x). f: its
   resieved primes, side in the top bit (nf > LA_CF: the host finds
   them itself) */
typedef struct {
	uint64_t key;
	int64_t a;
	int64_t b;		/* > 0 */
	uint32_t nlp;		/* rational low 4 bits, algebraic next 4 */
	uint32_t nf;
	uint32_t f[LA_CF];
} la_cand_t;

#endif /* !_GNFS_SIEVE_LASIEVE_CORE_H_ */
