/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
gnfs/sieve/sieve_core.cu)
--------------------------------------------------------------------*/

/* shared by the GPU line siever's kernels (sieve_core.cu) and its host
   driver (sieve_line_gpu.c) */

#ifndef _GNFS_SIEVE_SIEVE_CORE_H_
#define _GNFS_SIEVE_SIEVE_CORE_H_

#include <stdint.h>

#define LS_BLOCK 65536		/* sieve_line.c's BLOCK_SIZE */
#define LS_SEG 16384		/* positions per GPU block */
#define LS_CHUNK (1 << 20)	/* positions per bucket pass */
#define LS_SEGS (LS_CHUNK / LS_SEG)
#define LS_THREADS 512
#define LS_TINY 256		/* below this: sieved position-parallel */
#define LS_SKIP 0xffffffffu	/* entry not sieved on this line */
#define LS_BUCKET_CAP 12288	/* large-prime hits per segment and side */
#define LS_CAND_FACTORS 20	/* resieved primes kept per side */
#define LS_POLY_WORDS 8		/* coefficients up to 255 bits */
#define LS_EVAL_WORDS 12	/* norms up to 383 bits */
#define LS_TAB 1024		/* bit size -> scaled log */
#define LS_MAX_DEG 8

#ifdef __CUDACC__
#define LS_PTR(t) t *
#else
#define LS_PTR(t) uint64_t
#endif

/* one factor base (side), in device memory */
typedef struct {
	LS_PTR(const uint32_t) p;
	LS_PTR(const uint32_t) r;
	LS_PTR(const uint32_t) rem0;	/* p - min_a % p, reduced */
	LS_PTR(const uint8_t) logp;
	LS_PTR(const uint32_t) pw_p;	/* prime powers */
	LS_PTR(const uint32_t) pw_r;
	LS_PTR(const uint32_t) pw_orig;
	LS_PTR(const uint32_t) pw_rem0;
	LS_PTR(const uint8_t) pw_logp;
	uint32_t num_entries;
	uint32_t num_powers;
	uint32_t num_direct;		/* entries with p <= LS_SEG */
	uint32_t small_size;		/* entries with p <= 1000 */
} ls_side_t;

/* a polynomial: two's complement words, low first */
typedef struct {
	uint32_t degree;
	uint32_t pad;
	uint32_t c[LS_MAX_DEG + 1][LS_POLY_WORDS];
} ls_poly_t;

/* one batch of lines (one logarithm base), in device memory */
typedef struct {
	uint32_t num_lines;
	uint32_t max_cands;
	int64_t min_a;
	LS_PTR(const uint64_t) lines;	/* b */
	LS_PTR(const uint64_t) ltot;	/* positions */
	LS_PTR(const uint8_t) proj_r;	/* projective biases */
	LS_PTR(const uint8_t) proj_a;
	int32_t sc2_r, sc3_r, sc2_a, sc3_a;	/* scaled cutoffs */
	ls_poly_t poly_r, poly_a;
	int32_t tab_r[LS_TAB], tab_a[LS_TAB];	/* bits -> fplog_eval_poly */
} ls_batch_t;

/* a candidate: its line and position, the number of large primes
   allowed per side (rational low 4 bits, algebraic next 4), and the
   resieved primes (side 0 rational, 1 algebraic; nf may exceed
   LS_CAND_FACTORS, then the host finds them itself) */
typedef struct {
	uint32_t line;
	uint32_t pos;
	uint32_t nlp;
	uint32_t nf[2];
	uint32_t f[2][LS_CAND_FACTORS];
} ls_cand_t;

#endif /* !_GNFS_SIEVE_SIEVE_CORE_H_ */
