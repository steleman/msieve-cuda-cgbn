/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.
--------------------------------------------------------------------*/

/* the line siever's structures (sieve_line.c), shared with its GPU
   driver (sieve_line_gpu.c) */

#ifndef _GNFS_SIEVE_SIEVE_LINE_INT_H_
#define _GNFS_SIEVE_SIEVE_LINE_INT_H_

#include "sieve.h"

/* sieving takes place in L1-cache-size blocks */
#define BLOCK_SIZE 65536
#define BLOCK_HASH(x) ((uint64)(x) / BLOCK_SIZE)

/* how often the log targets for the algebraic sieve are
   recalculated. Currently the rational sieve is only
   updated once per block */
#define A_SAMPLE_RATE 512

/* structure used to sieve with one factor base prime */
typedef struct {
	uint32 p;
	uint32 r;
	uint32 next;
	uint16 offset;
	uint8 logp;
	uint8 skip;
} fb_sieve_entry_t;

/* used for sieving by powers of factor base primes */
typedef struct {
	uint16 p;	/* the power of a factor base prime */
	uint16 r;	 /* sieve root (does not change during sieving) */
	uint16 offset; /* the offset of the next update */
	uint8 logp;	/* scaled logarithm of 'prime' */
	uint8 orig_p;	/* the original factor base prime */
	uint8 skip;	/* if nonzero, do not sieve with this entry */
} fb_power_t;

#define FB_POWER_LIMIT 40

/* packed version of fb_entry_t, used for large FB primes */
typedef struct {
	uint32 p;		/* factor base prime */
	uint16 offset;	 /* the offset of the current update */
	uint8 logp;	/* scaled logarithm of 'prime' */
} packed_fb_t;

/* structure for projective roots */
typedef struct {
	uint32 p;			 /* factor base prime */
} projective_t;
	
/* structure used for resieving. We buffer at most
   MAX_RESIEVE_ENTRIES sieve values at a time, and 
   each sieve value uses up two resieve_t structures */

#define MAX_RESIEVE_ENTRIES 255
#define RESIEVE_INVALID 0xff
#define NUM_RESIEVE_FACTORS 19

typedef struct {
	uint16 offset;
	uint16 num_factors;
	uint32 factors[NUM_RESIEVE_FACTORS];
} resieve_t;
	
/* main structure controlling rational or algebraic sieving */
typedef struct {
	mpz_poly_t *poly;	/* the sieve polynomial */

	uint8 *sieve_block;	/* piece of sieve interval (one block worth) */
	uint32 *bucket_list; /* head of linked list of updates (per bucket) */

	uint32 curr_num_lp;	/* the number of large primes for this side */
	uint32 LP1_max;		/* single large prime max bound */

	uint32 cutoff2;	 	/* fudge factor for trial factoring */
	uint32 scaled_cutoff2;	/* fudge factor for trial factoring */
	mpz_t LP2_min;		/* double large prime min bound */
	mpz_t LP2_max;		/* double large prime max bound */

	uint32 cutoff3;	 	/* fudge factor for trial factoring */
	uint32 scaled_cutoff3;	/* fudge factor for trial factoring */
	mpz_t LP3_min;		/* triple large prime min bound */
	mpz_t LP3_max;		/* triple large prime max bound */

	mpz_t res;     /* temporaries */
	mpz_t tmp2;
	mpz_t tmp3;

	double base;	/* base used for logs of sieve values and primes */
	double log_base;	/* logarithm of the above */

	uint32 med_fb_size; /* number of entries with primes below BLOCK_SIZE */
	uint32 small_fb_size;/* number of entries that are never resieved */
	uint32 fb_size;      /* total number of factor base entries */

	projective_t proj_entries[20]; /* projective roots */
	uint32 proj_fb_size;		 /* number of projective roots */
	uint8 proj_bias;	 /* sum of logs of projective roots that divide
					the current b value */ 

	fb_sieve_entry_t *fb_entries; /* the factor base */

	fb_power_t *fb_powers;   /* used for powers of factor base primes */
	uint32 powers_fb_size;   /* the number of such entries */
	uint32 powers_fb_alloc;  /* space allocated for list of powers */

	packed_fb_t *update_list; /* list of updates to current block */
	uint32 num_updates_alloc; /* number of allocated updates */
	uint32 num_updates;	  /* the current number of updates */
} sieve_t;
	
/* main sieving structure */
typedef struct {
	msieve_obj *obj;
	factor_base_t *fb;

	int64 min_a;
	int64 max_a;
	uint64 min_b;
	uint64 max_b;

	uint32 num_buckets;
	sieve_t sieve_rfb;
	sieve_t sieve_afb;

	resieve_t *resieve_array;

	relation_batch_t relation_batch;

} sieve_job_t;


/* how often (in lines) the logarithm bases are recomputed */
#define LOG_UPDATE_RATE 100

#ifdef HAVE_CUDA
/* the GPU line siever (sieve_line_gpu.c): nonzero if it ran the job,
   with the relations found and the last b completed */
uint32 lattice_sieve_gpu(sieve_job_t *job, double skew,
		uint32 *relations_found, uint32 max_relations,
		uint64 *last_q);
uint32 line_sieve_gpu(sieve_job_t *job, uint32 *relations_found,
			uint32 max_relations, uint64 *last_b);
#endif

#endif /* !_GNFS_SIEVE_SIEVE_LINE_INT_H_ */
