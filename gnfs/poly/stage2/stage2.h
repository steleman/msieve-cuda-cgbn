/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: stage2.h 732 2012-08-04 02:32:46Z jasonp_sf $
--------------------------------------------------------------------*/

#ifndef _STAGE2_H_
#define _STAGE2_H_

#include <poly_skew.h>
#include "line_sieve_core.h"
#include "murphy_core.h"
#include "alpha_core.h"
#include "lattice_core.h"
#include <stage1_core_gpu/sizeopt_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------*/
/* data used in the current polynomial */

typedef struct {
	mpz_t gmp_a[MAX_POLY_DEGREE + 1];
	mpz_t gmp_b[MAX_POLY_DEGREE + 1];
	mpz_t gmp_c[MAX_POLY_DEGREE + 1];
	mpz_t gmp_lina[2];
	mpz_t gmp_linb[2];
	mpz_t gmp_linc[2];
	mpz_t gmp_help1;
	mpz_t gmp_help2;
	mpz_t gmp_help3;
	mpz_t gmp_help4;
	mpz_t gmp_p;
	mpz_t gmp_d;
} curr_poly_t;

/*-----------------------------------------------------------------------*/
/* data for rating polynomial yield */

typedef struct {
	integrate_t integ_aux;
	dickman_t dickman_aux;
} assess_t;

void assess_init(assess_t *a);

void assess_free(assess_t *a);

uint32 stage2_root_score(uint32 deg1, mpz_t *coeff1, 
       			uint32 prime_bound, double *score,
       			uint32 projective_only);

/*-----------------------------------------------------------------------*/
/* routines for optimizing polynomials */

void optimize_initial(curr_poly_t *data, uint32 deg, double *pol_norm,
			uint32 skew_only);

double optimize_initial_deg6(double best[MAX_VARS], 
			curr_poly_t *c, uint32 degree);

void optimize_final(mpz_t x, mpz_t y, int64 z, poly_rootopt_t *data);

/* optimize_final() of every rotation in a root heap; the Murphy E
   optimizations together, on the CPU unless a thread sets the hook
   (the GPU engine; murphy_core.h) */
typedef void (*murphy_batch_hook_t)(const mf_in_t *in, uint32 n,
				mf_out_t *out);
#if defined(_MSC_VER)
extern __declspec(thread) murphy_batch_hook_t murphy_batch_hook;
#else
extern _Thread_local murphy_batch_hook_t murphy_batch_hook;
#endif

/* when set for a thread, optimize_basic() hands its work to this
   function instead: the GPU root optimization (root_sieve_gpu.c)
   runs whole root optimizations as coroutines on its threads and
   batches their optimize_basic() calls for the GPU */
typedef double (*optimize_basic_hook_t)(dpoly_t *apoly,
				double *best_skewness,
				double *best_translation);
#if defined(_MSC_VER)
extern __declspec(thread) optimize_basic_hook_t optimize_basic_hook;
#else
extern _Thread_local optimize_basic_hook_t optimize_basic_hook;
#endif

double optimize_basic(dpoly_t *apoly, double *best_skewness,
				double *best_translation);

/* optimize_basic() on n polynomials; the batched root sieve
   (root_sieve_batch.c) makes its calls through this. Runs them one
   after the other, unless a thread sets the hook (the GPU engine) */
typedef struct {
	double norm;
	double skewness;
	double translation;
} basic_result_t;

typedef void (*optimize_basic_batch_hook_t)(dpoly_t *in, uint32 n,
				basic_result_t *out);
#if defined(_MSC_VER)
extern __declspec(thread) optimize_basic_batch_hook_t
				optimize_basic_batch_hook;
#else
extern _Thread_local optimize_basic_batch_hook_t optimize_basic_batch_hook;
#endif

void optimize_basic_batch(dpoly_t *in, uint32 n, basic_result_t *out);

/*-----------------------------------------------------------------------*/
/* data for the root sieve */

#define MAX_SIEVE_PRIME 100
#define ROOT_HEAP_SIZE 200
#define LOG_SCALE_FACTOR 1000
#define ROOT_SCORE_COARSE_MIN (-4.0)

typedef struct {
	uint16 resclass;
	uint16 start;
	uint16 step;
} sieve_root_t;

typedef struct {
	uint32 power;
	uint32 num_roots;
	double root_contrib;
	uint16 sieve_contrib;
	sieve_root_t *roots;
} sieve_power_t;

typedef struct {
	uint32 prime;
	uint32 num_powers;
	sieve_power_t *powers;

	uint32 contrib_array_size;
	uint32 contrib_array_offset;
	uint16 *contrib_array;
} sieve_prime_t;

/* a rotation: x and y as int64 when they fit (big = 0), which the GPU
   paths use without GMP; otherwise in the mpz_t fields. rotation_mpz()
   fills the mpz_t fields for the CPU code */
typedef struct {
	int64 xs, ys;
	uint32 big;
	mpz_t x;
	mpz_t y;
	int64 z;
	double score;
} rotation_t;

static INLINE void rotation_mpz(rotation_t *r) {
	if (!r->big) {
		int64_2gmp(r->xs, r->x);
		int64_2gmp(r->ys, r->y);
	}
}

typedef struct {
	uint32 num_entries;
	uint32 max_entries;
	rotation_t *entries;
	void *extra;
} root_heap_t;

/* definitions for root sieve */

#define MAX_CRT_FACTORS 10

typedef struct {
	uint16 score;
	uint64 x;
	uint64 y;
	uint64 z;
} lattice_t;

#define MAX_ROOTS 64

typedef struct {
	uint8 power;
	uint8 num_roots;
	uint16 score[MAX_ROOTS];
	uint8 roots[MAX_ROOTS][3];
} hit_t;


void compute_lattices(hit_t *hitlist, uint32 num_lattice_primes,
			lattice_t *lattices, uint64 lattice_size,
			uint32 num_lattices, uint32 dim);
void compute_line_size(double max_norm, dpoly_t *apoly,
		  double dbl_p, double dbl_d, double direction[3],
		  double last_line_min_in, double last_line_max_in,
		  double *line_min, double *line_max);


/* sieve for selecting XYZ triplets */

typedef struct {
	uint64 lattice_size;
	sieve_prime_t lattice_primes[MAX_CRT_FACTORS];
	uint32 num_lattice_primes;

	uint32 num_lattices;
	lattice_t *lattices;

	int64 z_base;
	uint32 z_blocks;
	double *y_line_min;
	double *y_line_max;
} sieve_xyz_t;

void sieve_xyz_alloc(sieve_xyz_t *xyz);
void sieve_xyz_free(sieve_xyz_t *xyz);

/* root sieve for selecting XY planes */

typedef struct {
	uint64 lattice_size;
	sieve_prime_t lattice_primes[MAX_CRT_FACTORS];
	uint32 num_lattice_primes;

	mpz_t y_base;
	uint32 y_blocks;

	mpz_t mp_lattice_size;
	double dbl_lattice_size;

	/* degrees 4 and 5: y_base as an int64 when it fits (y_ok), and the
	   mpz_t fields made from it and lattice_size only when CPU code
	   needs them (mp_stale: rs_xy_mpz) */
	int64 y_base_si;
	uint32 y_ok;
	uint32 mp_stale;

	/* degree 6, a plane in words (d6_words): the xy lattice size
	   l_tot = lattice_size * the xyz one, y_base_si, and the plane's
	   residue classes; the mpz_t fields only for CPU code (d6_mpz) */
	uint32 d6_words;
	uint64 l_tot;
	uint64 rx_si, ry_si;

	/* degree 6 only */
	uint16 curr_score;
	dpoly_t apoly;
	mpz_t crt0;
	mpz_t crt1;
	mpz_t resclass_x;
	mpz_t resclass_y;
	mpz_t tmp1, tmp2, tmp3, tmp4;

	/* degree 4 and 5 only */
	uint32 num_lattices;
	lattice_t *lattices;
	double *x_line_min;
	double *x_line_max;

} sieve_xy_t;

void sieve_xy_alloc(sieve_xy_t *xy);
void sieve_xy_free(sieve_xy_t *xy);


/* root sieve for selecting X lines */

typedef struct {
	uint64 lattice_size;
	sieve_prime_t lattice_primes[MAX_CRT_FACTORS];
	uint32 num_lattice_primes;

	dpoly_t apoly;

	double last_line_min;
	double last_line_max;
	uint16 curr_score;

	mpz_t x_base;
	uint32 x_blocks;

	mpz_t mp_lattice_size;
	double dbl_lattice_size;
	mpz_t crt0;
	mpz_t crt1;
	mpz_t resclass;
	mpz_t tmp1;

	/* degrees 4 and 5: mp_lattice_size (xy lattice_size times
	   lattice_size) made only when CPU code needs it (rs_x_mpz) */
	uint32 mp_stale;

	/* degree 6 in words: the CRT constants (128 bits, low first);
	   the mpz_t fields stale until d6_mpz() */
	uint64 crt_w0[2], crt_w1[2];
	uint32 d6_stale;
} sieve_x_t;

void sieve_x_alloc(sieve_x_t *x);
void sieve_x_free(sieve_x_t *x);

#define UNROLL 4
#define DEFAULT_BLOCK_SIZE  8192

typedef struct {
	poly_rootopt_t *data;

	uint32 num_primes;
	sieve_prime_t *primes;

	double sieve_bias; 
	double random_root_score;

	uint16 *sieve_block;

	dpoly_t apoly;
	double dbl_p;
	double dbl_d;
	double max_norm;

	root_heap_t root_heap;

	sieve_xyz_t xyzdata;
	sieve_xy_t xydata;
	sieve_x_t xdata;

	mpz_t curr_x, curr_y;
	int64 curr_z;

	/* the batched root sieve (root_sieve_batch.c): on for this run,
	   and its state */
	uint32 batched;
	void *batch;
	uint32 roots_on_host;	/* 0: the GPU setup kept them there */
} root_sieve_t;

/* batched root sieve (root_sieve_batch.c). The same search as the
   original code, reorganized so each polynomial has many independent
   optimize_basic() calls at a time, made through optimize_basic_batch():
   - every norm bound's line size is searched at once (each from the
     original's first starting interval, not from the previous bound);
   - degree 5: the x-line sizes of all y-blocks are searched in three
     rounds (every 64th block, every 8th, the rest), each started from
     the interpolated sizes of the round before instead of from the
     previous block;
   - the candidates of all sieved lines are scored together.
   On for a thread that sets root_sieve_batched (the GPU engine), or
   for every thread with MSIEVE_ROOTOPT_BATCHED=1 */

#if defined(_MSC_VER)
extern __declspec(thread) uint32 root_sieve_batched;
#else
extern _Thread_local uint32 root_sieve_batched;
#endif

/* one compute_line_size() search: input and result */
typedef struct {
	dpoly_t apoly;
	double direction[3];
	double max_norm;
	double last_min, last_max;	/* starting interval */
	double line_min, line_max;	/* result; line_min >= line_max
					   if the line is empty */
} line_job_t;

/* run n compute_line_size() searches together; each evaluates the
   same points as compute_line_size() with the same input */
void compute_line_sizes(line_job_t *jobs, uint32 n,
			double dbl_p, double dbl_d);

uint32 rs_batch_wanted(void);
void rs_batch_free(root_sieve_t *rs);
void rs_xlines_deg5(root_sieve_t *rs, uint32 y_blocks, int64 y0,
			uint64 lattice_size);
void rs_ylines_deg6(root_sieve_t *rs, uint32 z_blocks, int64 z0,
		uint64 lattice_size);
void rs_cand_add(root_sieve_t *rs, dpoly_t *apoly, double alpha);
void rs_cand_flush(root_sieve_t *rs);
void rs_cand_reset(root_sieve_t *rs);
uint32 rs_line_add(root_sieve_t *rs);

/* GPU line sieve: a run of lines whose coordinates the GPU forms from
   these constants and each line's parts (line_sieve_core.h); NULL is 0,
   for crt1 1. rs_line_parts() returns 0 if the line must be given as
   numbers instead (x_base, resclass, curr_y set as before) */
void rs_xy_mpz(sieve_xy_t *xy);
void rs_x_mpz(root_sieve_t *rs);
double rs_lattice_d(uint64 a, uint64 b);
double rs_i128_d(int64 a, uint64 b, uint64 c, uint64 d);
void rs_line_run(root_sieve_t *rs, uint32 ok, int64 y_base, uint64 ly,
		uint64 l1, uint64 l2,
		const uint64 *crt0, const uint64 *crt1);
void rs_lattice_crt(uint64 lxy, uint64 lx, uint64 *crt0, uint64 *crt1);
uint32 rs_line_parts(root_sieve_t *rs, uint64 ry, uint32 k, double line_min,
		uint64 r1, uint64 rx);
void rs_line_flush(root_sieve_t *rs);

/* the line sieve of the batched root sieve on n lines of one
   polynomial (line_sieve_core.h); runs on the CPU unless a thread sets
   this hook (the GPU engine) */
typedef void (*line_sieve_batch_hook_t)(const ls_poly_t *poly,
			const ls_prime_t *primes, const ls_power_t *powers,
			const ls_root_t *roots, const ls_line_t *lines,
			uint32 n, ls_out_t *out,
			const ls_run_t *runs, uint32 num_runs,
			const ls_cand_t **cand, const basic_result_t **norm,
			const so_basic_in_t **cand_poly);
#if defined(_MSC_VER)
extern __declspec(thread) line_sieve_batch_hook_t line_sieve_batch_hook;
#else
extern _Thread_local line_sieve_batch_hook_t line_sieve_batch_hook;
#endif

void root_sieve_init(root_sieve_t *rs);
void root_sieve_free(root_sieve_t *rs);
void root_sieve_run(poly_rootopt_t *data, double curr_norm,
				double alpha_proj);

uint64 find_lattice_size_z(double line_length);
void sieve_xyz_run_deg6(root_sieve_t *rs, uint64 lattice_size,
			double line_min, double line_max);

uint64 find_lattice_size_y(double line_length);
void sieve_xy_run_deg5(root_sieve_t *rs, uint64 lattice_size,
			double line_min, double line_max);
void sieve_xy_run_deg6(root_sieve_t *rs);

uint64 find_lattice_size_x(uint64 prev_lattice_size,
				double line_length);
void sieve_x_run_deg4(root_sieve_t *rs, uint64 lattice_size,
			double line_min, double line_max);
void sieve_x_run_deg5(root_sieve_t *rs);
void sieve_x_run_deg6(root_sieve_t *rs);

/* degree 6, batched: a plane's numbers in words, for the x-lines of
   all planes searched together (sieve_x_run_deg6_batch) */
typedef struct {
	dpoly_t apoly;			/* xy->apoly */
	int64 y_base_si;
	uint64 rx_si, ry_si;
	uint32 y_blocks;
	uint16 curr_score;
	int64 curr_z;
} d6_plane_t;

void sieve_x_run_deg6_batch(root_sieve_t *rs, d6_plane_t *planes,
				uint32 num_planes);

void root_sieve_line(root_sieve_t *rs);

void save_rotation_si(root_heap_t *heap, int64 x, int64 y,
			int64 z, float score);
void save_rotation(root_heap_t *heap, mpz_t x, mpz_t y,
		int64 z, float score);

void optimize_final_batch(root_heap_t *heap, poly_rootopt_t *data);

/* the root properties (and size scores) of n rotations of one
   polynomial (alpha_core.h;
   pl->pad holds the width, 512 or 1024 bits); set by the GPU engine */
typedef void (*alpha_batch_hook_t)(const ra_poly_t *pl,
				const ra_rot_t *rots, uint32 n,
				ra_res_t *res, mf_in_t *keep);

/* the polynomials to keep, rotated and translated exactly (rf_final) */
typedef void (*final_batch_hook_t)(const ra_poly_t *pl,
				const ra_final_t *jobs, uint32 n,
				ra_final_out_t *out);
#if defined(_MSC_VER)
extern __declspec(thread) final_batch_hook_t final_batch_hook;
#else
extern _Thread_local final_batch_hook_t final_batch_hook;
#endif
#if defined(_MSC_VER)
extern __declspec(thread) alpha_batch_hook_t alpha_batch_hook;
#else
extern _Thread_local alpha_batch_hook_t alpha_batch_hook;
#endif
/* v as two's complement 32-bit limbs of a bits-wide CGBN value */
uint32 ra_limbs(uint32_t *out, mpz_t v, uint32 bits);

/* the root sieve's setup on the GPU (alpha_core.h): roots of every
   prime power for the polynomial pl, in init_sieve()'s order; set by
   the GPU engine */
typedef void (*rs_setup_hook_t)(const ra_poly_t *pl, ls_root_t *roots,
				rs_setup_out_t *out);
#if defined(_MSC_VER)
extern __declspec(thread) rs_setup_hook_t rs_setup_hook;
#else
extern _Thread_local rs_setup_hook_t rs_setup_hook;
#endif
uint32 rs_setup_gpu(curr_poly_t *c, root_sieve_t *rs, uint32 deg,
			double sieve_bias);

/* the roots on the host, for the CPU code that needs them, if the GPU
   setup kept them on the GPU */
void rs_roots_need(root_sieve_t *rs);

/* the lattice choice on the GPU (lattice_core.h; built with
   RS_GPU_LATTICE). Each returns 0 if the GPU cannot do it (then the
   CPU code runs, after rs_roots_need) */
typedef void (*lattice_xy_hook_t)(const lt_set_t *set, lt_hit_t *hits);
typedef void (*lattice_x_hook_t)(const lt_xreq_t *req, const lt_lat_t *lats,
				lt_xout_t *out);
typedef void (*roots_fetch_hook_t)(ls_root_t *roots, uint32 n);
#if defined(_MSC_VER)
extern __declspec(thread) lattice_xy_hook_t lattice_xy_hook;
extern __declspec(thread) lattice_x_hook_t lattice_x_hook;
extern __declspec(thread) roots_fetch_hook_t roots_fetch_hook;
#else
extern _Thread_local lattice_xy_hook_t lattice_xy_hook;
extern _Thread_local lattice_x_hook_t lattice_x_hook;
extern _Thread_local roots_fetch_hook_t roots_fetch_hook;
#endif
uint32 rs_lattice_xy(root_sieve_t *rs, sieve_prime_t *lattice_primes,
			uint32 num_lattice_primes, hit_t *hitlist);
uint32 rs_lattice_x(root_sieve_t *rs, sieve_prime_t *lattice_primes,
			uint32 num_lattice_primes, uint32 y_ok, int64 y_base,
			uint64 xy_lattice_size, const lattice_t *lattices,
			uint32 num_lattices, uint32 y_blocks,
			uint64 x_lattice_size, lt_xout_t *out);

void optimize_final_batch_cpu(assess_t *assess, const mf_in_t *in, uint32 n,
				mf_out_t *out);

/*-------------------------------------------------------------------------*/

/* data for optimizing a single (ad, p, d) triplet */

typedef struct {
	curr_poly_t curr_poly;
	root_sieve_t root_sieve;
	assess_t assess;
	double size_cutoff;
} stage2_curr_data_t;

#ifdef __cplusplus
}
#endif

/* polynomial selection's wall-clock end (0: none; poly_skew.c sets it):
   past it the GPU root optimization starts no more jobs, and near it
   only jobs that can end in time; the jobs dropped (both in poly_skew.c,
   which every build compiles) */
extern double rootopt_wall_end;
extern uint64_t rootopt_dropped;

#endif /* !_STAGE2_H_ */
