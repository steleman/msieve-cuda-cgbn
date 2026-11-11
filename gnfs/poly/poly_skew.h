/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: poly_skew.h 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#ifndef _GNFS_POLY_POLY_SKEW_H_
#define _GNFS_POLY_POLY_SKEW_H_

#include "poly.h"

#ifdef __cplusplus
extern "C" {
#endif

/* external interface to skewed polynomial selector */

/* interface to stage 1 */

typedef void (*stage1_callback_t)(mpz_t ad, mpz_t p, mpz_t m, void *extra);

/* a polynomial size-optimized in stage 1 (on the GPU), for the next
   stage; extra is the callback data of the stage 2 thread */
typedef void (*stage1_sizeopt_callback_t)(void *extra, uint32 deg,
			mpz_t *alg_coeffs, mpz_t *rat_coeffs,
			double sizeopt_norm, double projective_alpha);

typedef struct {
	mpz_t gmp_N;
	mpz_t gmp_high_coeff_begin;
	mpz_t gmp_high_coeff_end;
	uint32 degree;
	double norm_max;
	uint32 deadline;
	stage1_callback_t callback;
	void *callback_data;

	/* optional: run stage 2 on num_stage2_threads threads, thread
	   i calling callback with stage2_callback_data[i] instead of
	   callback_data (0 = one thread, callback_data) */
	uint32 num_stage2_threads;
	void **stage2_callback_data;

	/* optional: size optimization inside stage 1 (GPU builds,
	   degrees 4 and 5). Hits then reach stage 2 already
	   optimized, through sizeopt_callback, and only if their
	   norm is at most sizeopt_norm */
	uint32 gpu_sizeopt;
	double sizeopt_norm;
	stage1_sizeopt_callback_t sizeopt_callback;
} poly_stage1_t;

void poly_stage1_init(poly_stage1_t *data, 
			stage1_callback_t callback,
			void *callback_data);
void poly_stage1_free(poly_stage1_t *data);
void poly_stage1_run(msieve_obj *obj, poly_stage1_t *data);


/* interface to size optimization */

typedef void (*sizeopt_callback_t)(uint32 deg, mpz_t *alg_coeffs, 
			mpz_t *rat_coeffs, double sizeopt_norm, 
			double projective_alpha, void *extra);
typedef struct {
	mpz_t gmp_N;
	uint32 degree;
	double max_stage1_norm;
	double max_sizeopt_norm;

	void *internal;

	sizeopt_callback_t callback;
	void *callback_data;
} poly_sizeopt_t;

void poly_sizeopt_init(poly_sizeopt_t *data,
		      sizeopt_callback_t callback,
		      void *callback_data);
void poly_sizeopt_free(poly_sizeopt_t *data);
void poly_sizeopt_run(poly_sizeopt_t *data, mpz_t ad, mpz_t p, mpz_t d);


/* interface to root optimization */

typedef void (*rootopt_callback_t)(void *extra, uint32 deg,
				mpz_t * coeff1, mpz_t * coeff2,
				double skewness, double size_score,
				double root_score, double combined_score,
				uint32 num_real_roots);
typedef struct {
	msieve_obj *obj;

	mpz_t gmp_N;
	uint32 degree;
	uint32 murphy_p_bound;
	double max_sizeopt_norm;
	double min_e;
	double min_e_bernstein;

	void *internal;

	rootopt_callback_t callback;
	void *callback_data;
} poly_rootopt_t;

void poly_rootopt_init(poly_rootopt_t *data, msieve_obj *obj,
		      rootopt_callback_t callback,
		      void *callback_data);
void poly_rootopt_free(poly_rootopt_t *data);
void poly_rootopt_run(poly_rootopt_t *data, mpz_t *alg_coeffs, 
			mpz_t *rat_coeffs, double sizeopt_norm, 
			double projective_alpha);

#ifdef __cplusplus
}
#endif

#endif /* _GNFS_POLY_POLY_SKEW_H_ */
