/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: optimize.c 839 2013-01-13 14:31:02Z jasonp_sf $
--------------------------------------------------------------------*/

#include "stage2.h"

typedef double (*norm_t)(double *a, uint32 degree, double s);

/*-------------------------------------------------------------------------*/
static double
ifs_rectangular(double *a, uint32 degree, double s)
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

static double
ifs_radial(double *a, uint32 degree, double s)
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
 
/*----------------------------------------------------------------------*/
uint32
stage2_root_score(uint32 deg1, mpz_t *coeff1, 
		 uint32 prime_bound, double *score,
		 uint32 projective_only)
{
	uint32 i;
	uint32 status;
	mpz_poly_t apoly;

	mpz_poly_init(&apoly);

	apoly.degree = deg1;
	for (i = 0; i <= deg1; i++)
		mpz_set(apoly.coeff[i], coeff1[i]);

	if (projective_only)
		status = analyze_poly_roots_projective(&apoly, 
						prime_bound, score);
	else
		status = analyze_poly_roots(&apoly, prime_bound, score);

	mpz_poly_free(&apoly);
	return status;
}

static const double xlate_weights[6+1][6+1] = {
	{  1.,  1.,  1.,  1.,  1.,  1.,  1.},
	{  0.,  1.,  2.,  3.,  4.,  5.,  6.},
	{  0.,  0.,  1.,  3.,  6., 10., 15.},
	{  0.,  0.,  0.,  1.,  4., 10., 20.},
	{  0.,  0.,  0.,  0.,  1.,  5., 15.},
	{  0.,  0.,  0.,  0.,  0.,  1.,  6.},
	{  0.,  0.,  0.,  0.,  0.,  0.,  1.},
};

/*-------------------------------------------------------------------------*/
static void
translate_d(double *c, double *d, uint32 deg, double x)
{
	uint32 i, j;

	for (i = 0, x = -x; i < deg; i++) {

		const double *weights = xlate_weights[i];
		double accum = d[deg] * weights[deg];

		for (j = deg; j > i; j--) {
			accum = accum * x + d[j-1] * weights[j-1];
		}
		c[i] = accum;
	}
	c[i] = d[i];
}

/*-------------------------------------------------------------------------*/
static void
translate_dd(dd_t *c, dd_t *d, uint32 deg, double x)
{
	uint32 i, j;

	for (i = 0, x = -x; i < deg; i++) {

		const double *weights = xlate_weights[i];
		dd_t accum = dd_mul_d(d[deg], weights[deg]);

		for (j = deg; j > i; j--) {
			accum = dd_add_dd(dd_mul_d(accum, x),
					  dd_mul_d(d[j-1], weights[j-1]));
		}
		c[i] = accum;
	}
	c[i] = d[i];
}

/*-------------------------------------------------------------------------*/
static void
translate_gmp(curr_poly_t *c, mpz_t *gmp_c, uint32 deg,
		mpz_t *lin, int64 k)
{
	/* note that unlike the other translation functions, 
	   gmp_c[] and lin[] are overwritten */

	uint32 i, j;

	int64_2gmp(-k, c->gmp_help1);

	for (i = 0; i < deg; i++) {

		const double *weights = xlate_weights[i];

		mpz_set_d(c->gmp_help2, weights[deg]);
		mpz_mul(c->gmp_help2, c->gmp_help2, gmp_c[deg]);

		for (j = deg; j > i; j--) {
			mpz_set_d(c->gmp_help3, weights[j-1]);
			mpz_mul(c->gmp_help3, c->gmp_help3, gmp_c[j-1]);
			mpz_addmul(c->gmp_help3, c->gmp_help2, c->gmp_help1);
			mpz_swap(c->gmp_help2, c->gmp_help3);
		}
		mpz_swap(gmp_c[i], c->gmp_help2);
	}

	mpz_addmul(lin[0], lin[1], c->gmp_help1);
}

/*-------------------------------------------------------------------------*/
#define SKEWNESS 0
#define TRANSLATE_SIZE 1
#define ROTATE0 2
#define ROTATE1 3
#define ROTATE2 4

typedef struct {
	uint32 rotate_dim;
	dpoly_t *drpoly;
	dpoly_t *dapoly;

	ddpoly_t *rpoly;
	ddpoly_t *apoly;
	integrate_t *integ_aux;

	dickman_t *dickman_aux;
	double root_score_r;
	double root_score_a;

	uint32 num_real_roots;
	norm_t norm_callback;
} opt_data_t;

static double poly_xlate_callback(double *v, void *extra)
{
	opt_data_t *opt = (opt_data_t *)extra;
	dpoly_t *apoly = opt->dapoly;
	double translated[MAX_POLY_DEGREE + 1];
	double t = floor(v[TRANSLATE_SIZE] + 0.5);
	double s = v[SKEWNESS];

	translate_d(translated, apoly->coeff, apoly->degree, t);

	return ifs_rectangular(translated, apoly->degree, s);
}

static double poly_skew_callback(double *v, void *extra)
{
	opt_data_t *opt = (opt_data_t *)extra;
	dpoly_t *apoly = opt->dapoly;
	double s = v[SKEWNESS];

	return opt->norm_callback(apoly->coeff, apoly->degree, s);
}

static double poly_rotate_callback(double *v, void *extra)
{
	uint32 i;
	opt_data_t *opt = (opt_data_t *)extra;
	dpoly_t apoly = *(opt->dapoly);
	double translated[MAX_POLY_DEGREE + 1];
	double s = floor(v[SKEWNESS] + 0.5);
	double t = floor(v[TRANSLATE_SIZE] + 0.5);
	double r0 = opt->drpoly->coeff[0];
	double r1 = opt->drpoly->coeff[1];

	if (s < 1.0)
		return 1e200;

	for (i = 0; i <= opt->rotate_dim; i++) {
		double c = floor(v[ROTATE0 + i] + 0.5);
		apoly.coeff[i] += r0 * c;
		apoly.coeff[i+1] += r1 * c;
	}
	
	translate_d(translated, apoly.coeff, apoly.degree, t);
	return opt->norm_callback(translated, apoly.degree, s);
}

static double poly_murphy_callback(double *v, void *extra)
{
	opt_data_t *opt = (opt_data_t *)extra;
	ddpoly_t *apoly = opt->apoly;
	ddpoly_t *rpoly = opt->rpoly;
	ddpoly_t new_rpoly, new_apoly;
	double t = floor(v[TRANSLATE_SIZE] + 0.5);
	double s = v[SKEWNESS];
	double score = 0;

	if (s < 0.5)
		return 1e50;

	new_rpoly.degree = 1;
	new_rpoly.coeff[1] = rpoly->coeff[1];
	new_rpoly.coeff[0] = dd_sub_dd(rpoly->coeff[0],
					dd_mul_d(rpoly->coeff[1], t));

	new_apoly.degree = apoly->degree;
	translate_dd(new_apoly.coeff, apoly->coeff, apoly->degree, t);

	analyze_poly_murphy(opt->integ_aux, opt->dickman_aux,
				&new_rpoly, opt->root_score_r,
				&new_apoly, opt->root_score_a,
				s, &score, &opt->num_real_roots);

	return -score;
}

/*-------------------------------------------------------------------------*/
void
optimize_initial(curr_poly_t *c, uint32 deg, double *pol_norm, uint32 skew_only)
{
	uint32 rotate_dim = deg - 4;
	uint32 num_vars = rotate_dim + 3;
	opt_data_t opt_data;
	uint32 i, j;
	double best[MAX_VARS];
	double score, last_score, tol;
	dpoly_t rpoly, apoly;
	objective_func objective = poly_rotate_callback;

	opt_data.rotate_dim = rotate_dim;
	opt_data.drpoly = &rpoly;
	opt_data.dapoly = &apoly;
	opt_data.norm_callback = ifs_radial;

	best[SKEWNESS] = 1000;
	best[TRANSLATE_SIZE] = 0;
	best[ROTATE0] = 0;
	best[ROTATE1] = 0;
	best[ROTATE2] = 0;
	if (skew_only) {
		num_vars = 1;
		objective = poly_skew_callback;
	}
	else if (deg == 6) {
		score = optimize_initial_deg6(best, c, deg);
	}

	score = 1e200;
	tol = 1e-5;
	rpoly.degree = 1;
	for (i = 0; i <= 1; i++)
		rpoly.coeff[i] = mpz_get_d(c->gmp_lina[i]);
	apoly.degree = deg;
	for (i = 0; i <= deg; i++)
		apoly.coeff[i] = mpz_get_d(c->gmp_a[i]);

	for (i = 0; i < 2; i++) {

		uint32 num_minimize = 0;

		do {
			last_score = score;
			score = minimize(best, num_vars, tol, 40, 
					objective, &opt_data);

			for (j = 0; j <= rotate_dim; j++) {
				double cj = floor(best[ROTATE0 + j] + 0.5);
				mpz_set_d(c->gmp_help1, cj);
				mpz_addmul(c->gmp_a[j+1], c->gmp_help1, 
						c->gmp_p);
				mpz_submul(c->gmp_a[j], c->gmp_help1, 
						c->gmp_d);
			}
			translate_gmp(c, c->gmp_a, deg, c->gmp_lina,
					(int64)(best[TRANSLATE_SIZE] + 0.5));

			mpz_neg(c->gmp_d, c->gmp_lina[0]);
			for (j = 0; j <= 1; j++)
				rpoly.coeff[j] = mpz_get_d(c->gmp_lina[j]);
			for (j = 0; j <= deg; j++)
				apoly.coeff[j] = mpz_get_d(c->gmp_a[j]);
			best[TRANSLATE_SIZE] = 0;
			best[ROTATE0] = 0;
			best[ROTATE1] = 0;
			best[ROTATE2] = 0;

			if (++num_minimize >= 5)
				break;

		} while (fabs(score - last_score) > .001 * fabs(score));

		if (i == 0) {
			opt_data.norm_callback = ifs_rectangular;
			tol = 1e-5;
			score = ifs_rectangular(apoly.coeff, apoly.degree,
						best[SKEWNESS]);
		}
	}

	*pol_norm = sqrt(fabs(score));
#if 0
	printf("norm %.7e skew %lf\n", *pol_norm, best[SKEWNESS]);
	for (i = 0; i < 2; i++)
		gmp_printf("%+Zd\n", c->gmp_lina[i]);
	for (i = 0; i <= deg; i++)
		gmp_printf("%+Zd\n", c->gmp_a[i]);
#endif
}

/*-------------------------------------------------------------------------*/
#if defined(_MSC_VER)
__declspec(thread) optimize_basic_hook_t optimize_basic_hook;
#else
_Thread_local optimize_basic_hook_t optimize_basic_hook;
#endif

double
optimize_basic(dpoly_t *apoly, double *best_skewness,
		double *best_translation)
{
	opt_data_t opt_data;
	double best[MAX_VARS];
	double score;

	if (optimize_basic_hook != NULL)
		return optimize_basic_hook(apoly, best_skewness,
					best_translation);

	opt_data.dapoly = apoly;
	best[TRANSLATE_SIZE] = 0;
	best[SKEWNESS] = 1000;

	score = minimize(best, 2, 1e-5, 40, poly_xlate_callback, &opt_data);

	*best_translation = floor(best[TRANSLATE_SIZE] + 0.5);
	*best_skewness  = best[SKEWNESS];
	return sqrt(score);
}

/*-------------------------------------------------------------------------*/
static void
optimize_final_core(curr_poly_t *c, assess_t *assess, uint32 deg,
			double root_score_r, double root_score_a,
			double *best_score_out,
			double *best_skewness_out, 
			uint32 *num_real_roots_out)
{
	uint32 i;
	opt_data_t opt_data;
	double best[MAX_VARS];
	double score;
	ddpoly_t rpoly, apoly;

	opt_data.rpoly = &rpoly;
	opt_data.apoly = &apoly;
	opt_data.integ_aux = &assess->integ_aux;
	opt_data.dickman_aux = &assess->dickman_aux;
	opt_data.root_score_r = root_score_r;
	opt_data.root_score_a = root_score_a;
	rpoly.degree = 1;
	apoly.degree = deg;

	best[TRANSLATE_SIZE] = 0;
	best[SKEWNESS] = 1000;

	for (i = 0; i <= 1; i++)
		rpoly.coeff[i] = dd_gmp2dd(c->gmp_linb[i]);
	for (i = 0; i <= deg; i++)
		apoly.coeff[i] = dd_gmp2dd(c->gmp_b[i]);

	score = minimize(best, 2, 1e-5, 40, 
			poly_murphy_callback, &opt_data);

	translate_gmp(c, c->gmp_b, deg, c->gmp_linb, 
			(int64)(best[TRANSLATE_SIZE] + 0.5));

	*best_score_out = fabs(score);
	*best_skewness_out = best[SKEWNESS];
	*num_real_roots_out = opt_data.num_real_roots;
}

/*-------------------------------------------------------------------------*/
static void
get_bernstein_score(curr_poly_t *c, assess_t *assess,
		uint32 deg, double root_score, double *eptr)
{
	uint32 i;
	double size_score;
	ddpoly_t rpoly, apoly;

	root_score = pow(exp(root_score), -2./(deg + 1));

	rpoly.degree = 1;
	for (i = 0; i <= 1; i++)
		rpoly.coeff[i] = dd_gmp2dd(c->gmp_linb[i]);

	apoly.degree = deg;
	for (i = 0; i <= deg; i++)
		apoly.coeff[i] = dd_gmp2dd(c->gmp_b[i]);

	analyze_poly_size(&assess->integ_aux, 
			&rpoly, &apoly, &size_score);
	if (size_score == 0.0)
		printf("error: size score computation failed\n");

	*eptr = size_score * root_score;
}

/*-------------------------------------------------------------------------*/
/* the first part of optimize_final(): the rotation (x, y, z) applied to
   c->gmp_b and c->gmp_linb, their root properties and Bernstein score;
   1 if the Murphy E optimization should follow */

static uint32
optimize_final_prep(mpz_t x, mpz_t y, int64 z, poly_rootopt_t *data,
		const double *alpha_a_in, const double *bscore_in,
		double *alpha_r_out, double *alpha_a_out, double *bscore_out)
{
	uint32 i;
	uint32 deg = data->degree;
	double alpha_r, alpha_a, bscore;
	stage2_curr_data_t *s = (stage2_curr_data_t *)data->internal;
	curr_poly_t *c = &s->curr_poly;
	assess_t *assess = &s->assess;

	for (i = 0; i <= 1; i++)
		mpz_set(c->gmp_linb[i], c->gmp_lina[i]);

	for (i = 0; i <= deg; i++)
		mpz_set(c->gmp_b[i], c->gmp_a[i]);

	int64_2gmp(z, c->gmp_help1);
	mpz_addmul(c->gmp_b[3], c->gmp_p, c->gmp_help1);
	mpz_submul(c->gmp_b[2], c->gmp_d, c->gmp_help1);

	mpz_addmul(c->gmp_b[2], c->gmp_p, y);
	mpz_submul(c->gmp_b[1], c->gmp_d, y);

	mpz_addmul(c->gmp_b[1], c->gmp_p, x);
	mpz_submul(c->gmp_b[0], c->gmp_d, x);

	/* alpha_a_in: already computed (on the GPU) */
	if (alpha_a_in != NULL)
		alpha_a = *alpha_a_in;
	else if (stage2_root_score(deg, c->gmp_b, 
			data->murphy_p_bound, &alpha_a, 0))
		return 0;

	if (stage2_root_score(1, c->gmp_linb, 
			data->murphy_p_bound, &alpha_r, 0))
		return 0;

	if (alpha_a > -4.5)
		return 0;

	/* bscore_in: already computed (from the GPU's size score) */
	if (bscore_in != NULL)
		bscore = *bscore_in;
	else
		get_bernstein_score(c, assess, deg, alpha_a, &bscore);

	*alpha_r_out = alpha_r;
	*alpha_a_out = alpha_a;
	*bscore_out = bscore;
	return (bscore > data->min_e_bernstein);
}

/*-------------------------------------------------------------------------*/
void
optimize_final(mpz_t x, mpz_t y, int64 z, poly_rootopt_t *data)
{
	uint32 deg = data->degree;
	uint32 num_real_roots;
	double alpha_r, alpha_a, skewness, bscore, combined_score;
	stage2_curr_data_t *s = (stage2_curr_data_t *)data->internal;
	curr_poly_t *c = &s->curr_poly;
	assess_t *assess = &s->assess;

	if (!optimize_final_prep(x, y, z, data, NULL, NULL,
			&alpha_r, &alpha_a, &bscore))
		return;

	optimize_final_core(c, assess, deg, alpha_r, alpha_a, 
			&combined_score, &skewness,
			&num_real_roots);

	if (combined_score > data->min_e) {
		data->callback(data->callback_data, deg, c->gmp_b, 
				c->gmp_linb, skewness, bscore,
				alpha_a, combined_score, 
				num_real_roots);
	}
}

/*-------------------------------------------------------------------------*/
/* optimize_final() of every rotation in a root heap, with the Murphy E
   optimizations of all of them done together (murphy_core.h): on the
   CPU, the original code; through the hook, on the GPU */

#if defined(_MSC_VER)
__declspec(thread) murphy_batch_hook_t murphy_batch_hook;
#else
_Thread_local murphy_batch_hook_t murphy_batch_hook;
#endif

/* optimize_final_core() without the translation at the end */
static void
optimize_final_dd(assess_t *assess, const mf_in_t *in, mf_out_t *out)
{
	uint32 i;
	opt_data_t opt_data;
	double best[MAX_VARS];
	double score;
	ddpoly_t rpoly, apoly;

	opt_data.rpoly = &rpoly;
	opt_data.apoly = &apoly;
	opt_data.integ_aux = &assess->integ_aux;
	opt_data.dickman_aux = &assess->dickman_aux;
	opt_data.root_score_r = in->root_score_r;
	opt_data.root_score_a = in->root_score_a;
	opt_data.num_real_roots = 0;
	rpoly.degree = 1;
	apoly.degree = in->degree;

	best[TRANSLATE_SIZE] = 0;
	best[SKEWNESS] = 1000;

	for (i = 0; i <= 1; i++)
		rpoly.coeff[i] = dd_set_dd(in->r_hi[i], in->r_lo[i]);
	for (i = 0; i <= in->degree; i++)
		apoly.coeff[i] = dd_set_dd(in->a_hi[i], in->a_lo[i]);

	score = minimize(best, 2, 1e-5, 40, 
			poly_murphy_callback, &opt_data);

	out->score = fabs(score);
	out->skewness = best[SKEWNESS];
	out->translation = best[TRANSLATE_SIZE];
	out->num_real_roots = opt_data.num_real_roots;
}

/* the Murphy E optimizations of a batch on the CPU, with the given
   integrator and Dickman tables, or (NULL) their own */
void
optimize_final_batch_cpu(assess_t *assess, const mf_in_t *in, uint32 n,
			mf_out_t *out)
{
	assess_t own;

	if (assess == NULL) {
		assess_init(&own);
		assess = &own;
	}
	for (uint32 i = 0; i < n; i++)
		optimize_final_dd(assess, in + i, out + i);
	if (assess == &own)
		assess_free(&own);
}

/* the root properties (alpha) of every rotation in a heap through the
   hook (alpha_core.h); NULL (computed one by one on the CPU) without
   it, or if the numbers do not fit */

#if defined(_MSC_VER)
__declspec(thread) alpha_batch_hook_t alpha_batch_hook;
#else
_Thread_local alpha_batch_hook_t alpha_batch_hook;
#endif

/* v as two's complement 32-bit limbs, low first; 0 if it needs more
   than bits - 2 bits */
uint32
ra_limbs(uint32_t *out, mpz_t v, uint32 bits)
{
	uint32 n = bits / 32;
	size_t count = 0;

	if (mpz_sizeinbase(v, 2) > bits - 2)
		return 0;
	memset(out, 0, n * sizeof(uint32_t));
	mpz_export(out, &count, -1, sizeof(uint32_t), 0, 0, v);
	if (mpz_sgn(v) < 0) {
		uint32 carry = 1;

		for (uint32 i = 0; i < n; i++) {
			uint64 w = (uint64)(uint32_t)~out[i] + carry;

			out[i] = (uint32_t)w;
			carry = (uint32)(w >> 32);
		}
	}
	return 1;
}

static ra_res_t *
rs_alpha_batch(root_heap_t *heap, poly_rootopt_t *data, mf_in_t *keep)
{
	stage2_curr_data_t *s = (stage2_curr_data_t *)data->internal;
	curr_poly_t *c = &s->curr_poly;
	uint32 deg = data->degree;
	uint32 n = heap->num_entries;
	uint32 bits, need = 0;
	ra_poly_t *pl;
	ra_rot_t *rots;
	ra_res_t *res;

	if (alpha_batch_hook == NULL || n == 0 || deg > RA_MAX_DEG ||
	    data->murphy_p_bound != PRIME_BOUND || sizeof(long) < 8)
		return NULL;

	/* the width: room for a_i and the rotation terms */
	for (uint32 i = 0; i <= deg; i++)
		need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_a[i], 2));
	need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_p, 2) + 64);
	need = MAX(need, (uint32)mpz_sizeinbase(c->gmp_d, 2) + 64);
	bits = (need + 8 <= 512) ? 512 : (need + 8 <= 1024) ? 1024 : 0;
	if (bits == 0)
		return NULL;

	pl = (ra_poly_t *)xcalloc(1, sizeof(ra_poly_t));
	rots = (ra_rot_t *)xmalloc(n * sizeof(ra_rot_t));
	res = (ra_res_t *)xmalloc(n * sizeof(ra_res_t));

	pl->degree = deg;
	pl->pad = bits;
	pl->min_e_bernstein = data->min_e_bernstein;
	for (uint32 i = 0; i <= deg; i++)
		ra_limbs(pl->a[i], c->gmp_a[i], bits);
	ra_limbs(pl->p, c->gmp_p, bits);
	ra_limbs(pl->d, c->gmp_d, bits);
	for (uint32 i = 0; i <= 1; i++) {
		dd_t r = dd_gmp2dd(c->gmp_lina[i]);

		pl->r_hi[i] = r.hi;
		pl->r_lo[i] = r.lo;
	}

	for (uint32 i = 0; i < n; i++) {
		rotation_t *r = heap->entries + i;

		if (r->big) {
			free(pl);
			free(rots);
			free(res);
			return NULL;
		}
		rots[i].x = r->xs;
		rots[i].y = r->ys;
		rots[i].z = r->z;
		rots[i].poly = 0;
		rots[i].mf_index = RA_NO_MF;
	}

	alpha_batch_hook(pl, rots, n, res, keep);
	free(pl);
	free(rots);
	return res;
}

#if defined(_MSC_VER)
__declspec(thread) final_batch_hook_t final_batch_hook;
#else
_Thread_local final_batch_hook_t final_batch_hook;
#endif

typedef struct {
	mpz_t b[MAX_POLY_DEGREE + 1];
	mpz_t linb[2];
	double alpha_a;
	double bscore;
	uint32 gpu;		/* made from the GPU's input: no mpz here */
	uint32 rot;		/* its rotation in the heap */
} final_cand_t;

/* a polynomial to keep, and what the callback reports of it */
typedef struct {
	uint32 rot;
	int64 k;
	double skewness, bscore, alpha_a, score;
	uint32 num_real_roots;
} final_keep_t;

/* a two's complement number of RF_LIMBS words, low first */
static void
rf_to_mpz(mpz_t r, const uint32_t *limbs)
{
	uint32_t a[RF_LIMBS];
	uint32 neg = limbs[RF_LIMBS - 1] >> 31, carry = 1;

	for (uint32 i = 0; i < RF_LIMBS; i++) {
		a[i] = limbs[i];
		if (neg) {
			uint64 w = (uint64)(uint32_t)~a[i] + carry;

			a[i] = (uint32_t)w;
			carry = (uint32)(w >> 32);
		}
	}
	mpz_import(r, RF_LIMBS, -1, sizeof(uint32_t), 0, 0, a);
	if (neg)
		mpz_neg(r, r);
}

/* hand the kept polynomials to the callback: rotated and translated on
   the GPU (rf_final), or else as optimize_final() does it */
static void
optimize_final_keep(root_heap_t *heap, poly_rootopt_t *data,
		const final_keep_t *kp, uint32 n)
{
	stage2_curr_data_t *s = (stage2_curr_data_t *)data->internal;
	curr_poly_t *c = &s->curr_poly;
	uint32 deg = data->degree;
	ra_final_out_t *out = NULL;

	if (n == 0)
		return;
	if (final_batch_hook != NULL && sizeof(long) >= 8) {
		ra_poly_t *pl = (ra_poly_t *)xcalloc(1, sizeof(ra_poly_t));
		ra_final_t *jobs = (ra_final_t *)xcalloc(n, sizeof(ra_final_t));
		uint32 ok = (deg <= RA_MAX_DEG);

		/* room for the translation: k^deg times the coefficients */
		for (uint32 i = 0; ok && i <= deg; i++)
			ok = ra_limbs(pl->a[i], c->gmp_a[i], 1024) &&
			     mpz_sizeinbase(c->gmp_a[i], 2) < 700;
		ok = ok && ra_limbs(pl->p, c->gmp_p, 1024) &&
			ra_limbs(pl->d, c->gmp_d, 1024);
		pl->degree = deg;
		for (uint32 i = 0; ok && i < n; i++) {
			rotation_t *r = heap->entries + kp[i].rot;
			int64 ak = kp[i].k < 0 ? -kp[i].k : kp[i].k;

			if (r->big || (ak >> 40) != 0) {
				ok = 0;
				break;
			}
			jobs[i].x = r->xs;
			jobs[i].y = r->ys;
			jobs[i].z = r->z;
			jobs[i].k = kp[i].k;
		}
		if (ok) {
			/* in parts a GPU batch can take */
			out = (ra_final_out_t *)xmalloc(n *
						sizeof(ra_final_out_t));
			for (uint32 i = 0; i < n; i += RF_MAX_JOBS)
				final_batch_hook(pl, jobs + i,
					MIN(RF_MAX_JOBS, n - i), out + i);
		}
		free(pl);
		free(jobs);
	}

	for (uint32 i = 0; i < n; i++) {
		const final_keep_t *f = kp + i;

		if (out != NULL) {
			for (uint32 j = 0; j <= deg; j++)
				rf_to_mpz(c->gmp_b[j], out[i].b[j]);
			rf_to_mpz(c->gmp_linb[0], out[i].lin0);
			mpz_set(c->gmp_linb[1], c->gmp_lina[1]);
		}
		else {
			rotation_t *r = heap->entries + f->rot;
			double ar, aa, bs;

			rotation_mpz(r);
			optimize_final_prep(r->x, r->y, r->z, data,
					&f->alpha_a, &f->bscore,
					&ar, &aa, &bs);
			translate_gmp(c, c->gmp_b, deg, c->gmp_linb, f->k);
		}
		data->callback(data->callback_data, deg, c->gmp_b,
				c->gmp_linb, f->skewness, f->bscore,
				f->alpha_a, f->score, f->num_real_roots);
	}
	free(out);
}

void
optimize_final_batch(root_heap_t *heap, poly_rootopt_t *data)
{
	uint32 i, j, n = 0;
	uint32 deg = data->degree;
	stage2_curr_data_t *s = (stage2_curr_data_t *)data->internal;
	curr_poly_t *c = &s->curr_poly;
	final_cand_t *cand;
	mf_in_t *in;
	mf_out_t *out;
	ra_res_t *gpu;
	mf_in_t *keep;
	final_keep_t *kp;
	uint32 nk = 0;

	if (heap->num_entries == 0)
		return;

	/* the root properties and size scores of all of them at once,
	   on the GPU; with the Murphy E inputs of those that pass */
	keep = (mf_in_t *)xcalloc(RA_KEEP, sizeof(mf_in_t));
	gpu = rs_alpha_batch(heap, data, keep);

	cand = (final_cand_t *)xmalloc(heap->num_entries *
					sizeof(final_cand_t));
	kp = (final_keep_t *)xmalloc(heap->num_entries *
					sizeof(final_keep_t));
	in = (mf_in_t *)xcalloc(heap->num_entries, sizeof(mf_in_t));
	out = (mf_out_t *)xcalloc(heap->num_entries, sizeof(mf_out_t));

	for (i = 0; i < heap->num_entries; i++) {
		rotation_t *r = heap->entries + i;
		final_cand_t *k = cand + n;
		mf_in_t *f = in + n;
		double alpha_r, alpha_a, bscore;
		dd_t rdd[2], add[MAX_POLY_DEGREE + 1];
		dd_complex_t roots[2 * MAX_POLY_DEGREE];

		const double *alpha_in = NULL, *bscore_in = NULL;
		double gpu_bscore;

		if (gpu != NULL && (gpu[i].flags & RA_MURPHY)) {
			/* all of it done on the GPU */
			if (gpu[i].score > data->min_e) {
				final_keep_t *q = kp + nk++;

				q->rot = i;
				q->k = (int64)(gpu[i].translation + 0.5);
				q->skewness = gpu[i].skewness;
				q->bscore = gpu[i].size *
					pow(exp(gpu[i].alpha), -2./(deg + 1));
				q->alpha_a = gpu[i].alpha;
				q->score = gpu[i].score;
				q->num_real_roots = gpu[i].num_real_roots;
			}
			continue;
		}
		if (gpu != NULL) {
			/* from the GPU: stop as optimize_final_prep()
			   would, before any big-integer work */
			alpha_in = &gpu[i].alpha;
			if (gpu[i].alpha > -4.5)
				continue;
			if (gpu[i].size >= 0) {
				/* get_bernstein_score() */
				if (gpu[i].size == 0.0)
					printf("error: size score computation "
						"failed\n");
				gpu_bscore = gpu[i].size *
					pow(exp(gpu[i].alpha), -2./(deg + 1));
				if (!(gpu_bscore > data->min_e_bernstein))
					continue;
				bscore_in = &gpu_bscore;

				/* its Murphy E input, made on the GPU */
				if ((gpu[i].flags & RA_PASSED) &&
				    gpu[i].keep < RA_KEEP) {
					*f = keep[gpu[i].keep];
					k->alpha_a = gpu[i].alpha;
					k->bscore = gpu_bscore;
					k->gpu = 1;
					k->rot = i;
					n++;
					continue;
				}
			}
		}
		rotation_mpz(r);
		if (!optimize_final_prep(r->x, r->y, r->z, data,
				alpha_in, bscore_in,
				&alpha_r, &alpha_a, &bscore))
			continue;

		f->degree = deg;
		f->root_score_r = alpha_r;
		f->root_score_a = alpha_a;
		for (j = 0; j <= 1; j++) {
			rdd[j] = dd_gmp2dd(c->gmp_linb[j]);
			f->r_hi[j] = rdd[j].hi;
			f->r_lo[j] = rdd[j].lo;
		}
		for (j = 0; j <= deg; j++) {
			add[j] = dd_gmp2dd(c->gmp_b[j]);
			f->a_hi[j] = add[j].hi;
			f->a_lo[j] = add[j].lo;
		}

		/* the roots, for the GPU; analyze_poly_murphy() gives
		   no score if the root finder fails */
		if (find_poly_roots(rdd, 1, roots) ||
		    find_poly_roots(add, deg, roots + 1))
			continue;
		for (j = 0; j < deg; j++) {
			if (dd_cmp_d(roots[1 + j].i, 0.0) == 0)
				f->num_real_roots++;
		}
		for (j = 0; j < deg + 1; j++) {
			if (roots[j].i.hi < 0.0)
				continue;
			f->roots[f->num_roots++] = roots[j].r.hi;
		}

		for (j = 0; j <= deg; j++)
			mpz_init_set(k->b[j], c->gmp_b[j]);
		for (j = 0; j <= 1; j++)
			mpz_init_set(k->linb[j], c->gmp_linb[j]);
		k->alpha_a = alpha_a;
		k->bscore = bscore;
		k->gpu = 0;
		k->rot = i;
		n++;
	}

	if (n > 0) {
		if (murphy_batch_hook != NULL) {
			murphy_batch_hook(in, n, out);
		}
		else {
			for (i = 0; i < n; i++)
				optimize_final_dd(&s->assess, in + i, out + i);
		}
	}

	for (i = 0; i < n; i++) {
		final_cand_t *k = cand + i;

		if (out[i].score > data->min_e) {
			final_keep_t *q = kp + nk++;

			q->rot = k->rot;
			q->k = (int64)(out[i].translation + 0.5);
			q->skewness = out[i].skewness;
			q->bscore = k->bscore;
			q->alpha_a = k->alpha_a;
			q->score = out[i].score;
			q->num_real_roots = out[i].num_real_roots;
		}
		if (!k->gpu) {
			for (j = 0; j <= deg; j++)
				mpz_clear(k->b[j]);
			for (j = 0; j <= 1; j++)
				mpz_clear(k->linb[j]);
		}
	}

	/* the polynomials kept, in the order of the heap */
	if (nk > 1) {
		for (i = 1; i < nk; i++) {
			final_keep_t t = kp[i];

			for (j = i; j > 0 && kp[j - 1].rot > t.rot; j--)
				kp[j] = kp[j - 1];
			kp[j] = t;
		}
	}
	optimize_final_keep(heap, data, kp, nk);

	free(kp);
	free(keep);
	free(cand);
	free(in);
	free(out);
	free(gpu);
}
