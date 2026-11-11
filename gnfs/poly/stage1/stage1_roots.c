/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: stage1_roots.c 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#include <stdbool.h>
#include <stdint.h>
#include <stage1.h>

#define P_PRIME_LIMIT 0xfffff000
#define MAX_P_FACTORS 7
#define MAX_POWERS 4

/* the compiler's native 128-bit unsigned integer; __extension__
   keeps -Wpedantic quiet about the non-ISO type */
__extension__ typedef unsigned __int128 uint128_native;

/* structure for building arithmetic progressions */

typedef struct {
	uint32_t power;
	uint32_t roots[MAX_POLYSELECT_DEGREE];
} aprog_power_t;

typedef struct {
	uint32_t p;
	double p_log;
	double p_log_inv;

	/* number of 'degree-th roots' of N (mod p) */
	uint32_t num_roots;

	/* largest e for which p^e < 2^32 */
	uint32_t max_power;

	/* power p^e_j and its 'degree-th roots' of N (mod p^e_j) */
	aprog_power_t powers[MAX_POWERS];

	/* maximum product of other p which may be
	   combined with this p */
	uint64_t cofactor_max;
} aprog_t;

typedef struct {
	aprog_t *aprogs;
	uint32_t num_aprogs;
	uint32_t num_aprogs_alloc;
} aprog_list_t;

/* structures for finding arithmetic progressions via
   explicit enumeration */

typedef struct {
	/* number of distinct primes p_i in the current enum product */
	uint32_t num_factors;

	/* index into aprog list for each distinct prime p_i */
	uint32_t curr_factor[MAX_P_FACTORS + 1];

	/* exponent e_i of each prime p_i */
	uint32_t curr_power[MAX_P_FACTORS + 1];

	/* the 'running product' of p_1^e_1..p_{i-1}^e_{i-1} for
	   each i, or 1 if i==1 */
	uint64_t curr_prod[MAX_P_FACTORS + 1];
	double curr_prod_log[MAX_P_FACTORS + 1];

	/* number of roots in each 'running product' */
	uint32_t curr_num_roots[MAX_P_FACTORS + 1];
} p_enum_t;

#define ALGO_ENUM  0x1
#define ALGO_PRIME 0x2

/* a factory for building arithmetic progressions */

typedef struct {
	uint32_t num_roots_min;
	uint32_t num_roots_max;
	uint32_t avail_algos;
	bool defer_lift;     /* p < 2^32: leave roots mod p (GPU lifts) */
	uint32_t part_count; /* > 1: keep only p in this thread's share */
	uint32_t part_id;
	bool fb_only;
	uint32_t degree;
	uint64_t p_min, p_max;
	double p_min_log, p_max_log;
	mpz_poly_t tmp_poly;

	aprog_list_t aprog_data;

	prime_sieve_t p_prime;

	p_enum_t p_enum;

	/* incremental CRT for enumerated p (get_enum_roots): level i
	   holds the product of the first i+1 factors and its roots, in
	   the order combine_roots produces them; valid for the
	   factor/power recorded there and every level below */
	uint32_t lvl_valid;
	uint32_t lvl_factor[MAX_P_FACTORS];
	uint32_t lvl_power[MAX_P_FACTORS];
	uint64_t lvl_prod[MAX_P_FACTORS];
	uint32_t lvl_num[MAX_P_FACTORS];
	uint64_t *lvl_roots[MAX_P_FACTORS];

	mpz_t p, pp, nmodpp, m0modpp, tmp1, tmp2;
	mpz_t accum[MAX_P_FACTORS + 1];
	mpz_t roots[MAX_ROOTS];
} sieve_fb_t;

/*------------------------------------------------------------------------*/
static inline uint32_t
lift_root_32(uint32_t n, uint32_t r, uint32_t old_power, 
		uint32_t p, uint32_t d)
{
	/* given r, a d_th root of n mod old_power, compute
	   the corresponding root mod (old_power*p) via Hensel lifting */

	uint32_t p2 = old_power * p;
	uint64_t rsave = r;

	uint32_t q = mp_modsub_1(n % p2, mp_expo_1(r, d, p2), p2) / old_power;
	r = mp_modmul_1(d, mp_expo_1(r % p, d - 1, p), p);
	r = mp_modmul_1(q, mp_modinv_1(r, p), p);
	return rsave + old_power * r;
}

/*------------------------------------------------------------------------*/
void
sieve_fb_free(void *s_in)
{
	sieve_fb_t *s = (sieve_fb_t *)s_in;
	aprog_list_t *list = &s->aprog_data;

	free_prime_sieve(&s->p_prime);
	free(list->aprogs);

	for (uint32_t k = 0; k < MAX_P_FACTORS; k++)
		free(s->lvl_roots[k]);
	mpz_clear(s->p);
	mpz_clear(s->pp);
	mpz_clear(s->nmodpp);
	mpz_clear(s->m0modpp);
	mpz_clear(s->tmp1);
	mpz_clear(s->tmp2);
	mpz_poly_free(&s->tmp_poly);
	for (uint32_t i = 0; i <= MAX_P_FACTORS; i++)
		mpz_clear(s->accum[i]);
	for (uint32_t i = 0; i < MAX_ROOTS; i++)
		mpz_clear(s->roots[i]);

	free(s);
}

/*------------------------------------------------------------------------*/
static uint32_t
get_prime_roots(const poly_coeff_t *c, uint32_t p, uint32_t *roots,
		mpz_poly_t *tmp_poly)
{
	/* find all nonzero roots of (N' - x^d) mod p, where 
	   d is the desired polynomial degree and N' is the 
	   transformed version of N modulo p. We throw out roots
	   of zero because a zero root implies p divides the
	   degree or the leading algebraic poly coefficient, and
	   neither of these is allowed in later stages */

	mpz_tdiv_r_ui(tmp_poly->coeff[0], c->trans_N, p);

	if (mpz_cmp_ui(tmp_poly->coeff[0], 0) == 0) {
		/* when p divides trans_N, only a root of zero
		   exists, so skip this p */
		return 0;
	}
	uint32_t i;
	for (i = 1; i < c->degree; i++)
		mpz_set_ui(tmp_poly->coeff[i], 0);

	tmp_poly->degree = i;
	mpz_set_ui(tmp_poly->coeff[i], p - 1);

	uint32_t high_coeff;
	return poly_get_zeros(roots, tmp_poly, p, &high_coeff, 0);
}

/*------------------------------------------------------------------------*/
static void
sieve_add_aprog(sieve_fb_t *s, const poly_coeff_t *c, uint32_t p, 
		uint32_t fb_roots_min, uint32_t fb_roots_max)
{
	uint32_t degree = c->degree;
	uint32_t roots[MAX_POLYSELECT_DEGREE];
	aprog_list_t *list = &s->aprog_data;

	/* p will be able to generate arithmetic progressions;
	   add it to our list of them... */

	if (list->num_aprogs == list->num_aprogs_alloc) {
		list->num_aprogs_alloc *= 2;
		list->aprogs = (aprog_t *)xrealloc(list->aprogs,
						list->num_aprogs_alloc *
						sizeof(aprog_t));
	}

	/* ...if trans_N has any degree_th roots mod p */

	aprog_t *a = list->aprogs + list->num_aprogs;
	a->p = p;
	a->p_log = log(p);
	a->p_log_inv = 1.0 / a->p_log;
	uint32_t num_roots = get_prime_roots(c, p, roots, &s->tmp_poly);

	if (num_roots == 0 ||
	    num_roots < fb_roots_min ||
	    num_roots > fb_roots_max)
		return;

	list->num_aprogs++;
	a->num_roots = num_roots;

	uint32_t power = p;
	uint32_t power_limit = (uint32_t)(-1) / p;
	uint32_t i;
	for (i = 1; i < MAX_POWERS && power < power_limit; i++)
		power *= p;

	a->max_power = i;
	a->powers[0].power = p;

	for (uint32_t j = 0; j < num_roots; j++)
		a->powers[0].roots[j] = roots[j];

	/* add powers of p as well */

	power = p;
	for (i = 1; i < a->max_power; i++) {

		power *= p;
		a->powers[i].power = power;

		uint32_t nmodp = mpz_tdiv_ui(c->trans_N,
						(unsigned long)power);

		for (uint32_t j = 0; j < num_roots; j++)
			a->powers[i].roots[j] = lift_root_32(nmodp,
						a->powers[i - 1].roots[j],
						a->powers[i - 1].power,
						p, degree);
	}
}

/*------------------------------------------------------------------------*/
void *
sieve_fb_alloc(void)
{
	sieve_fb_t *s = (sieve_fb_t *)xcalloc(1, sizeof(sieve_fb_t));
	for (uint32_t k = 0; k < MAX_P_FACTORS; k++)
		s->lvl_roots[k] = (uint64_t *)xmalloc(MAX_ROOTS *
						sizeof(uint64_t));
	aprog_list_t *aprog = &s->aprog_data;

	mpz_init(s->p);
	mpz_init(s->pp);
	mpz_init(s->nmodpp);
	mpz_init(s->m0modpp);
	mpz_init(s->tmp1);
	mpz_init(s->tmp2);
	mpz_poly_init(&s->tmp_poly);
	for (uint32_t i = 0; i <= MAX_P_FACTORS; i++)
		mpz_init(s->accum[i]);
	for (uint32_t i = 0; i < MAX_ROOTS; i++)
		mpz_init(s->roots[i]);

	aprog->num_aprogs_alloc = 500;
	aprog->aprogs = (aprog_t *)xmalloc(aprog->num_aprogs_alloc *
						sizeof(aprog_t));
	return s;
}

/*------------------------------------------------------------------------*/
void 
sieve_fb_init(void *s_in, poly_coeff_t *c,
		uint32 factor_min, uint32 factor_max,
		uint32 fb_roots_min, uint32 fb_roots_max,
		uint32 fb_only)
{
	sieve_fb_t *s = (sieve_fb_t *)s_in;
	aprog_list_t *aprog = &s->aprog_data;

	s->degree = c->degree;
	s->fb_only = fb_only;
	s->lvl_valid = 0;               /* new aprogs: CRT cache stale */

	factor_max = MIN(factor_max, 1000000);

	if (factor_max <= factor_min)
		return;

	aprog->num_aprogs = 0;
	prime_sieve_t prime_sieve;
	init_prime_sieve(&prime_sieve, factor_min, factor_max);

	while (1) {
		uint32_t p = get_next_prime(&prime_sieve);

		if (p > factor_max)
			break;

		sieve_add_aprog(s, c, p, fb_roots_min, fb_roots_max);
	}

	free_prime_sieve(&prime_sieve);
}

/*------------------------------------------------------------------------*/
void 
sieve_fb_reset(void *s_in, uint64 p_min, uint64 p_max,
		uint32 num_roots_min, uint32 num_roots_max)
{
	sieve_fb_t *s = (sieve_fb_t *)s_in;
	aprog_t *aprogs = s->aprog_data.aprogs;
	uint32_t num_aprogs = s->aprog_data.num_aprogs;

	if (num_roots_max > MAX_ROOTS) {
		printf("error: num_roots_max exceeds %d\n", MAX_ROOTS);
		exit(-1);
	}

	s->p_min = p_min;
	s->p_max = p_max;
	s->p_min_log = log(p_min);
	s->p_max_log = log(p_max);
	s->num_roots_min = num_roots_min;
	s->num_roots_max = num_roots_max;
	s->avail_algos = 0;
	s->lvl_valid = 0;

	if (num_roots_max == 0) {
		/* nothing to do */
		return;
	}

	/* set up for finding arithmetic progressions by
	   enumerating combinations of small factors p_i
	   from the aprog list */

	if (num_aprogs > 0) {
		p_enum_t *p_enum = &s->p_enum; 

		s->avail_algos |= ALGO_ENUM;

		/* clear the enum state */

		p_enum->curr_factor[0] = 0;
		p_enum->num_factors = 0;
		p_enum->curr_power[0] = 0;
		p_enum->curr_prod[0] = 1;
		p_enum->curr_prod_log[0] = 0;
		p_enum->curr_num_roots[0] = 1;

		for (uint32_t i = 0; i < num_aprogs; i++) {
			aprog_t *a = aprogs + i;

			a->cofactor_max = p_max / a->p;
		}
	}

	/* set up for finding arithmetic progressions where
	   p is a large prime exceeding factor_max; do so if
	   that's allowed and the range of such p is higher
	   than what's already in the aprog list */

	if (!s->fb_only &&
	    p_min < P_PRIME_LIMIT &&
	    s->degree >= num_roots_min) {
		uint32_t last_p;

		if (num_aprogs == 0)
			last_p = 0;
		else
			last_p = aprogs[num_aprogs - 1].p;

		if (p_max > last_p) {
			s->avail_algos |= ALGO_PRIME;

			free_prime_sieve(&s->p_prime);
			init_prime_sieve(&s->p_prime, MAX(p_min, last_p + 1),
					 MIN(p_max, P_PRIME_LIMIT));
		}
	}
}

/*------------------------------------------------------------------------*/
/* 64-bit arithmetic for p < 2^32 (so p^2 < 2^64): the same values the
   mpz code computes, without GMP calls for one-word numbers */

static inline uint64_t
mulmod64(uint64_t a, uint64_t b, uint64_t m)
{
	return (uint64_t)((uint128_native)a * b % m);
}

static inline uint64_t
powmod64(uint64_t b, uint32_t e, uint64_t m)
{
	uint64_t r = 1 % m;
	b %= m;
	while (e) {
		if (e & 1)
			r = mulmod64(r, b, m);
		b = mulmod64(b, b, m);
		e >>= 1;
	}
	return r;
}

/* a^-1 mod m for m < 2^32; returns 0 when gcd(a, m) != 1 */
static inline uint64_t
invmod64(uint64_t a, uint64_t m)
{
	int64_t t0 = 0, t1 = 1;
	int64_t r0 = (int64_t)m, r1 = (int64_t)(a % m);

	while (r1 != 0) {
		int64_t q = r0 / r1, tmp;
		tmp = r0 - q * r1; r0 = r1; r1 = tmp;
		tmp = t0 - q * t1; t0 = t1; t1 = tmp;
	}
	if (r0 != 1)
		return 0;
	return (uint64_t)(t0 < 0 ? t0 + (int64_t)m : t0);
}

/* Montgomery arithmetic for an odd modulus m < 2^63 (R = 2^64):
   no divisions in the inner loops */

typedef struct {
	uint64_t m;       /* odd modulus, m < 2^63 */
	uint64_t minv;    /* -m^-1 mod 2^64 */
	uint64_t r2;      /* R^2 mod m */
	uint64_t one;     /* R mod m (1 in Montgomery form) */
} mont64_t;

static inline void
mont64_init(mont64_t *M, uint64_t m)
{
	uint64_t x = m;                         /* Newton: m * x = 1 */
	for (int i = 0; i < 5; i++)
		x *= 2 - m * x;
	M->m = m;
	M->minv = (uint64_t)0 - x;
	M->one = ((uint64_t)0 - m) % m;           /* 2^64 mod m */
	M->r2 = mulmod64(M->one, M->one, m);    /* 2^128 mod m */
}

/* a * b / R mod m, for a, b < m < 2^63 (so the sum below fits) */
static inline uint64_t
mont64_mul(const mont64_t *M, uint64_t a, uint64_t b)
{
	uint128_native t = (uint128_native)a * b;
	uint64_t u = (uint64_t)t * M->minv;
	uint64_t r = (uint64_t)((t + (uint128_native)u * M->m) >> 64);
	return (r >= M->m) ? r - M->m : r;
}

static inline uint64_t
mont64_to(const mont64_t *M, uint64_t a)
{
	return mont64_mul(M, a % M->m, M->r2);
}

static inline uint64_t
mont64_from(const mont64_t *M, uint64_t a)
{
	return mont64_mul(M, a, 1);
}

/* b^e in Montgomery form, b already in Montgomery form */
static inline uint64_t
mont64_pow(const mont64_t *M, uint64_t b, uint32_t e)
{
	uint64_t r = M->one;
	while (e) {
		if (e & 1)
			r = mont64_mul(M, r, b);
		b = mont64_mul(M, b, b);
		e >>= 1;
	}
	return r;
}

/* lift_roots for p < 2^32 with trans_N > 0 and trans_m0 >= 0;
   returns false (nothing changed) if a root is not invertible, so
   the caller can use the mpz code instead.

   For odd p (so p^2 is odd and below 2^63 or 2^64): Montgomery
   arithmetic mod p^2 and mod p, the exact division by p as a
   multiply by p^-1 mod 2^64, and one modular inverse per p for
   all its roots (Montgomery's batch inversion) */
static bool
lift_roots_64(sieve_fb_t *s, const poly_coeff_t *c, uint64_t p,
		uint32_t num_roots)
{
	uint32_t degree = s->degree;
	uint64_t pp = p * p;
	uint64_t nmodpp = mpz_fdiv_ui(c->trans_N, (unsigned long)pp);
	uint64_t m0modpp = mpz_fdiv_ui(c->trans_m0, (unsigned long)pp);
	uint64_t out[MAX_ROOTS];

	if ((p & 1) && pp < ((uint64_t)1 << 63)) {
		mont64_t Mpp, Mp;
		uint64_t t1[MAX_ROOTS], t2m[MAX_ROOTS], pre[MAX_ROOTS];

		mont64_init(&Mpp, pp);
		mont64_init(&Mp, p);
		uint64_t pinv = (uint64_t)0 - Mp.minv;  /* p^-1 mod 2^64 */

		/* t1 = (N - r^d mod p^2) / p, exact; t2 = d * r^(d-1)
		   mod p, kept in Montgomery form for the batch inverse */
		for (uint32_t i = 0; i < num_roots; i++) {
			uint64_t r = mpz_get_ui(s->roots[i]);
			uint64_t x = mont64_from(&Mpp,
				mont64_pow(&Mpp, mont64_to(&Mpp, r), degree));
			uint64_t d = (nmodpp >= x) ? nmodpp - x : nmodpp + pp - x;
			uint64_t rm = mont64_to(&Mp, r);

			t1[i] = d * pinv;               /* exact: p | d */
			t2m[i] = mont64_mul(&Mp, mont64_pow(&Mp, rm, degree - 1),
					mont64_to(&Mp, degree));
			out[i] = r;
		}

		/* batch inverse of all t2 mod p */
		uint64_t prodm = Mp.one;
		for (uint32_t i = 0; i < num_roots; i++) {
			pre[i] = prodm;
			prodm = mont64_mul(&Mp, prodm, t2m[i]);
		}
		uint64_t inv = invmod64(mont64_from(&Mp, prodm), p);
		if (inv == 0)
			return false;
		inv = mont64_to(&Mp, inv);              /* (prod)^-1, Mont. form */
		for (uint32_t i = num_roots; i-- > 0; ) {
			uint64_t inv_i = mont64_mul(&Mp, inv, pre[i]);  /* t2_i^-1 */

			inv = mont64_mul(&Mp, inv, t2m[i]);
			/* t1 * t2^-1 mod p: Montgomery product of a plain and
			   a Montgomery-form value gives the plain result */
			uint64_t root = mont64_mul(&Mp, t1[i], inv_i);  /* t1 < p */
			root = out[i] + root * p;
			out[i] = (root >= m0modpp) ? root - m0modpp :
						root + pp - m0modpp;
		}
	}
	else {
		for (uint32_t i = 0; i < num_roots; i++) {
			uint64_t r = mpz_get_ui(s->roots[i]);
			uint64_t x = powmod64(r, degree, pp);
			uint64_t t1 = (nmodpp >= x) ? nmodpp - x : nmodpp + pp - x;

			t1 /= p;
			uint64_t t2 = mulmod64(powmod64(r, degree - 1, p),
						degree, p);
			t2 = invmod64(t2, p);
			if (t2 == 0)
				return false;
			t1 = mulmod64(t1, t2, p);
			uint64_t root = r + t1 * p;
			root = (root >= m0modpp) ? root - m0modpp :
						root + pp - m0modpp;
			out[i] = root;
		}
	}
	for (uint32_t i = 0; i < num_roots; i++)
		mpz_set_ui(s->roots[i], (unsigned long)out[i]);
	return true;
}

/*------------------------------------------------------------------------*/
static void
lift_roots(sieve_fb_t *s, const poly_coeff_t *c, uint64_t p,
		uint32_t num_roots)
{
	/* we have num_roots arithmetic progressions mod p;
	   convert the progressions to be mod p^2, using
	   Hensel lifting, and then move the origin of the
	   result trans_m0 units to the left. */

	unsigned long degree = s->degree;

	if (p < ((uint64_t)1 << 32) && mpz_sgn(c->trans_N) > 0 &&
	    mpz_sgn(c->trans_m0) >= 0 && lift_roots_64(s, c, p, num_roots))
		return;

	uint64_2gmp(p, s->p);
	mpz_mul(s->pp, s->p, s->p);
	mpz_tdiv_r(s->nmodpp, c->trans_N, s->pp);
	mpz_tdiv_r(s->m0modpp, c->trans_m0, s->pp);

	for (uint32_t i = 0; i < num_roots; i++) {

		mpz_powm_ui(s->tmp1, s->roots[i], degree, s->pp);
		mpz_sub(s->tmp1, s->nmodpp, s->tmp1);
		if (mpz_cmp_ui(s->tmp1, (unsigned long)0) < 0)
			mpz_add(s->tmp1, s->tmp1, s->pp);
		mpz_tdiv_q(s->tmp1, s->tmp1, s->p);

		mpz_powm_ui(s->tmp2, s->roots[i], degree-1, s->p);
		mpz_mul_ui(s->tmp2, s->tmp2, degree);
		mpz_invert(s->tmp2, s->tmp2, s->p);

		mpz_mul(s->tmp1, s->tmp1, s->tmp2);
		mpz_tdiv_r(s->tmp1, s->tmp1, s->p);
		mpz_addmul(s->roots[i], s->tmp1, s->p);
		mpz_sub(s->roots[i], s->roots[i], s->m0modpp);
		if (mpz_cmp_ui(s->roots[i], (unsigned long)0) < 0)
			mpz_add(s->roots[i], s->roots[i], s->pp);
	}
}

/*------------------------------------------------------------------------*/
static uint32_t
combine_roots(sieve_fb_t *s, uint64_t p,
		uint32_t num_factors, const uint32_t p_i[MAX_P_FACTORS],
		const uint32_t num_roots[MAX_P_FACTORS],
		uint32_t roots[MAX_P_FACTORS][MAX_POLYSELECT_DEGREE])
{
	/* given a composite p and its factors p_i,
	   combine the roots mod p_i into roots mod
	   p using the Chinese Remainder Theorem */

	if (num_factors == 1) { 
		/* no CRT needed */

		for (uint32_t i = 0; i < num_roots[0]; i++)
			mpz_set_ui(s->roots[i], (unsigned long)roots[0][i]);

		return MIN(num_roots[0], s->num_roots_max);
	}

	/* fill in auxiliary CRT quantities */

	uint64_t prod[MAX_P_FACTORS];
	for (uint32_t i = 0; i < num_factors; i++) {
		prod[i] = p / p_i[i];
		prod[i] *= mp_modinv_1((uint32_t)(prod[i] % p_i[i]), p_i[i]);
	}

	/* p < 2^32: each term prod * root is below 2^64 and a sum of
	   up to 7 fits in 128 bits; same roots, same order */

	if (p < ((uint64_t)1 << 32)) {
		uint128_native acc[MAX_P_FACTORS + 1];
		uint32_t idx[MAX_P_FACTORS];
		uint32_t n = 0;

		acc[num_factors] = 0;
		for (uint32_t k = 0; k < num_factors; k++)
			idx[k] = num_roots[k] - 1;
		for (uint32_t k = num_factors; k-- > 0; )
			acc[k] = acc[k + 1] +
				(uint128_native)prod[k] * roots[k][idx[k]];
		while (1) {
			mpz_set_ui(s->roots[n++],
				(unsigned long)(uint64_t)(acc[0] % p));
			if (n == s->num_roots_max)
				return n;

			/* step the innermost index down, carrying outward
			   like the nested loops (each runs high to low) */

			uint32_t k;
			for (k = 0; k < num_factors; k++) {
				if (idx[k] > 0) {
					idx[k]--;
					break;
				}
				idx[k] = num_roots[k] - 1;
			}
			if (k == num_factors)
				return n;
			for (; ; k--) {
				acc[k] = acc[k + 1] + (uint128_native)
						prod[k] * roots[k][idx[k]];
				if (k == 0)
					break;
			}
		}
	}

	mpz_set_ui(s->accum[num_factors], (unsigned long)0);
	uint64_2gmp(p, s->p);

#if MAX_P_FACTORS > 7
#error "MAX_P_FACTORS exceeds 7"
#endif
	/* loop over all combinations of roots, changing one
	   root at a time. The accumulator value in the innermost
	   loop will exceed p by a few bits, so we need the
	   accum array to have wide integers. */

	uint32_t i0, i1, i2, i3, i4, i5, i6, i;
	i0 = i1 = i2 = i3 = i4 = i5 = i6 = i = 0;
	switch (num_factors) {
	case 7:
		for (i6 = num_roots[6] - 1; (int32_t)i6 >= 0; i6--) {
			uint64_2gmp(prod[6], s->accum[6]);
			mpz_mul_ui(s->accum[6], s->accum[6], 
						(unsigned long)roots[6][i6]);
			mpz_add(s->accum[6], s->accum[6], s->accum[7]);
	case 6:
		for (i5 = num_roots[5] - 1; (int32_t)i5 >= 0; i5--) {
			uint64_2gmp(prod[5], s->accum[5]);
			mpz_mul_ui(s->accum[5], s->accum[5], 
						(unsigned long)roots[5][i5]);
			mpz_add(s->accum[5], s->accum[5], s->accum[6]);
	case 5:
		for (i4 = num_roots[4] - 1; (int32_t)i4 >= 0; i4--) {
			uint64_2gmp(prod[4], s->accum[4]);
			mpz_mul_ui(s->accum[4], s->accum[4], 
						(unsigned long)roots[4][i4]);
			mpz_add(s->accum[4], s->accum[4], s->accum[5]);
	case 4:
		for (i3 = num_roots[3] - 1; (int32_t)i3 >= 0; i3--) {
			uint64_2gmp(prod[3], s->accum[3]);
			mpz_mul_ui(s->accum[3], s->accum[3], 
						(unsigned long)roots[3][i3]);
			mpz_add(s->accum[3], s->accum[3], s->accum[4]);
	case 3:
		for (i2 = num_roots[2] - 1; (int32_t)i2 >= 0; i2--) {
			uint64_2gmp(prod[2], s->accum[2]);
			mpz_mul_ui(s->accum[2], s->accum[2], 
						(unsigned long)roots[2][i2]);
			mpz_add(s->accum[2], s->accum[2], s->accum[3]);
	case 2:
		for (i1 = num_roots[1] - 1; (int32_t)i1 >= 0; i1--) {
			uint64_2gmp(prod[1], s->accum[1]);
			mpz_mul_ui(s->accum[1], s->accum[1], 
						(unsigned long)roots[1][i1]);
			mpz_add(s->accum[1], s->accum[1], s->accum[2]);

		for (i0 = num_roots[0] - 1; (int32_t)i0 >= 0; i0--) {
			uint64_2gmp(prod[0], s->accum[0]);
			mpz_mul_ui(s->accum[0], s->accum[0], 
						(unsigned long)roots[0][i0]);
			mpz_add(s->accum[0], s->accum[0], s->accum[1]);

			mpz_tdiv_r(s->accum[0], s->accum[0], s->p);
			mpz_set(s->roots[i++], s->accum[0]);

			if (i == s->num_roots_max)
				goto finished;
		}}}}}}}
	}

finished:
	return i;
}

/*------------------------------------------------------------------------*/
static uint32_t
get_enum_roots_crt(sieve_fb_t *s)
{
	/* incremental CRT (Garner): the enumeration changes only the
	   top of its stack between calls, so keep each level's product
	   and roots and recombine from the first level that changed.
	   Level i: P_i = P_{i-1} * p_i, and for each root s of p_i
	   (highest index first), for each root r of level i-1 in order:
	   r + P_{i-1} * ((s - r) * P_{i-1}^-1 mod p_i). That is the
	   order of combine_roots' nested loops, and keeping only the
	   first num_roots_max at each level keeps its truncation */

	aprog_t *aprogs = s->aprog_data.aprogs;
	p_enum_t *p_enum = &s->p_enum;
	uint32_t num_factors = p_enum->num_factors;
	uint32_t max = s->num_roots_max;

	for (uint32_t i = 0; i < num_factors; i++) {
		uint32_t f = p_enum->curr_factor[i];
		uint32_t e = p_enum->curr_power[i];
		aprog_t *a = aprogs + f;
		uint32_t pi = a->powers[e].power;
		uint32_t nr = a->num_roots;
		const uint32_t *ri = a->powers[e].roots;
		uint64_t *out = s->lvl_roots[i];
		uint32_t n = 0;

		if (i < s->lvl_valid && s->lvl_factor[i] == f &&
		    s->lvl_power[i] == e)
			continue;

		s->lvl_valid = i;       /* this level and above are stale */

		if (i == 0) {
			for (uint32_t j = nr; j-- > 0 && n < max; )
				out[n++] = ri[j];
			s->lvl_prod[0] = pi;
		}
		else {
			uint32_t P = (uint32_t)s->lvl_prod[i - 1];  /* < 2^32 */
			const uint64_t *prev = s->lvl_roots[i - 1];
			uint32_t nprev = s->lvl_num[i - 1];
			uint32_t inv = mp_modinv_1(P % pi, pi);

			for (uint32_t j = nr; j-- > 0 && n < max; ) {
				uint32_t sj = ri[j];
				for (uint32_t k = 0; k < nprev && n < max; k++) {
					uint32_t rm = (uint32_t)prev[k] % pi;
					uint32_t d = (sj >= rm) ? sj - rm :
							sj + pi - rm;
					uint64_t t = (uint64_t)d * inv % pi;
					out[n++] = prev[k] + (uint64_t)P * t;
				}
			}
			s->lvl_prod[i] = (uint64_t)P * pi;
		}
		s->lvl_num[i] = n;
		s->lvl_factor[i] = f;
		s->lvl_power[i] = e;
		s->lvl_valid = i + 1;
	}

	uint32_t k = s->lvl_num[num_factors - 1];
	for (uint32_t i = 0; i < k; i++)
		mpz_set_ui(s->roots[i],
			(unsigned long)s->lvl_roots[num_factors - 1][i]);
	return k;
}

/*------------------------------------------------------------------------*/
static uint32_t
get_enum_roots(sieve_fb_t *s)
{
	aprog_t *aprogs = s->aprog_data.aprogs;
	p_enum_t *p_enum = &s->p_enum;
	const uint32_t *curr_factor = p_enum->curr_factor;
	const uint32_t *curr_power = p_enum->curr_power;
	uint32_t num_factors = p_enum->num_factors;
	uint64_t p = p_enum->curr_prod[num_factors];

	/* several factors, p < 2^32: the incremental CRT. One factor
	   keeps the original path (its roots come out lowest first) */

	static int old_crt = -1;        /* MSIEVE_OLD_CRT=1: to compare */
	if (old_crt < 0)
		old_crt = (getenv("MSIEVE_OLD_CRT") != NULL);

	if (num_factors > 1 && p < ((uint64_t)1 << 32) && !old_crt)
		return get_enum_roots_crt(s);

	uint32_t p_i[MAX_P_FACTORS];
	uint32_t num_roots[MAX_P_FACTORS];
	uint32_t roots[MAX_P_FACTORS][MAX_POLYSELECT_DEGREE];

	for (uint32_t i = 0; i < num_factors; i++) {
		aprog_t *a = aprogs + curr_factor[i];

		/* p_i may be a power */

		p_i[i] = a->powers[curr_power[i]].power;
		num_roots[i] = a->num_roots;
		for (uint32_t j = 0; j < num_roots[i]; j++)
			roots[i][j] = a->powers[curr_power[i]].roots[j];
	}

	return combine_roots(s, p, num_factors, p_i, num_roots, roots);
}

/*------------------------------------------------------------------------*/
static uint64_t
get_next_enum(sieve_fb_t *s)
{
	/* find the next p by enumerating combinations of
	   aprogs p_i. We do this by maintaining a "stack"
	   of unique chosen p_i which are strictly in
	   non-increasing order, when read first-to-last.
	   While maintaining this ordering, and respecting
	   the maximum size of p, we "push" a new p_i
	   whenever possible. Otherwise, we "pop" p_i and,
	   as long as ordering can be maintained, "push"
	   the successor of the p_i which was last popped.
	   Multiple copies of the same p_i are combined
	   together into a single p_i (to simplify the CRT
	   later).

	   enumeration ends when there are no more aprogs
	   p_i available to be pushed

	   this method allows the p_i to be recovered
	   directly by reading the stack; no trial division
	   is necessary */

	p_enum_t *p_enum = &s->p_enum;
	uint32_t *curr_factor = p_enum->curr_factor;
	uint32_t *curr_power = p_enum->curr_power;
	uint32_t *curr_num_roots = p_enum->curr_num_roots;
	uint64_t *curr_prod = p_enum->curr_prod;
	double *curr_prod_log = p_enum->curr_prod_log;

	while (1) {

		uint32_t i = p_enum->num_factors;
		bool power_up = (i && curr_factor[i] == curr_factor[i - 1]);
		uint32_t num_roots = curr_num_roots[i];
		aprog_t *a = NULL;

		if (curr_factor[i] < s->aprog_data.num_aprogs)
			a = s->aprog_data.aprogs + curr_factor[i];

		if (a != NULL && curr_prod[i] <= a->cofactor_max
		    && !(power_up && ++curr_power[i - 1] >= a->max_power)
		    && (power_up || i < MAX_P_FACTORS)) {

			uint64_t p = curr_prod[i] * a->p;

			if (!power_up) {
				p_enum->num_factors = ++i;
				num_roots *= a->num_roots;
				curr_num_roots[i] = num_roots;
			}

			curr_prod[i] = p;
			curr_prod_log[i] = curr_prod_log[i - 1] +
					a->p_log * (curr_power[i - 1] + 1);
			curr_factor[i] = 0;
			curr_power[i] = 0;

			if (p >= s->p_min && num_roots >= s->num_roots_min)
				return p;
		}
		else if (i) {

			i--;
			a = s->aprog_data.aprogs + curr_factor[i];
			double p0_log_inv = s->aprog_data.aprogs[0].p_log_inv;
			double target = floor((s->p_max_log - curr_prod_log[i]
					- a->p_log) * p0_log_inv);
			uint32_t factor_max = (i ? curr_factor[i - 1]
					: s->aprog_data.num_aprogs) - 1;

			uint32_t j;
			for (j = curr_factor[i] + 1; j <= factor_max; j++) {
				a = s->aprog_data.aprogs + j;
				double cmp = floor((s->p_min_log
					- curr_prod_log[i]) * a->p_log_inv);

				if (cmp <= target)
					break;
			}
			curr_factor[i] = j;
			curr_power[i] = 0;
			p_enum->num_factors = i;
		}
		else {
			return P_SEARCH_DONE;
		}
	}
}

/*------------------------------------------------------------------------*/
uint64
sieve_fb_next(void *s_in, poly_coeff_t *c,
		root_callback callback, void *extra)
{
	/* main external interface */

	sieve_fb_t *s = (sieve_fb_t *)s_in;

	while (1) {
		uint64_t p;
		uint32_t num_roots;

		if (s->avail_algos & ALGO_ENUM) {

			/* first attempt to find a p by 
			   combining smaller suitable aprogs p_i */

			p = get_next_enum(s);

			if (p == P_SEARCH_DONE) {
				s->avail_algos &= ~ALGO_ENUM;
				continue;
			}

			/* producer threads split the enumeration by its
			   bottom factor; the others' p cost only the walk */
			if (s->part_count > 1 &&
			    s->p_enum.curr_factor[0] % s->part_count !=
							s->part_id)
				continue;

			num_roots = get_enum_roots(s);
		}
		else if (s->avail_algos & ALGO_PRIME) {

			/* then try to find a large prime p */

			uint32_t roots[MAX_POLYSELECT_DEGREE];

			p = get_next_prime(&s->p_prime);

			if (p >= s->p_max || p >= P_PRIME_LIMIT) {
				s->avail_algos &= ~ALGO_PRIME;
				continue;
			}
			if (s->part_count > 1 && p % s->part_count != s->part_id)
				continue;

			num_roots = get_prime_roots(c, p, roots,
						    &s->tmp_poly);
			num_roots = MIN(num_roots, s->num_roots_max);

			if (num_roots == 0 ||
			    num_roots < s->num_roots_min)
				continue;

			for (uint32_t i = 0; i < num_roots; i++)
				mpz_set_ui(s->roots[i],
					(unsigned long)roots[i]);
		}
		else {
			return P_SEARCH_DONE;
		}

		/* p found; generate all the arithmetic
		   progressions it allows and postprocess the
		   whole batch */

		if (!(s->defer_lift && p < ((uint64_t)1 << 32)))
			lift_roots(s, c, p, num_roots);
		callback(p, num_roots, s->roots, extra);

		return p;
	}
}

/*------------------------------------------------------------------------*/
void
sieve_fb_set_defer_lift(void *s_in, uint32 defer)
{
	((sieve_fb_t *)s_in)->defer_lift = defer;
}

/*------------------------------------------------------------------------*/
/* an independent copy of an initialized factory (its own aprog list,
   CRT cache and scratch), for a producer thread; part_count/part_id
   give it a 1/part_count share of what sieve_fb_next produces */
void *
sieve_fb_clone(void *s_in, uint32 part_count, uint32 part_id)
{
	sieve_fb_t *src = (sieve_fb_t *)s_in;
	sieve_fb_t *s = (sieve_fb_t *)sieve_fb_alloc();
	aprog_list_t *a = &s->aprog_data;

	s->degree = src->degree;
	s->fb_only = src->fb_only;
	s->defer_lift = src->defer_lift;
	s->part_count = part_count;
	s->part_id = part_id;
	s->lvl_valid = 0;

	a->num_aprogs = src->aprog_data.num_aprogs;
	a->num_aprogs_alloc = MAX(1, a->num_aprogs);
	free(a->aprogs);
	a->aprogs = (aprog_t *)xmalloc(a->num_aprogs_alloc * sizeof(aprog_t));
	memcpy(a->aprogs, src->aprog_data.aprogs,
			a->num_aprogs * sizeof(aprog_t));
	return s;
}
