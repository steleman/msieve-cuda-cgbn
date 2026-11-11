/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: relation.c 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#include <common.h>
#include "gnfs.h"
#include "filter/relpar.h"

/*--------------------------------------------------------------------*/
static INLINE
#ifdef __GNUC__
__attribute__((always_inline))
#endif
uint32 divide_factor_out(mpz_t polyval, uint64 p, 
			uint8 *factors, uint32 *array_size_in,
			uint32 *num_factors, uint32 compress,
			mpz_t tmp1, mpz_t tmp2, mpz_t tmp3) {

	/* read the rational factors. Note that the following
	   will work whether a given factor appears only once
	   or whether its full multiplicity is in the relation */

	uint32 i = *num_factors;
	uint32 array_size = *array_size_in;
	uint32 multiplicity = 0;

	if (p < ((uint64)1 << 32)) {
		while (1) {
			uint32 rem = mpz_tdiv_q_ui(tmp1, polyval, p);
			if (rem != 0 || mpz_cmp_ui(tmp1, 0) == 0)
				break;

			multiplicity++;
			mpz_swap(tmp1, polyval);
		}
	}
	else {
		uint64_2gmp(p, tmp1);
		while (1) {
			mpz_tdiv_qr(tmp2, tmp3, polyval, tmp1);
			if (mpz_cmp_ui(tmp3, 0) != 0 || 
			    mpz_cmp_ui(tmp2, 0) == 0)
				break;

			multiplicity++;
			mpz_swap(tmp2, polyval);
		}
	}
	if (i + multiplicity >= TEMP_FACTOR_LIST_SIZE)
		return 1;

	if (compress) {
		if (multiplicity & 1) {
			array_size = compress_p(factors, p, array_size);
			i++;
		}
	}
	else if (multiplicity) {
		i += multiplicity;
		while (multiplicity--)
			array_size = compress_p(factors, p, array_size);
	}
	*num_factors = i;
	*array_size_in = array_size;
	return 0;
}

/*--------------------------------------------------------------------*/
#define RELATION_TF_BOUND 1000

int32 nfs_read_relation(char *buf, factor_base_t *fb, 
			relation_t *r, uint32 *array_size_out,
			uint32 compress, mpz_t polyval,
			uint32 test_primality) {

	/* note that only the polynomials within the factor
	   base need to be initialized */

	uint32 i; 
	uint64 p;
	int64 a, atmp;
	uint64 b;
	char *tmp, *next_field;
	mpz_poly_t *rpoly = &fb->rfb.poly;
	mpz_poly_t *apoly = &fb->afb.poly;
	uint32 num_factors_r;
	uint32 num_factors_a;
	uint32 array_size = 0;
	uint8 *factors = r->factors;

	/* read the relation coordinates */

	a = strtoll(buf, &next_field, 10);
	tmp = next_field;
	if (tmp[0] != ',' || !isdigit(tmp[1]))
		return -1;

	b = strtoull(tmp+1, &next_field, 10);
	tmp = next_field;

	num_factors_r = 0;
	num_factors_a = 0;
	r->a = a;
	r->b = b;

	/* for free relations, store the roots and not
	   the prime factors */

	if (b == 0) {
		uint32 i;
		uint32 roots[MAX_POLY_DEGREE];
		uint32 high_coeff, num_roots;

		/* note that the finite field rootfinder must
		   be given p < 2^32 */

		p = (uint64)a;
		if (p == 0 || p >= ((uint64)1 << 32))
			return -2;

		if (test_primality && !mp_is_prime_1((uint32)p))
			return -98;

		array_size = compress_p(factors, p, array_size);

		num_roots = poly_get_zeros(roots, apoly,
						(uint32)p, &high_coeff, 0);
		if (num_roots != apoly->degree || high_coeff == 0)
			return -4;
		for (i = 0; i < num_roots; i++) {
			array_size = compress_p(factors, (uint64)roots[i], 
						array_size);
		}

		r->num_factors_r = 1;
		r->num_factors_a = num_roots;
		*array_size_out = array_size;
		return 0;
	}

	if (tmp[0] != ':')
		return -5;
	
	atmp = a % (int64)b;
	if (atmp < 0)
		atmp += b;

	if (mp_gcd_2((uint64)atmp, b) != 1)
		return -6;

	/* handle a rational factor of -1 */

	eval_poly(polyval, a, b, rpoly);
	if (mpz_cmp_ui(polyval, 0) == 0)
		return -6;
	if (mpz_cmp_ui(polyval, 0) < 0) {
		array_size = compress_p(factors, 0, array_size);
		num_factors_r++;
		mpz_abs(polyval, polyval);
	}

	/* read the rational factors (possibly an empty list) */

	if (isxdigit(tmp[1])) {
		do {
			p = strtoull(tmp + 1, &next_field, 16);

			if (test_primality && 
			    p > RELATION_TF_BOUND && 
			    p < ((uint64)1 << 32) &&
	    		    !mp_is_prime_1((uint32)p))
				return -98;

			if (p > 1 && divide_factor_out(polyval, p, 
						factors, &array_size,
						&num_factors_r, compress,
						rpoly->tmp1, rpoly->tmp2,
						rpoly->tmp3)) {
				return -8;
			}
			tmp = next_field;
		} while (tmp[0] == ',' && isxdigit(tmp[1]));
	}
	else {
		tmp++;
	}

	if (tmp[0] != ':')
		return -9;

	/* if there are rational factors still to be accounted
	   for, assume they are small and find them by trial division */

	for (i = p = 0; mpz_cmp_ui(polyval, 1) != 0 && 
				p < RELATION_TF_BOUND; i++) {

		p += prime_delta[i];
		if (divide_factor_out(polyval, p, factors, 
				&array_size, &num_factors_r, 
				compress, rpoly->tmp1, 
				rpoly->tmp2, rpoly->tmp3)) {
			return -10;
		}
	}

	if (mpz_cmp_ui(polyval, 1) != 0)
		return -11;

	/* read the algebraic factors */

	eval_poly(polyval, a, b, apoly);
	if (mpz_cmp_ui(polyval, 0) == 0)
		return -12;
	mpz_abs(polyval, polyval);

	if (isxdigit(tmp[1])) {
		do {
			p = strtoull(tmp + 1, &next_field, 16);

			if (test_primality &&
			    p > RELATION_TF_BOUND && 
			    p < ((uint64)1 << 32) &&
	    		    !mp_is_prime_1((uint32)p))
				return -98;

			if (p > 1 && divide_factor_out(polyval, p, 
						factors, &array_size,
						&num_factors_a, compress,
						apoly->tmp1, apoly->tmp2,
						apoly->tmp3)) {
				return -13;
			}
			tmp = next_field;
		} while (tmp[0] == ',' && isxdigit(tmp[1]));
	}

	/* if there are algebraic factors still to be accounted
	   for, assume they are small and find them by trial division */

	for (i = p = 0; mpz_cmp_ui(polyval, 1) != 0 && 
					p < RELATION_TF_BOUND; i++) {

		p += prime_delta[i];
		if (divide_factor_out(polyval, p, factors, 
				&array_size, &num_factors_a, 
				compress, apoly->tmp1,
				apoly->tmp2, apoly->tmp3)) {
			return -14;
		}
	}

	if (mpz_cmp_ui(polyval, 1) != 0)
		return -15;
	
	r->num_factors_r = num_factors_r;
	r->num_factors_a = num_factors_a;
	*array_size_out = array_size;
	return 0;
}

/*--------------------------------------------------------------------*/
uint32 find_large_ideals(relation_t *rel, 
			relation_lp_t *out, 
			uint32 filtmin_r, uint32 filtmin_a) {
	uint32 i;
	uint32 num_ideals = 0;
	uint32 array_size = 0;
	uint32 num_factors_r;
	int64 a = rel->a;
	uint64 b = rel->b;

	out->gf2_factors = 0;

	/* handle free relations */

	if (b == 0) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		if (p > filtmin_r) {
			ideal_t *ideal = out->ideal_list + num_ideals;

			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)p;
			ideal->r_hi = (uint16)(p >> 32);
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors++;
		}

		if (p > filtmin_a) {
			for (i = 0; i < rel->num_factors_a; i++) {
				ideal_t *ideal = out->ideal_list + 
							num_ideals + i;
				uint64 root = decompress_p(rel->factors,
							&array_size);

				ideal->p_lo = (uint32)compressed_p;
				ideal->p_hi = (uint16)(compressed_p >> 32);
				ideal->rat_or_alg = ALGEBRAIC_IDEAL;
				ideal->r_lo = (uint32)root;
				ideal->r_hi = (uint16)(root >> 32);
			}
			num_ideals += rel->num_factors_a;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors += rel->num_factors_a;
		}

		out->ideal_count = num_ideals;
		return num_ideals;
	}

	/* find the large rational ideals */

	num_factors_r = rel->num_factors_r;

	for (i = 0; i < num_factors_r; i++) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		/* if processing all the ideals, make up a
		   separate unique entry for rational factors of -1 */

		if (p == 0 && filtmin_r == 0) {
			ideal_t *ideal = out->ideal_list + num_ideals;
			ideal->p_lo = (uint32)(-1);
			ideal->p_hi = 0x7fff;
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)(-1);
			ideal->r_hi = 0xffff;
			num_ideals++;
			continue;
		}

		if (p > filtmin_r) {

			/* make a single unique entry for p, instead
			   of finding the exact number r for which
			   rational_poly(r) mod p is zero */

			ideal_t *ideal = out->ideal_list + num_ideals;

			if (num_ideals >= TEMP_FACTOR_LIST_SIZE)
				return TEMP_FACTOR_LIST_SIZE + 1;

			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)p;
			ideal->r_hi = (uint16)(p >> 32);
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {

			/* we only keep a count of the ideals that are
			   too small to list explicitly. NFS filtering
			   will work a little better if we completely
			   ignore the smallest ideals */

			out->gf2_factors++;
		}
	}

	/* repeat for the large algebraic ideals */

	for (i = 0; i < (uint32)rel->num_factors_a; i++) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		if (p > filtmin_a) {
			ideal_t *ideal = out->ideal_list + num_ideals;
			uint32 bmodp;

			if (num_ideals >= TEMP_FACTOR_LIST_SIZE)
				return TEMP_FACTOR_LIST_SIZE + 1;

			/* this time we have to find the exact r */

			bmodp = b % p;
			if (bmodp == 0) {
				ideal->r_lo = (uint32)p;
				ideal->r_hi = (uint16)(p >> 32);
			}
			else {
				uint64 root;
				int64 mapped_a = a % (int64)p;
				if (mapped_a < 0)
					mapped_a += p;

				root = (uint64)mapped_a;
				if (p < ((uint64)1 << 32)) {
					root = mp_modmul_1((uint32)root, 
						    mp_modinv_1(bmodp, 
						    	(uint32)p), (uint32)p);
				}
				else {
					root = mp_modmul_2(root, 
						    mp_modinv_2(bmodp, p), p);
				}
				ideal->r_lo = (uint32)root;
				ideal->r_hi = (uint16)(root >> 32);

			}
			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = ALGEBRAIC_IDEAL;
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors++;
		}
	}

	out->ideal_count = num_ideals;
	return num_ideals;
}

/*--------------------------------------------------------------------*/
static int bsearch_relation(const void *key, const void *rel) {
	relation_t *r = (relation_t *)rel;
	uint32 *k = (uint32 *)key;

	if ((*k) < r->rel_index)
		return -1;
	if ((*k) > r->rel_index)
		return 1;
	return 0;
}

static void remap_relation_numbers(msieve_obj *obj, 
				uint32 num_cycles, 
				la_col_t *cycle_list, 
				uint32 num_relations,
				relation_t *rlist) {
	uint32 i, j;

	/* walk through the list of cycles and convert
	   each occurence of a line number in the savefile
	   to an offset in the relation array */

	for (i = 0; i < num_cycles; i++) {
		la_col_t *c = cycle_list + i;

		for (j = 0; j < c->cycle.num_relations; j++) {

			/* since relations were read in order of increasing
			   relation index (= savefile line number), use 
			   binary search to locate relation j for this
			   cycle, then save a pointer to it */

			relation_t *rptr = (relation_t *)bsearch(
						c->cycle.list + j,
						rlist,
						(size_t)num_relations,
						sizeof(relation_t),
						bsearch_relation);
			if (rptr == NULL) {
				/* this cycle is corrupt somehow */
				logprintf(obj, "error: cannot locate "
						"relation %u\n", 
						c->cycle.list[j]);
				exit(-1);
			}
			else {
				c->cycle.list[j] = rptr - rlist;
			}
		}
	}
}

/*--------------------------------------------------------------------*/
static int compare_uint32(const void *x, const void *y) {
	uint32 *xx = (uint32 *)x;
	uint32 *yy = (uint32 *)y;
	if (*xx > *yy)
		return 1;
	if (*xx < *yy)
		return -1;
	return 0;
}

typedef struct {
	uint32 relidx;
	uint32 count;
} relcount_t;

/*--------------------------------------------------------------------*/
/* read relations relidx_list[0, num) (increasing relation numbers) from
   the relation store or the savefile; the count read */
static uint32 read_relation_list(msieve_obj *obj, factor_base_t *fb,
				uint32 *relidx_list, uint32 num_unique_relidx,
				uint32 compress, relation_t **rlist_out) {
	uint32 i, j, n;
	relation_t *rlist;
	savefile_t *savefile = &obj->savefile;
	relpar_t rp;
	relsrc_t src;

	relsrc_open(&src, savefile);

	/* read the list of relations */

	rlist = (relation_t *)xmalloc(num_unique_relidx * sizeof(relation_t));

	i = (uint32)(-1);
	j = 0;
	n = 0;
	relpar_init(&rp, fb, compress, 0, 0, 0, 0);
	for (;;) {
		uint32 done = n >= num_unique_relidx || !relsrc_next(&src);
		uint32 k;

		if (!done) {
			if (++i < relidx_list[n]) {

				/* relation not needed */

				continue;
			}
			n++;
			if (!relsrc_add(&src, &rp, i))
				continue;
		}

		/* read the block's relations on all CPUs, keep them in
		   order */

		relpar_run(&rp);
		for (k = 0; k < rp.num; k++) {
			relpar_item_t *it = rp.items + k;
			relation_t *r;

			if (it->status) {
				/* at this point, if the relation couldn't be
				   read then the filtering stage should have
				   found that out and skipped it */

				logprintf(obj, "error: relation %u corrupt\n",
						it->rel_index);
				exit(-1);
			}
			r = rlist + j++;
			*r = it->rel;
			r->rel_index = it->rel_index;
			r->factors = (uint8 *)xmalloc(it->array_size *
							sizeof(uint8));
			memcpy(r->factors, it->factors,
					it->array_size * sizeof(uint8));
		}
		if (done)
			break;
	}
	relpar_free(&rp);

	relsrc_close(&src);
	*rlist_out = rlist;
	return j;
}

static void nfs_get_cycle_relations(msieve_obj *obj, 
				factor_base_t *fb, uint32 num_cycles, 
				la_col_t *cycle_list, 
				uint32 *num_relations_out,
				relation_t **rlist_out,
				uint32 compress,
				uint32 dependency) {
	uint32 i, j;
	relation_t *rlist;

	hashtable_t unique_relidx;
	uint32 num_unique_relidx;
	uint32 *relidx_list;
	relcount_t *entry;

	hashtable_init(&unique_relidx, 
			(uint32)WORDS_IN(relcount_t), 
			(uint32)1);

	/* fill the hashtable */

	for (i = 0; i < num_cycles; i++) {
		la_col_t *c = cycle_list + i;
		uint32 num_relations = c->cycle.num_relations;
		uint32 *list = c->cycle.list;

		for (j = 0; j < num_relations; j++) {
			uint32 already_seen;
			entry = (relcount_t *)hashtable_find(
						&unique_relidx, list + j, 
						NULL, &already_seen);
			if (!already_seen)
				entry->count = 1;
			else
				entry->count++;
		}
	}

	/* convert the internal list of hashtable entries into
	   a list of 32-bit relation numbers. If reading in just
	   the relations in one dependency, squeeze out relations
	   that appear an even number of times */

	hashtable_close(&unique_relidx);
	num_unique_relidx = hashtable_get_num(&unique_relidx);
	entry = (relcount_t *)hashtable_get_first(&unique_relidx);
	relidx_list = unique_relidx.match_array;

	for (i = j = 0; i < num_unique_relidx; i++) {
		if (dependency == 0 || entry->count % 2 != 0)
			relidx_list[j++] = entry->relidx;

		entry = (relcount_t *)hashtable_get_next(
					&unique_relidx, entry);
	}
	num_unique_relidx = j;

	/* sort the list in order of increasing relation number */

	qsort(relidx_list, (size_t)num_unique_relidx, 
		sizeof(uint32), compare_uint32);

	logprintf(obj, "cycles contain %u unique relations\n", 
				num_unique_relidx);

	j = read_relation_list(obj, fb, relidx_list, num_unique_relidx,
				compress, &rlist);

	num_unique_relidx = *num_relations_out = j;
	logprintf(obj, "read %u relations\n", j);
	hashtable_free(&unique_relidx);
	*rlist_out = rlist;
}

/*--------------------------------------------------------------------*/
void nfs_read_cycles(msieve_obj *obj, 
			factor_base_t *fb,
			uint32 *num_cycles_out, 
			la_col_t **cycle_list_out, 
			uint32 *num_relations_out,
			relation_t **rlist_out,
			uint32 compress,
			uint32 dependency) {

	uint32 num_cycles;
	uint32 num_relations;
	la_col_t *cycle_list = NULL;
	relation_t *rlist;

	/* read the raw list of relation numbers for each cycle */

	read_cycles(obj, &num_cycles, &cycle_list, dependency, NULL);

	if (num_cycles == 0) {
		free(cycle_list);
		if (num_cycles_out != NULL)
			*num_cycles_out = 0;

		if (cycle_list_out != NULL)
			*cycle_list_out = NULL;

		if (num_relations_out != NULL)
			*num_relations_out = 0;

		if (rlist_out != NULL)
			*rlist_out = NULL;
		return;
	}

	/* finish if caller doesn't want the relations as well */

	if (fb == NULL || num_relations_out == NULL || rlist_out == NULL) {
		*num_cycles_out = num_cycles;
		*cycle_list_out = cycle_list;
		return;
	}

	/* now read the list of relations needed by the
	   list of cycles */

	nfs_get_cycle_relations(obj, fb, num_cycles, cycle_list, 
				&num_relations, &rlist, compress,
				dependency);

	*num_relations_out = num_relations;
	*rlist_out = rlist;

	/* if both the cycles and relations are wanted by 
	   callers, then modify the cycles to point to the
	   relations in memory and not on disk */

	if (num_cycles_out != NULL && cycle_list_out != NULL) {

		remap_relation_numbers(obj, num_cycles, cycle_list,
					num_relations, rlist);

		*num_cycles_out = num_cycles;
		*cycle_list_out = cycle_list;
	}
	else {
		free_cycle_list(cycle_list, num_cycles);
	}
}

/*--------------------------------------------------------------------*/
/* the bytes of a relation's compressed factor list */
static uint32 relation_factor_bytes(relation_t *r) {

	uint32 k, size = 0;
	uint32 count = (r->b == 0) ? (uint32)1 + r->num_factors_a :
			(uint32)r->num_factors_r + r->num_factors_a;

	for (k = 0; k < count; k++)
		decompress_p(r->factors, &size);
	return size;
}

/*--------------------------------------------------------------------*/
void nfs_read_dependencies(msieve_obj *obj, factor_base_t *fb,
			uint32 num_deps, const uint32 *deps,
			uint32 *num_relations_out, relation_t **rlist_out) {

	/* as nfs_read_cycles(compress = 0) for each dependency, but the
	   relations are read in one pass over the savefile */

	uint32 k, i, j;
	uint32 **lists = (uint32 **)xcalloc(num_deps, sizeof(uint32 *));
	uint32 *counts = (uint32 *)xcalloc(num_deps, sizeof(uint32));
	uint32 *all, num_all = 0, total = 0;
	relation_t *rall;

	for (k = 0; k < num_deps; k++) {
		uint32 num_cycles;
		la_col_t *cycle_list = NULL;
		hashtable_t unique_relidx;
		relcount_t *entry;
		uint32 n;

		num_relations_out[k] = 0;
		rlist_out[k] = NULL;
		read_cycles(obj, &num_cycles, &cycle_list, deps[k], NULL);
		if (num_cycles == 0) {
			free(cycle_list);
			continue;
		}

		/* the relations in an odd number of the dependency's
		   cycles, as nfs_get_cycle_relations finds them */

		hashtable_init(&unique_relidx, (uint32)WORDS_IN(relcount_t),
				(uint32)1);
		for (i = 0; i < num_cycles; i++) {
			la_col_t *c = cycle_list + i;

			for (j = 0; j < c->cycle.num_relations; j++) {
				uint32 already_seen;

				entry = (relcount_t *)hashtable_find(
						&unique_relidx,
						c->cycle.list + j,
						NULL, &already_seen);
				if (!already_seen)
					entry->count = 1;
				else
					entry->count++;
			}
		}
		hashtable_close(&unique_relidx);
		n = hashtable_get_num(&unique_relidx);
		lists[k] = (uint32 *)xmalloc(MAX(n, 1) * sizeof(uint32));
		entry = (relcount_t *)hashtable_get_first(&unique_relidx);
		for (i = j = 0; i < n; i++) {
			if (entry->count % 2 != 0)
				lists[k][j++] = entry->relidx;
			entry = (relcount_t *)hashtable_get_next(
						&unique_relidx, entry);
		}
		counts[k] = j;
		qsort(lists[k], (size_t)j, sizeof(uint32), compare_uint32);
		logprintf(obj, "dependency %u: cycles contain %u unique "
				"relations\n", deps[k], j);
		hashtable_free(&unique_relidx);
		free_cycle_list(cycle_list, num_cycles);
		total += j;
	}

	/* their union, read once */

	all = (uint32 *)xmalloc(MAX(total, 1) * sizeof(uint32));
	for (k = 0; k < num_deps; k++) {
		memcpy(all + num_all, lists[k], counts[k] * sizeof(uint32));
		num_all += counts[k];
	}
	qsort(all, (size_t)num_all, sizeof(uint32), compare_uint32);
	for (i = j = 0; i < num_all; i++) {
		if (j == 0 || all[i] != all[j - 1])
			all[j++] = all[i];
	}
	num_all = j;
	num_all = read_relation_list(obj, fb, all, num_all, 0, &rall);
	logprintf(obj, "read %u relations for %u dependencies\n",
			num_all, num_deps);

	/* each dependency's relations, copied (both lists increase) */

	for (k = 0; k < num_deps; k++) {
		relation_t *rl;
		uint32 m = 0;

		if (counts[k] == 0)
			continue;
		rl = (relation_t *)xmalloc(counts[k] * sizeof(relation_t));
		for (i = j = 0; i < counts[k]; i++) {
			while (j < num_all && rall[j].rel_index < lists[k][i])
				j++;
			if (j == num_all || rall[j].rel_index != lists[k][i]) {
				logprintf(obj, "error: relation %u missing\n",
						lists[k][i]);
				exit(-1);
			}
			{
				uint32 size = relation_factor_bytes(rall + j);

				rl[m] = rall[j];
				rl[m].factors = (uint8 *)xmalloc(MAX(size, 1));
				memcpy(rl[m].factors, rall[j].factors, size);
				m++;
			}
		}
		num_relations_out[k] = m;
		rlist_out[k] = rl;
	}

	nfs_free_relation_list(rall, num_all);
	for (k = 0; k < num_deps; k++)
		free(lists[k]);
	free(lists);
	free(counts);
	free(all);
}

/*--------------------------------------------------------------------*/
void nfs_free_relation_list(relation_t *rlist, uint32 num_relations) {

	uint32 i;

	for (i = 0; i < num_relations; i++)
		free(rlist[i].factors);
	free(rlist);
}

