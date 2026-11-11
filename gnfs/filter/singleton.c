/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: singleton.c 967 2014-06-26 02:48:22Z jasonp_sf $
--------------------------------------------------------------------*/

#include "filter.h"
#include "relpar.h"

/*--------------------------------------------------------------------*/
void nfs_write_lp_file(msieve_obj *obj, factor_base_t *fb,
			filter_t *filter, uint32 max_relations,
			uint32 pass) {

	/* read through the relation file and form a packed 
	   array of relation_ideal_t structures. This is the
	   first step to get the NFS relations into the 
	   algorithm-independent form that the rest of the
	   filtering will use */

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *relation_fp;
	FILE *final_fp;
	char buf[LINE_BUF_SIZE];
	size_t header_words;
	uint32 next_relation;
	uint32 curr_relation;
	uint32 num_relations;
	hashtable_t unique_ideals;
	relation_ideal_t packed_ideal;
	uint32 have_skip_list = (pass == 0);
	mpz_t scratch;
	relpar_t rp;
	relsrc_t src;

	logprintf(obj, "commencing singleton removal, initial pass\n");

	relsrc_open(&src, savefile);
	sprintf(buf, "%s.d", savefile->name);
	relation_fp = fopen(buf, "rb");
	if (relation_fp == NULL) {
		logprintf(obj, "error: can't open dup file\n");
		exit(-1);
	}
	sprintf(buf, "%s.lp", savefile->name);
	final_fp = fopen(buf, "wb");
	if (final_fp == NULL) {
		logprintf(obj, "error: can't open output LP file\n");
		exit(-1);
	}

	hashtable_init(&unique_ideals, (uint32)WORDS_IN(ideal_t), 0);
	header_words = (sizeof(relation_ideal_t) - 
			sizeof(packed_ideal.ideal_list)) / sizeof(uint32);

	/* for each relation that survived the duplicate removal */

	curr_relation = (uint32)(-1);
	next_relation = (uint32)(-1);
	num_relations = 0;
	mpz_init(scratch);
	relpar_init(&rp, fb, 1, 0, 1, filter->filtmin_r, filter->filtmin_a);
	fread(&next_relation, (size_t)1, 
			sizeof(uint32), relation_fp);

	for (;;) {
		uint32 eof = !relsrc_next(&src);
		uint32 stop = 0;
		uint32 k;

		if (!eof) {
			curr_relation++;
			if (max_relations && curr_relation >= max_relations) {
				stop = 1;
				goto run_block;
			}

			if (have_skip_list) {
				if (curr_relation == next_relation) {
					fread(&next_relation, sizeof(uint32), 
							(size_t)1, relation_fp);
					continue;
				}
			}
			else {
				if (curr_relation < next_relation)
					continue;
				fread(&next_relation, sizeof(uint32), 
						(size_t)1, relation_fp);
			}

			if (!relsrc_add(&src, &rp, curr_relation))
				continue;
		}

	run_block:
		/* read the block's relations and find their large
		   ideals on all CPUs, then number the ideals and write
		   the relations in order */

		relpar_run(&rp);
		for (k = 0; k < rp.num; k++) {
			relpar_item_t *it = rp.items + k;
			relation_lp_t *tmp_ideal = &it->lp;

			if (it->status != 0)
				continue;
			num_relations++;

			packed_ideal.rel_index = it->rel_index;
			packed_ideal.gf2_factors = tmp_ideal->gf2_factors;
			packed_ideal.ideal_count = tmp_ideal->ideal_count;

			/* map each ideal to a unique integer */

			for (i = 0; i < tmp_ideal->ideal_count; i++) {
				ideal_t *ideal = tmp_ideal->ideal_list + i;

				hashtable_find(&unique_ideals, ideal,
						packed_ideal.ideal_list + i,
						NULL);
			}

			/* dump the relation to disk */

			fwrite(&packed_ideal, sizeof(uint32),
				header_words + tmp_ideal->ideal_count, 
				final_fp);
		}
		if (eof || stop)
			break;
	}
	relpar_free(&rp);

	mpz_clear(scratch);
	filter->num_relations = num_relations;
	filter->num_ideals = hashtable_get_num(&unique_ideals);
	filter->relation_array = NULL;
	logprintf(obj, "memory use: %.1f MB\n",
			(double)hashtable_sizeof(&unique_ideals) / 1048576);
	hashtable_free(&unique_ideals);
	relsrc_close(&src);
	fclose(relation_fp);
	fclose(final_fp);

	sprintf(buf, "%s.lp", savefile->name);
	filter->lp_file_size = get_file_size(buf);

	sprintf(buf, "%s.d", savefile->name);
	if (remove(buf) != 0) {
		logprintf(obj, "error: can't delete dup file\n");
		exit(-1);
	}
}
