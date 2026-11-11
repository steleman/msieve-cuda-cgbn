/* MIT License, Copyright (c) 2026 Stefan Teleman (full text: gnfs/sieve/sieve_core.cu) */
/* batchreplay <dump> <reps> <min_prime> <max_prime> <lp_r> <lp_a>

   Feeds the cofactors a sieve run sent to batch factoring
   (MSIEVE_BATCH_DUMP=<dump> with -ns "lattice ...") through msieve's
   relation_batch_* as the siever does (a batch run every
   target_relations), <reps> times: the first time in the dump's order,
   then each time shuffled again. Prints the relations found and an
   order-independent hash of their (a, b). Batch factoring must give the
   same relations in any order; this is how its zero-remainder bug was
   found (2 of 20 orders lost 8 relations).

   The bounds are those of relation_batch_init in sieve_line.c: the
   smaller factor base maximum, MIN(3 << 27, MAX(LP1_max) / 4) and the
   two LP1_max (the c95: 2120429 16777216 67108864 67108864).

   Build (in the msieve directory, after make):
   gcc -std=c11 -D_GNU_SOURCE -O3 -Iinclude test/sieve/batchreplay.c \
       -o batchreplay libmsieve.a <the libraries msieve links> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <batch_factor.h>

static uint32 found;
static uint64 abhash;

static void
count_relation(savefile_t *savefile, int64 a, uint64 b,
		uint32 *factors_r, uint32 num_factors_r,
		uint32 lp_r[MAX_LARGE_PRIMES],
		uint32 *factors_a, uint32 num_factors_a,
		uint32 lp_a[MAX_LARGE_PRIMES])
{
	uint64 h = (uint64)a * 0x9e3779b97f4a7c15ull ^
			b * 0xc2b2ae3d27d4eb4full;

	(void)savefile; (void)factors_r; (void)num_factors_r; (void)lp_r;
	(void)factors_a; (void)num_factors_a; (void)lp_a;
	found++;
	abhash += h ^ (h >> 29);
}

int
main(int argc, char **argv)
{
	msieve_obj *obj;
	relation_batch_t rb;
	mpz_t r, alg;
	char line[512], **lines = NULL;
	uint32 nl = 0, al = 0;
	FILE *f;

	if (argc < 7) {
		fprintf(stderr, "usage: batchreplay <dump> <reps> <min_prime> "
				"<max_prime> <lp_r> <lp_a>\n");
		return 1;
	}
	f = fopen(argv[1], "r");
	if (f == NULL) {
		perror(argv[1]);
		return 1;
	}
	while (fgets(line, sizeof(line), f)) {
		if (nl == al) {
			al = al ? 2 * al : 1u << 20;
			lines = (char **)realloc(lines, al * sizeof(char *));
		}
		lines[nl++] = strdup(line);
	}
	fclose(f);

	obj = msieve_obj_new("1", 0, NULL, "batchreplay.log", NULL, 1, 2,
			0, cpu_generic, 0, 0, 1, 0, NULL);
	relation_batch_init(obj, &rb, (uint32)strtoul(argv[3], NULL, 10),
			(uint32)strtoul(argv[4], NULL, 10),
			(uint32)strtoul(argv[5], NULL, 10),
			(uint32)strtoul(argv[6], NULL, 10),
			NULL, count_relation);
	mpz_init(r);
	mpz_init(alg);

	for (int k = 0; k < atoi(argv[2]); k++) {
		uint32 total = 0;
		uint64 s = (uint64)(k + 1) * 0x9e3779b97f4a7c15ull;

		for (uint32 i = nl - 1; k > 0 && i > 0; i--) {
			uint32 j;
			char *tmp;

			s = s * 6364136223846793005ull + 1442695040888963407ull;
			j = (uint32)((s >> 33) % (i + 1));
			tmp = lines[i];
			lines[i] = lines[j];
			lines[j] = tmp;
		}
		found = 0;
		abhash = 0;
		for (uint32 i = 0; i < nl; i++) {
			long long a;
			unsigned long long b;
			char rs[256], as[256];

			if (sscanf(lines[i], "%lld %llu %255s %255s", &a, &b,
					rs, as) != 4)
				continue;
			mpz_set_str(r, rs, 10);
			mpz_set_str(alg, as, 10);
			relation_batch_add(a, b, NULL, 0, r, NULL, 0, alg, &rb);
			if (rb.num_relations >= rb.target_relations)
				total += relation_batch_run(&rb);
		}
		total += relation_batch_run(&rb);
		printf("order %d: %u cofactors, %u relations, hash %016llx\n",
			k, nl, total, (unsigned long long)abhash);
		fflush(stdout);
	}
	relation_batch_free(&rb);
	mpz_clear(r);
	mpz_clear(alg);
	msieve_obj_free(obj);
	return 0;
}
