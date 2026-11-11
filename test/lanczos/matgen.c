/* MIT License, Copyright (c) 2026 Stefan Teleman (full text: common/lanczos/gpu/lanczos_gpu.c) */
/* matgen <savefile> <nrows> <excess cols> <weight> <skew> <seed>:
   a random GF(2) matrix in msieve's .mat format (dump_matrix), NFS-like:
   column weights around <weight>, rows chosen as nrows * u^skew (dense
   at low rows), sorted and without repeats */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <msieve.h>
#include <common.h>

static uint64 s = 88172645463325252ull;
static uint64 xr(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static int cmp(const void *a, const void *b) {
	uint32 x = *(const uint32 *)a, y = *(const uint32 *)b;
	return (x > y) - (x < y);
}

int main(int argc, char **argv) {
	char *save = argv[1];
	uint32 nrows = atoi(argv[2]), ncols = nrows + atoi(argv[3]);
	uint32 w = atoi(argv[4]);
	double skew = atof(argv[5]);
	uint64 total = 0;
	s ^= (uint64)atoi(argv[6]) * 0x9E3779B97F4A7C15ull;

	msieve_obj *obj = msieve_obj_new("1", 0, save, "matgen.log", NULL,
			1, 2, 0, cpu_generic, 32768, 1048576, 1, 0, NULL);
	la_col_t *cols = (la_col_t *)calloc(ncols, sizeof(la_col_t));
	for (uint32 c = 0; c < ncols; c++) {
		uint32 k = w / 2 + (uint32)(xr() % (w + 1));	/* w/2 .. 3w/2 */
		uint32 *d = (uint32 *)malloc(k * sizeof(uint32)), m = 0;
		for (uint32 j = 0; j < k; j++) {
			double u = (double)(xr() >> 11) / 9007199254740992.0;
			uint32 r = (uint32)(nrows * pow(u, skew));
			d[j] = (r >= nrows) ? nrows - 1 : r;
		}
		qsort(d, k, sizeof(uint32), cmp);
		for (uint32 j = 0; j < k; j++)
			if (m == 0 || d[j] != d[m - 1]) d[m++] = d[j];
		cols[c].data = d;
		cols[c].weight = m;
		total += m;
	}
	dump_matrix(obj, nrows, 0, ncols, cols, total);
	printf("%s: %u x %u, %llu nonzeros (%.1f per column)\n", save, nrows,
		ncols, (unsigned long long)total, (double)total / ncols);
	return 0;
}
