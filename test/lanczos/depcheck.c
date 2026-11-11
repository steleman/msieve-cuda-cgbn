/* MIT License, Copyright (c) 2026 Stefan Teleman (full text: common/lanczos/gpu/lanczos_gpu.c) */
/* depcheck <savefile>: every dependency in <savefile>.dep must combine
   the columns of <savefile>.mat into zero on every row */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int main(int argc, char **argv) {
	char buf[512];
	uint32_t nrows, nd, ncols, i;
	snprintf(buf, sizeof(buf), "%s.mat", argv[1]);
	FILE *m = fopen(buf, "rb");
	snprintf(buf, sizeof(buf), "%s.dep", argv[1]);
	FILE *d = fopen(buf, "rb");
	if (!m || !d) { printf("cannot open files\n"); return 2; }
	fread(&nrows, 4, 1, m); fread(&nd, 4, 1, m); fread(&ncols, 4, 1, m);
	uint64_t *dep = malloc((size_t)ncols * 8), *acc = calloc(nrows, 8), used = 0;
	if (fread(dep, 8, ncols, d) != ncols) { printf("short .dep\n"); return 2; }
	uint32_t dw = (nd + 31) / 32, *col = malloc(1 << 24);
	for (i = 0; i < ncols; i++) {
		uint32_t wgt;
		fread(&wgt, 4, 1, m); fread(col, 4, wgt + dw, m);
		used |= dep[i];
		for (uint32_t j = 0; j < wgt; j++) acc[col[j]] ^= dep[i];
		for (uint32_t j = 0; j < nd; j++)
			if ((col[wgt + j / 32] >> (j % 32)) & 1) acc[j] ^= dep[i];
	}
	uint64_t bad = 0;
	for (i = 0; i < nrows; i++) bad |= acc[i];
	int ndeps = __builtin_popcountll(used), good = __builtin_popcountll(used & ~bad);
	printf("%s: %d dependencies, %d of them null vectors%s\n", argv[1], ndeps,
		good, (ndeps > 0 && good == ndeps) ? " (all correct)" : " (FAILED)");
	return !(ndeps > 0 && good == ndeps);
}
