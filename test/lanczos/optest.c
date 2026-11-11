/* MIT License, Copyright (c) 2026 Stefan Teleman (full text: common/lanczos/gpu/lanczos_gpu.c) */
/* optest <nrows> <dense rows>: every GPU backend operation against the
   CPU backend's on the same random matrix and vectors */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <msieve.h>
#include <common.h>
#include "common/lanczos/lanczos.h"
#include "common/lanczos/gpu/lanczos_gpu.h"

static uint64 s = 88172645463325252ull;
static uint64 xr(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static int cmp(const void *a, const void *b) {
	uint32 x = *(const uint32 *)a, y = *(const uint32 *)b; return (x > y) - (x < y); }

static la_col_t *make(uint32 nrows, uint32 ncols, uint32 nd) {
	la_col_t *c = calloc(ncols, sizeof(la_col_t));
	uint32 dw = (nd + 31) / 32;
	s = 12345;
	for (uint32 i = 0; i < ncols; i++) {
		uint32 k = 30 + xr() % 61, m = 0;
		uint32 *d = malloc((k + dw) * sizeof(uint32));
		for (uint32 j = 0; j < k; j++) {
			double u = (double)(xr() >> 11) / 9007199254740992.0;
			d[j] = nd + (uint32)((nrows - nd) * u * u);
			if (d[j] >= nrows) d[j] = nrows - 1;
		}
		qsort(d, k, 4, cmp);
		for (uint32 j = 0; j < k; j++) if (!m || d[j] != d[m-1]) d[m++] = d[j];
		for (uint32 j = 0; j < dw; j++) {
			uint32 b = (uint32)xr() & (uint32)xr();	/* ~25% */
			if (j == dw - 1 && nd % 32) b &= (1u << (nd % 32)) - 1;
			d[m + j] = b;
		}
		c[i].data = d; c[i].weight = m;
	}
	return c;
}

static void setup(packed_matrix_t *p, la_col_t *A, uint32 nrows, uint32 ncols, uint32 nd) {
	memset(p, 0, sizeof(*p));
	p->unpacked_cols = A; p->nrows = p->max_nrows = nrows;
	p->ncols = p->max_ncols = ncols; p->num_dense_rows = nd; p->num_threads = 1;
}

static int same(const char *what, v_t *a, v_t *b, uint32 n) {
	uint32 bad = 0, first = 0;
	for (uint32 i = 0; i < n; i++)
		for (uint32 w = 0; w < VWORDS; w++)
			if (a[i].w[w] != b[i].w[w]) { if (!bad) first = i; bad++; break; }
	printf("%-28s %s", what, bad ? "DIFFER" : "same");
	if (bad) printf(" (%u of %u, first %u)", bad, n, first);
	printf("\n");
	return bad == 0;
}

int main(int argc, char **argv) {
	uint32 nrows = atoi(argv[1]), nd = atoi(argv[2]), ncols = (argc > 3) ? (uint32)atoi(argv[3]) : nrows + 100, n = MAX(nrows, ncols);
	msieve_obj *obj = msieve_obj_new("1", 0, "optest", "optest.log", NULL,
			1, 2, 0, cpu_generic, 32768, 1048576, 1, 0, NULL);
	packed_matrix_t pg, pc;
	setup(&pg, make(nrows, ncols, nd), nrows, ncols, nd);
	setup(&pc, make(nrows, ncols, nd), nrows, ncols, nd);
	if (!gpu_matrix_extra_init(obj, &pg)) { printf("no GPU\n"); return 1; }
	cpu_matrix_extra_init(obj, &pc, NUM_MEDIUM_ROWS);
	printf("matrix %u x %u, %u dense rows, CPU %s\n", nrows, ncols, nd,
		pc.unpacked_cols ? "unpacked" : "packed");

	v_t *hx = malloc(n * sizeof(v_t)), *hy = malloc(n * sizeof(v_t));
	v_t *r1 = malloc(n * sizeof(v_t)), *r2 = malloc(n * sizeof(v_t));
	v_t X[VBITS], o1[VBITS], o2[VBITS], mask;
	for (uint32 i = 0; i < n; i++) for (uint32 w = 0; w < VWORDS; w++) { hx[i].w[w] = xr(); hy[i].w[w] = xr(); }
	for (uint32 i = 0; i < VBITS; i++) for (uint32 w = 0; w < VWORDS; w++) X[i].w[w] = xr();
	for (uint32 w = 0; w < VWORDS; w++) mask.w[w] = xr();

	void *gx = gpu_vv_alloc(n, pg.extra), *gy = gpu_vv_alloc(n, pg.extra), *gb = gpu_vv_alloc(n, pg.extra);
	v_t *cx = cpu_vv_alloc(n, pc.extra), *cy = cpu_vv_alloc(n, pc.extra), *cb = cpu_vv_alloc(n, pc.extra);
	gpu_vv_copyin(gx, hx, n); gpu_vv_copyin(gy, hy, n);
	cpu_vv_copyin(cx, hx, n); cpu_vv_copyin(cy, hy, n);

	gpu_vv_clear(gb, n); cpu_vv_clear(cb, n);
	gpu_mul_core(&pg, gx, gb); cpu_mul_core(&pc, cx, cb);
	gpu_vv_copyout(r1, gb, nrows); cpu_vv_copyout(r2, cb, nrows);
	same("A x", r1, r2, nrows);
	{
		/* the reference, from a fresh copy */
		la_col_t *A = make(nrows, ncols, nd);
		uint32 dw = (nd + 31) / 32;
		v_t *ref = calloc(n, sizeof(v_t));
		for (uint32 i = 0; i < ncols; i++) {
			for (uint32 j = 0; j < A[i].weight; j++)
				ref[A[i].data[j]] = v_xor(ref[A[i].data[j]], hx[i]);
			for (uint32 j = 0; j < nd; j++)
				if ((A[i].data[A[i].weight + j / 32] >> (j % 32)) & 1)
					ref[j] = v_xor(ref[j], hx[i]);
		}
		same("  GPU A x vs reference", r1, ref, nrows);
		same("  CPU A x vs reference", r2, ref, nrows);
		memset(ref, 0, n * sizeof(v_t));
		for (uint32 i = 0; i < ncols; i++) {
			v_t a = v_zero;
			for (uint32 j = 0; j < A[i].weight; j++) a = v_xor(a, hx[A[i].data[j]]);
			for (uint32 j = 0; j < nd; j++)
				if ((A[i].data[A[i].weight + j / 32] >> (j % 32)) & 1) a = v_xor(a, hx[j]);
			ref[i] = a;
		}
		gpu_mul_trans_core(&pg, gx, gb); cpu_mul_trans_core(&pc, cx, cb);
		gpu_vv_copyout(r1, gb, ncols); cpu_vv_copyout(r2, cb, ncols);
		same("  GPU A^T x vs reference", r1, ref, ncols);
		same("  CPU A^T x vs reference", r2, ref, ncols);
		(void)dw;
		{
			/* GPU A x again, against sparse-only and transposed */
			v_t *rs = calloc(n, sizeof(v_t));
			for (uint32 i = 0; i < ncols; i++)
				for (uint32 j = 0; j < A[i].weight; j++)
					rs[A[i].data[j]] = v_xor(rs[A[i].data[j]], hx[i]);
			gpu_mul_core(&pg, gx, gb);
			gpu_vv_copyout(r1, gb, nrows);
			same("  GPU A x vs sparse only", r1, rs, nrows);
			same("  GPU A x vs A^T x ref", r1, ref, nrows);
			printf("  row 100: gpu %016llx ref %016llx; x[0] %016llx\n",
				(unsigned long long)r1[100].w[0], (unsigned long long)rs[100].w[0],
				(unsigned long long)hx[0].w[0]);
		}
	}

	gpu_mul_trans_core(&pg, gx, gb); cpu_mul_trans_core(&pc, cx, cb);
	gpu_vv_copyout(r1, gb, ncols); cpu_vv_copyout(r2, cb, ncols);
	same("A^T x", r1, r2, ncols);

	gpu_vv_mul_BxN_NxB(&pg, gx, gy, o1, n); cpu_vv_mul_BxN_NxB(&pc, cx, cy, o2, n);
	same("x^T y", o1, o2, VBITS);

	gpu_vv_copyin(gb, hy, n); cpu_vv_copyin(cb, hy, n);
	gpu_vv_mul_NxB_BxB_acc(&pg, gx, X, gb, n); cpu_vv_mul_NxB_BxB_acc(&pc, cx, X, cb, n);
	gpu_vv_copyout(r1, gb, n); cpu_vv_copyout(r2, cb, n);
	same("y ^= x X", r1, r2, n);

	gpu_vv_xor(gb, gx, n); cpu_vv_xor(cb, cx, n);
	gpu_vv_copyout(r1, gb, n); cpu_vv_copyout(r2, cb, n);
	same("b ^= x", r1, r2, n);

	gpu_vv_mask(gb, mask, n); cpu_vv_mask(cb, mask, n);
	gpu_vv_copyout(r1, gb, n); cpu_vv_copyout(r2, cb, n);
	same("b &= mask", r1, r2, n);
	return 0;
}
