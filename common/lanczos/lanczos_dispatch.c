/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (see gpu/lanczos_gpu.c)
--------------------------------------------------------------------*/

/* GPU builds: the block Lanczos backend interface of lanczos.h, sent
   to the GPU backend (gpu/lanczos_gpu.c) or the CPU's (cpu/, whose
   functions are renamed cpu_* in these builds). The choice is made per
   matrix in matrix_extra_init: the GPU unless MSIEVE_GPU_LA=0, the
   matrix has fewer than MSIEVE_GPU_LA_MIN columns (default 100000),
   there is no usable GPU or the matrix does not fit on it */

#include "lanczos.h"
#include "gpu/lanczos_gpu.h"

static uint32 la_on_gpu;

uint32 la_vectors_on_gpu(void) {
	return la_on_gpu;
}

void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p,
			uint32 first_block_size) {
	const char *e = getenv("MSIEVE_GPU_LA");
	const char *m = getenv("MSIEVE_GPU_LA_MIN");
	uint32 min_cols = (m != NULL) ? (uint32)strtoul(m, NULL, 10) : 100000;

	la_on_gpu = 0;
	if (!(e != NULL && strcmp(e, "0") == 0) && p->max_ncols >= min_cols &&
	    gpu_matrix_extra_init(obj, p)) {
		la_on_gpu = 1;
		return;
	}
	cpu_matrix_extra_init(obj, p, first_block_size);
}

void matrix_extra_free(packed_matrix_t *p) {
	if (la_on_gpu)
		gpu_matrix_extra_free(p);
	else
		cpu_matrix_extra_free(p);
	la_on_gpu = 0;
}

size_t packed_matrix_sizeof(packed_matrix_t *p) {
	return la_on_gpu ? gpu_packed_matrix_sizeof(p) :
			   cpu_packed_matrix_sizeof(p);
}

void mul_core(packed_matrix_t *A, void *x, void *b) {
	if (la_on_gpu) gpu_mul_core(A, x, b); else cpu_mul_core(A, x, b);
}

void mul_trans_core(packed_matrix_t *A, void *x, void *b) {
	if (la_on_gpu) gpu_mul_trans_core(A, x, b);
	else cpu_mul_trans_core(A, x, b);
}

void *vv_alloc(uint32 n, void *extra) {
	return la_on_gpu ? gpu_vv_alloc(n, extra) : cpu_vv_alloc(n, extra);
}

void vv_free(void *v) {
	if (la_on_gpu) gpu_vv_free(v); else cpu_vv_free(v);
}

void vv_copyin(void *dest, v_t *src, uint32 n) {
	if (la_on_gpu) gpu_vv_copyin(dest, src, n);
	else cpu_vv_copyin(dest, src, n);
}

void vv_copy(void *dest, void *src, uint32 n) {
	if (la_on_gpu) gpu_vv_copy(dest, src, n);
	else cpu_vv_copy(dest, src, n);
}

void vv_copyout(v_t *dest, void *src, uint32 n) {
	if (la_on_gpu) gpu_vv_copyout(dest, src, n);
	else cpu_vv_copyout(dest, src, n);
}

void vv_clear(void *v, uint32 n) {
	if (la_on_gpu) gpu_vv_clear(v, n); else cpu_vv_clear(v, n);
}

void vv_xor(void *dest, void *src, uint32 n) {
	if (la_on_gpu) gpu_vv_xor(dest, src, n);
	else cpu_vv_xor(dest, src, n);
}

void vv_mask(void *v, v_t mask, uint32 n) {
	if (la_on_gpu) gpu_vv_mask(v, mask, n); else cpu_vv_mask(v, mask, n);
}

void vv_mul_NxB_BxB_acc(packed_matrix_t *A, void *v, v_t *x,
			void *y, uint32 n) {
	if (la_on_gpu) gpu_vv_mul_NxB_BxB_acc(A, v, x, y, n);
	else cpu_vv_mul_NxB_BxB_acc(A, v, x, y, n);
}

void vv_mul_BxN_NxB(packed_matrix_t *A, void *x, void *y,
			v_t *xy, uint32 n) {
	if (la_on_gpu) gpu_vv_mul_BxN_NxB(A, x, y, xy, n);
	else cpu_vv_mul_BxN_NxB(A, x, y, xy, n);
}
