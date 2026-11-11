/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (see lanczos_gpu.c)
--------------------------------------------------------------------*/

/* the GPU backend of the block Lanczos iteration (lanczos_gpu.c) and
   the CPU backend's names in GPU builds (cpu/lanczos_cpu.h renames
   them); lanczos_dispatch.c chooses between the two per matrix */

#ifndef _COMMON_LANCZOS_GPU_LANCZOS_GPU_H_
#define _COMMON_LANCZOS_GPU_LANCZOS_GPU_H_

#include "../lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32 gpu_matrix_extra_init(msieve_obj *obj, packed_matrix_t *p);
void gpu_matrix_extra_free(packed_matrix_t *p);
size_t gpu_packed_matrix_sizeof(packed_matrix_t *p);
void gpu_mul_core(packed_matrix_t *A, void *x, void *b);
void gpu_mul_trans_core(packed_matrix_t *A, void *x, void *b);
void *gpu_vv_alloc(uint32 n, void *extra);
void gpu_vv_free(void *v);
void gpu_vv_copyin(void *dest, v_t *src, uint32 n);
void gpu_vv_copy(void *dest, void *src, uint32 n);
void gpu_vv_copyout(v_t *dest, void *src, uint32 n);
void gpu_vv_clear(void *v, uint32 n);
void gpu_vv_xor(void *dest, void *src, uint32 n);
void gpu_vv_mask(void *v, v_t mask, uint32 n);
void gpu_vv_mul_NxB_BxB_acc(packed_matrix_t *A, void *v, v_t *x,
			void *y, uint32 n);
void gpu_vv_mul_BxN_NxB(packed_matrix_t *A, void *x, void *y,
			v_t *xy, uint32 n);

void cpu_matrix_extra_init(msieve_obj *obj, packed_matrix_t *p,
			uint32 first_block_size);
void cpu_matrix_extra_free(packed_matrix_t *p);
size_t cpu_packed_matrix_sizeof(packed_matrix_t *p);
void cpu_mul_core(packed_matrix_t *A, void *x, void *b);
void cpu_mul_trans_core(packed_matrix_t *A, void *x, void *b);
void *cpu_vv_alloc(uint32 n, void *extra);
void cpu_vv_free(void *v);
void cpu_vv_copyin(void *dest, v_t *src, uint32 n);
void cpu_vv_copy(void *dest, void *src, uint32 n);
void cpu_vv_copyout(v_t *dest, void *src, uint32 n);
void cpu_vv_clear(void *v, uint32 n);
void cpu_vv_xor(void *dest, void *src, uint32 n);
void cpu_vv_mask(void *v, v_t mask, uint32 n);
void cpu_vv_mul_NxB_BxB_acc(packed_matrix_t *A, void *v, v_t *x,
			void *y, uint32 n);
void cpu_vv_mul_BxN_NxB(packed_matrix_t *A, void *x, void *y,
			v_t *xy, uint32 n);

/* a pinned host buffer (0, 1 or 2) of at least n vectors */
v_t *la_gpu_host_buf(uint32 which, size_t n);

/* CUDA-aware MPI: the grid operations take device vectors (and the
   GPU's work must be finished before an MPI call reads them) */
uint32 la_gpu_cuda_mpi(void);
void la_gpu_sync(void);
void la_gpu_uuid(uint8 out[16]);	/* the process's GPU */

#ifdef HAVE_MPI
/* the grid operations as rings (lanczos_ring.c): NCCL
   (MSIEVE_LA_NCCL=1, NCCL builds) or MPI on host buffers
   (MSIEVE_LA_RING_TEST=1); la_ring_mode() 0 when off */
uint32 la_ring_init(msieve_obj *obj, packed_matrix_t *p);
void la_ring_free(void);
uint32 la_ring_mode(void);
void la_ring_mul(packed_matrix_t *A, void *x, void *b, void *scratch,
		uint32 sym);
#endif

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_GPU_LANCZOS_GPU_H_ */
