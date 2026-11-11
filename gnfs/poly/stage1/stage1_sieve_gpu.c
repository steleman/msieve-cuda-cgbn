/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: stage1_sieve_gpu.c 1112 2026-11-02 02:13:30Z jasonp_sf $
--------------------------------------------------------------------*/

#include <stdbool.h>
#include <stdint.h>
#include <stage1.h>
#include <cpu_intrinsics.h>
#include <stage1_core_gpu/stage1_core.h>
#include "stage1_sizeopt_gpu.h"

/*#define GPU_DEBUG*/

#define LIFT_MAX_LIMBS 64   /* must match stage1_core.cu */

/* fewest roots a special-q must have: the GPU code has always used
   the polynomial degree (msieve's choice); the CPU code uses 1 */
#ifndef Q_ROOTS_MIN
#define Q_ROOTS_MIN degree
#endif

enum {
	GPU_BATCH_INVERT_P16_Q32 = 0,
	GPU_BATCH_INVERT_P16_Q64,
	GPU_BATCH_INVERT_P32_Q32,
	GPU_BATCH_INVERT_P32_Q64,
	GPU_TRANSFORM_P16_Q32_K32_D16,
	GPU_TRANSFORM_P16_Q32_K64_D32,
	GPU_TRANSFORM_P16_Q64_K32_D16,
	GPU_TRANSFORM_P16_Q64_K64_D32,
	GPU_TRANSFORM_P32_Q32_K64_D32,
	GPU_TRANSFORM_P32_Q64_K64_D32,
	GPU_BUCKET_PASS1_32,
	GPU_BUCKET_PASS1_64,
	GPU_BUCKET_PASS2,
	GPU_BUCKET_RECOVER_32,
	GPU_BUCKET_RECOVER_64,
	GPU_TRANSPOSE_32,
	GPU_LIFT_QROOTS_Q32,
	GPU_COLLISION_UNSORT32_SMALL,
	GPU_COLLISION_UNSORT32_LARGE,
	NUM_GPU_FUNCTIONS /* must be last */
};

static const char * gpu_kernel_names[] = 
{
	"batch_invert_p16_q32",
	"batch_invert_p16_q64",
	"batch_invert_p32_q32",
	"batch_invert_p32_q64",
	"transform_p16_q32_k32_d16",
	"transform_p16_q32_k64_d32",
	"transform_p16_q64_k32_d16",
	"transform_p16_q64_k64_d32",
	"transform_p32_q32_k64_d32",
	"transform_p32_q64_k64_d32",
	"bucket_sort_pass1_32",
	"bucket_sort_pass1_64",
	"bucket_sort_pass2",
	"bucket_recover_32",
	"bucket_recover_64",
	"transpose_32",
	"lift_qroots_q32",
	"collision_search_unsorted32_small",
	"collision_search_unsorted32_large",
};

/*------------------------------------------------------------------------*/
typedef struct {
	uint32_t num_p_alloc;
	CUdeviceptr p;
	CUdeviceptr mont_w;
	CUdeviceptr mont_r0;
	CUdeviceptr mont_r1;
	CUdeviceptr mont_r2;
	CUdeviceptr mont_r3;
	CUdeviceptr ss_mod_pp;

	uint32_t num_roots_alloc;
	CUdeviceptr which_p;
	CUdeviceptr roots;

	uint32_t num_inv_alloc;
	CUdeviceptr q_inv;
	CUdeviceptr scratch;
} dev_soa_t;

static void 
dev_soa_free(dev_soa_t * soa) {
	CUDA_TRY(cuMemFree(soa->p))
	CUDA_TRY(cuMemFree(soa->mont_w))
	CUDA_TRY(cuMemFree(soa->mont_r0))
	CUDA_TRY(cuMemFree(soa->mont_r1))
	CUDA_TRY(cuMemFree(soa->mont_r2))
	CUDA_TRY(cuMemFree(soa->mont_r3))
	CUDA_TRY(cuMemFree(soa->ss_mod_pp))
	CUDA_TRY(cuMemFree(soa->which_p))
	CUDA_TRY(cuMemFree(soa->roots))
	CUDA_TRY(cuMemFree(soa->q_inv))
	CUDA_TRY(cuMemFree(soa->scratch))
}

static void 
dev_soa_realloc(dev_soa_t * soa, uint32_t s0, 
		uint32_t s1, size_t sizeof_p)
{
	size_t sizeof_root = 2 * sizeof_p;
	if (soa->num_p_alloc < s0) {
		soa->num_p_alloc = s0;
		CUDA_TRY(cuMemFree(soa->p))
		CUDA_TRY(cuMemFree(soa->mont_w))
		CUDA_TRY(cuMemFree(soa->mont_r0))
		CUDA_TRY(cuMemFree(soa->mont_r1))
		CUDA_TRY(cuMemFree(soa->mont_r2))
		CUDA_TRY(cuMemFree(soa->mont_r3))
		CUDA_TRY(cuMemFree(soa->ss_mod_pp))
		CUDA_TRY(cuMemAlloc(&soa->p, s0 * sizeof_p))
		CUDA_TRY(cuMemAlloc(&soa->mont_w, s0 * sizeof_root))
		CUDA_TRY(cuMemAlloc(&soa->mont_r0, s0 * sizeof_root))
		CUDA_TRY(cuMemAlloc(&soa->mont_r1, s0 * sizeof_root))
		CUDA_TRY(cuMemAlloc(&soa->mont_r2, s0 * sizeof_root))
		CUDA_TRY(cuMemAlloc(&soa->mont_r3, s0 * sizeof_root))
		CUDA_TRY(cuMemAlloc(&soa->ss_mod_pp, s0 * sizeof_root))
	}
	if (soa->num_roots_alloc < s1) {
		soa->num_roots_alloc = s1;
		CUDA_TRY(cuMemFree(soa->which_p))
		CUDA_TRY(cuMemFree(soa->roots))
		CUDA_TRY(cuMemAlloc(&soa->which_p, s1 * sizeof(uint32_t)))
		CUDA_TRY(cuMemAlloc(&soa->roots, s1 * sizeof_root))
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	size_t sizeof_p;

	uint32_t num_p;
	uint32_t num_p_alloc;
	void *p;
	void *mont_w;
	void *mont_r0;
	void *mont_r1;
	void *mont_r2;
	void *mont_r3;
	void *ss_mod_pp;

	uint32_t num_roots;
	uint32_t num_roots_alloc;
	uint32_t *which_p;
	void *roots;

	dev_soa_t dev_soa;
} p_soa_t;

static void
p_soa_init(p_soa_t * soa, size_t sizeof_p)
{
	uint32_t t0 = 1000;
	uint32_t t1 = 10000;
	size_t sizeof_root = 2 * sizeof_p;

	soa->sizeof_p = sizeof_p;
	soa->num_p_alloc = t0;
	soa->p = xmalloc(t0 * sizeof_p);
	soa->mont_w = xmalloc(t0 * sizeof_root);
	soa->mont_r0 = xmalloc(t0 * sizeof_root);
	soa->mont_r1 = xmalloc(t0 * sizeof_root);
	soa->mont_r2 = xmalloc(t0 * sizeof_root);
	soa->mont_r3 = xmalloc(t0 * sizeof_root);
	soa->ss_mod_pp = xmalloc(t0 * sizeof_root);

	soa->num_roots_alloc = t1;
	soa->which_p = (uint32_t *)xmalloc(t1 * sizeof(uint32_t));
	soa->roots = xmalloc(t1 * sizeof_root);
}

static void
p_soa_free(p_soa_t * soa)
{
	free(soa->p);
	free(soa->mont_w);
	free(soa->mont_r0);
	free(soa->mont_r1);
	free(soa->mont_r2);
	free(soa->mont_r3);
	free(soa->ss_mod_pp);
	free(soa->which_p);
	free(soa->roots);
	dev_soa_free(&soa->dev_soa);
}

static void
p_soa_copy(p_soa_t * soa, CUstream stream)
{
	if (soa->num_p) {
		size_t sizeof_root = 2 * soa->sizeof_p;
		dev_soa_realloc(&soa->dev_soa, soa->num_p_alloc,
				soa->num_roots_alloc, soa->sizeof_p);

		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.p, soa->p,
				soa->num_p * soa->sizeof_p, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.mont_w, soa->mont_w,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.mont_r0, soa->mont_r0,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.mont_r1, soa->mont_r1,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.mont_r2, soa->mont_r2,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.mont_r3, soa->mont_r3,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.ss_mod_pp, soa->ss_mod_pp,
				soa->num_p * sizeof_root, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.which_p, soa->which_p,
				soa->num_roots * sizeof(uint32_t), stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.roots, soa->roots,
				soa->num_roots * sizeof_root, stream))
	}
}

static void
p_soa_realloc_host(p_soa_t * soa, uint32_t num_roots)
{
	uint32_t curr_p = soa->num_p;
	uint32_t curr_roots = soa->num_roots;
	size_t sizeof_root = 2 * soa->sizeof_p;

	if (curr_p >= soa->num_p_alloc) {
		uint32_t s0 = 2 * soa->num_p_alloc;
		soa->num_p_alloc = s0;
		soa->p = xrealloc(soa->p, s0 * soa->sizeof_p);
		soa->mont_w = xrealloc(soa->mont_w, s0 * sizeof_root);
		soa->mont_r0 = xrealloc(soa->mont_r0, s0 * sizeof_root);
		soa->mont_r1 = xrealloc(soa->mont_r1, s0 * sizeof_root);
		soa->mont_r2 = xrealloc(soa->mont_r2, s0 * sizeof_root);
		soa->mont_r3 = xrealloc(soa->mont_r3, s0 * sizeof_root);
		soa->ss_mod_pp = xrealloc(soa->ss_mod_pp, s0 * sizeof_root);
	}
	if (curr_roots + num_roots >= soa->num_roots_alloc) {
		uint32_t s0 = 2 * (curr_roots + num_roots);
		soa->num_roots_alloc = s0;
		soa->which_p = (uint32_t *)xrealloc(soa->which_p,
					s0 * sizeof(uint32_t));
		soa->roots = xrealloc(soa->roots, s0 * sizeof_root);
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint64_t sieve_size;
	p_soa_t soa16;
	p_soa_t soa32;
} p_soa_array_t;

static p_soa_array_t *
p_soa_array_init(void)
{
	p_soa_array_t *s = (p_soa_array_t *)xcalloc(1, 
					sizeof(p_soa_array_t));
	p_soa_init(&s->soa16, sizeof(uint16_t));
	p_soa_init(&s->soa32, sizeof(uint32_t));
	return s;
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	p_soa_free(&s->soa16);
	p_soa_free(&s->soa32);
	free(s);
}

static void
p_soa_array_reset(p_soa_array_t *s, uint64_t sieve_size)
{
	s->sieve_size = sieve_size;
	s->soa16.num_p = s->soa32.num_p = 0;
	s->soa16.num_roots = s->soa32.num_roots = 0;
}

static void
p_soa_array_start(p_soa_array_t *s, CUstream stream)
{
	p_soa_copy(&s->soa16, stream);
	p_soa_copy(&s->soa32, stream);
}

static void
store_p_soa(uint64 p, uint32 num_roots, mpz_t *roots, void *extra)
{
	p_soa_array_t *s = (p_soa_array_t *)extra;
	uint64_t sieve_size = s->sieve_size;

	if (p < 65536) {
		p_soa_t *soa = &s->soa16;
		uint32_t curr_p = soa->num_p;
		uint32_t curr_roots = soa->num_roots;

		p_soa_realloc_host(soa, num_roots);

		/* form 2^32, 2^64, 2^96 and 2^160 mod p^2 */

		uint32_t pp = (uint32_t)p * (uint32_t)p;
		uint32_t t0 = -pp % pp;
		uint32_t t1 = mp_modmul_1(t0, t0, pp);
		uint32_t t2 = mp_modmul_1(t1, t1, pp);

		((uint16_t *)soa->p)[curr_p] = (uint16_t)p;
		((uint32_t *)soa->mont_w)[curr_p] = montmul32_w(pp);
		((uint32_t *)soa->mont_r0)[curr_p] = t0;
		((uint32_t *)soa->mont_r1)[curr_p] = t1;
		((uint32_t *)soa->mont_r2)[curr_p] = mp_modmul_1(t1, t0, pp);
		((uint32_t *)soa->mont_r3)[curr_p] = mp_modmul_1(t2, t0, pp);
		((uint32_t *)soa->ss_mod_pp)[curr_p] =
					pp - (uint32_t)((sieve_size / 2) % pp);

		for (uint32_t i = 0; i < num_roots; i++) {
			soa->which_p[curr_roots + i] = curr_p;
			((uint32_t *)soa->roots)[curr_roots + i] = 
					mpz_get_ui(roots[i]);
		}
		soa->num_p++;
		soa->num_roots += num_roots;
	}
	else {
		p_soa_t *soa = &s->soa32;
		uint32_t curr_p = soa->num_p;
		uint32_t curr_roots = soa->num_roots;

		p_soa_realloc_host(soa, num_roots);

		/* form 2^64, 2^128 and 2^192 mod p^2 */

		uint64_t pp = p * p;
		uint64_t t0 = -pp % pp;
		uint64_t t1 = mp_modmul_2(t0, t0, pp);

		((uint32_t *)soa->p)[curr_p] = (uint32_t)p;
		((uint64_t *)soa->mont_w)[curr_p] = montmul64_w(pp);
		((uint64_t *)soa->mont_r0)[curr_p] = t0;
		((uint64_t *)soa->mont_r1)[curr_p] = t1;
		((uint64_t *)soa->mont_r2)[curr_p] = 0;
		((uint64_t *)soa->mont_r3)[curr_p] = mp_modmul_2(t1, t0, pp);
		((uint64_t *)soa->ss_mod_pp)[curr_p] = 
				pp - (sieve_size / 2) % pp;

		for (uint32_t i = 0; i < num_roots; i++) {
			soa->which_p[curr_roots + i] = curr_p;
			((uint64_t *)soa->roots)[curr_roots + i] = 
					gmp2uint64(roots[i]);
		}
		soa->num_p++;
		soa->num_roots += num_roots;
	}
}

/*------------------------------------------------------------------------*/

typedef struct {
	uint32_t num_q_alloc;
	CUdeviceptr q;

	uint32_t num_roots_alloc;
	CUdeviceptr which_q;
	CUdeviceptr roots;
} dev_q_soa_t;

static void 
dev_q_soa_free(dev_q_soa_t * soa) 
{
	CUDA_TRY(cuMemFree(soa->q))
	CUDA_TRY(cuMemFree(soa->which_q))
	CUDA_TRY(cuMemFree(soa->roots))
}

static void 
dev_q_soa_realloc(dev_q_soa_t * soa, uint32_t s0, 
		uint32_t s1, size_t sizeof_q) 
{
	size_t sizeof_root = 2 * sizeof_q;
	if (soa->num_q_alloc < s0) {
		soa->num_q_alloc = s0;
		CUDA_TRY(cuMemFree(soa->q))
		CUDA_TRY(cuMemAlloc(&soa->q, s0 * sizeof_root))
	}
	if (soa->num_roots_alloc < s1) {
		soa->num_roots_alloc = s1;
		CUDA_TRY(cuMemFree(soa->which_q))
		CUDA_TRY(cuMemFree(soa->roots))
		CUDA_TRY(cuMemAlloc(&soa->which_q, s1 * sizeof(uint32_t)))
		CUDA_TRY(cuMemAlloc(&soa->roots, s1 * sizeof_root))
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	size_t sizeof_q;

	uint32_t num_q;
	uint32_t num_q_alloc;
	void * q;

	uint32_t num_roots;
	uint32_t num_roots_alloc;
	uint32_t * which_q;
	void * roots;

	dev_q_soa_t dev_soa;
} q_soa_t;

static void
q_soa_init(q_soa_t * soa, size_t sizeof_q)
{
	uint32_t t0 = 1000;
	uint32_t t1 = 5000;
	size_t sizeof_root = 2 * sizeof_q;

	soa->sizeof_q = sizeof_q;
	soa->num_q_alloc = t0;
	soa->q = xmalloc(t0 * sizeof_q);

	soa->num_roots_alloc = t1;
	soa->which_q = (uint32_t *)xmalloc(t1 * sizeof(uint32_t));
	soa->roots = xmalloc(t1 * sizeof_root);
}

static void
q_soa_free(q_soa_t * soa)
{
	free(soa->q);
	free(soa->which_q);
	free(soa->roots);
	dev_q_soa_free(&soa->dev_soa);
}

static void
q_soa_copy(q_soa_t * soa, CUstream stream)
{
	if (soa->num_q) {
		size_t sizeof_root = 2 * soa->sizeof_q;
		dev_q_soa_realloc(&soa->dev_soa, soa->num_q_alloc,
				soa->num_roots_alloc, soa->sizeof_q);

		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.q, soa->q,
				soa->num_q * soa->sizeof_q, stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.which_q, soa->which_q,
				soa->num_roots * sizeof(uint32_t), stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_soa.roots, soa->roots,
				soa->num_roots * sizeof_root, stream))
	}
}

static void
q_soa_realloc_host(q_soa_t * soa, uint32_t num_roots)
{
	uint32_t curr_q = soa->num_q;
	uint32_t curr_roots = soa->num_roots;
	size_t sizeof_root = 2 * soa->sizeof_q;

	if (curr_q >= soa->num_q_alloc) {
		uint32_t s0 = 2 * soa->num_q_alloc;
		soa->num_q_alloc = s0;
		soa->q = xrealloc(soa->q, s0 * soa->sizeof_q);
	}
	if (curr_roots + num_roots >= soa->num_roots_alloc) {
		uint32_t s0 = 2 * (curr_roots + num_roots);
		soa->num_roots_alloc = s0;
		soa->which_q = (uint32_t *)xrealloc(soa->which_q,
					s0 * sizeof(uint32_t));
		soa->roots = xrealloc(soa->roots, s0 * sizeof_root);
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	q_soa_t soa32;
	q_soa_t soa64;
} specialq_array_t;

static specialq_array_t *
specialq_array_init(void)
{
	specialq_array_t *sq = (specialq_array_t *)xcalloc(1,
					sizeof(specialq_array_t));

	q_soa_init(&sq->soa32, sizeof(uint32_t));
	q_soa_init(&sq->soa64, sizeof(uint64_t));
	return sq;
}

static void
specialq_array_free(specialq_array_t *sq)
{
	q_soa_free(&sq->soa32);
	q_soa_free(&sq->soa64);
	free(sq);
}

static void
specialq_array_reset(specialq_array_t *sq)
{
	sq->soa32.num_q = 0; sq->soa64.num_q = 0;
	sq->soa32.num_roots = 0; sq->soa64.num_roots = 0;
}

static void
store_specialq(uint64 q, uint32 num_roots, mpz_t *roots, void *extra)
{
	specialq_array_t *sq = (specialq_array_t *)extra;

	if (q < ((uint64_t)1 << 32)) {
		q_soa_t * soa = &sq->soa32;
		uint32_t curr_q = soa->num_q;
		uint32_t curr_roots = soa->num_roots;

		q_soa_realloc_host(soa, num_roots);

		for (uint32_t i = 0; i < num_roots; i++) {
			((uint64_t *)soa->roots)[curr_roots + i] =
					gmp2uint64(roots[i]);
			soa->which_q[curr_roots + i] = curr_q;
		}

		((uint32_t *)soa->q)[curr_q] = (uint32_t)q;
		soa->num_q++;
		soa->num_roots += num_roots;
	}
	else {
		q_soa_t * soa = &sq->soa64;
		uint32_t curr_q = soa->num_q;
		uint32_t curr_roots = soa->num_roots;

		q_soa_realloc_host(soa, num_roots);

		memset((uint128 *)(soa->roots) + curr_roots, 
			0, num_roots * sizeof(uint128));
		for (uint32_t i = 0; i < num_roots; i++) {
			mpz_export(((uint128 *)soa->roots)[curr_roots + i].w, 
					NULL, -1, sizeof(uint32_t), 0, 0, roots[i]);
			soa->which_q[curr_roots + i] = curr_q;
		}

		((uint64_t *)soa->q)[curr_q] = q;
		soa->num_q++;
		soa->num_roots += num_roots;
	}
}

typedef struct {

	CUcontext gpu_context;
	CUmodule gpu_module;

	p_soa_array_t *p_array;
	specialq_array_t *q_array;

	CUstream stream;

	gpu_launch_t *launch;

	uint32_t num_gpu_arrays_alloc;
	CUdeviceptr gpu_array_size;

	size_t indata_alloc;
	CUdeviceptr gpu_all_indata;
	CUdeviceptr gpu_key;
	CUdeviceptr gpu_key_prefix;
	CUdeviceptr gpu_data;

	size_t collision_scratch_alloc;
	uint32_t collision_list_size;
	uint32_t min_collision_list_size;
	uint64_t pass2_dropped;
	CUdeviceptr gpu_collision_scratch;

	CUdeviceptr gpu_found_array;
	found_t *found_array;

	/* special-q roots lifted on the GPU (lift_qroots_q32) */
	bool gpu_lift;
	CUdeviceptr gpu_lift_failed;

	/* size optimization on the GPU (stage1_sizeopt_gpu.c);
	   so_on: for the current leading coefficient */
	so_ctx_t *so;
	bool so_on;

	CUevent start_event;
	CUevent end_event;
#ifdef GPU_DEBUG
	CUevent debug_start_event;
	CUevent debug_end_event;
#endif
	double gpu_elapsed;

	/* this thread's GPU, and its share of that GPU's memory */
	const gpu_info_t *gpu_info;
	size_t max_collision_workspace;

} device_thread_data_t;

/* stage 1's GPUs: thread t runs on lane t mod num_lanes */
typedef struct {

	gpu_info_t *gpu_info;		/* one per lane */
	uint32_t num_lanes;
	uint32_t num_threads;
	size_t *max_collision_workspace;	/* per lane, per thread */
} device_data_t;

/*------------------------------------------------------------------------*/
/* the bucket sort scratch space holds the bucket offsets from
   pass 1, their transpose, and then the list of pass 2
   collisions, whose first entry holds the list size */

static CUdeviceptr
get_collision_list(const device_thread_data_t *t,
		uint32_t qroot_batch_size,
		uint32_t num_subarrays,
		uint32_t bucket_bits)
{
	return t->gpu_collision_scratch + ((2 * sizeof(uint32_t) *
			num_subarrays * qroot_batch_size) << bucket_bits);
}

/*------------------------------------------------------------------------*/
static void
check_found_array(task_data_t *task, 
		uint32_t threadid, 
		q_soa_t * q_soa)
{
	stage1_sieve_data_t *d = task->d;
	stage1_sieve_thread_data_t *t0 = d->threads + threadid;
	device_thread_data_t *t = (device_thread_data_t *)t0->hw_thread_data;
	found_t *found_array = t->found_array;
	int64_t sieve_size = (int64_t)t->p_array->sieve_size;

	/* copy the counts first, then only the slots in use (the
	   array holds FOUND_ARRAY_SIZE entries, ~24 MB) */

	CUDA_TRY(cuMemcpyDtoHAsync(found_array, t->gpu_found_array,
			sizeof(found_t), t->stream))
	CUDA_TRY(cuStreamSynchronize(t->stream))

	if (*(uint32_t *)found_array > FOUND_ARRAY_SIZE - 1) {
		logprintf(task->obj, "warning: %u collisions found in one "
				"batch; only %u kept\n",
				*(uint32_t *)found_array, FOUND_ARRAY_SIZE - 1);
	}
	uint32_t found_array_size = MIN(FOUND_ARRAY_SIZE - 1,
				*(uint32_t *)found_array);

	/* size optimization on the GPU: the collisions never leave it,
	   and the host loop below has nothing to do */

	if (t->so_on) {
		so_run_batch(t->so, d, t->stream, t->gpu_found_array,
				found_array_size, q_soa->dev_soa.q,
				q_soa->dev_soa.which_q, q_soa->dev_soa.roots,
				(uint32_t)q_soa->sizeof_q, t->p_array->sieve_size);
	}
	else if (found_array_size > 0) {
		CUDA_TRY(cuMemcpyDtoHAsync(found_array + 1,
				t->gpu_found_array + sizeof(found_t),
				found_array_size * sizeof(found_t), t->stream))
		CUDA_TRY(cuStreamSynchronize(t->stream))
	}

	uint32_t num_lost = found_array[0].p1;
	t->pass2_dropped += found_array[0].p2;
	if (found_array_size == 0 && num_lost == 0 &&
			found_array[0].p2 == 0)
		return;

	/* clear only the first element */
	CUDA_TRY(cuMemsetD8Async(t->gpu_found_array, 0, 
				sizeof(found_t), t->stream))

	/* the bucket sort found more collisions than it had room
	   for in one batch; the next batches get a larger list */

	if (num_lost > 0) {
		uint32_t new_size = MIN(2 * num_lost, MAX_COLLISION_LIST_SIZE);

		if (new_size > t->min_collision_list_size) {
			logprintf(task->obj, "warning: %u collisions overflowed "
					"a list of %u; enlarging to %u\n",
					num_lost, t->collision_list_size,
					new_size);
			t->min_collision_list_size = new_size;
		}
	}

	for (uint32_t i = 0; i < (t->so_on ? 0 : found_array_size); i++) {
		found_t *f = found_array + i + 1;
		int64_t res = (int64_t)(f->offset) - sieve_size / 2;
		uint64_t p = (uint64_t)f->p1 * f->p2;
		uint32_t which_qroot = f->which_qroot;
		uint32_t which_q = q_soa->which_q[which_qroot];
		uint64_t q;
		uint128 qroot;

		if (q_soa->sizeof_q == 4) {
			uint64_t root64 = ((uint64_t *)q_soa->roots)[which_qroot];

			q = ((uint32_t *)q_soa->q)[which_q];
			qroot.w[0] = (uint32_t)root64;
			qroot.w[1] = (uint32_t)(root64 >> 32);
			qroot.w[2] = qroot.w[3] = 0;
		}
		else {
			q = ((uint64_t *)q_soa->q)[which_q];
			qroot = ((uint128 *)q_soa->roots)[which_qroot];
		}
		handle_collision(task, p, q, qroot, res);
	}
}

#ifdef GPU_DEBUG  /*------------------------------------------------------*/

static void
check_bucket2(uint32_t qroot_batch_size, 
	uint32_t num_subarrays,
	uint32_t bucket_bits, 
	device_thread_data_t *t)
{
	uint32_t num_collisions;
	CUdeviceptr collision_list = get_collision_list(t,
					qroot_batch_size, num_subarrays,
					bucket_bits);

	CUDA_TRY(cuMemcpyDtoH(&num_collisions, collision_list,
			sizeof(uint32_t)))
	printf("pass 2 collisions: %u%s\n", num_collisions,
			(num_collisions >= t->collision_list_size) ?
			" (list overflowed)" : "");
}

#endif  /*------------------------------------------------------*/

/*------------------------------------------------------------------------*/
static void
handle_batch_invert(p_soa_t * p_soa, 
		q_soa_t * q_soa,
		gpu_launch_t * gpu_kernel,
		uint32_t target_blocks,
		CUstream stream)
{
	/* handle batch inversion, computing inv(q^2) mod p^2 */

	uint32_t threads_x = MIN(TRANSFORM_NUM_THREADS, 
				gpu_kernel->threads_per_block);
	uint32_t num_p = p_soa->num_p;
	uint32_t num_q = q_soa->num_q;
	uint32_t blocks_x = (num_p + threads_x - 1) / threads_x;
	uint32_t blocks_y = 1;
	uint32_t q_batch = num_q;

	/* make sure there is GPU memory allocated to receive
	   num_p * num_q inverse results; pad the size to 
	   attempt to only do this once */

	if (p_soa->dev_soa.num_inv_alloc < num_p * num_q) {
		uint32_t t0 = num_p * (num_q * 13 / 10);

		p_soa->dev_soa.num_inv_alloc = t0;
		CUDA_TRY(cuMemFree(p_soa->dev_soa.q_inv))
		CUDA_TRY(cuMemFree(p_soa->dev_soa.scratch))
		CUDA_TRY(cuMemAlloc(&p_soa->dev_soa.q_inv,
				t0 * 2 * p_soa->sizeof_p))
		CUDA_TRY(cuMemAlloc(&p_soa->dev_soa.scratch,
				t0 * 2 * p_soa->sizeof_p))
	}

	/* we start by assuming each CUDA thread handles all q
	   for a single p, and leave that alone if the resulting
	   number of blocks will occupy the GPU sufficiently.
	   When there are too few p for that to happen, reduce
	   the number of q per thread with a floor that tries
	   to keep the batch inversion efficient */

	blocks_x = MIN(blocks_x, 1000);
	blocks_x = MAX(blocks_x, 1);
	if (blocks_x * blocks_y < target_blocks) {
		uint32_t target_y = MAX(1, target_blocks / blocks_x);

		q_batch = MAX(10, num_q / target_y);
		q_batch = MIN(SPECIALQ_BATCH_SIZE, q_batch);
		blocks_y = (num_q + q_batch - 1) / q_batch;

		/* avoid making the last y block too small */
		if (blocks_y > 1) {
			uint32_t rem = num_q % q_batch;
			if (rem > 0 && rem < q_batch / 2) {
				q_batch += (q_batch / 2) / 
					(blocks_y - 1);
				blocks_y = (num_q + q_batch - 1) / 
					q_batch;
			}
		}
	}

	/* format kernel arguments and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &p_soa->dev_soa.p;
	args[1] = &p_soa->dev_soa.mont_w;
	args[2] = &p_soa->dev_soa.mont_r1;

	/* pick the constant that will make Montgomery remainder
	   operations work given the selected operand sizes. The GPU
	   kernel needs to compute q mod p^2, and when q and p^2
	   have the same number of 32-bit words the mod operation is a 
	   single Montgomery multiply. When q has more words 
	   than p^2 then we need a different constant for the 
	   multi-word Montgomery modulo operation */

	switch (q_soa->sizeof_q << 3 | p_soa->sizeof_p) {
	case 4 << 3 | 2: args[3] = &p_soa->dev_soa.mont_r0; break;
	case 4 << 3 | 4: args[3] = &p_soa->dev_soa.mont_r0; break;
	case 8 << 3 | 2: args[3] = &p_soa->dev_soa.mont_r2; break;
	case 8 << 3 | 4: args[3] = &p_soa->dev_soa.mont_r0; break;
	default:
		printf("unsupported operand sizes\n");
		exit(-1);
	}
	args[4] = &q_soa->dev_soa.q;
	args[5] = &p_soa->dev_soa.q_inv;
	args[6] = &p_soa->dev_soa.scratch;
	args[7] = &q_batch;
	args[8] = &num_p;
	args[9] = &num_q;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		blocks_x, blocks_y, 1, threads_x, 1, 1, 
		0, stream, args, NULL))

#ifdef GPU_DEBUG
	printf("p%lu %u, q %u x %u x %u roots, threads %u, blocks %u x %u\n", 
			8 * p_soa->sizeof_p, num_p, num_q, 
			q_batch, q_soa->num_roots,
			threads_x, blocks_y, blocks_x);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_transform(p_soa_t * p_soa, 
		q_soa_t * q_soa,
		gpu_launch_t * gpu_kernel,
		uint32_t qroot_batch_start,
		uint32_t qroot_batch_size,
		uint32_t max_out_array_size,
		uint64_t sieve_size,
		uint32_t target_blocks,
		device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	uint32_t num_proots = p_soa->num_roots;
	uint32_t blocks_x = (num_proots + threads_x - 1) / threads_x;
	uint32_t blocks_y = 1;
	uint32_t arrays_per_block = qroot_batch_size;

	/* we start by assuming each CUDA thread handles all 
	   the q roots for a single p root, and leave that alone 
	   if the resulting number of blocks will occupy the 
	   GPU sufficiently.  When there are too few p roots 
	   for that to happen, reduce the number of q roots per 
	   thread */

	blocks_x = MIN(blocks_x, 1000);
	blocks_x = MAX(blocks_x, 1);
	if (blocks_x * blocks_y < target_blocks) {
		uint32_t target_y = MAX(1, target_blocks / blocks_x);

		arrays_per_block = MAX(5, qroot_batch_size / target_y);
		arrays_per_block = MIN(SPECIALQ_BATCH_SIZE, 
					arrays_per_block);
		blocks_y = (qroot_batch_size + arrays_per_block - 1) / 
				arrays_per_block;

		/* avoid making the last y block too small */
		if (blocks_y > 1) {
			uint32_t rem = qroot_batch_size % arrays_per_block;
			if (rem > 0 && rem < arrays_per_block / 2) {
				arrays_per_block += (arrays_per_block / 2) / 
						(blocks_y - 1);
				blocks_y = (qroot_batch_size + 
						arrays_per_block - 1) / 
						arrays_per_block;
			}
		}
	}

	/* format kernel arguments and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &p_soa->dev_soa.p;
	args[1] = &p_soa->dev_soa.mont_w;

	/* pick the constant that will make Montgomery remainder
	   operations work given the selected operand sizes. The GPU
	   kernel must compute q_root mod p^2, where q_root is
	   bounded by q^2. When this has the same number of 32-bit
	   words as p^2 the modulo operation is a single Montgomery
	   multiply, otherwise (3 cases out of 4) we need multi-
	   digit Montgomery modular reduction, and the constant
	   depends on how many 32-bit or 64-bit words q_root needs */

	switch (q_soa->sizeof_q << 3 | p_soa->sizeof_p) {
	case 4 << 3 | 2: args[2] = &p_soa->dev_soa.mont_r2; break;
	case 4 << 3 | 4: args[2] = &p_soa->dev_soa.mont_r0; break;
	case 8 << 3 | 2: args[2] = &p_soa->dev_soa.mont_r3; break;
	case 8 << 3 | 4: args[2] = &p_soa->dev_soa.mont_r3; break;
	default:
		printf("unsupported operand sizes\n");
		exit(-1);
	}
	args[3] = &p_soa->dev_soa.ss_mod_pp;
	args[4] = &p_soa->dev_soa.q_inv;
	args[5] = &p_soa->dev_soa.which_p;
	args[6] = &p_soa->dev_soa.roots;
	args[7] = &q_soa->dev_soa.which_q;
	args[8] = &q_soa->dev_soa.roots;
	args[9] = &qroot_batch_start;
	args[10] = &sieve_size;
	args[11] = &p_soa->num_p;
	args[12] = &num_proots;
	args[13] = &qroot_batch_size;
	args[14] = &t->gpu_array_size;
	args[15] = &t->gpu_key;
	args[16] = &t->gpu_data;
	args[17] = &max_out_array_size;
	args[18] = &arrays_per_block;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		blocks_x, blocks_y, 1, threads_x, 1, 1, 
		0, t->stream, args, NULL))

#ifdef GPU_DEBUG
	printf("p%luq%lu p roots %u q roots %u-%u, blocks %u x %u qbatch %u\n", 
			8 * p_soa->sizeof_p, 8 * q_soa->sizeof_q,
			num_proots, qroot_batch_start, 
			qroot_batch_start + qroot_batch_size, 
			blocks_y, blocks_x, arrays_per_block);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_collision_search(
	uint32_t qroot_batch_start,
	uint32_t qroot_batch_size,
	uint32_t max_out_array_size,
	gpu_launch_t * gpu_kernel,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	uint32_t blocks_x = 1;
	uint32_t blocks_y = MIN(target_blocks, qroot_batch_size);
	uint32_t arrays_per_block = (qroot_batch_size + 
			blocks_y - 1) / blocks_y;

	if (blocks_x * blocks_y < target_blocks) {
		uint32_t target_y = MAX(1, target_blocks / blocks_x);

		arrays_per_block = MAX(5, qroot_batch_size / target_y);
		arrays_per_block = MIN(SPECIALQ_BATCH_SIZE, 
					arrays_per_block);
		blocks_y = (qroot_batch_size + arrays_per_block - 1) / 
				arrays_per_block;

		/* avoid making the last y block too small */
		if (blocks_y > 1) {
			uint32_t rem = qroot_batch_size % arrays_per_block;
			if (rem > 0 && rem < arrays_per_block / 2) {
				arrays_per_block += (arrays_per_block / 2) / 
						(blocks_y - 1);
				blocks_y = (qroot_batch_size + 
						arrays_per_block - 1) / 
						arrays_per_block;
			}
		}
	}

	/* format kernel args and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &t->gpu_key;
	args[1] = &t->gpu_data;
	args[2] = &t->gpu_key_prefix;
	args[3] = &qroot_batch_start;
	args[4] = &max_out_array_size;
	args[5] = &t->gpu_array_size;
	args[6] = &qroot_batch_size;
	args[7] = &arrays_per_block;
	args[8] = &t->gpu_found_array;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		blocks_x, blocks_y, 1, threads_x, 1, 1, 
		0, t->stream, args, NULL))

#ifdef GPU_DEBUG
	printf("collision: %u arrays of %u keys, blocks %u x %u qbatch %u\n", 
		qroot_batch_size, max_out_array_size,
		blocks_y, blocks_x, arrays_per_block);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_bucket_pass1(
	uint32_t qroot_batch_size,
	uint32_t max_out_array_size,
	uint32_t num_subarrays,
	uint32_t bucket_bits,
	uint32_t prefix_bits,
	uint32_t loc_shift,
	uint32_t cache_keys,
	gpu_launch_t * gpu_kernel,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	uint32_t blocks_x = num_subarrays;
	uint32_t blocks_y = 1;
	uint32_t arrays_per_block = qroot_batch_size;

	if (blocks_x * blocks_y < target_blocks) {
		uint32_t target_y = MAX(1, target_blocks / blocks_x);

		arrays_per_block = MAX(5, qroot_batch_size / target_y);
		arrays_per_block = MIN(SPECIALQ_BATCH_SIZE, 
					arrays_per_block);
		blocks_y = (qroot_batch_size + arrays_per_block - 1) / 
				arrays_per_block;

		/* avoid making the last y block too small */
		if (blocks_y > 1) {
			uint32_t rem = qroot_batch_size % arrays_per_block;
			if (rem > 0 && rem < arrays_per_block / 2) {
				arrays_per_block += (arrays_per_block / 2) / 
						(blocks_y - 1);
				blocks_y = (qroot_batch_size + 
						arrays_per_block - 1) / 
						arrays_per_block;
			}
		}
	}

	/* format kernel args and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &t->gpu_key;
	args[1] = &t->gpu_key_prefix;
	args[2] = &t->gpu_collision_scratch;
	args[3] = &t->gpu_array_size;
	args[4] = &max_out_array_size;
	args[5] = &bucket_bits;
	args[6] = &prefix_bits;
	args[7] = &loc_shift;
	args[8] = &qroot_batch_size;
	args[9] = &arrays_per_block;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		blocks_x, blocks_y, 1, threads_x, 1, 1,
		sizeof(uint32_t) * ((1 << bucket_bits) + cache_keys), t->stream,
		args, NULL))

#ifdef GPU_DEBUG
	printf("bucket_pass1: %u arrays of %u keys, blocks %u x %u qbatch %u\n", 
		qroot_batch_size, max_out_array_size,
		blocks_y, blocks_x, arrays_per_block);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_transpose(
	uint32_t qroot_batch_size,
	uint32_t num_subarrays,
	uint32_t bucket_bits,
	gpu_launch_t * gpu_kernel,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	uint32_t num_buckets = 1 << bucket_bits;
	CUdeviceptr res = t->gpu_collision_scratch + 
			num_buckets * num_subarrays * 
			qroot_batch_size * sizeof(uint32_t);
	

	/* format kernel args and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &t->gpu_collision_scratch;
	args[1] = &res;
	args[2] = &num_subarrays;
	args[3] = &num_buckets;
	args[4] = &qroot_batch_size;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		target_blocks, 1, 1, threads_x, 1, 1, 
		0, t->stream, args, NULL))

#ifdef GPU_DEBUG
	printf("transpose: %u arrays of %u x %u\n",
		qroot_batch_size, num_subarrays, num_buckets);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_bucket_pass2(
	uint32_t qroot_batch_size,
	uint32_t max_out_array_size,
	uint32_t num_subarrays,
	uint32_t bucket_bits,
	uint32_t prefix_bits,
	gpu_launch_t * gpu_kernel,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	uint32_t num_buckets = 1 << bucket_bits;

	CUdeviceptr xpose_array = t->gpu_collision_scratch + 
					sizeof(uint32_t) * num_buckets * 
					num_subarrays * qroot_batch_size;
	CUdeviceptr collision_list = get_collision_list(t,
					qroot_batch_size, num_subarrays,
					bucket_bits);

	uint32_t blocks_x = num_buckets;
	uint32_t blocks_y = 1;
	uint32_t arrays_per_block = qroot_batch_size;

	if (blocks_x * blocks_y < target_blocks) {
		uint32_t target_y = MAX(1, target_blocks / blocks_x);

		arrays_per_block = MAX(5, qroot_batch_size / target_y);
		arrays_per_block = MIN(SPECIALQ_BATCH_SIZE, 
					arrays_per_block);
		blocks_y = (qroot_batch_size + arrays_per_block - 1) / 
				arrays_per_block;

		/* avoid making the last y block too small */
		if (blocks_y > 1) {
			uint32_t rem = qroot_batch_size % arrays_per_block;
			if (rem > 0 && rem < arrays_per_block / 2) {
				arrays_per_block += (arrays_per_block / 2) / 
						(blocks_y - 1);
				blocks_y = (qroot_batch_size + 
						arrays_per_block - 1) / 
						arrays_per_block;
			}
		}
	}

	CUDA_TRY(cuMemsetD32Async(collision_list, 0, 1, t->stream))

	/* format kernel args and launch */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &t->gpu_key_prefix;
	args[1] = &xpose_array;
	args[2] = &collision_list;
	args[3] = &t->collision_list_size;
	args[4] = &t->gpu_found_array;
	args[5] = &max_out_array_size;
	args[6] = &num_subarrays;
	args[7] = &bucket_bits;
	args[8] = &prefix_bits;
	args[9] = &qroot_batch_size;
	args[10] = &arrays_per_block;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		blocks_x, blocks_y, 1, threads_x, 1, 1, 
		0, t->stream, args, NULL))

#ifdef GPU_DEBUG
	printf("bucket_pass2: %u arrays of %u keys, blocks %u x %u qbatch %u\n", 
		qroot_batch_size, max_out_array_size,
		blocks_y, blocks_x, arrays_per_block);
#endif
}

/*------------------------------------------------------------------------*/
static void
handle_bucket_recover(
	uint32_t qroot_batch_start,
	uint32_t qroot_batch_size,
	uint32_t max_out_array_size,
	uint32_t num_subarrays,
	uint32_t bucket_bits,
	uint32_t prefix_bits,
	uint32_t loc_shift,
	gpu_launch_t * gpu_kernel,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
	uint32_t threads_x = TRANSFORM_NUM_THREADS; 
	CUdeviceptr collision_list = get_collision_list(t,
					qroot_batch_size, num_subarrays,
					bucket_bits);

	/* the number of collisions is only known on the GPU, 
	   so launch enough blocks to fill the card and have 
	   each block loop over its share of collisions. Blocks
	   with nothing to do exit immediately */

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &t->gpu_key;
	args[1] = &t->gpu_data;
	args[2] = &t->gpu_collision_scratch; /* pass 1 bucket offsets */
	args[3] = &collision_list;
	args[4] = &t->collision_list_size;
	args[5] = &max_out_array_size;
	args[6] = &num_subarrays;
	args[7] = &bucket_bits;
	args[8] = &prefix_bits;
	args[9] = &loc_shift;
	args[10] = &qroot_batch_start;
	args[11] = &t->gpu_found_array;

	CUDA_TRY(cuLaunchKernel(gpu_kernel->kernel_func,
		target_blocks, 1, 1, threads_x, 1, 1, 
		0, t->stream, args, NULL))

#ifdef GPU_DEBUG
	printf("bucket_recover: blocks %u\n", target_blocks);
#endif
}

/*------------------------------------------------------------------------*/
/* how the collision search handles key arrays of a given
   maximum size; computed in one place, so that sizing the
   batches and running them cannot disagree */

typedef struct {
	gpu_launch_t *search_kernel;   /* direct hash search, or NULL */
	gpu_launch_t *pass1_kernel;    /* bucket sort, or NULL */
	gpu_launch_t *recover_kernel;
	uint32_t bucket_bits;
	uint32_t num_subarrays;
	uint32_t max_subarray_size;
	uint32_t prefix_bits;
	uint32_t loc_shift;
} collision_plan_t;

static void
plan_collision_search(uint32_t max_out_array_size,
		uint32_t key_bits,
		const device_thread_data_t *t,
		collision_plan_t *plan)
{
	memset(plan, 0, sizeof(collision_plan_t));

	/* choose the collision search. Measurements on a 
	   Quadro P1000 (sm_61) found:

	   - for keys of up to 32 bits, the direct hash search
	     beats the bucket sort for arrays of at least 7000 
	     keys (no larger 32-bit arrays were available)

	   - pass 2 of the bucket sort loses collisions abruptly
	     once a bucket holds too many keys, because its fixed
	     size overflow list fills up. With 15-bit stage 1 
	     tables and a 1000-entry list there were no losses 
	     at 5300 keys per bucket and heavy losses at 8700 
	     (a simple model puts the limit near 6400). Fewer 
	     buckets are otherwise faster, so use the fewest 
	     buckets that stay below MAX_BUCKET_KEYS; pass 2 
	     reports any entries it drops */

	if (key_bits <= 32) {
		if (max_out_array_size < 8192) {
			plan->search_kernel = t->launch +
					GPU_COLLISION_UNSORT32_SMALL;
			return;
		}
		else if (t->gpu_info->compute_version_major >= 7 &&
			max_out_array_size < 30000) {
			plan->search_kernel = t->launch +
					GPU_COLLISION_UNSORT32_LARGE;
			return;
		}
		plan->pass1_kernel = t->launch + GPU_BUCKET_PASS1_32;
		plan->recover_kernel = t->launch + GPU_BUCKET_RECOVER_32;
	}
	else {
		plan->pass1_kernel = t->launch + GPU_BUCKET_PASS1_64;
		plan->recover_kernel = t->launch + GPU_BUCKET_RECOVER_64;
	}

	uint32_t bucket_bits = 5;
	while (bucket_bits < MAX_BUCKET_BITS &&
			(max_out_array_size >> bucket_bits) >
			MAX_BUCKET_KEYS) {
		bucket_bits++;
	}
	uint32_t num_buckets = 1 << bucket_bits;

	/* pass 1 sorts each subarray in shared memory, so 
	   subarrays can be no larger than PASS1_CACHE_KEYS.
	   Pass 2 runs fastest when each bucket of a subarray 
	   averages about 32 keys, and runs much slower when 
	   subarrays are smaller than that */

	uint32_t subarray_size = MIN(32 * num_buckets, PASS1_CACHE_KEYS);
	uint32_t num_subarrays = (max_out_array_size + subarray_size - 1) / 
				subarray_size;
	num_subarrays = MIN(num_subarrays, MAX_SUBARRAYS);

	/* pass 1 stores a prefix of each key above the bucket
	   bits, and uses the rest of the word for the location
	   of the key in its subarray, shifted right by loc_shift.
	   Choose a single loc_shift for the whole batch, so that
	   the location of any key in the largest possible
	   subarray fits; pass 1 and the recovery of collisions
	   then cannot disagree about it */

	uint32_t prefix_bits = MIN(28, key_bits - bucket_bits);
	uint32_t max_subarray_size = (max_out_array_size + num_subarrays - 1) / 
				num_subarrays;
	uint32_t loc_shift = 0;
	while ((max_subarray_size - 1) >> loc_shift >=
			(uint32_t)1 << (32 - prefix_bits)) {
		loc_shift++;
	}

	plan->bucket_bits = bucket_bits;
	plan->num_subarrays = num_subarrays;
	plan->max_subarray_size = max_subarray_size;
	plan->prefix_bits = prefix_bits;
	plan->loc_shift = loc_shift;
}

/*------------------------------------------------------------------------*/
static uint32_t
get_collision_list_size(const device_thread_data_t *t,
		const collision_plan_t *plan,
		uint32_t max_out_array_size,
		uint32_t num_arrays)
{
	/* size the list of pass 2 collisions. For random keys,
	   the expected number of pairs of matching prefixes in a
	   key array of size n is n^2/2 divided by 2^(bucket_bits +
	   prefix_bits). Real key arrays have many more than that
	   (up to 70x more in experiments), so leave a lot of room;
	   if the list ever overflows, check_found_array enlarges
	   the minimum size used here */

	double expected = ldexp((double)max_out_array_size * 
			max_out_array_size / 2, 
			-(int32_t)(plan->bucket_bits + plan->prefix_bits));
	expected = MIN(256 * expected * num_arrays,
			MAX_COLLISION_LIST_SIZE);
	return MAX(t->min_collision_list_size, (uint32_t)expected);
}

/*------------------------------------------------------------------------*/
/* the scratch space for the bucket sort holds the offsets
   of each bucket in each subarray, their transpose, and
   the list of pass 2 collisions */

static size_t
get_bucket_scratch_size(const collision_plan_t *plan,
		uint32_t num_arrays,
		uint32_t collision_list_size)
{
	if (plan->pass1_kernel == NULL)
		return 0;

	return 2 * sizeof(uint32_t) * num_arrays *
		((size_t)plan->num_subarrays << plan->bucket_bits) +
		sizeof(collision_t) * collision_list_size;
}

/*------------------------------------------------------------------------*/
static void
handle_final(
	uint32_t qroot_batch_start,
	uint32_t qroot_batch_size,
	uint32_t max_out_array_size,
	uint32_t key_bits,
	uint32_t target_blocks,
	device_thread_data_t *t)
{
#ifdef GPU_DEBUG
	float elapsed_ms;
#endif
	collision_plan_t plan;

	plan_collision_search(max_out_array_size, key_bits, t, &plan);
	uint32_t bucket_bits = plan.bucket_bits;
	uint32_t num_subarrays = plan.num_subarrays;
	uint32_t max_subarray_size = plan.max_subarray_size;
	uint32_t prefix_bits = plan.prefix_bits;
	uint32_t loc_shift = plan.loc_shift;

	if (plan.pass1_kernel != NULL) {

		/* allocate scratch space if necessary */

		t->collision_list_size = get_collision_list_size(t, &plan,
					max_out_array_size, qroot_batch_size);
		size_t scratch_size = get_bucket_scratch_size(&plan, 
					qroot_batch_size, t->collision_list_size);

		if (t->collision_scratch_alloc < scratch_size) {
			t->collision_scratch_alloc = scratch_size * 11 / 10;
			CUDA_TRY(cuMemFree(t->gpu_collision_scratch))
			CUDA_TRY(cuMemAlloc(&t->gpu_collision_scratch,
					t->collision_scratch_alloc))
		}

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif

		handle_bucket_pass1(qroot_batch_size,
				max_out_array_size,
				num_subarrays,
				bucket_bits,
				prefix_bits, loc_shift,
				max_subarray_size,
				plan.pass1_kernel,
				target_blocks, t);

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
				t->debug_start_event, t->debug_end_event))
		printf("bucket1 %.3f ms\n", elapsed_ms);
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif

		handle_transpose(qroot_batch_size, 
				num_subarrays,
				bucket_bits, 
				t->launch + GPU_TRANSPOSE_32,
				target_blocks, t);

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
				t->debug_start_event, t->debug_end_event))
		printf("xpose %.3f ms\n", elapsed_ms);
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif

		handle_bucket_pass2(qroot_batch_size,
				max_out_array_size,
				num_subarrays,
				bucket_bits,
				prefix_bits,
				t->launch + GPU_BUCKET_PASS2,
				target_blocks, t);

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
				t->debug_start_event, t->debug_end_event))
		printf("bucket2 %.3f ms\n", elapsed_ms);
		check_bucket2(qroot_batch_size, num_subarrays,
				bucket_bits, t);
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif

		handle_bucket_recover(qroot_batch_start,
				qroot_batch_size,
				max_out_array_size,
				num_subarrays,
				bucket_bits,
				prefix_bits, loc_shift,
				plan.recover_kernel,
				target_blocks, t);

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms,
				t->debug_start_event, t->debug_end_event))
		printf("recover %.3f ms\n", elapsed_ms);
#endif
	}
	else {
#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif
		handle_collision_search(qroot_batch_start,
				qroot_batch_size, max_out_array_size,
				plan.search_kernel, target_blocks, t);

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
				t->debug_start_event, t->debug_end_event))
		printf("collision %.3f ms\n", elapsed_ms);
#endif
	}
}

/*------------------------------------------------------------------------*/
static uint32_t
handle_special_q_batch(task_data_t *task, uint32_t threadid,
		q_soa_t * q_soa, uint32_t qroot_batch_size, 
		uint32_t q_search_array_size, uint32_t sort_key_bits)
{
	float elapsed_ms;
	msieve_obj * obj = task->obj;
	stage1_sieve_data_t *d = task->d;
	stage1_sieve_thread_data_t *t0 = d->threads + threadid;
	device_thread_data_t *t = (device_thread_data_t *)t0->hw_thread_data;
	p_soa_array_t *p_array = t->p_array;
	uint32_t target_blocks = 8 * t->gpu_info->num_compute_units;
	uint32_t qroot_batch_start = 0;

	CUDA_TRY(cuEventRecord(t->start_event, t->stream))

	q_soa_copy(q_soa, t->stream);

	/* lift the special-q roots mod q to q^2 (and subtract m0) on the
	   GPU, in place, before anything reads them */

	if (t->gpu_lift && q_soa->sizeof_q == 4 && q_soa->num_roots > 0) {
		gpu_launch_t *k = t->launch + GPU_LIFT_QROOTS_Q32;
		uint32_t threads = MIN(256, k->threads_per_block);
		uint32_t n = q_soa->num_roots;
		uint32_t blocks = MIN((n + threads - 1) / threads,
			    (uint32_t)(32 * t->gpu_info->num_compute_units));
		void *args[GPU_MAX_KERNEL_ARGS];

		args[0] = &q_soa->dev_soa.q;
		args[1] = &q_soa->dev_soa.which_q;
		args[2] = &q_soa->dev_soa.roots;
		args[3] = &n;
		args[4] = &t->gpu_lift_failed;
		CUDA_TRY(cuLaunchKernel(k->kernel_func, blocks, 1, 1,
				threads, 1, 1, 0, t->stream, args, NULL))

		/* the host rebuilds each hit's polynomial from its own
		   copy of the root (check_found_array), so it needs the
		   lifted values too */
		CUDA_TRY(cuMemcpyDtoHAsync(q_soa->roots, q_soa->dev_soa.roots,
				(size_t)n * 2 * q_soa->sizeof_q, t->stream))
	}

	/* compute modular inverses of all the q */

	if (p_array->soa16.num_p > 0) {
		handle_batch_invert(&p_array->soa16, q_soa,
				t->launch + ((q_soa->sizeof_q == 4) ?
					GPU_BATCH_INVERT_P16_Q32 :
					GPU_BATCH_INVERT_P16_Q64),
				target_blocks, t->stream);
	}
	if (p_array->soa32.num_p > 0) {
		handle_batch_invert(&p_array->soa32, q_soa,
				t->launch + ((q_soa->sizeof_q == 4) ?
					GPU_BATCH_INVERT_P32_Q32 :
					GPU_BATCH_INVERT_P32_Q64),
				target_blocks, t->stream);
	}

	/* transform the roots of each q in batches */

	while (qroot_batch_start < q_soa->num_roots) {
		uint32_t curr_batch_size = MIN(qroot_batch_size,
				q_soa->num_roots - qroot_batch_start);

		/* if the final batch will be too small,
		   shrink the second-to-last batch */

		if (qroot_batch_start + curr_batch_size < 
				q_soa->num_roots &&
		    q_soa->num_roots - qroot_batch_start <
				qroot_batch_size * 3 / 2) {
			curr_batch_size = qroot_batch_size * 3 / 4;
		}

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_start_event, t->stream))
#endif
		/* zero out the sizes of the current batch of arrays */

		CUDA_TRY(cuMemsetD32Async(t->gpu_array_size, 0, 
					curr_batch_size, t->stream))

		/* transform current batch; if all the p are less than
		   2^16 then the search arrays can be half the size */

		if (p_array->soa16.num_p > 0) {
			uint32_t which_kernel = (q_soa->sizeof_q == 4) ?
						GPU_TRANSFORM_P16_Q32_K32_D16 :
						GPU_TRANSFORM_P16_Q64_K32_D16;
			if (p_array->soa32.num_p > 0) {
				which_kernel = (q_soa->sizeof_q == 4) ?
						GPU_TRANSFORM_P16_Q32_K64_D32 :
						GPU_TRANSFORM_P16_Q64_K64_D32;
			}
			handle_transform(&p_array->soa16, q_soa,
					t->launch + which_kernel,
					qroot_batch_start, 
					curr_batch_size,
					q_search_array_size,
					p_array->sieve_size,
					target_blocks, 
					t);
		}
		if (p_array->soa32.num_p > 0) {
			handle_transform(&p_array->soa32, q_soa,
					t->launch + ((q_soa->sizeof_q == 4) ?
						GPU_TRANSFORM_P32_Q32_K64_D32 :
						GPU_TRANSFORM_P32_Q64_K64_D32),
					qroot_batch_start, 
					curr_batch_size,
					q_search_array_size,
					p_array->sieve_size,
					target_blocks, 
					t);
		}

#ifdef GPU_DEBUG
		CUDA_TRY(cuEventRecord(t->debug_end_event, t->stream))
		CUDA_TRY(cuEventSynchronize(t->debug_end_event))
		CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
				t->debug_start_event, t->debug_end_event))
		printf("transform %.3f ms\n", elapsed_ms);
#endif

		handle_final(qroot_batch_start, curr_batch_size, 
				q_search_array_size, sort_key_bits, 
				target_blocks, t);

		qroot_batch_start += curr_batch_size;

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			break;
	}

	check_found_array(task, threadid, q_soa);

	CUDA_TRY(cuEventRecord(t->end_event, t->stream))
	CUDA_TRY(cuEventSynchronize(t->end_event))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms, 
			t->start_event, t->end_event))
	printf("full batch %.3f ms\n", elapsed_ms);
	return (obj->flags & MSIEVE_FLAG_STOP_SIEVING) != 0;
}

/*------------------------------------------------------------------------*/
void
gpu_thread_data_init(void *data, int threadid)
{
	stage1_sieve_data_t *sd = (stage1_sieve_data_t *)data;
	device_data_t *d = (device_data_t *)(sd->hw_data);
	device_thread_data_t *t = (device_thread_data_t *)xcalloc(1,
					sizeof(device_thread_data_t));
	uint32_t lane = (uint32_t)threadid % d->num_lanes;

	t->gpu_info = d->gpu_info + lane;
	t->max_collision_workspace = d->max_collision_workspace[lane];

	/* every thread needs its own context; making all
	   threads share the same context causes problems
	   with the sort engine, because apparently it
	   changes the GPU cache size on the fly */

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000 /* API change */
	CUDA_TRY(cuCtxCreate(&t->gpu_context, NULL,
			CU_CTX_BLOCKING_SYNC,
			t->gpu_info->device_handle))
#else
	CUDA_TRY(cuCtxCreate(&t->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			t->gpu_info->device_handle))
#endif

	/* load GPU kernels */

	if (t->gpu_info->compute_version_major == 5) {
		if (t->gpu_info->compute_version_minor < 2)
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm50.ptx"))
		else
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm52.ptx"))
	}
	else if (t->gpu_info->compute_version_major == 6) {
		if (t->gpu_info->compute_version_minor < 1)
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm60.ptx"))
		else
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm61.ptx"))
	}
	else if (t->gpu_info->compute_version_major == 7) {
		if (t->gpu_info->compute_version_minor < 5)
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm70.ptx"))
		else
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm75.ptx"))
	}
	else if (t->gpu_info->compute_version_major >= 12) {
		/* Blackwell (sm_120): PTX built for it, not JIT from sm_80 */
		CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm120.ptx"))
	}
	else if (t->gpu_info->compute_version_major >= 10) {
		/* data-center Blackwell (B200, sm_100) */
		CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm100.ptx"))
	}
	else if (t->gpu_info->compute_version_major == 9) {
		/* Hopper (H100/H200, sm_90); it used to get sm_80 */
		CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm90.ptx"))
	}
	else if (t->gpu_info->compute_version_major >= 8) {
		if (t->gpu_info->compute_version_minor < 6)
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm80.ptx"))
		else if (t->gpu_info->compute_version_minor < 9)
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm86.ptx"))
		else
			CUDA_TRY(cuModuleLoad(&t->gpu_module, "stage1_core_sm89.ptx"))
	}
	else {
		printf("sorry, Nvidia doesn't want to support your card\n");
		logprintf(sd->obj, "sorry, Nvidia doesn't want to support your card\n");
		exit(-1);
	}

	t->launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	uint32_t num_kernels = NUM_GPU_FUNCTIONS;
	if (t->gpu_info->compute_version_major < 7)
		num_kernels--; 

	for (uint32_t i = 0; i < num_kernels; i++) {
		gpu_launch_init(t->gpu_module, gpu_kernel_names[i],
				t->launch + i);
		if (threadid == 0) {
			logprintf(sd->obj, "kernel %s: %u regs, %u smem\n",
					gpu_kernel_names[i],
					t->launch[i].num_regs,
					t->launch[i].static_shared_mem);
		}
	}

	/* threads each send a stream of kernel calls */

	CUDA_TRY(cuStreamCreate(&t->stream, 0))

	if (sd->poly->gpu_sizeopt) {
		t->so = so_init(sd->obj, t->gpu_info->compute_version_major,
				t->gpu_info->compute_version_minor,
				t->gpu_info->num_compute_units);
	}

	/* set up found array */

	CUDA_TRY(cuMemAlloc(&t->gpu_found_array, sizeof(found_t) *
			FOUND_ARRAY_SIZE))
	t->found_array = (found_t *)xmalloc(sizeof(found_t) * 
			FOUND_ARRAY_SIZE);

	CUDA_TRY(cuMemsetD8(t->gpu_found_array, 0, sizeof(found_t)))
	t->min_collision_list_size = MIN_COLLISION_LIST_SIZE;

	CUDA_TRY(cuMemAlloc(&t->gpu_lift_failed, sizeof(uint32_t)))
	CUDA_TRY(cuMemsetD8(t->gpu_lift_failed, 0, sizeof(uint32_t)))

	/* set up root generation arrays */

	t->p_array = p_soa_array_init();
	t->q_array = specialq_array_init();

	CUDA_TRY(cuEventCreate(&t->start_event, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&t->end_event, CU_EVENT_BLOCKING_SYNC))
#ifdef GPU_DEBUG
	CUDA_TRY(cuEventCreate(&t->debug_start_event, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&t->debug_end_event, CU_EVENT_BLOCKING_SYNC))
#endif

	sd->threads[threadid].hw_thread_data = t;
}

/*------------------------------------------------------------------------*/
void
gpu_thread_data_free(void *data, int threadid)
{
	stage1_sieve_data_t *sd = (stage1_sieve_data_t *)data;
	device_thread_data_t *t = sd->threads[threadid].hw_thread_data;

	CUDA_TRY(cuEventDestroy(t->start_event))
	CUDA_TRY(cuEventDestroy(t->end_event))
#ifdef GPU_DEBUG
	CUDA_TRY(cuEventDestroy(t->debug_start_event))
	CUDA_TRY(cuEventDestroy(t->debug_end_event))
#endif

	CUDA_TRY(cuMemFree(t->gpu_array_size))
	CUDA_TRY(cuMemFree(t->gpu_all_indata))
	CUDA_TRY(cuMemFree(t->gpu_collision_scratch))
	so_free(t->so);
	CUDA_TRY(cuStreamDestroy(t->stream))

	free(t->found_array);
	CUDA_TRY(cuMemFree(t->gpu_found_array))
	CUDA_TRY(cuMemFree(t->gpu_lift_failed))

	free(t->launch);

	p_soa_array_free(t->p_array);
	specialq_array_free(t->q_array);

	CUDA_TRY(cuCtxDestroy(t->gpu_context)) 
	free(t);
}

/*------------------------------------------------------------------------*/
void *
gpu_sieve_data_init(msieve_obj *obj, uint32 *num_threads)
{
	gpu_config_t gpu_config;
	uint32_t devs[MAX_GPU], ndev = 0, nlanes, nthreads, sharers = 1;
	const char *e = getenv("MSIEVE_SIMULATE_GPUS");

	gpu_init(&gpu_config);
	if (gpu_config.num_gpu == 0) {
		printf("error: no CUDA-enabled GPUs found\n");
		exit(-1);
	}
	{
		uint32 lr = 0, ls = 1;

#ifdef HAVE_MPI
		lr = obj->mpi_local_rank;
		ls = obj->mpi_local_size;
#endif
		ndev = gpu_select(&gpu_config, obj->which_gpu, lr, ls, devs,
				&sharers);
	}
	if (ndev == 0) {
		printf("error: GPU %u does not exist "
			"or is not CUDA-enabled\n", obj->which_gpu);
		exit(-1);
	}

	/* MSIEVE_SIMULATE_GPUS=n (testing): n lanes on the GPU(s) as
	   i % (devices) */
	nlanes = ndev;
	if (e != NULL && atoi(e) > 0)
		nlanes = MIN((uint32_t)atoi(e), MAX_GPU);

	/* one to four threads per GPU (each thread a leading
	   coefficient at a time) */
	nthreads = MAX(nlanes, MIN(4 * nlanes, *num_threads));
	*num_threads = nthreads;

	device_data_t *d = (device_data_t *)xcalloc(1, sizeof(device_data_t));
	d->num_lanes = nlanes;
	d->num_threads = nthreads;
	d->gpu_info = (gpu_info_t *)xmalloc(nlanes * sizeof(gpu_info_t));
	d->max_collision_workspace = (size_t *)xmalloc(nlanes *
						sizeof(size_t));

	for (uint32_t l = 0; l < nlanes; l++) {
		gpu_info_t *gpu_info = d->gpu_info + l;
		uint32_t dev = devs[l % ndev];
		uint32_t on_lane = nthreads / nlanes +
					(l < nthreads % nlanes);

		memcpy(gpu_info, gpu_config.info + dev, sizeof(gpu_info_t));
		if (l < ndev || l == 0) {
			logprintf(obj, "using GPU %u (%s)\n", dev,
					gpu_info->name);
			logprintf(obj, "selected card has CUDA arch %d.%d\n",
					gpu_info->compute_version_major,
					gpu_info->compute_version_minor);
		}

		size_t gpu_mem = 0.3 * gpu_info->global_mem_size;
		if (obj->nfs_args != NULL) {

			char *tmp = strstr(obj->nfs_args, "gpu_mem_mb=");
			if (tmp != NULL) {
				size_t m = strtoul(tmp + 11, NULL, 10);

				gpu_mem = MIN(gpu_info->global_mem_size,
						m * 1048576);
				gpu_mem = MAX(10 * 1048576, gpu_mem);
				if (l == 0)
					logprintf(obj, "setting max GPU mem "
						"use to %.1lf MB\n",
						(double)gpu_mem / 1048576);
			}
		}

		/* MPI processes sharing the card share its budget */
		gpu_mem /= MAX(1, sharers);

		/* simulated lanes share their card's memory */
		if (nlanes > ndev)
			gpu_mem /= (nlanes / ndev + (l % ndev <
						nlanes % ndev));

		/* each of the GPU's threads gets an equal share of the
		   memory used on it */
		d->max_collision_workspace[l] = gpu_mem / MAX(1, on_lane);
	}
	if (nlanes > 1)
		logprintf(obj, "stage 1 on %u GPU%s%s, %u threads\n", nlanes,
				"s", nlanes > ndev ? " (simulated)" : "",
				nthreads);
	return d;
}

/*------------------------------------------------------------------------*/
void gpu_sieve_data_free(void *gpu_data)
{
	device_data_t *d = (device_data_t *)gpu_data;

	free(d->gpu_info);
	free(d->max_collision_workspace);
	free(d);
}

/*------------------------------------------------------------------------*/
/* special-q producer threads (MSIEVE_Q_THREADS=n): each owns a clone
   of the special-q factory yielding 1/n of its output, and fills
   batches into a small ring of slots that the GPU thread drains, so
   producing special-q overlaps the GPU work. Producers touch only host
   memory; the GPU thread does every CUDA call */

#include <pthread.h>

enum { QSLOT_FREE, QSLOT_FILLING, QSLOT_FULL, QSLOT_DRAINING };

typedef struct {
	pthread_mutex_t lock;
	pthread_cond_t cv_full;
	pthread_cond_t cv_free;
	uint32_t nslots;
	specialq_array_t **slot;
	uint32_t *state;
	uint32_t producers_left;
	volatile uint32_t stop;
	poly_coeff_t *c;
	uint32_t q_batch_size;
} qqueue_t;

typedef struct {
	qqueue_t *q;
	void *fb;
} qprod_arg_t;

static void *
q_producer(void *arg)
{
	qprod_arg_t *pa = (qprod_arg_t *)arg;
	qqueue_t *q = pa->q;
	bool done = false;

	while (!done) {
		uint32_t k = 0;

		pthread_mutex_lock(&q->lock);
		for (;;) {
			if (q->stop)
				break;
			for (k = 0; k < q->nslots; k++)
				if (q->state[k] == QSLOT_FREE)
					break;
			if (k < q->nslots)
				break;
			pthread_cond_wait(&q->cv_free, &q->lock);
		}
		if (q->stop) {
			pthread_mutex_unlock(&q->lock);
			break;
		}
		q->state[k] = QSLOT_FILLING;
		pthread_mutex_unlock(&q->lock);

		specialq_array_t *sq = q->slot[k];
		specialq_array_reset(sq);
		while (sq->soa32.num_q + sq->soa64.num_q < SPECIALQ_BATCH_SIZE ||
		       sq->soa32.num_roots + sq->soa64.num_roots <
							q->q_batch_size) {
			if (sieve_fb_next(pa->fb, q->c, store_specialq,
					sq) == P_SEARCH_DONE) {
				done = true;
				break;
			}
			if (q->stop)
				break;
		}

		pthread_mutex_lock(&q->lock);
		q->state[k] = (sq->soa32.num_roots + sq->soa64.num_roots > 0) ?
					QSLOT_FULL : QSLOT_FREE;
		pthread_cond_broadcast(&q->cv_full);
		pthread_cond_broadcast(&q->cv_free);
		pthread_mutex_unlock(&q->lock);
		if (q->stop)
			break;
	}

	pthread_mutex_lock(&q->lock);
	q->producers_left--;
	pthread_cond_broadcast(&q->cv_full);
	pthread_mutex_unlock(&q->lock);
	return NULL;
}

static uint32_t
q_producer_count(void)
{
	/* default 4: more producers measured no faster */
	const char *e = getenv("MSIEVE_Q_THREADS");
	long n = e ? strtol(e, NULL, 10) : 4;
	if (n < 0) n = 0;
	if (n > 256) n = 256;
	return (uint32_t)n;
}

/*------------------------------------------------------------------------*/
void
stage1_specialq_gpu(task_data_t *task, uint32 threadid,
		uint64 special_q_min, uint64 special_q_max, 
		uint32 p_min, uint32 p_max)
{
	stage1_sieve_data_t *d = task->d;
	poly_coeff_t *c = task->c;
	stage1_sieve_thread_data_t *t0 = d->threads + threadid;
	device_thread_data_t *t = (device_thread_data_t *)t0->hw_thread_data;
	uint32_t degree = d->poly->degree;

	void *p_fb = t0->sieve_p_fb;
	void *q_fb = t0->sieve_q_fb;
	p_soa_array_t *p_array = t->p_array;
	specialq_array_t *sq = t->q_array;
	double cpu_start_time = get_cpu_time();
	double elapsed = 0;

	t->gpu_elapsed = 0;
	t->so_on = (t->so != NULL) ? so_set_coeff(t->so, d->poly, c) : false;

	/* we will be looking for collisions by sorting.
	   Each of the p values needs at least one entry 
	   in the list to search, so the 'sieve size' implicit 
	   in the dataset is the square of the largest p, 
	   and the sieve starts at -sieve_size / 2 */

	/* round up to an even number; rounding down would make
	   sieve_size smaller than p_max^2 when the latter is odd,
	   so that a p equal to p_max would have no room in the
	   sieve (the GPU code then loops forever) */

	uint64_t sieve_size = ((uint64_t)p_max * p_max + 1) & (uint64_t)(~1);

	/* build all the arithmetic progressions */

	p_soa_array_reset(p_array, sieve_size);
	sieve_fb_reset(p_fb, p_min, p_max, 1, MAX_ROOTS);
	while (sieve_fb_next(p_fb, c, store_p_soa,
			p_array) != P_SEARCH_DONE) {
		;
	}

	uint32_t num_p = p_array->soa16.num_p + p_array->soa32.num_p;
	uint32_t num_p_roots = p_array->soa16.num_roots + 
			p_array->soa32.num_roots;
	if (num_p_roots == 0)
		return;

	/* copy to GPU once */

	p_soa_array_start(p_array, t->stream);
#if 1
	printf("pprogs: %u+%u=%u entries, %u+%u=%u roots\n", 
		p_array->soa16.num_p, p_array->soa32.num_p, 
		num_p,
		p_array->soa16.num_roots, p_array->soa32.num_roots, 
		num_p_roots);
#endif
	/* we need to be sorting many sieve arrays in order
	   to get maximum throughput

	   First find the maximum number of bits in sort keys */

	uint32_t sort_key_bits = 64;
	while (!(sieve_size & ((uint64_t)1 << (sort_key_bits - 1))))
		sort_key_bits--;

	/* another limit on the number of special-q is the
	   size of the array we want to sort. When p_min
	   is about half of p_max, each root in the p batch
	   will yield anywhere between 1 to 4 offsets less
	   than sieve_size for each special-q. Experience shows 
	   that on average each special-q generates about 
	   2.5 - 3 times num_p_roots entries to sort. Earlier 
	   versions of this code assumed just one offset per root, 
	   which allows simple GPU code but requires several
	   times as many special-q values for a given number 
	   of collisions. Especially as inputs get larger the 
	   collision rate drops very rapidly, and increasing the
	   number of offsets by 2.5x inceases the probability of
	   a collision 6.25x due to the birthday paradox. So 
	   throughput will probably be higher if we do the 
	   work to find all the offsets we are entitled to */

	uint32_t q_search_array_size = 3.0 * num_p_roots;

	/* we have enough information to decide on the size
	   of a single batch. The initial problem setup needs
	   either a 32-bit key/16-bit p or 64-bit key / 32-bit p,
	   and the collision search works on a 32-bit prefix of 
	   each key. The bucket sort also needs scratch space that
	   grows with the number of key arrays, plus a list of
	   collisions; use the largest batch that fits */

	size_t collision_bytes = (sort_key_bits <= 32) ? 
			(sizeof(uint32_t) + sizeof(uint16_t) +
			 sizeof(uint32_t)) :
			(sizeof(uint64_t) + sizeof(uint32_t) +
			 sizeof(uint32_t));

	collision_plan_t plan;
	plan_collision_search(q_search_array_size, sort_key_bits, 
				t, &plan);
	uint32_t q_batch_size = t->max_collision_workspace / 
			(collision_bytes * q_search_array_size);
	while (q_batch_size > 1 &&
		(size_t)q_batch_size * q_search_array_size * 
			collision_bytes +
		get_bucket_scratch_size(&plan, q_batch_size,
			get_collision_list_size(t, &plan, 
				q_search_array_size, q_batch_size)) >
		t->max_collision_workspace) {
		q_batch_size -= MAX(1, q_batch_size / 256);
	}
	
	/* set up the collision arrays */

	if (t->indata_alloc < q_batch_size * 
			q_search_array_size * collision_bytes) {
		t->indata_alloc = q_batch_size * 
				q_search_array_size * collision_bytes *
				11 / 10;
		CUDA_TRY(cuMemFree(t->gpu_all_indata))
		CUDA_TRY(cuMemAlloc(&t->gpu_all_indata, 
				t->indata_alloc))
	}
	t->gpu_key = t->gpu_all_indata;
	t->gpu_key_prefix = t->gpu_key + q_batch_size * 
			q_search_array_size * ((sort_key_bits <= 32) ? 
			sizeof(uint32_t) : sizeof(uint64_t));
	t->gpu_data = t->gpu_key_prefix + q_batch_size * 
			q_search_array_size * sizeof(uint32_t);

	if (t->num_gpu_arrays_alloc < q_batch_size) {
		t->num_gpu_arrays_alloc = 3 * q_batch_size / 2;
		CUDA_TRY(cuMemFree(t->gpu_array_size))
		CUDA_TRY(cuMemAlloc(&t->gpu_array_size, 
				t->num_gpu_arrays_alloc * sizeof(uint32_t)))
	}

#if 1
	printf("max q batch %u q array size %u\n", 
			q_batch_size, q_search_array_size);
#endif
	/* initialize special-q factory */

	/* lift special-q roots on the GPU when the kernel's fast path
	   applies (trans_N > 0, trans_m0 >= 0, both fit its constant
	   arrays); MSIEVE_CPU_LIFT=1 keeps the CPU lifting, to compare */

	t->gpu_lift = false;
	if (getenv("MSIEVE_CPU_LIFT") == NULL &&
	    mpz_sgn(c->trans_N) > 0 && mpz_sgn(c->trans_m0) >= 0 &&
	    mpz_sizeinbase(c->trans_N, 2) <= 32 * LIFT_MAX_LIMBS &&
	    mpz_sizeinbase(c->trans_m0, 2) <= 32 * LIFT_MAX_LIMBS) {
		uint32_t nl[LIFT_MAX_LIMBS] = {0}, ml[LIFT_MAX_LIMBS] = {0};
		size_t nn = 0, nm = 0;
		uint32_t nn32, nm32, deg32 = degree;
		CUdeviceptr ptr;
		size_t bytes;

		mpz_export(nl, &nn, -1, sizeof(uint32_t), 0, 0, c->trans_N);
		mpz_export(ml, &nm, -1, sizeof(uint32_t), 0, 0, c->trans_m0);
		nn32 = (uint32_t)nn;
		nm32 = (uint32_t)nm;
		CUDA_TRY(cuModuleGetGlobal(&ptr, &bytes, t->gpu_module, "c_lift_N"))
		CUDA_TRY(cuMemcpyHtoD(ptr, nl, sizeof(nl)))
		CUDA_TRY(cuModuleGetGlobal(&ptr, &bytes, t->gpu_module, "c_lift_m0"))
		CUDA_TRY(cuMemcpyHtoD(ptr, ml, sizeof(ml)))
		CUDA_TRY(cuModuleGetGlobal(&ptr, &bytes, t->gpu_module, "c_lift_nlimbs"))
		CUDA_TRY(cuMemcpyHtoD(ptr, &nn32, sizeof(uint32_t)))
		CUDA_TRY(cuModuleGetGlobal(&ptr, &bytes, t->gpu_module, "c_lift_m0limbs"))
		CUDA_TRY(cuMemcpyHtoD(ptr, &nm32, sizeof(uint32_t)))
		CUDA_TRY(cuModuleGetGlobal(&ptr, &bytes, t->gpu_module, "c_lift_degree"))
		CUDA_TRY(cuMemcpyHtoD(ptr, &deg32, sizeof(uint32_t)))
		t->gpu_lift = true;
	}
	sieve_fb_set_defer_lift(q_fb, t->gpu_lift);

	sieve_fb_reset(q_fb, special_q_min, 
			special_q_max, Q_ROOTS_MIN, MAX_ROOTS);

	bool quit = false;
	uint32_t nprod = q_producer_count();
	if (nprod == 0) {
		bool all_q_done = false;

		while (!quit && !all_q_done) {

			specialq_array_reset(sq);

			while (sq->soa32.num_q + 
					sq->soa64.num_q < SPECIALQ_BATCH_SIZE ||
				sq->soa32.num_roots +
					sq->soa64.num_roots < q_batch_size) {
				if (sieve_fb_next(q_fb, c,
					store_specialq, sq) == P_SEARCH_DONE) {

					all_q_done = true;
					break;
				}
			}
			if (sq->soa32.num_roots + sq->soa64.num_roots == 0)
				break;
	#if 1
			printf("q batch %u+%u=%u, q roots %u+%u=%u x %u-bit\n",
				sq->soa32.num_q, sq->soa64.num_q,
				sq->soa32.num_q + sq->soa64.num_q,
				sq->soa32.num_roots, sq->soa64.num_roots,
				sq->soa32.num_roots + sq->soa64.num_roots,
				sort_key_bits);
	#endif
			if (sq->soa32.num_q > 0) {
				quit |= handle_special_q_batch(task, threadid, 
							&sq->soa32, q_batch_size, 
							q_search_array_size, 
							sort_key_bits);
			}
			if (!quit && sq->soa64.num_q > 0) {
				quit |= handle_special_q_batch(task, threadid, 
							&sq->soa64, q_batch_size, 
							q_search_array_size, 
							sort_key_bits);
			}

			elapsed = get_cpu_time() - cpu_start_time + t->gpu_elapsed;
			if (elapsed > task->coeff_deadline ||
			    (task->d->wall_end &&
			     stage1_wall_now() > task->d->wall_end))
				quit = true;
		}
	}
	else {
		/* producer threads fill batches; this thread drains them */
		qqueue_t qq = {
			.nslots = nprod + 2,
			.producers_left = nprod,
			.stop = 0,
			.c = c,
			.q_batch_size = q_batch_size,
		};

		pthread_mutex_init(&qq.lock, NULL);
		pthread_cond_init(&qq.cv_full, NULL);
		pthread_cond_init(&qq.cv_free, NULL);
		qq.slot = (specialq_array_t **)xmalloc(qq.nslots *
					sizeof(specialq_array_t *));
		qq.state = (uint32_t *)xcalloc(qq.nslots, sizeof(uint32_t));
		for (uint32_t k = 0; k < qq.nslots; k++)
			qq.slot[k] = specialq_array_init();

		qprod_arg_t *pa = (qprod_arg_t *)xmalloc(nprod * 
						sizeof(qprod_arg_t));
		pthread_t *tid = (pthread_t *)xmalloc(nprod * sizeof(pthread_t));
		for (uint32_t k = 0; k < nprod; k++) {
			pa[k].q = &qq;
			pa[k].fb = sieve_fb_clone(q_fb, nprod, k);
			sieve_fb_reset(pa[k].fb, special_q_min,
					special_q_max, Q_ROOTS_MIN, MAX_ROOTS);
			pthread_create(tid + k, NULL, q_producer, pa + k);
		}

		while (!quit) {
			uint32_t k;

			pthread_mutex_lock(&qq.lock);
			for (;;) {
				for (k = 0; k < qq.nslots; k++)
					if (qq.state[k] == QSLOT_FULL)
						break;
				if (k < qq.nslots || qq.producers_left == 0)
					break;
				pthread_cond_wait(&qq.cv_full, &qq.lock);
			}
			if (k == qq.nslots) {           /* all done */
				pthread_mutex_unlock(&qq.lock);
				break;
			}
			qq.state[k] = QSLOT_DRAINING;
			pthread_mutex_unlock(&qq.lock);

			specialq_array_t *bsq = qq.slot[k];
			printf("q batch %u+%u=%u, q roots %u+%u=%u x %u-bit\n",
				bsq->soa32.num_q, bsq->soa64.num_q,
				bsq->soa32.num_q + bsq->soa64.num_q,
				bsq->soa32.num_roots, bsq->soa64.num_roots,
				bsq->soa32.num_roots + bsq->soa64.num_roots,
				sort_key_bits);
			if (bsq->soa32.num_q > 0) {
				quit |= handle_special_q_batch(task, threadid,
						&bsq->soa32, q_batch_size,
						q_search_array_size,
						sort_key_bits);
			}
			if (!quit && bsq->soa64.num_q > 0) {
				quit |= handle_special_q_batch(task, threadid,
						&bsq->soa64, q_batch_size,
						q_search_array_size,
						sort_key_bits);
			}

			pthread_mutex_lock(&qq.lock);
			qq.state[k] = QSLOT_FREE;
			pthread_cond_broadcast(&qq.cv_free);
			pthread_mutex_unlock(&qq.lock);

			elapsed = get_cpu_time() - cpu_start_time +
						t->gpu_elapsed;
			if (elapsed > task->coeff_deadline ||
			    (task->d->wall_end &&
			     stage1_wall_now() > task->d->wall_end))
				quit = true;
		}

		pthread_mutex_lock(&qq.lock);
		qq.stop = 1;
		pthread_cond_broadcast(&qq.cv_free);
		pthread_cond_broadcast(&qq.cv_full);
		pthread_mutex_unlock(&qq.lock);
		for (uint32_t k = 0; k < nprod; k++) {
			pthread_join(tid[k], NULL);
			sieve_fb_free(pa[k].fb);
		}
		for (uint32_t k = 0; k < qq.nslots; k++)
			specialq_array_free(qq.slot[k]);
		free(qq.slot);
		free(qq.state);
		free(pa);
		free(tid);
		pthread_mutex_destroy(&qq.lock);
		pthread_cond_destroy(&qq.cv_full);
		pthread_cond_destroy(&qq.cv_free);
	}

	/* size-optimize what this coefficient's batches collected */
	if (t->so_on)
		so_flush(t->so);

	t0->cumulative_elapsed += elapsed;

	sieve_fb_set_defer_lift(q_fb, 0);
	if (t->gpu_lift) {
		uint32_t failed = 0;
		CUDA_TRY(cuMemcpyDtoH(&failed, t->gpu_lift_failed, sizeof(uint32_t)))
		if (failed > 0) {
			logprintf(task->obj, "warning: %u special-q roots could "
					"not be lifted on the GPU\n", failed);
			CUDA_TRY(cuMemsetD8(t->gpu_lift_failed, 0, sizeof(uint32_t)))
		}
	}

	if (t->pass2_dropped > 0) {
		logprintf(task->obj, "warning: bucket sort pass 2 dropped "
				"%" PRIu64 " entries; some collisions were missed\n",
				t->pass2_dropped);
		t->pass2_dropped = 0;
	}
}

