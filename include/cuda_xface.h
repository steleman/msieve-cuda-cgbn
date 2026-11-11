/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: cuda_xface.h 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#ifndef _GPU_XFACE_H
#define _GPU_XFACE_H

#if defined(HAVE_CUDA)

#include <util.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_GPU 64	/* devices gpu_init() records (it used to be 4,
			   and more devices overran config->info) */

typedef struct {
	char name[32];
	int32 compute_version_major;
	int32 compute_version_minor;
	int32 clock_speed; /* in kHz */
	int32 num_compute_units;
	int32 constant_mem_size;
	int32 shared_mem_size;
	size_t global_mem_size;
	int32 registers_per_block;
	int32 max_threads_per_block;
	int32 can_overlap;
	int32 warp_size;
	int32 max_thread_dim[3];
	int32 max_grid_size[3];
	int32 has_timeout;
	CUdevice device_handle;
} gpu_info_t;

typedef struct {
	int32 num_gpu;
	gpu_info_t info[MAX_GPU];
} gpu_config_t;

void gpu_init(gpu_config_t *config);

/* the devices a process uses (devs, returns their number): which_gpu
   alone, or with which_gpu = 0xffffffff (MSIEVE_ALL_GPUS) its share
   of the node's GPUs: local rank r of L gets devices r, r + L, ...;
   with fewer GPUs than processes, they share (r mod GPUs), and
   *sharers (if not NULL) is how many of the node's processes use the
   device (1 otherwise: their memory budgets divide by it). Returns 0
   if which_gpu does not exist */
uint32 gpu_select(const gpu_config_t *config, uint32 which_gpu,
			uint32 local_rank, uint32 local_size, uint32 *devs,
			uint32 *sharers);

#define CUDA_TRY(func) \
	{ 			 				\
		CUresult status = func;				\
		if (status != CUDA_SUCCESS) {			\
			const char *err_str = NULL;		\
			cuGetErrorString(status, &err_str);	\
			printf("CUDA error (%s: %d): %s\n", 	\
				__FILE__, __LINE__, err_str);	\
			exit(-1);				\
		}						\
	}

#define GPU_MAX_KERNEL_ARGS 20

typedef struct {
	CUfunction kernel_func;
	int32 threads_per_block;
	int32 num_regs;
	int32 static_shared_mem;
} gpu_launch_t;

void gpu_launch_init(CUmodule gpu_module, const char *func_name,
			gpu_launch_t *launch);

#ifdef __cplusplus
}
#endif

#endif /* HAVE_CUDA */

#endif /* !_CUDA_XFACE_H_ */
