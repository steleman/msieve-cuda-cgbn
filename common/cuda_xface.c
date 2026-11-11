/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: cuda_xface.c 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#include <cuda_xface.h>

#ifdef  HAVE_CUDA

/*------------------------------------------------------------------------*/
void
gpu_init(gpu_config_t *config)
{
	int32 i, j;
	static const CUdevice_attribute block_dim_attr[3] = {
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
	};
	static const CUdevice_attribute grid_dim_attr[3] = {
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
	};

	/* determine the specifics of CUDA GPUs using the 14 different
	   methods in the Nvidia documentation */

	memset(config, 0, sizeof(gpu_config_t));

	CUDA_TRY(cuInit(0))
	CUDA_TRY(cuDeviceGetCount(&config->num_gpu))
	if (config->num_gpu == 0)
		return;
	if (config->num_gpu > MAX_GPU)
		config->num_gpu = MAX_GPU;

	for (i = 0; i < (int32)config->num_gpu; i++) {
		CUdevice device;
		gpu_info_t *info = config->info + i;

		CUDA_TRY(cuDeviceGet(&device, i))

		info->device_handle = device;

		CUDA_TRY(cuDeviceGetName(info->name,
				sizeof(info->name), device))

		/* cuDeviceComputeCapability and cuDeviceGetProperties
		   are deprecated; query the same values as attributes */

		CUDA_TRY(cuDeviceGetAttribute(&info->compute_version_major,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				device))
		CUDA_TRY(cuDeviceGetAttribute(&info->compute_version_minor,
				CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				device))
		CUDA_TRY(cuDeviceGetAttribute(&info->clock_speed,
				CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device))
		CUDA_TRY(cuDeviceGetAttribute(&info->constant_mem_size,
				CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device))
		CUDA_TRY(cuDeviceGetAttribute(&info->shared_mem_size,
				CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
				device))
		CUDA_TRY(cuDeviceGetAttribute(&info->registers_per_block,
				CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
				device))
		CUDA_TRY(cuDeviceGetAttribute(&info->max_threads_per_block,
				CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				device))
		CUDA_TRY(cuDeviceGetAttribute(&info->warp_size,
				CU_DEVICE_ATTRIBUTE_WARP_SIZE, device))
		for (j = 0; j < 3; j++) {
			CUDA_TRY(cuDeviceGetAttribute(
				&info->max_thread_dim[j],
				block_dim_attr[j], device))
			CUDA_TRY(cuDeviceGetAttribute(
				&info->max_grid_size[j],
				grid_dim_attr[j], device))
		}
		
		CUDA_TRY(cuDeviceTotalMem(
			&info->global_mem_size, device))

		CUDA_TRY(cuDeviceGetAttribute(&info->can_overlap,
				CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device))

		CUDA_TRY(cuDeviceGetAttribute(&info->num_compute_units,
				CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
				device))

		CUDA_TRY(cuDeviceGetAttribute(&info->has_timeout,
				CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
				device))
	}
}

/*------------------------------------------------------------------------*/
void gpu_launch_init(CUmodule gpu_module, const char *func_name,
			gpu_launch_t *launch)
{
	memset(launch, 0, sizeof(gpu_launch_t));

	CUDA_TRY(cuModuleGetFunction(&launch->kernel_func,
			gpu_module, func_name))

	CUDA_TRY(cuFuncGetAttribute(&launch->threads_per_block,
			CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			launch->kernel_func))

	CUDA_TRY(cuFuncGetAttribute(&launch->num_regs,
			CU_FUNC_ATTRIBUTE_NUM_REGS,
			launch->kernel_func))

	CUDA_TRY(cuFuncGetAttribute(&launch->static_shared_mem,
			CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
			launch->kernel_func))
}

/*------------------------------------------------------------------------*/
uint32
gpu_select(const gpu_config_t *config, uint32 which_gpu,
		uint32 local_rank, uint32 local_size, uint32 *devs,
		uint32 *sharers)
{
	uint32 n = 0, ngpu = (uint32)MAX(0, config->num_gpu);

	local_size = MAX(1, local_size);
	if (sharers != NULL)
		*sharers = 1;
	if (which_gpu != 0xffffffffu) {
		/* -g: every process of the node on that one */
		if (which_gpu >= ngpu)
			return 0;
		devs[0] = which_gpu;
		if (sharers != NULL)
			*sharers = local_size;
		return 1;
	}
	if (ngpu == 0)
		return 0;
	if (ngpu < local_size) {
		devs[0] = local_rank % ngpu;
		if (sharers != NULL)
			*sharers = local_size / ngpu +
				(devs[0] < local_size % ngpu);
		return 1;
	}
	for (uint32 d = local_rank; d < ngpu; d += local_size)
		devs[n++] = d;
	return n;
}

#else

/* ISO C forbids an empty translation unit */
typedef int cuda_xface_not_needed;

#endif
