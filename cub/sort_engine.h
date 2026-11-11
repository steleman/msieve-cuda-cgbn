/* simple DSO interface for parallel segmented sort;
   keys are 32- or 64-bit unsigned integers and values are
   32-bit unsigned integers. The input struct assumes
   many small-to-medium size arrays are batched together,
   and the start and end offsets into a single unified
   array are provided */

#ifndef _SORT_ENGINE_H_
#define _SORT_ENGINE_H_

#include <stdlib.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	CUdeviceptr start_offsets;
	CUdeviceptr end_offsets;

	CUdeviceptr keys_in;
	CUdeviceptr keys_in_scratch;
	CUdeviceptr data_in;
	CUdeviceptr data_in_scratch;
	CUstream stream;

	size_t num_arrays;
	size_t max_array_size;
	int key_bits;
	int sort_bits;
} sort_data_t;

typedef void * (*sort_engine_init_func)(void);

typedef void (*sort_engine_free_func)(void * engine);

typedef void (*sort_engine_run_func)(void * engine, 
				sort_data_t * sort_data);

#ifdef __cplusplus
}
#endif

#endif /* !_SORT_ENGINE_H_ */
