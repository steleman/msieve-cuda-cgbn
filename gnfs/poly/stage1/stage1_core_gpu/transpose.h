/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id$
--------------------------------------------------------------------*/

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <cub/cub.cuh>
#include "cuda_intrinsics.h"
#include "stage1_core.h"

template<size_t Bytes> struct XposeCacheTraits {
	static const uint32 CacheRows = 32;
	static const uint32 CacheCols = 32;
};

template<> 
struct XposeCacheTraits<8> {
	static const uint32 CacheRows = 16;
	static const uint32 CacheCols = 32;
};

template<typename T, uint32 THREADS_PER_BLOCK>
__device__ void TransposeOneBlock(
	const T * __restrict__ in,
	T * __restrict__ out,
	uint32 rows,
	uint32 cols,
	uint32 r0,
	uint32 c0,
	uint32 rcount,
	uint32 ccount)
{
	// read in(r0..(r0+rcount),c0..(c0+ccount)) with 
	// unit stride, transpose in shared memory, 
	// write out(c0..(c0+ccount),r0..(r0+rcount)) with 
	// unit stride
	//
	// standard trick: make shared memory cache rows
	// one element too long, to avoid bank conflicts
	// writing down each column

	const uint32 CR = XposeCacheTraits<sizeof(T)>::CacheRows;
	const uint32 CC = XposeCacheTraits<sizeof(T)>::CacheCols;
	__shared__ T cache[CC][CR+1];

	uint32 block_count = rcount * ccount;

	const T * block_in = in + r0 * cols + c0;
	for (uint32 i = threadIdx.x; i < block_count; i += THREADS_PER_BLOCK) {
		uint32 r, c;
		if (ccount == CC) { // fast path
			r = i / CC;
			c = i % CC;
		}
		else {
			r = i / ccount;
			c = i % ccount;
		}
		cache[c][r] = block_in[r * cols + c];
	}

	__syncthreads();

	T * block_out = out + c0 * rows + r0;
	for (uint32 i = threadIdx.x; i < block_count; i += THREADS_PER_BLOCK) {
		uint32 r, c;
		if (rcount == CR) { // fast path
			c = i / CR;
			r = i % CR;
		}
		else {
			c = i / rcount;
			r = i % rcount;
		}
		block_out[c * rows + r] = cache[c][r];
	}

	// prevent shared memory from being corrupted if this function
	// is called in a loop

	__syncthreads();
}

template<typename T, uint32 THREADS_PER_BLOCK>
__device__ void Transpose(
	const T * __restrict__ in,
	T * __restrict__ out,
	uint32 rows,
	uint32 cols,
	uint32 num_arrays)
{
	// get the preferred dimensions of a cache block, which
	// the host should not have to know

	const uint32 CR = XposeCacheTraits<sizeof(T)>::CacheRows;
	const uint32 CC = XposeCacheTraits<sizeof(T)>::CacheCols;

	// number of cache blocks in one array

	const uint32 block_rows = (rows + CR - 1) / CR;
	const uint32 block_cols = (cols + CC - 1) / CC;
	const uint32 array_blocks = block_rows * block_cols;

	// dimension of whole problem

	const uint32 total_blocks = num_arrays * array_blocks;
	const uint32 stride = rows * cols;

	// proceed a grid at a time

	for (uint32 i = blockIdx.x; i < total_blocks; i += gridDim.x) {

		// locate the array that block i operates on and
		// the bounds within that array

		uint32 curr_array = i / array_blocks;
		uint32 curr_block = i % array_blocks;
		uint32 curr_row_start = CR * (curr_block / block_cols);
		uint32 curr_col_start = CC * (curr_block % block_cols);

		TransposeOneBlock<T, THREADS_PER_BLOCK>(
				in + curr_array * stride,
				out + curr_array * stride,
				rows, cols,
				curr_row_start, curr_col_start,
				min(CR, rows - curr_row_start),
				min(CC, cols - curr_col_start));
	}
}

extern "C"
{

__global__ void
transpose_32(
	const uint32 * __restrict__ in,
	uint32 * __restrict__ out,
	uint32 rows,
	uint32 cols,
	uint32 num_arrays)
{
	Transpose<uint32, TRANSFORM_NUM_THREADS>(
			in, out, rows, cols, num_arrays);
}

} // extern "C"

#endif // !TRANSPOSE_H
