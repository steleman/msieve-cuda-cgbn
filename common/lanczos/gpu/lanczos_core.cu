/*--------------------------------------------------------------------
MIT License

Copyright (c) 2026 Stefan Teleman

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--------------------------------------------------------------------*/

/* GPU kernels of the block Lanczos iteration (lanczos_gpu.c): GF(2)
   sparse matrix times a block of VBITS vectors, and the vector-vector
   operations. A vector element is a v_t of VWORDS 64-bit words. The
   matrix is kept twice, by rows for b = A x and by columns for
   b = A^T x (a warp per row or column: measured 770-860 GB/s effective
   on an RTX PRO 4500, 8-12x faster than cuSPARSE, which has no GF(2)
   arithmetic); its first num_dense rows are bitmaps, dw 32-bit words
   per column */

#ifndef VBITS
#error "VBITS not set"
#endif
#define VWORDS ((VBITS + 63) / 64)

typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef struct {
	uint64 w[VWORDS];
} v_t;

__device__ __forceinline__ void
v_xor_into(v_t &a, const v_t &b)
{
	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		a.w[i] ^= b.w[i];
}

__device__ __forceinline__ v_t
v_load(const v_t *p)
{
	v_t r;
	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		r.w[i] = __ldg(&p->w[i]);
	return r;
}

__device__ __forceinline__ v_t
v_shfl_xor(v_t a, int s)
{
	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		a.w[i] = __shfl_xor_sync(0xffffffff, a.w[i], s);
	return a;
}

__device__ __forceinline__ v_t
v_shfl(v_t a, int lane)
{
	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		a.w[i] = __shfl_sync(0xffffffff, a.w[i], lane);
	return a;
}

/* the XOR of a over the warp, in every lane */
__device__ __forceinline__ v_t
v_warp_xor(v_t a)
{
	for (int s = 16; s > 0; s /= 2)
		v_xor_into(a, v_shfl_xor(a, s));
	return a;
}

/*------------------------------------------------------------------*/
/* b[r] = XOR of x[c] over the sparse entries of row r (rows are
   0..nrows-1; the dense rows' bits are added by la_dense_rows) */

extern "C" __global__ void
la_spmv_rows(const uint64 *off, const uint32 *col, const v_t *x, v_t *b,
		uint32 nrows)
{
	uint32 r = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	uint32 lane = threadIdx.x % 32;
	v_t acc;

	if (r >= nrows)
		return;
	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		acc.w[i] = 0;
	for (uint64 k = off[r] + lane; k < off[r + 1]; k += 32)
		v_xor_into(acc, v_load(x + col[k]));
	acc = v_warp_xor(acc);
	if (lane == 0)
		b[r] = acc;
}

/*------------------------------------------------------------------*/
/* the dense rows of b = A x: b[j] ^= XOR of x[c] over the columns c
   whose bitmap has bit j. Lane l of a warp takes row 32 w + l of each
   bitmap word w, over the warp's columns, 32 at a time */

extern "C" __global__ void
la_dense_rows(const uint32 *bits, uint32 dw, uint32 num_dense,
		const v_t *x, v_t *b, uint32 ncols)
{
	uint32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	uint32 nwarps = (gridDim.x * blockDim.x) / 32;
	uint32 lane = threadIdx.x % 32;

	for (uint32 w = 0; w < dw; w++) {
		v_t acc;

		#pragma unroll
		for (int i = 0; i < VWORDS; i++)
			acc.w[i] = 0;
		for (uint32 c0 = warp * 32; c0 < ncols; c0 += nwarps * 32) {
			uint32 c = c0 + lane;
			v_t xc;
			uint32 bw = 0;

			if (c < ncols) {
				xc = v_load(x + c);
				bw = __ldg(bits + (uint64)c * dw + w);
			}
			else {
				#pragma unroll
				for (int i = 0; i < VWORDS; i++)
					xc.w[i] = 0;
			}
			for (int t = 0; t < 32; t++) {
				v_t xt = v_shfl(xc, t);
				uint32 bt = __shfl_sync(0xffffffff, bw, t);

				if ((bt >> lane) & 1)
					v_xor_into(acc, xt);
			}
		}
		if (32 * w + lane < num_dense) {
			#pragma unroll
			for (int i = 0; i < VWORDS; i++) {
				if (acc.w[i])
					atomicXor(&b[32 * w + lane].w[i],
						acc.w[i]);
			}
		}
	}
}

/*------------------------------------------------------------------*/
/* b[c] = XOR of x[r] over the sparse entries of column c, and of
   x[j] over the dense rows j whose bit the column has (x[0..num_dense)
   in shared memory) */

extern "C" __global__ void
la_spmv_cols(const uint64 *off, const uint32 *row, const uint32 *bits,
		uint32 dw, uint32 num_dense, const v_t *x, v_t *b,
		uint32 ncols)
{
	extern __shared__ v_t xd[];
	uint32 c = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	uint32 lane = threadIdx.x % 32;
	v_t acc;

	for (uint32 j = threadIdx.x; j < num_dense; j += blockDim.x)
		xd[j] = x[j];
	__syncthreads();
	if (c >= ncols)
		return;

	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		acc.w[i] = 0;
	for (uint64 k = off[c] + lane; k < off[c + 1]; k += 32)
		v_xor_into(acc, v_load(x + row[k]));
	for (uint32 j = lane; j < num_dense; j += 32) {
		if ((__ldg(bits + (uint64)c * dw + j / 32) >> (j % 32)) & 1)
			v_xor_into(acc, xd[j]);
	}
	acc = v_warp_xor(acc);
	if (lane == 0)
		b[c] = acc;
}

/*------------------------------------------------------------------*/
extern "C" __global__ void
la_xor(v_t *dst, const v_t *src, uint32 n)
{
	for (uint32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x) {
		v_t a = dst[i];

		v_xor_into(a, src[i]);
		dst[i] = a;
	}
}

extern "C" __global__ void
la_mask(v_t *v, v_t mask, uint32 n)
{
	for (uint32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x) {
		v_t a = v[i];

		#pragma unroll
		for (int k = 0; k < VWORDS; k++)
			a.w[k] &= mask.w[k];
		v[i] = a;
	}
}

/*------------------------------------------------------------------*/
/* y[i] ^= v[i] * X, X a VBITS x VBITS matrix (X[k] is row k): with
   nibble tables, T[p][q] = XOR of X[4p + k] over the bits k of q */

#define NIB (VBITS / 4)

extern "C" __global__ void
la_mul_nxb(const v_t *v, const v_t *X, v_t *y, uint32 n)
{
	__shared__ v_t T[NIB][16];

	for (uint32 t = threadIdx.x; t < NIB * 16; t += blockDim.x) {
		uint32 p = t / 16, q = t % 16;
		v_t a;

		#pragma unroll
		for (int i = 0; i < VWORDS; i++)
			a.w[i] = 0;
		for (int k = 0; k < 4; k++) {
			if ((q >> k) & 1)
				v_xor_into(a, X[4 * p + k]);
		}
		T[p][q] = a;
	}
	__syncthreads();

	for (uint32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x) {
		v_t vi = v[i], acc = y[i];

		#pragma unroll
		for (uint32 p = 0; p < NIB; p++) {
			uint32 q = (uint32)(vi.w[p / 16] >> (4 * (p % 16))) & 15;

			v_xor_into(acc, T[p][q]);
		}
		y[i] = acc;
	}
}

/*------------------------------------------------------------------*/
/* out[k] ^= XOR over i of y[i] where bit k of x[i] is set, i.e.
   out = x^T y (VBITS x VBITS). A block takes tiles of RB elements into
   shared memory; thread t accumulates row k = t % VBITS over the tile
   elements congruent to t / VBITS; the partial rows are combined in
   shared memory and XORed into out (zeroed by the caller) */

#define RB 256

extern "C" __global__ void __launch_bounds__(RB)
la_mul_bxn(const v_t *x, const v_t *y, v_t *out, uint32 n)
{
	__shared__ v_t xs[RB], ys[RB];
	__shared__ v_t part[RB];
	const uint32 k = threadIdx.x % VBITS;
	const uint32 sub = threadIdx.x / VBITS;
	const uint32 nsub = RB / VBITS;
	v_t acc;

	#pragma unroll
	for (int i = 0; i < VWORDS; i++)
		acc.w[i] = 0;

	for (uint32 base = blockIdx.x * RB; base < n;
	     base += gridDim.x * RB) {
		uint32 i = base + threadIdx.x;

		__syncthreads();
		if (i < n) {
			xs[threadIdx.x] = x[i];
			ys[threadIdx.x] = y[i];
		}
		else {
			#pragma unroll
			for (int w = 0; w < VWORDS; w++)
				xs[threadIdx.x].w[w] = ys[threadIdx.x].w[w] = 0;
		}
		__syncthreads();
		for (uint32 e = sub; e < RB; e += nsub) {
			uint64 m = 0 - ((xs[e].w[k / 64] >> (k % 64)) & 1);

			#pragma unroll
			for (int w = 0; w < VWORDS; w++)
				acc.w[w] ^= ys[e].w[w] & m;
		}
	}

	__syncthreads();
	part[threadIdx.x] = acc;
	__syncthreads();
	if (threadIdx.x < VBITS) {
		v_t a = part[threadIdx.x];

		for (uint32 s = 1; s < nsub; s++)
			v_xor_into(a, part[s * VBITS + threadIdx.x]);
		#pragma unroll
		for (int w = 0; w < VWORDS; w++) {
			if (a.w[w])
				atomicXor(&out[threadIdx.x].w[w], a.w[w]);
		}
	}
}
