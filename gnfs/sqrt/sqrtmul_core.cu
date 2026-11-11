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

/* GPU kernels of the large integer multiplication of the NFS square
   root (sqrt_gpu_mul.c): a number theoretic transform modulo three
   primes c 2^40 + 1 below 2^62, the operands cut into 64-bit pieces;
   each coefficient of the convolution is below n 2^128, far below the
   product of the primes, so a CRT (Garner) gives it exactly, and the
   host propagates the carries. The product is mpz_mul's.

   Ported from the author's GPU square root multiplication for
   CADO-NFS (sqrt/sqrt_gpu_mul.cu of the cado-nfs-gpu fork), written by
   the same author and released here under the MIT license: the
   kernels are the same, the host side uses the driver API and loads
   this file as PTX, as msieve's other GPU code does. */

#include "sqrtmul_core.h"

typedef unsigned long long u64;

__device__ const unsigned int sm_abi[2] =
		{ SM_ABI, (unsigned int)sizeof(sm_crt_t) };

__constant__ sm_prime_t c_pr[SM_NP];
__constant__ sm_crt_t c_crt;

/* Montgomery arithmetic, m odd, m < 2^63 */
__device__ __forceinline__ u64
redc(u64 hi, u64 lo, u64 m, u64 minv)
{
	const u64 q = lo * minv;
	const u64 r = hi + __umul64hi(q, m) + (lo != 0);
	return r >= m ? r - m : r;
}

__device__ __forceinline__ u64
mulredc(u64 a, u64 b, u64 m, u64 minv)
{
	return redc(__umul64hi(a, b), a * b, m, minv);
}

__device__ __forceinline__ u64
addm(u64 a, u64 b, u64 m)
{
	u64 r = a + b;
	return r >= m ? r - m : r;
}

__device__ __forceinline__ u64
subm(u64 a, u64 b, u64 m)
{
	return a >= b ? a - b : a + m - b;
}

/* the operand, reduced mod each prime, in Montgomery form, padded to N
   (blockIdx.y: the prime) */
extern "C" __global__ void
sm_load(u64 *A, const u64 *a, u64 n, u64 N)
{
	const int j = blockIdx.y;
	const sm_prime_t P = c_pr[j];

	for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < N;
	     i += (u64)gridDim.x * blockDim.x) {
		const u64 x = i < n ? a[i] % P.p : 0;

		A[j * N + i] = mulredc(x, P.r2, P.p, P.minv);
	}
}

/* T[k] = w^k (Montgomery), k < N/2, w of order N (or its inverse) */
extern "C" __global__ void
sm_twiddle(u64 *T, u64 half, const u64 *wR)
{
	const int j = blockIdx.y;
	const sm_prime_t P = c_pr[j];

	for (u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x; k < half;
	     k += (u64)gridDim.x * blockDim.x) {
		u64 r = mulredc(1, P.r2, P.p, P.minv);
		u64 b = wR[j];

		for (u64 e = k; e; e >>= 1, b = mulredc(b, b, P.p, P.minv))
			if (e & 1)
				r = mulredc(r, b, P.p, P.minv);
		T[j * half + k] = r;
	}
}

/* one stage of the forward transform (decimation in frequency): blocks
   of len = 2 h, natural order in, bit-reversed order out */
extern "C" __global__ void
sm_dif(u64 *A, u64 N, u64 h, const u64 *T)
{
	const int j = blockIdx.y;
	const sm_prime_t P = c_pr[j];
	u64 *a = A + j * N;
	const u64 *t = T + j * (N / 2);
	const u64 stride = N / (2 * h);

	for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < N / 2;
	     i += (u64)gridDim.x * blockDim.x) {
		const u64 jj = i % h, blk = i / h;
		const u64 i0 = blk * 2 * h + jj, i1 = i0 + h;
		const u64 u = a[i0], v = a[i1];

		a[i0] = addm(u, v, P.p);
		a[i1] = mulredc(subm(u, v, P.p), t[jj * stride], P.p, P.minv);
	}
}

/* one stage of the inverse transform (decimation in time) */
extern "C" __global__ void
sm_dit(u64 *A, u64 N, u64 h, const u64 *T)
{
	const int j = blockIdx.y;
	const sm_prime_t P = c_pr[j];
	u64 *a = A + j * N;
	const u64 *t = T + j * (N / 2);
	const u64 stride = N / (2 * h);

	for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < N / 2;
	     i += (u64)gridDim.x * blockDim.x) {
		const u64 jj = i % h, blk = i / h;
		const u64 i0 = blk * 2 * h + jj, i1 = i0 + h;
		const u64 u = a[i0];
		const u64 v = mulredc(a[i1], t[jj * stride], P.p, P.minv);

		a[i0] = addm(u, v, P.p);
		a[i1] = subm(u, v, P.p);
	}
}

extern "C" __global__ void
sm_pointwise(u64 *A, const u64 *B, u64 N)
{
	const int j = blockIdx.y;
	const sm_prime_t P = c_pr[j];

	for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < N;
	     i += (u64)gridDim.x * blockDim.x)
		A[j * N + i] = mulredc(A[j * N + i], B[j * N + i], P.p, P.minv);
}

/* coefficient i of the convolution, as 3 limbs */
extern "C" __global__ void
sm_crt(u64 *out, const u64 *A, u64 N, u64 nr, sm_ninv_t ninv)
{
	const sm_prime_t P1 = c_pr[0], P2 = c_pr[1], P3 = c_pr[2];

	for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < nr;
	     i += (u64)gridDim.x * blockDim.x) {
		/* residues: Montgomery form v R; times N^-1 and out of it */
		const u64 r1 = mulredc(A[i], ninv.v[0], P1.p, P1.minv);
		const u64 r2 = mulredc(A[N + i], ninv.v[1], P2.p, P2.minv);
		const u64 r3 = mulredc(A[2 * N + i], ninv.v[2], P3.p, P3.minv);
		/* Garner: x = x1 + p1 x2 + p1 p2 x3 */
		const u64 x1 = r1;
		const u64 x2 = mulredc(subm(r2, x1 % P2.p, P2.p),
				c_crt.p1_inv_mod_p2R, P2.p, P2.minv);
		const u64 t = addm(x1 % P3.p, mulredc(x2 % P3.p,
				c_crt.p1_mod_p3R, P3.p, P3.minv), P3.p);
		const u64 x3 = mulredc(subm(r3, t, P3.p),
				c_crt.p1p2_inv_mod_p3R, P3.p, P3.minv);
		/* p1 x2 (2 limbs) + p1 p2 x3 (3 limbs) + x1 */
		u64 l0 = P1.p * x2, l1 = __umul64hi(P1.p, x2), l2;
		u64 s, c, cc, m0, m1, c1, m2;

		s = l0 + x1; l1 += s < l0; l0 = s;
		m0 = c_crt.p1p2[0] * x3;
		m1 = __umul64hi(c_crt.p1p2[0], x3) + c_crt.p1p2[1] * x3;
		c1 = m1 < c_crt.p1p2[1] * x3;
		m2 = __umul64hi(c_crt.p1p2[1], x3) + c1;
		s = l0 + m0; c = s < l0; l0 = s;
		s = l1 + m1; cc = s < l1; s += c; cc += s < c; l1 = s;
		l2 = m2 + cc;
		out[3 * i] = l0;
		out[3 * i + 1] = l1;
		out[3 * i + 2] = l2;
	}
}
