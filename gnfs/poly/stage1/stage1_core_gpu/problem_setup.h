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

#ifndef PROBLEM_SETUP_H
#define PROBLEM_SETUP_H

#include <cub/cub.cuh>
#include "cuda_intrinsics.h"
#include "stage1_core.h"

// for integer p, the type of a root mod p^2
template<typename PType> struct RootTraits;
template<> struct RootTraits<uint16> { typedef uint32 RootType; };
template<> struct RootTraits<uint32> { typedef uint64 RootType; };
template<> struct RootTraits<uint64> { typedef uint128 RootType; };

// arithmetic with p root type PType and q root type QType
template<typename PType, typename QType> 
struct Arith
{
	typedef typename RootTraits<PType>::RootType PPType;
	typedef typename RootTraits<QType>::RootType QQType;

	static inline __device__ PPType sqr(PType p) {
		return static_cast<PPType>(p) *
			static_cast<PPType>(p);
	}

	static inline __device__ PType gcd(PType p, QType q) {
		return ::gcd(q, static_cast<QType>(p));
	}

	static inline __device__ PPType modsub(
				PPType p, PPType q, PPType n) {
		return ::modsub(p, q, n);
	}

	static inline __device__ PPType modinv(PPType p, PPType q) {
		return ::modinv(p, q);
	}

	static inline __device__ PPType montmul(PPType x, PPType y, 
					PPType n, PPType w) {
		return ::montmul(x, y, n, w);
	}

	// for mod operations, when the numerator has a wider
	// type than denominator we use multi-word Montgomery 
	// arithmetic. When the numerator has number of words 
	// <= denominator words then a single Montgomery multiply 
	// by 2^(bits in PPType) is enough to compute the mod
	//
	// use constexpr to avoid compile errors using static_cast 
	// for the homegrown uint128 type
	//
	// Note that t has to be chosen so that the Montgomery
	// arithmetic works correctly for the specific sizes of
	// x and n below

	template<typename NType>
	static inline __device__ PPType mod(NType x, PPType t, 
						PPType n, PPType w) {
		if constexpr (sizeof(NType) <= sizeof(PPType)) {
			PPType xx = static_cast<PPType>(x);
			if (xx < n)
				return xx;
			return ::montmul(xx, t, n, w);
		}
		else {
			return ::mod(x, t, n, w);
		}
	}
};

// specializations

template<> inline __device__ uint64 
Arith<uint32, uint64>::sqr(uint32 p)
{
	return wide_sqr32(p);
}

// ------------- generic batch modular inversion ----------------------

template<typename PType, typename QType> 
struct Inverter
{
	typedef Arith<PType, QType> arith;
	typedef typename arith::PPType PPType;
	typedef typename arith::QQType QQType;

	static __device__ void batch_invert(
		const PType * __restrict__ p_in, 
		const PPType * __restrict__ mont_w, 
		const PPType * __restrict__ mont_r, 
		const PPType * __restrict__ mont_t, 
		const QType * __restrict__ q_in, 
		PPType * __restrict__ q_inv, 
		PPType * __restrict__ q_inv_scratch,
		uint32 q_batch_size, 
		uint32 num_p, 
		uint32 num_q)
	{
		uint32 num_threads = gridDim.x * blockDim.x;
		uint32 q_min = q_batch_size * blockIdx.y;
		uint32 q_max = min(num_q, q_min + q_batch_size);

		for (uint32 my_p = blockIdx.x * blockDim.x + threadIdx.x;
				my_p < num_p; my_p += num_threads) {

			uint32 j;
			PType p = p_in[my_p];
			PPType pp = arith::sqr(p);
			PPType pp_w = mont_w[my_p];
			PPType pp_r = mont_r[my_p];
			PPType pp_t = mont_t[my_p];
			PPType * inv_ptr = q_inv + q_min * num_p + my_p;
			PPType * qsave_ptr = q_inv_scratch + q_min * num_p + my_p;
			PPType invprod = 0;
			uint32 first = 0;

			for (j = q_min; j < q_max; j++, inv_ptr += num_p, 
						qsave_ptr += num_p) {

				// reduce q mod pp

				QType q = q_in[j];
				PPType qmodpp = arith::mod(q, pp_t, pp, pp_w);

				// current p and current q must be coprime
				if (arith::gcd(p, q) != 1) {
					*qsave_ptr = 0;
				}
				else {
					// save q and accumulate without converting
					// to Montgomery form. This means invprod
					// is scaled by an increasing negative power 
					// of 2

					*qsave_ptr = qmodpp;
					if (invprod == 0) {
						first = j;
						invprod = qmodpp;
					}
					else {
						invprod = arith::montmul(invprod, 
								qmodpp, pp, pp_w);
					}
				}
				*inv_ptr = invprod;
			}

			j--;
			inv_ptr -= num_p;
			qsave_ptr -= num_p;
			if (invprod) {
				// invert the accumulated product and convert
			 	// (once) to Montgomery form. This turns the
			 	// negative power of 2 scale factor positive 

				invprod = arith::modinv(invprod, pp);
				invprod = arith::montmul(invprod, pp_r, pp, pp_w);

				// peel off results in reverse order

				for (; j > first; j--, inv_ptr -= num_p, 
							qsave_ptr -= num_p) {

					PPType *t = inv_ptr - num_p;
					PPType res;
					PPType curr_q = *qsave_ptr;

					if (curr_q == 0) {
						*inv_ptr = 0;
						continue;
					}

					// multiplying by the previous invprod
			 		// will cancel out all the previous q's 
			 		// and almost all the powers of 2, leaving 
			 		// the inverse of the j_th q in Montgomery 
			 		// form. Square that to compute the j_th 
			 		// result, then remove the inverse of the 
			 		// current q from invprod

					res = arith::montmul(invprod, *t, pp, pp_w);
					*inv_ptr = arith::montmul(res, res, pp, pp_w);
					invprod = arith::montmul(invprod, curr_q, pp, pp_w);
				}

				// initial nonzero result 
				*inv_ptr = arith::montmul(invprod, invprod, pp, pp_w);
				j--; inv_ptr -= num_p;
			}
			for (; (int32)j >= (int32)q_min; j--, inv_ptr -= num_p)
				*inv_ptr = 0;
		}
	}
};

extern "C" {

__global__ void
batch_invert_p16_q32(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_r, 
	const uint32 * __restrict__ mont_t, 
	const uint32 * __restrict__ q_in, 
	uint32 * __restrict__ q_inv, 
	uint32 * __restrict__ q_inv_scratch,
	uint32 q_batch_size, uint32 num_p, uint32 num_q)
{
	Inverter<uint16, uint32>::batch_invert(
		p_in, mont_w, mont_r, mont_t, q_in, 
		q_inv, q_inv_scratch, q_batch_size, num_p, num_q);
}

__global__ void
batch_invert_p32_q32(
	const uint32 * __restrict__ p_in, 
	const uint64 * __restrict__ mont_w, 
	const uint64 * __restrict__ mont_r, 
	const uint64 * __restrict__ mont_t, 
	const uint32 * __restrict__ q_in, 
	uint64 * q_inv, 
	uint64 * q_inv_scratch,
	uint32 q_batch_size, uint32 num_p, uint32 num_q)
{
	Inverter<uint32, uint32>::batch_invert(
		p_in, mont_w, mont_r, mont_t, q_in, 
		q_inv, q_inv_scratch, q_batch_size, num_p, num_q);
}

__global__ void
batch_invert_p16_q64(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_r, 
	const uint32 * __restrict__ mont_t, 
	const uint64 * __restrict__ q_in, 
	uint32 * __restrict__ q_inv, 
	uint32 * __restrict__ q_inv_scratch,
	uint32 q_batch_size, uint32 num_p, uint32 num_q)
{
	Inverter<uint16, uint64>::batch_invert(
		p_in, mont_w, mont_r, mont_t, q_in, 
		q_inv, q_inv_scratch, q_batch_size, num_p, num_q);
}

__global__ void
batch_invert_p32_q64(
	const uint32 * __restrict__ p_in, 
	const uint64 * __restrict__ mont_w, 
	const uint64 * __restrict__ mont_r, 
	const uint64 * __restrict__ mont_t, 
	const uint64 * __restrict__ q_in, 
	uint64 * __restrict__ q_inv, 
	uint64 * __restrict__ q_inv_scratch,
	uint32 q_batch_size, uint32 num_p, uint32 num_q)
{
	Inverter<uint32, uint64>::batch_invert(
		p_in, mont_w, mont_r, mont_t, q_in, 
		q_inv, q_inv_scratch, q_batch_size, num_p, num_q);
}

} // extern "C"


//------------------ generic batch root transform ------------------------

template<typename PType, typename QType, 
		typename OutKeyType, typename OutPType> 
struct Transform
{
	typedef Arith<PType, QType> arith;
	typedef typename arith::PPType PPType;
	typedef typename arith::QQType QQType;

	static __device__ void batch_transform(
		const PType * __restrict__ p_in, 
		const PPType * __restrict__ mont_w, 
		const PPType * __restrict__ mont_t, 
		const PPType * __restrict__ ss_mod_pp,
		const PPType * __restrict__ q_inv_mod_pp,
		const uint32 * __restrict__ which_p,
		const PPType * __restrict__ p_roots,
		const uint32 * __restrict__ which_q,
		const QQType * __restrict__ q_roots,
		uint32 q_root_start,
		uint64 sieve_size,
		uint32 num_p,
		uint32 num_p_roots,
		uint32 num_q_roots,
		uint32 * __restrict__ out_array_sizes,
		OutKeyType * __restrict__ out_array_key,
		OutPType * __restrict__ out_array_data,
		uint32 max_out_array_size,
		uint32 out_array_batch_size)
	{
		uint32 num_threads = gridDim.x * blockDim.x;
		uint32 out_min = out_array_batch_size * blockIdx.y;
		uint32 out_max = min(num_q_roots, 
					out_min + out_array_batch_size);

		typedef cub::BlockScan<uint32, TRANSFORM_NUM_THREADS, 
				cub::BLOCK_SCAN_RAKING> PrefixSum;
		__shared__ PrefixSum::TempStorage prefix_tmp;
		__shared__ uint32 store_bcast;

		// loop over p roots; since the inner loop includes 
		// a thread sync and CUB prefix sum, we must have a 
		// complete thread block participate

		for (uint32 proot_start = blockIdx.x * blockDim.x;
				proot_start < num_p_roots;
				proot_start += num_threads) {

			uint32 my_proot = proot_start + threadIdx.x;
			uint32 my_proot_clamp = min(my_proot, num_p_roots - 1);

			uint32 p_idx = which_p[my_proot_clamp];
			PPType pp_root = p_roots[my_proot_clamp];

			PType p = p_in[p_idx];
			PPType pp = arith::sqr(p);
			PPType pp_w = mont_w[p_idx];
			PPType pp_t = mont_t[p_idx];
			PPType ss_pp = ss_mod_pp[p_idx];
			OutKeyType * curr_out_key = out_array_key + 
					max_out_array_size * out_min;
			OutPType * curr_out_data = out_array_data + 
					max_out_array_size * out_min;

			// loop over q roots

			for (uint32 i = out_min; i < out_max; i++) {

				// transform q root i for a block of p roots 

				PPType res;
				uint32 my_out_offset;
				uint32 block_sum;
				uint32 count = 0;

				PPType q_inv = q_inv_mod_pp[p_idx + num_p * 
							which_q[q_root_start + i]];
				QQType q_root = q_roots[q_root_start + i];
				PPType q_root_mod = arith::mod(q_root, pp_t, pp, pp_w);

				res = arith::modsub(pp_root, q_root_mod, pp);
				res = arith::montmul(res, q_inv, pp, pp_w);
				res = arith::modsub(res, ss_pp, pp);

				if (q_inv && my_proot < num_p_roots) {
					// p and q are coprime, and res is guaranteed 
					// to be less than sieve_size and thus is 
					// usable for the collision search

					// res + k * pp may also be < sieve_size for 
					// a few k, and this represents almost free extra
					// raw material for the collision search. 
					// Unfortunately each thread is entitled to
					// a different value of k, and storing them
					// contiguously requires the whole CUDA block
					// to cooperate 

					// when OutKeyType is uint32 we assume sieve_size
					// is close to 2^32 and need to take care to 
					// avoid overflow of intermediate quantities

					OutKeyType t = static_cast<OutKeyType>(res);
					OutKeyType ppk = static_cast<OutKeyType>(pp);
					OutKeyType limit = static_cast<OutKeyType>(
								sieve_size) - ppk;
					count = 1;
					while (t < limit) {
						count++;
						t += ppk;
					}
				}

				PrefixSum(prefix_tmp).ExclusiveSum(count, 
						my_out_offset, block_sum);

				// have thread 0 reserve space in the output arrays
				// for all the other threads 

				if (threadIdx.x == 0) {
					store_bcast = atomicAdd(out_array_sizes + i, 
								block_sum);
				}
				__syncthreads();

				// store results 
				{
					uint32 out_off = store_bcast;
					OutKeyType t = static_cast<OutKeyType>(res);
					OutKeyType ppk = static_cast<OutKeyType>(pp);
					for (uint32 j = 0; j < count; j++) {
						uint32 off = out_off + my_out_offset + j;
						if (off < max_out_array_size) {
							curr_out_key[off] = t;
							curr_out_data[off] = 
								static_cast<OutPType>(p);
						}
						t += ppk;
					}
				}
				curr_out_key += max_out_array_size;
				curr_out_data += max_out_array_size;
			}
		}
	}
};

/*------------------------------------------------------------------------*/
extern "C"
{

__global__ void
transform_p16_q32_k32_d16(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_t, 
	const uint32 * __restrict__ ss_mod_pp,
	const uint32 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p,
	const uint32 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint64 * __restrict__ q_roots, 
	uint32 q_root_start, 
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint32 * __restrict__ out_array_key, 
	uint16 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint16, uint32, uint32, uint16>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

__global__ void
transform_p16_q32_k64_d32(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_t, 
	const uint32 * __restrict__ ss_mod_pp,
	const uint32 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p, 
	const uint32 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint64 * __restrict__ q_roots, 
	uint32 q_root_start,
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint64 * __restrict__ out_array_key, 
	uint32 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint16, uint32, uint64, uint32>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

__global__ void
transform_p16_q64_k32_d16(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_t, 
	const uint32 * __restrict__ ss_mod_pp,
	const uint32 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p, 
	const uint32 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint128 * __restrict__ q_roots, 
	uint32 q_root_start,
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint32 * __restrict__ out_array_key, 
	uint16 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint16, uint64, uint32, uint16>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

__global__ void
transform_p16_q64_k64_d32(
	const uint16 * __restrict__ p_in, 
	const uint32 * __restrict__ mont_w, 
	const uint32 * __restrict__ mont_t, 
	const uint32 * __restrict__ ss_mod_pp,
	const uint32 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p, 
	const uint32 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint128 * __restrict__ q_roots, 
	uint32 q_root_start,
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint64 * __restrict__ out_array_key, 
	uint32 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint16, uint64, uint64, uint32>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

__global__ void
transform_p32_q32_k64_d32(
	const uint32 * __restrict__ p_in, 
	const uint64 * __restrict__ mont_w, 
	const uint64 * __restrict__ mont_t, 
	const uint64 * __restrict__ ss_mod_pp,
	const uint64 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p, 
	const uint64 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint64 * __restrict__ q_roots, 
	uint32 q_root_start,
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint64 * __restrict__ out_array_key, 
	uint32 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint32, uint32, uint64, uint32>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

__global__ void
transform_p32_q64_k64_d32(
	const uint32 * __restrict__ p_in, 
	const uint64 * __restrict__ mont_w, 
	const uint64 * __restrict__ mont_t, 
	const uint64 * __restrict__ ss_mod_pp,
	const uint64 * __restrict__ q_inv_mod_pp, 
	const uint32 * __restrict__ which_p, 
	const uint64 * __restrict__ p_roots,
	const uint32 * __restrict__ which_q, 
	const uint128 * __restrict__ q_roots, 
	uint32 q_root_start,
	uint64 sieve_size, 
	uint32 num_p, 
	uint32 num_p_roots,
	uint32 num_q_roots, 
	uint32 * __restrict__ out_array_sizes,
	uint64 * __restrict__ out_array_key, 
	uint32 * __restrict__ out_array_data,
	uint32 max_out_array_size, 
	uint32 out_array_batch_size)
{
	Transform<uint32, uint64, uint64, uint32>::batch_transform(
		p_in, mont_w, mont_t, ss_mod_pp, q_inv_mod_pp, which_p,
		p_roots, which_q, q_roots, q_root_start, sieve_size,
		num_p, num_p_roots, num_q_roots, out_array_sizes, out_array_key,
		out_array_data, max_out_array_size, out_array_batch_size);
}

} // extern "C"

#endif // !PROBLEM_SETUP_H
