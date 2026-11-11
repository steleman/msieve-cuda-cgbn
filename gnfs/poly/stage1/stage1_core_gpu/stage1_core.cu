/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: stage1_core.cu 1112 2026-11-02 02:13:30Z jasonp_sf $
--------------------------------------------------------------------*/

#include <cub/cub.cuh>
#include "cuda_intrinsics.h"
#include "stage1_core.h"
#include "problem_setup.h"
#include "transpose.h"

/*------------------------------------------------------------------------*/
template<typename KeyType, uint32 THREADS_PER_BLOCK>
__device__ void BucketSortPass1_OneBlock(
	const KeyType * __restrict__ keys,
	uint32 * __restrict__ key_prefix,
	uint32 * __restrict__ sorted_offsets,
	uint32 array_size,
	uint32 subarray_size,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 bucket_hash_bits)
{
	uint32 key_start = blockIdx.x * subarray_size;
	uint32 key_end = min(array_size, key_start + subarray_size);
	const uint32 num_bins = 1 << bucket_hash_bits;
	const uint32 bin_mask = num_bins - 1;

	typedef cub::BlockScan<uint32, THREADS_PER_BLOCK, 
				cub::BLOCK_SCAN_RAKING> PrefixSum;

	extern __shared__ uint32 buckets[]; // num_bins entries
	uint32 * cache = buckets + num_bins; // subarray_size entries

	__shared__ struct {
		typename PrefixSum::TempStorage prefix_tmp;
	} shared_data;

	// clear the bucket sizes

	for (uint32 i = threadIdx.x; i < num_bins; 
				i += THREADS_PER_BLOCK) {
		buckets[i] = 0;
	}
	__syncthreads();

	// count the keys in each bin

	for (uint32 i = key_start + threadIdx.x; i < key_end; 
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bin = static_cast<uint32>(key) & bin_mask;
		atomicAdd(buckets + bin, 1);
	}

	// proceed through the bins THREADS_PER_BLOCK at a time
	// (the prefix sum needs all threads contributing)

	uint32 key_count = 0;

	for (uint32 batch_start = 0; batch_start < num_bins;
			batch_start += THREADS_PER_BLOCK) {

		// fence off shared memory from previous loop

		__syncthreads();

		uint32 my_bin = batch_start + threadIdx.x;
		uint32 my_size = 0;

		if (my_bin < num_bins)
			my_size = buckets[my_bin];

		// prefix sum the bin sizes

		uint32 my_off;
		uint32 curr_count;
		PrefixSum(shared_data.prefix_tmp).ExclusiveSum(
				my_size, my_off, curr_count);

		if (my_bin < num_bins) {
			// save the bin offset

			buckets[my_bin] = key_count + my_off;

			// save the global array offset and
			// the count in the current bin; we pack
			// into a single word to avoid needing
			// the next word to determine the size,
			// making all these entries independent
			// (but limiting the array size to 2^24)

			sorted_offsets[blockIdx.x * num_bins + my_bin] = 
					(my_size << 24) |
					(key_start + key_count + my_off);
		}
		key_count += curr_count;
	}

	__syncthreads();

	// store a prefix of the key above bucket_hash_bits in 
	// their respective bin; reuse the top half of each 
	// bucket entry for the current offset of the chosen bin,
	// to avoid needing a separate shared memory array
	//
	// We also need a way to locate key i if it turns out to 
	// participate in a collision. One option to do that is 
	// to store (i - key_start) in a separate array, but 
	// the extra memory traffic this generates makes pass 1 
	// take 50% longer. So instead we store a smaller prefix 
	// of key i and concatenate the top few bits of 
	// (i - key_start). This will narrow down the location of 
	// i to a small enough range that it can be searched 
	// exhaustively the few times that a collision occurs.
	// The host chooses prefix_bits and loc_shift so that the
	// location of every key in the largest subarray fits

	uint32 prefix_mask = (1 << prefix_bits) - 1;

	for (uint32 i = key_start + threadIdx.x; i < key_end; 
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bin = (uint32)key & bin_mask;
		uint32 bin_val = atomicAdd(buckets + bin, 1 << 16);
		uint32 offset = (bin_val & 0xffff) + (bin_val >> 16);
		uint32 word = ((i - key_start) >> loc_shift) << prefix_bits |
				(prefix_mask & static_cast<uint32>(
				key >> bucket_hash_bits));

		cache[offset] = word;
	}

	// writing each word straight to global memory would
	// scatter the writes over as many places as there are
	// bins, and with many bins those writes combine poorly.
	// So the subarray is sorted in shared memory above, and
	// written out in order here

	{
		uint32 num_keys = (key_end > key_start) ? 
					key_end - key_start : 0;

		__syncthreads();
		for (uint32 i = threadIdx.x; i < num_keys; 
					i += THREADS_PER_BLOCK) {
			key_prefix[key_start + i] = cache[i];
		}
	}

	// prevent shared memory from being corrupted if this function
	// is called in a loop

	__syncthreads();
}

template<typename KeyType, uint32 THREADS_PER_BLOCK>
__device__ void BucketSortPass1(
	const KeyType * __restrict__ keys,
	uint32 * __restrict__ key_prefix,
	uint32 * __restrict__ sorted_offsets,
	const uint32 * __restrict__ array_sizes,
	uint32 max_array_size,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 num_key_arrays,
	uint32 key_array_batch)
{
	uint32 array_min = key_array_batch * blockIdx.y;
	uint32 array_max = min(num_key_arrays, array_min + key_array_batch);

	for (uint32 i = array_min; i < array_max; i++) {

		// clip the size of array i, divide it into
		// a fixed number of pieces and process each
		// independently

		uint32 array_size = min(array_sizes[i], max_array_size);
		uint32 subarray_size = (array_size + gridDim.x - 1) / gridDim.x;

		BucketSortPass1_OneBlock<KeyType, THREADS_PER_BLOCK>(
				keys + i * max_array_size,
				key_prefix + i * max_array_size,
				sorted_offsets + i * 
					(gridDim.x << bucket_hash_bits),
				array_size, subarray_size, prefix_bits,
				loc_shift, bucket_hash_bits);
	}
}

extern "C"
{

__global__ void
bucket_sort_pass1_32(
	const uint32 * __restrict__ keys,
	uint32 * __restrict__ key_prefix,
	uint32 * __restrict__ sorted_offsets,
	const uint32 * __restrict__ array_sizes,
	uint32 max_array_size,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 num_key_arrays,
	uint32 key_array_batch)
{
	BucketSortPass1<uint32, TRANSFORM_NUM_THREADS>(
			keys, key_prefix,
			sorted_offsets, array_sizes, 
			max_array_size, bucket_hash_bits, 
			prefix_bits, loc_shift, 
			num_key_arrays, key_array_batch);
}

__global__ void
bucket_sort_pass1_64(
	const uint64 * __restrict__ keys,
	uint32 * __restrict__ key_prefix,
	uint32 * __restrict__ sorted_offsets,
	const uint32 * __restrict__ array_sizes,
	uint32 max_array_size,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 num_key_arrays,
	uint32 key_array_batch)
{
	BucketSortPass1<uint64, TRANSFORM_NUM_THREADS>(
			keys, key_prefix,
			sorted_offsets, array_sizes, 
			max_array_size, bucket_hash_bits, 
			prefix_bits, loc_shift, 
			num_key_arrays, key_array_batch);
}

} // extern "C"

/*------------------------------------------------------------------------*/
// pass 2 overflow list entry. 'next' indexes the overflow
// list (1-based), 'subarray' is 1-based and 'subarray_off'
// is the position in the run of keys from that subarray

#define BUCKET_NEXT_BITS 11
#define BUCKET_SUBARRAY_BITS 13

union bucket_entry {
	uint32 word;
	struct {
		uint32 next : BUCKET_NEXT_BITS;
		uint32 subarray : BUCKET_SUBARRAY_BITS;
		uint32 subarray_off : 8;
	} llist;
};

template<uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS, uint32 OVERFLOW_SIZE>
__device__ void BucketSortPass2_OneBin(
	const uint32 * __restrict__ key_prefix,
	const uint32 * __restrict__ sorted_offsets,
	collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	found_t * __restrict__ found,
	uint32 num_subarrays,
	uint32 which_array,
	uint32 which_bin,
	uint32 prefix_bits)
{
	static_assert(OVERFLOW_SIZE < (1 << BUCKET_NEXT_BITS),
			"overflow list too large for bucket_entry");

	typedef cub::BlockScan<uint16, THREADS_PER_BLOCK, 
				cub::BLOCK_SCAN_RAKING> PrefixSum;

	__shared__ struct {
		union {
			uint32 stage1_bits0[1 << (HASH1_BITS-5)];
			bucket_entry hashtable[1 << HASH2_BITS];
		} d0;
		union {
			uint32 stage1_bits1[1 << (HASH1_BITS-5)];
			typename PrefixSum::TempStorage prefix_tmp;
		} d1;
	} shared_data;

	__shared__ struct {
		uint32 num_overflow;
		bucket_entry overflow[OVERFLOW_SIZE];
	} shared_collision;

	// stage1: set a bit in the first stage 1 table based
	// on the HASH1_BITS low-order bits of each key prefix;
	// set the corresponding bit in the second
	// table if a bit in the first table is set 2+ times

	// clear shared mem
	{
		uint32 * s = (uint32 *)&shared_data;
		uint32 shared_words = sizeof(shared_data) / sizeof(uint32);
		for (uint32 i = threadIdx.x; i < shared_words; 
				i += THREADS_PER_BLOCK) {
			s[i] = 0;
		}
		if (threadIdx.x == 0)
			shared_collision.num_overflow = 0;
		__syncthreads();
	}

	// fill the stage 1 bit arrays; give each warp a batch
	// of (unit stride) key prefixes to handle. Only the low
	// prefix_bits of each prefix word are key bits; the rest
	// are location bits and must not affect the hashing

	const uint32 prefix_mask = (1 << prefix_bits) - 1;
	const uint32 stage1_mask = (1 << HASH1_BITS) - 1;
	const uint32 warp_size = 32;
	const uint32 my_warp = threadIdx.x / warp_size;
	const uint32 my_warp_lane = threadIdx.x % warp_size;

	for (uint32 i = my_warp; i < num_subarrays;
				i += THREADS_PER_BLOCK / warp_size) {
		uint32 batch = sorted_offsets[i];
		uint32 batch_off = batch & 0xffffff;
		uint32 batch_size = batch >> 24;
		for (uint32 j = my_warp_lane; j < batch_size; j += warp_size) {
			uint32 prefix = key_prefix[batch_off + j] & prefix_mask;
			uint32 bit = prefix & stage1_mask;
			uint32 update_loc = bit / 32;
			uint32 update_mask = 1 << (bit % 32);
			uint32 last_mask = atomicOr(shared_data.d0.stage1_bits0 + 
							update_loc, update_mask);
			if (last_mask & update_mask) {
				atomicOr(shared_data.d1.stage1_bits1 + 
						update_loc, update_mask);
			}
		}
	}
	__syncthreads(); // necessary for reading final bitfields

	// stage 2 is limited to keys whose stage1 mask occurs in
	// the second table; these are hashed and keys with the 
	// same hashes are linked into lists in parallel
	//
	// The hashtable is a concatenation of 1<<HASH2_BITS heads
	// of linked lists

	// clear the linked list heads

	bucket_entry * b = shared_data.d0.hashtable;
	const uint32 num_bins = 1 << HASH2_BITS;
	for (uint32 i = threadIdx.x; i < num_bins; i += THREADS_PER_BLOCK)
		b[i].word = 0;
	__syncthreads();

	// repeat the batch scan

	uint32 stage2_mask = (1 << HASH2_BITS) - 1;

	for (uint32 i = my_warp; i < num_subarrays;
				i += THREADS_PER_BLOCK / warp_size) {
		uint32 batch = sorted_offsets[i];
		uint32 batch_off = batch & 0xffffff;
		uint32 batch_size = batch >> 24;
		for (uint32 j = my_warp_lane; j < batch_size; j += warp_size) {
			uint32 prefix = key_prefix[batch_off + j] & prefix_mask;
			uint32 bit = prefix & stage1_mask;
			uint32 update_loc = bit / 32;
			uint32 update_mask = 1 << (bit % 32);

			if (shared_data.d1.stage1_bits1[update_loc] & update_mask) {

				// add prefix to the hashtable. We want the
				// hashtable bins to also store something, as that
				// increases the capacity of the table greatly

				// calculate the stage 2 bin to use

				uint32 bin = (prefix >> HASH1_BITS) & stage2_mask;

				// form the entry to add. We start by assuming 
				// bin is empty, and form a linked list entry
				// with one link (next = 0). The offsets to store
				// are 1-based, to distinguish them from an
				// empty entry that has offset 0

				bucket_entry new_entry;
				new_entry.llist.next = 0;
				new_entry.llist.subarray = i + 1;
				new_entry.llist.subarray_off = j;

				// attempt to add new_entry. If the linked list
				// head for the hashtable bin is empty (value 0), 
				// this will succeed and we are done

				bucket_entry curr_entry;
				curr_entry.word = atomicCAS(&b[bin].word, 0, 
							new_entry.word);
				if (curr_entry.llist.subarray != 0) {

					// linked list was not empty; reserve an
					// entry in the overflow array

					uint32 new_loc = atomicAdd(
							&shared_collision.num_overflow, 1);

					if (new_loc < OVERFLOW_SIZE) {
						// point to it, make new_entry the
						// new head of list and push the 
						// previous head to overflow.
						// Note that curr_entry may have
						// changed in the meantime, but that's 
						// okay since we will always have 
						// to store something. The overflow
						// offset is 1-based

						new_entry.llist.next = new_loc + 1;
						curr_entry.word = atomicExch(&b[bin].word, 
									new_entry.word);
						shared_collision.overflow[new_loc] = curr_entry;
					}
				}
			}
		}
	}

	__syncthreads();

	// we need overflow entries to have a chance of collisions

	uint32 num_overflow = shared_collision.num_overflow;
	if (num_overflow == 0)
		return;

	// proceed through the bins THREADS_PER_BLOCK at a time
	// (the prefix sum needs all threads contributing)

	for (uint32 batch_start = 0; batch_start < num_bins;
			batch_start += THREADS_PER_BLOCK) {

		// squeeze out bins with <= 1 entry 
		// (nothing to collide with)

		uint32 curr_bin = batch_start + threadIdx.x;
		uint16 save_bin = 0;

		if (curr_bin < num_bins && b[curr_bin].llist.next > 0)
			save_bin = 1;

		// prefix sum the nonempty bin numbers

		uint16 my_off;
		uint16 total_nonempty;
		PrefixSum(shared_data.d1.prefix_tmp).ExclusiveSum(
				save_bin, my_off, total_nonempty);

		// augment the overflow array with the surviving
		// hashtable bins

		if (save_bin && num_overflow + 
				my_off < OVERFLOW_SIZE) {
			shared_collision.overflow[num_overflow + my_off] =
							b[curr_bin];
		}

		// account for larger array size, prepare shared
		// memory for the next iteration

		num_overflow += total_nonempty;
		__syncthreads();
	}

	// if the overflow list filled up, keys were dropped and
	// collisions may have been missed; tell the host

	if (num_overflow > OVERFLOW_SIZE && threadIdx.x == 0)
		atomicAdd(&found->p2, num_overflow - OVERFLOW_SIZE);

	num_overflow = min(num_overflow, OVERFLOW_SIZE);

	// start at each saved key and compare to all the others
	// downstream in its hashtable chain. We expect hashtable
	// chains to be rare and very short, and so use all the
	// threads in the block to balance the load of potentially
	// a quadratic number of comparisons

	for (uint32 i = threadIdx.x; i < num_overflow;
				i += THREADS_PER_BLOCK) {

		bucket_entry b0 = shared_collision.overflow[i];
		uint32 subarray0 = b0.llist.subarray - 1;
		uint32 batch0 = sorted_offsets[subarray0];
		uint32 offset0 = (batch0 & 0xffffff) + 
					b0.llist.subarray_off;
		uint32 key0 = key_prefix[offset0];

		uint16 next = b0.llist.next;

		while (next != 0) {
			bucket_entry b1 = shared_collision.overflow[next - 1];
			uint32 subarray1 = b1.llist.subarray - 1;
			uint32 batch1 = sorted_offsets[subarray1];
			uint32 offset1 = (batch1 & 0xffffff) + 
						b1.llist.subarray_off;
			uint32 key1 = key_prefix[offset1];

			if ((key0 & prefix_mask) == (key1 & prefix_mask)) {

				// the prefixes match; save enough to find
				// the original keys later. This is rare, so
				// write straight to global memory. Entry 0
				// of the list holds the list size

				uint32 my_id = atomicAdd((uint32 *)collision_list,
							1) + 1;
				if (my_id < collision_list_size) {
					collision_t *c = collision_list + my_id;
					c->which_array_bin = which_array << 16 |
								which_bin;
					c->subarrays = subarray1 << 16 | subarray0;
					c->prefix_word0 = key0;
					c->prefix_word1 = key1;
				}
			}
			next = b1.llist.next;
		}
	}
	__syncthreads(); // make count of collision array visible
}

template<uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS, uint32 OVERFLOW_SIZE>
__device__ void BucketSortPass2(
	const uint32 * __restrict__ key_prefix,
	const uint32 * __restrict__ sorted_offsets,
	collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	found_t * __restrict__ found,
	uint32 max_array_size,
	uint32 num_subarrays,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 num_key_arrays,
	uint32 key_array_batch)
{
	uint32 array_min = key_array_batch * blockIdx.y;
	uint32 array_max = min(num_key_arrays, array_min + key_array_batch);
	uint32 num_buckets = 1 << bucket_hash_bits;

	for (uint32 i = array_min; i < array_max; i++) {
		for (uint32 j = blockIdx.x; j < num_buckets; j += gridDim.x) {
			BucketSortPass2_OneBin<THREADS_PER_BLOCK,
					HASH1_BITS, HASH2_BITS, OVERFLOW_SIZE>(
					key_prefix + i * max_array_size,
					sorted_offsets + (i * num_buckets +
							j) * num_subarrays,
					collision_list, collision_list_size, found,
					num_subarrays, i, j, prefix_bits);
		}
	}
}

extern "C"
{

// pass 2 only looks at 32-bit key prefixes, so the same
// kernel works for any size of key. The sizes of the
// stage 1 tables and the overflow list determine how many
// keys a bucket can hold before collisions get dropped

__global__ void
bucket_sort_pass2(
	const uint32 * __restrict__ key_prefix,
	const uint32 * __restrict__ sorted_offsets,
	collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	found_t * __restrict__ found,
	uint32 max_array_size,
	uint32 num_subarrays,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 num_key_arrays,
	uint32 key_array_batch)
{
	BucketSortPass2<TRANSFORM_NUM_THREADS, 15, 9, 1000>(
			key_prefix, sorted_offsets,
			collision_list, collision_list_size, found,
			max_array_size, num_subarrays,
			bucket_hash_bits, prefix_bits,
			num_key_arrays, key_array_batch);
}

} // extern "C"

/*------------------------------------------------------------------------*/
// pass 1 divides each key array into subarrays, and stores
// the prefix of key i somewhere in the range of offsets
// occupied by the subarray containing i. The prefix word
// also contains the top bits of the position of i in its
// subarray, so each key of a colliding pair lies in a
// small window of its subarray. Subarrays have fewer than
// 2^16 keys (pass 1 already depends on this), so the window
// bounds relative to the start of the subarray fit in 16 bits

struct recover_window {
	uint32 prefix;
	uint16 start;
	uint16 end;
};

// Each block handles one collision from pass 2 at a time.
// All the threads in the block scan the two windows of keys
// that contain the colliding pair, then compare all the
// candidates found in one window to those in the other.
// Collisions are rare and the windows are small, so it is
// better to finish each collision quickly with a full block
// than to stretch the work out over fewer threads

template<typename KeyType, typename DataType,
	uint32 THREADS_PER_BLOCK>
__device__ void BucketRecover(
	const KeyType * __restrict__ keys,
	const DataType * __restrict__ data,
	const uint32 * __restrict__ sorted_offsets,
	const collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	uint32 max_array_size,
	uint32 num_subarrays,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 qroot_batch_start,
	found_t * __restrict__ found)
{
	uint32 num_collisions = *(const uint32 *)collision_list;
	uint32 num_bins = 1 << bucket_hash_bits;
	uint32 bin_mask = num_bins - 1;
	uint32 prefix_mask = (1 << prefix_bits) - 1;

	__shared__ uint32 num_match[2];
	__shared__ uint32 match[2][RECOVER_MATCH_SIZE];

	// if pass 2 found more collisions than fit in the list,
	// tell the host how many there were

	if (num_collisions >= collision_list_size) {
		if (blockIdx.x == 0 && threadIdx.x == 0)
			atomicMax(&found->p1, num_collisions);
		num_collisions = collision_list_size - 1;
	}

	for (uint32 i = blockIdx.x; i < num_collisions; i += gridDim.x) {

		collision_t c = collision_list[i + 1];
		uint32 which_array = c.which_array_bin >> 16;
		uint32 which_bin = c.which_array_bin & 0xffff;
		uint32 array_start = which_array * max_array_size;
		const KeyType * curr_keys = keys + array_start;
		const uint32 * curr_offsets = sorted_offsets +
				which_array * (num_subarrays << bucket_hash_bits);
		uint32 base[2];
		recover_window w[2];

		// pass 1 saved the offset and size of each bin in
		// each subarray, so bin 0 begins the subarray and
		// the last bin ends it

		for (uint32 j = 0; j < 2; j++) {
			uint32 subarray = (j == 0) ? (c.subarrays & 0xffff) :
						(c.subarrays >> 16);
			uint32 word = (j == 0) ? c.prefix_word0 :
						c.prefix_word1;
			const uint32 * s = curr_offsets + subarray * num_bins;
			uint32 last = s[num_bins - 1];
			uint32 start = (word >> prefix_bits) << loc_shift;

			base[j] = s[0] & 0xffffff;
			w[j].prefix = word & prefix_mask;
			w[j].start = start;
			w[j].end = min((last & 0xffffff) + (last >> 24) - base[j],
					start + (1 << loc_shift));
		}

		if (threadIdx.x < 2)
			num_match[threadIdx.x] = 0;
		__syncthreads();

		// find the keys in each window that are in the
		// correct bin and have the correct prefix

		for (uint32 j = 0; j < 2; j++) {
			uint32 k_end = base[j] + w[j].end;

			for (uint32 k = base[j] + w[j].start + threadIdx.x;
					k < k_end; k += THREADS_PER_BLOCK) {
				KeyType key = curr_keys[k];
				uint32 bin = static_cast<uint32>(key) & bin_mask;
				uint32 prefix = prefix_mask & static_cast<uint32>(
							key >> bucket_hash_bits);

				if (bin == which_bin && prefix == w[j].prefix) {
					uint32 my_id = atomicAdd(num_match + j, 1);
					if (my_id < RECOVER_MATCH_SIZE)
						match[j][my_id] = k;
				}
			}
		}
		__syncthreads();

		// compare every candidate in the first window to
		// every candidate in the second. If both windows are
		// the same then each pair appears twice, so only
		// consider one ordering

		uint32 n0 = min(num_match[0], RECOVER_MATCH_SIZE);
		uint32 n1 = min(num_match[1], RECOVER_MATCH_SIZE);
		bool same_window = (base[0] == base[1] &&
					w[0].start == w[1].start);

		for (uint32 j = threadIdx.x; j < n0 * n1;
					j += THREADS_PER_BLOCK) {
			uint32 k0 = match[0][j / n1];
			uint32 k1 = match[1][j % n1];

			if (k0 == k1 || (same_window && k0 > k1))
				continue;

			KeyType key0 = curr_keys[k0];
			KeyType key1 = curr_keys[k1];

			if (key0 == key1) {
				uint32 p0 = static_cast<uint32>(
						data[array_start + k0]);
				uint32 p1 = static_cast<uint32>(
						data[array_start + k1]);
				if (gcd(p0, p1) == 1) {
					// youppi!
					uint32 my_id = atomicAdd((uint32 *)found, 1) + 1;
					if (my_id < FOUND_ARRAY_SIZE) {
						found_t *f = found + my_id;
						f->offset = static_cast<uint64>(key0);
						f->p1 = p0;
						f->p2 = p1;
						f->which_qroot = qroot_batch_start +
								which_array;
					}
				}
			}
		}

		// the next collision reuses the match lists

		__syncthreads();
	}
}

extern "C"
{

__global__ void
bucket_recover_32(
	const uint32 * __restrict__ keys,
	const uint16 * __restrict__ data,
	const uint32 * __restrict__ sorted_offsets,
	const collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	uint32 max_array_size,
	uint32 num_subarrays,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 qroot_batch_start,
	found_t * __restrict__ found)
{
	BucketRecover<uint32, uint16, TRANSFORM_NUM_THREADS>(
			keys, data, sorted_offsets,
			collision_list, collision_list_size,
			max_array_size, num_subarrays,
			bucket_hash_bits, prefix_bits, loc_shift,
			qroot_batch_start, found);
}

__global__ void
bucket_recover_64(
	const uint64 * __restrict__ keys,
	const uint32 * __restrict__ data,
	const uint32 * __restrict__ sorted_offsets,
	const collision_t * __restrict__ collision_list,
	uint32 collision_list_size,
	uint32 max_array_size,
	uint32 num_subarrays,
	uint32 bucket_hash_bits,
	uint32 prefix_bits,
	uint32 loc_shift,
	uint32 qroot_batch_start,
	found_t * __restrict__ found)
{
	BucketRecover<uint64, uint32, TRANSFORM_NUM_THREADS>(
			keys, data, sorted_offsets,
			collision_list, collision_list_size,
			max_array_size, num_subarrays,
			bucket_hash_bits, prefix_bits, loc_shift,
			qroot_batch_start, found);
}

} // extern "C"

/*------------------------------------------------------------------------*/
union hash_entry {
	uint32 word;
	struct {
		uint16 key_off;
		uint16 next;
	} llist;
};

// abbreviated collision search, intended for a number
// of keys signficantly smaller than 1 << HASH1_BITS
template<typename KeyType, typename DataType,
	uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS>
__device__ void CollisionSearchOneBinSmall(
	const KeyType * __restrict__ keys,
	const DataType * __restrict__ data,
	hash_entry * __restrict__ overflow,
	uint32 start_off,
	uint32 end_off,
	uint32 sorted_bits,
	uint32 which_qroot,
	found_t * __restrict__ found)
{
	typedef cub::BlockScan<uint16, THREADS_PER_BLOCK, 
				cub::BLOCK_SCAN_RAKING> PrefixSum;

	__shared__ struct {
		uint32 num_overflow;
		union {
			uint32 stage1_bits0[1 << (HASH1_BITS-5)];
			hash_entry hashtable[1 << HASH2_BITS];
		} d0;
		union {
			uint32 stage1_bits1[1 << (HASH1_BITS-5)];
			typename PrefixSum::TempStorage prefix_tmp;
		} d1;
	} shared_data;

	// stage1: set a bit in the first stage 1 table based
	// on the HASH1_BITS low-order bits after sorted_bits
	// of each key; set the corresponding bit in the second
	// table if a bit in the first table is set 2+ times

	// clear shared mem
	{
		uint32 * s = (uint32 *)&shared_data;
		uint32 shared_words = sizeof(shared_data) / sizeof(uint32);
		for (uint32 i = threadIdx.x; i < shared_words; 
				i += THREADS_PER_BLOCK) {
			s[i] = 0;
		}
		__syncthreads();
	}

	// fill the stage 1 bit arrays

	uint32 stage1_mask = (1 << HASH1_BITS) - 1;

	for (uint32 i = start_off + threadIdx.x; i < end_off;
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bit = (uint32)(key >> sorted_bits) & stage1_mask;
		uint32 update_loc = bit / 32;
		uint32 update_mask = 1 << (bit % 32);
		uint32 last_mask = atomicOr(shared_data.d0.stage1_bits0 + 
						update_loc, update_mask);
		if (last_mask & update_mask) {
			atomicOr(shared_data.d1.stage1_bits1 + 
					update_loc, update_mask);
		}
	}
	__syncthreads(); // necessary for reading final bitfields

	// stage 2 is limited to keys whose stage1 mask occurs in
	// the second table; these are hashed and keys with the 
	// same hashes are linked into lists in parallel
	//
	// The hashtable is a concatenation of 1<<HASH2_BITS heads
	// of linked lists

	// clear the linked list heads

	hash_entry * h = shared_data.d0.hashtable;
	const uint32 num_bins = 1 << HASH2_BITS;
	for (uint32 i = threadIdx.x; i < num_bins; i += THREADS_PER_BLOCK)
		h[i].word = 0;
	__syncthreads();

	uint32 stage2_mask = (1 << HASH2_BITS) - 1;
	uint32 stage2_shift = sorted_bits + HASH1_BITS;

	for (uint32 i = start_off + threadIdx.x; i < end_off;
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bit = (uint32)(key >> sorted_bits) & stage1_mask;
		uint32 update_loc = bit / 32;
		uint32 update_mask = 1 << (bit % 32);

		if (shared_data.d1.stage1_bits1[update_loc] & update_mask) {

			// add offset i to the hashtable. We want the
			// hashtable bins to also store keys, as that
			// increases the capacity of the table greatly

			// calculate the bin to use

			uint32 bin = (uint32)(key >> stage2_shift) & stage2_mask;

			// form the entry to add. We start by assuming 
			// bin is empty, and form a linked list entry
			// with one link (next = 0). The offsets to store
			// are 1-based, to distinguish them from an
			// empty entry that has offset 0

			hash_entry new_entry;
			new_entry.llist.key_off = i - start_off + 1;
			new_entry.llist.next = 0;

			// attempt to add new_entry. If the linked list
			// head for the hashtable bin is empty (value 0), 
			// this will succeed and we are done

			hash_entry curr_entry;
			curr_entry.word = atomicCAS(&h[bin].word, 0, 
						new_entry.word);
			if (curr_entry.llist.key_off != 0) {

				// linked list was not empty; reserve an
				// entry in the overflow array

				uint32 new_loc = atomicAdd(
						&shared_data.num_overflow, 1);

				// point to it, make new_entry the
				// new head of list and push the 
				// previous head to overflow.
				// Note that curr_entry may have
				// changed in the meantime, but that's 
				// okay since we will always have 
				// to store something. The overflow
				// offset is 1-based

				new_entry.llist.next = new_loc + 1;
				curr_entry.word = atomicExch(&h[bin].word, 
							new_entry.word);
				overflow[start_off + new_loc] = curr_entry;
			}
		}
	}

	__syncthreads();

	// we need overflow entries to have a chance of collisions

	uint32 num_overflow = shared_data.num_overflow;
	if (num_overflow == 0)
		return;

	// proceed through the bins THREADS_PER_BLOCK at a time
	// (the prefix sum needs all threads contributing)

	for (uint32 batch_start = 0; batch_start < num_bins;
			batch_start += THREADS_PER_BLOCK) {

		// squeeze out bins with <= 1 entry 
		// (nothing to collide with)

		uint32 curr_bin = batch_start + threadIdx.x;
		uint16 save_bin = 0;

		if (curr_bin < num_bins && h[curr_bin].llist.next > 0)
			save_bin = 1;

		// prefix sum the nonempty bin numbers

		uint16 my_off;
		uint16 total_nonempty;
		PrefixSum(shared_data.d1.prefix_tmp).ExclusiveSum(
				save_bin, my_off, total_nonempty);

		// augment the overflow array with the surviving
		// hashtable bins

		if (save_bin) {
			overflow[start_off + num_overflow + 
					my_off] = h[curr_bin];
		}

		// account for larger array size, prepare shared
		// memory for the next iteration

		num_overflow += total_nonempty;
		__syncthreads();
	}

	// start at each saved key and compare to all the others
	// downstream in its hashtable chain. We expect hashtable
	// chains to be rare and very short, and so use all the 
	// threads in the block to balance the load of potentially 
	// a quadratic number of comparisons

	for (uint32 i = threadIdx.x; i < num_overflow; 
				i += THREADS_PER_BLOCK) {

		hash_entry h0 = overflow[start_off + i];
		uint32 offset0 = start_off + h0.llist.key_off - 1;
		KeyType key0 = keys[offset0];

		uint16 next = h0.llist.next;

		while (next != 0) {
			hash_entry h1 = overflow[start_off + next - 1];
			uint32 offset1 = start_off + h1.llist.key_off - 1;
			KeyType key1 = keys[offset1];

			if (key0 == key1) {
				uint32 p0 = static_cast<uint32>(data[offset0]);
				uint32 p1 = static_cast<uint32>(data[offset1]);
				if (gcd(p0, p1) == 1) {
					// youppi!
					uint32 my_id = atomicAdd((uint32 *)found, 1) + 1;
					if (my_id < FOUND_ARRAY_SIZE) {
						found_t *f = found + my_id;
						f->offset = static_cast<uint64>(key1);
						f->p1 = p0;
						f->p2 = p1;
						f->which_qroot = which_qroot;
					}
				}
			}
			next = h1.llist.next;
		}
	}
	__syncthreads(); // make count of found array visible
}

template<typename KeyType, typename DataType,
	uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS>
__device__ void CollisionSearchUnsortedSmall(
	const KeyType * __restrict__ keys,
	const DataType * __restrict__ data,
	uint32 * __restrict__ collision_scratch,
	uint32 qroot_batch_start,
	uint32 max_array_size,
	const uint32 * __restrict__ array_sizes,
	uint32 num_key_arrays,
	uint32 key_array_batch,
	found_t * __restrict__ found)
{
	uint32 array_min = key_array_batch * blockIdx.y;
	uint32 array_max = min(num_key_arrays, array_min + key_array_batch);

	// Note that the number of elements in array i below
	// was not previously clamped to the maximum, so do
	// that here

	for (uint32 i = array_min; i < array_max; i++) {

		CollisionSearchOneBinSmall<KeyType, DataType, THREADS_PER_BLOCK,
					HASH1_BITS, HASH2_BITS>(
				keys + i * max_array_size,
				data + i * max_array_size,
				(hash_entry *)(collision_scratch + i * max_array_size),
				0, min(array_sizes[i], max_array_size), 
				0, qroot_batch_start + i, found);
	}
}

extern "C"
{

__global__ void
collision_search_unsorted32_small(
	const uint32 * __restrict__ keys, 
	const uint16 * __restrict__ data,
	uint32 * __restrict__ collision_scratch,
	uint32 qroot_batch_start,
	uint32 max_array_size,
	const uint32 * __restrict__ array_sizes, 
	uint32 num_key_arrays,
	uint32 key_array_batch,
	found_t * __restrict__ found)
{
	CollisionSearchUnsortedSmall<uint32, uint16, 
			TRANSFORM_NUM_THREADS, 14, 9>(
				keys, data, collision_scratch, 
				qroot_batch_start, max_array_size, 
				array_sizes, num_key_arrays, 
				key_array_batch, found);
}

} // extern "C"


/*------------------------------------------------------------------------*/
/* Lifting special-q roots on the GPU (lift_qroots_q32). For each root
   r of x^d = N (mod q), q < 2^32, compute the Hensel lift to q^2 and
   subtract m0 (mod q^2): the same result as lift_roots in
   stage1_roots.c. One thread per root. Odd q with q^2 < 2^63 uses
   Montgomery arithmetic (R = 2^64); other q use 128-bit remainders */

#define LIFT_MAX_LIMBS 64

__constant__ uint32 c_lift_N[LIFT_MAX_LIMBS];   /* trans_N, LS limb first */
__constant__ uint32 c_lift_m0[LIFT_MAX_LIMBS];  /* trans_m0 */
__constant__ uint32 c_lift_nlimbs;
__constant__ uint32 c_lift_m0limbs;
__constant__ uint32 c_lift_degree;

struct lift_mont {
	uint64 m, minv, r2, one;
};

__device__ __forceinline__ uint64
lift_mont_mul(const lift_mont &M, uint64 a, uint64 b)
{
	/* a*b/R mod m for a, b < m < 2^63 */
	uint64 lo = a * b;
	uint64 hi = __umul64hi(a, b);
	uint64 u = lo * M.minv;
	uint64 lo2 = u * M.m;
	uint64 hi2 = __umul64hi(u, M.m);
	uint64 r = hi + hi2 + (lo + lo2 < lo ? 1 : 0);
	return (r >= M.m) ? r - M.m : r;
}

__device__ void
lift_mont_init(lift_mont &M, uint64 m)
{
	uint64 x = m, r;
	int i;
	for (i = 0; i < 5; i++)
		x *= 2 - m * x;
	M.m = m;
	M.minv = 0 - x;
	r = (0 - m) % m;                        /* R mod m */
	M.one = r;
	for (i = 0; i < 64; i++) {              /* R^2 mod m by doubling */
		r <<= 1;                        /* r < m < 2^63: no wrap */
		if (r >= m)
			r -= m;
	}
	M.r2 = r;
}

__device__ __forceinline__ uint64
lift_mont_to(const lift_mont &M, uint64 a)
{
	return lift_mont_mul(M, (a < M.m) ? a : a % M.m, M.r2);
}

__device__ uint64
lift_mont_pow(const lift_mont &M, uint64 b, uint32 e)
{
	uint64 r = M.one;
	while (e) {
		if (e & 1)
			r = lift_mont_mul(M, r, b);
		b = lift_mont_mul(M, b, b);
		e >>= 1;
	}
	return r;
}

/* value of a little-endian limb array mod m, Montgomery (odd m) */
__device__ uint64
lift_mont_bigmod(const lift_mont &M, const uint32 *limbs, uint32 n)
{
	uint64 t32 = lift_mont_to(M, (uint64)1 << 32);   /* 2^32 * R mod m */
	uint64 r = 0;
	int32 i;
	for (i = (int32)n - 1; i >= 0; i--) {
		uint64 l = limbs[i];
		r = lift_mont_mul(M, r, t32);           /* r * 2^32 mod m */
		if (l >= M.m)
			l %= M.m;
		r += l;
		if (r >= M.m)
			r -= M.m;
	}
	return r;
}

/* generic (any m < 2^64) 128-bit remainder versions */
__device__ __forceinline__ uint64
lift_mulmod(uint64 a, uint64 b, uint64 m)
{
	return (uint64)((unsigned __int128)a * b % m);
}

__device__ uint64
lift_powmod(uint64 b, uint32 e, uint64 m)
{
	uint64 r = 1 % m;
	b %= m;
	while (e) {
		if (e & 1)
			r = lift_mulmod(r, b, m);
		b = lift_mulmod(b, b, m);
		e >>= 1;
	}
	return r;
}

__device__ uint64
lift_bigmod(const uint32 *limbs, uint32 n, uint64 m)
{
	uint64 r = 0;
	int32 i;
	for (i = (int32)n - 1; i >= 0; i--)
		r = (uint64)((((unsigned __int128)r << 32) | limbs[i]) % m);
	return r;
}

/* a^-1 mod q (q < 2^32), 0 if not invertible */
__device__ uint32
lift_inv32(uint32 a, uint32 q)
{
	int64 t0 = 0, t1 = 1;
	int64 r0 = q, r1 = a % q;
	while (r1 != 0) {
		int64 k = r0 / r1, tmp;
		tmp = r0 - k * r1; r0 = r1; r1 = tmp;
		tmp = t0 - k * t1; t0 = t1; t1 = tmp;
	}
	if (r0 != 1)
		return 0;
	return (uint32)(t0 < 0 ? t0 + q : t0);
}

extern "C"
{

__global__ void
lift_qroots_q32(const uint32 * __restrict__ q_array,
		const uint32 * __restrict__ which_q,
		uint64 * __restrict__ roots,
		uint32 num_roots,
		uint32 * __restrict__ num_failed)
{
	uint32 degree = c_lift_degree;
	uint32 i;

	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < num_roots;
				i += gridDim.x * blockDim.x) {
		uint64 q = q_array[which_q[i]];
		uint64 qq = q * q;
		uint64 r = roots[i];            /* root mod q */
		uint64 nmod, m0mod, x, t1, t2, root;
		uint32 inv;

		if ((q & 1) && qq < ((uint64)1 << 63)) {
			lift_mont Mqq, Mq;
			lift_mont_init(Mqq, qq);
			lift_mont_init(Mq, q);
			nmod = lift_mont_bigmod(Mqq, c_lift_N, c_lift_nlimbs);
			m0mod = lift_mont_bigmod(Mqq, c_lift_m0, c_lift_m0limbs);
			/* both plain residues: each step's R^-1 cancels
			   the R in t32 */
			x = lift_mont_mul(Mqq, lift_mont_pow(Mqq,
					lift_mont_to(Mqq, r), degree), 1);
			t1 = (nmod >= x) ? nmod - x : nmod + qq - x;
			{
				uint64 qinv = 0 - Mq.minv;      /* q^-1 mod 2^64 */
				t1 *= qinv;                     /* exact: q | t1 */
			}
			t2 = lift_mont_mul(Mq, lift_mont_mul(Mq,
				lift_mont_pow(Mq, lift_mont_to(Mq, r), degree - 1),
				lift_mont_to(Mq, degree)), 1);
			inv = lift_inv32((uint32)t2, (uint32)q);
			if (inv == 0) {
				atomicAdd(num_failed, 1);
				continue;
			}
			t1 = lift_mont_mul(Mq, t1, lift_mont_to(Mq, inv));
		}
		else {
			nmod = lift_bigmod(c_lift_N, c_lift_nlimbs, qq);
			m0mod = lift_bigmod(c_lift_m0, c_lift_m0limbs, qq);
			x = lift_powmod(r, degree, qq);
			t1 = (nmod >= x) ? nmod - x : nmod + qq - x;
			t1 /= q;
			t2 = lift_mulmod(lift_powmod(r, degree - 1, q), degree, q);
			inv = lift_inv32((uint32)t2, (uint32)q);
			if (inv == 0) {
				atomicAdd(num_failed, 1);
				continue;
			}
			t1 = lift_mulmod(t1, inv, q);
		}
		root = r + t1 * q;
		root = (root >= m0mod) ? root - m0mod : root + qq - m0mod;
		roots[i] = root;
	}
}

} // extern "C"

#if __CUDA_ARCH__ >= 700  //-------  older cards will not have __match_any_sync
template<typename KeyType, typename DataType,
	uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS>
__device__ void CollisionSearchOneBinLarge(
	const KeyType * __restrict__ keys,
	const DataType * __restrict__ data,
	uint32 * __restrict__ sorted_offsets,
	uint32 start_off,
	uint32 end_off,
	uint32 sorted_bits,
	uint32 which_qroot,
	found_t * __restrict__ found)
{
	typedef cub::BlockScan<uint32, THREADS_PER_BLOCK, 
				cub::BLOCK_SCAN_RAKING> PrefixSum;

	__shared__ struct {
		uint32 num_overflow;
		union {
			uint32 stage1_bits0[1 << (HASH1_BITS-5)];
			uint32 bucket_size[(1 << HASH2_BITS) + 1];
		} d0;
		union {
			uint32 stage1_bits1[1 << (HASH1_BITS-5)];
		} d1;
		typename PrefixSum::TempStorage prefix_tmp;
	} shared_data;

	// stage1: set a bit in the first stage 1 table based
	// on the HASH1_BITS low-order bits after sorted_bits
	// of each key; set the corresponding bit in the second
	// table if a bit in the first table is set 2+ times

	uint32 stage1_mask = (1 << HASH1_BITS) - 1;

	// clear shared mem
	{
		uint32 * s = (uint32 *)&shared_data;
		uint32 shared_words = sizeof(shared_data) / sizeof(uint32);
		for (uint32 i = threadIdx.x; i < shared_words; 
				i += THREADS_PER_BLOCK) {
			s[i] = 0;
		}
		__syncthreads();
	}

	// fill the stage 1 bit arrays

	for (uint32 i = start_off + threadIdx.x; i < end_off; i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bit = (uint32)(key >> sorted_bits) & stage1_mask;
		uint32 update_loc = bit / 32;
		uint32 update_mask = 1 << (bit % 32);
		uint32 last_mask = atomicOr(shared_data.d0.stage1_bits0 + 
						update_loc, update_mask);
		if (last_mask & update_mask) {
			atomicOr(shared_data.d1.stage1_bits1 + 
					update_loc, update_mask);
		}
	}
	__syncthreads(); // necessary for reading final bitfields

	// stage 2 is limited to keys whose stage1 mask occurs in
	// the second table; these are hashed and keys with the 
	// same hashes are bucket sorted

	// clear the bucket sizes

	uint32 * buckets = shared_data.d0.bucket_size;
	const uint32 num_bins = 1 << HASH2_BITS;
	for (uint32 i = threadIdx.x; i < num_bins; 
				i += THREADS_PER_BLOCK) {
		buckets[i] = 0;
	}
	__syncthreads();

	uint32 stage2_mask = (1 << HASH2_BITS) - 1;

	// count the surviving keys in each bin

	for (uint32 i = start_off + threadIdx.x; i < end_off; 
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bit = (uint32)(key >> sorted_bits) & stage1_mask;
		uint32 update_loc = bit / 32;
		uint32 update_mask = 1 << (bit % 32);

		if (shared_data.d1.stage1_bits1[update_loc] & update_mask) {
			uint32 bin = (uint32)(key >> sorted_bits) & stage2_mask;
			atomicAdd(buckets + bin, 1);
		}
	}

	// proceed through the bins THREADS_PER_BLOCK at a time
	// (the prefix sum needs all threads contributing)

	uint32 num_survivors = 0;

	for (uint32 batch_start = 0; batch_start < num_bins;
			batch_start += THREADS_PER_BLOCK) {

		// fence off shared memory from previous loop

		__syncthreads();

		// bins must have more than 1 entry (need
		// something to collide with)

		uint32 my_bin = batch_start + threadIdx.x;
		uint32 my_size = 0;

		if (my_bin < num_bins) {
			my_size = buckets[my_bin];
			if (my_size == 1)
				my_size = 0;
		}

		// prefix sum the bin sizes

		uint32 my_off;
		uint32 curr_survivors;
		PrefixSum(shared_data.prefix_tmp).ExclusiveSum(
				my_size, my_off, curr_survivors);

		// save the resulting offset

		if (my_bin < num_bins)
			buckets[my_bin] = num_survivors + my_off;
		num_survivors += curr_survivors;
	}

	// save last offset
	if (threadIdx.x == 0)
		buckets[num_bins] = num_survivors;
	__syncthreads();

	// place offsets of survivors in their respective bin,
	// along with the 16 bits above HASH2_BITS;
	// reuse the top half of each bucket entry for the current
	// offset, to allow binning in parallel

	uint32 shift2 = sorted_bits + HASH2_BITS;
	for (uint32 i = start_off + threadIdx.x; i < end_off; 
				i += THREADS_PER_BLOCK) {
		KeyType key = keys[i];
		uint32 bit = (uint32)(key >> sorted_bits) & stage1_mask;
		uint32 update_loc = bit / 32;
		uint32 update_mask = 1 << (bit % 32);

		if (shared_data.d1.stage1_bits1[update_loc] & update_mask) {
			uint32 bin = (uint32)(key >> sorted_bits) & stage2_mask;
			uint32 my_bin_off = atomicAdd(buckets + bin, 1 << 16);
			uint32 bin_off = my_bin_off & 0xffff;
			uint32 res = (i - start_off) << 16 | 
					((uint32)(key >> shift2) & 0xffff);
			sorted_offsets[start_off + bin_off + 
					(my_bin_off >> 16)] = res;
		}
	}

	__syncthreads();
	
	// use a warp to process each bin

	const uint32 warp_size = 32;
	uint32 my_warp_lane = threadIdx.x % warp_size;

	for (uint32 bin = threadIdx.x / warp_size; bin < num_bins; 
			bin += THREADS_PER_BLOCK / warp_size) {

		uint32 bin_start = buckets[bin] & 0xffff;
		uint32 bin_end = buckets[bin + 1] & 0xffff;
		uint32 num_keys = min(warp_size, bin_end - bin_start);

		if (my_warp_lane >= num_keys)
			continue;

		// perform warp-wide all-against-all compare,
		// continue only if there is a match in a higher-
		// numbered lane

		uint32 read_val0 = sorted_offsets[start_off + 
					bin_start + my_warp_lane];
		uint32 match_mask = __match_any_sync(
					0xffffffff >> (32 - num_keys), 
					read_val0 & 0xffff);
		uint32 curr_match_lane = my_warp_lane;
		uint32 curr_match_mask = (match_mask >> curr_match_lane) & (~1);
		if (!curr_match_mask)
			continue;

		// at least two of the keys in the bin have
		// a 16-bit match, and current thread owns
		// the lower-indexed one; test all the other
		// match candidates owned by higher warp
	    	// lanes for exact equality

		uint32 key_off0 = start_off + (read_val0 >> 16);
		KeyType key0 = keys[key_off0];
		do {

			uint32 next_lane = curr_match_lane + 
						__ffs(curr_match_mask) - 1;
			uint32 read_val1 = sorted_offsets[start_off + 
						bin_start + next_lane];
			uint32 key_off1 = start_off + (read_val1 >> 16);
			KeyType key1 = keys[key_off1];

			if (key0 == key1) {
				uint32 p0 = static_cast<uint32>(data[key_off0]);
				uint32 p1 = static_cast<uint32>(data[key_off1]);
				if (gcd(p0, p1) == 1) {
					// youppi!
					uint32 my_id = atomicAdd((uint32 *)found, 1) + 1;
					if (my_id < FOUND_ARRAY_SIZE) {
						found_t *f = found + my_id;
						f->offset = static_cast<uint64>(key1);
						f->p1 = p0;
						f->p2 = p1;
						f->which_qroot = which_qroot;
					}
				}
			}

			curr_match_lane = next_lane;
			curr_match_mask = (match_mask >> curr_match_lane) & (~1);
		} while (curr_match_mask);
	}
	__syncthreads(); // make count of found array visible
}

template<typename KeyType, typename DataType,
	uint32 THREADS_PER_BLOCK,
	uint32 HASH1_BITS, uint32 HASH2_BITS>
__device__ void CollisionSearchUnsortedLarge(
	const KeyType * __restrict__ keys,
	const DataType * __restrict__ data,
	uint32 * __restrict__ collision_scratch,
	uint32 qroot_batch_start,
	uint32 max_array_size,
	const uint32 * __restrict__ array_sizes,
	uint32 num_key_arrays,
	uint32 key_array_batch,
	found_t * __restrict__ found)
{
	uint32 array_min = key_array_batch * blockIdx.y;
	uint32 array_max = min(num_key_arrays, array_min + key_array_batch);

	// Note that the number of elements in array i below
	// was not previously clamped to the maximum, so do
	// that here

	for (uint32 i = array_min; i < array_max; i++) {

		CollisionSearchOneBinLarge<KeyType, DataType, THREADS_PER_BLOCK,
					HASH1_BITS, HASH2_BITS>(
				keys + i * max_array_size,
				data + i * max_array_size,
				collision_scratch + i * max_array_size,
				0, min(array_sizes[i], max_array_size), 
				0, qroot_batch_start + i, found);
	}
}

extern "C"
{

__global__ void
collision_search_unsorted32_large(
	const uint32 * __restrict__ keys, 
	const uint16 * __restrict__ data,
	uint32 * __restrict__ collision_scratch,
	uint32 qroot_batch_start,
	uint32 max_array_size,
	const uint32 * __restrict__ array_sizes, 
	uint32 num_key_arrays,
	uint32 key_array_batch,
	found_t * __restrict__ found)
{
	CollisionSearchUnsortedLarge<uint32, uint16, 
			TRANSFORM_NUM_THREADS, 14, 9>(
				keys, data, collision_scratch, 
				qroot_batch_start, max_array_size, 
				array_sizes, num_key_arrays, 
				key_array_batch, found);
}

} // extern "C"

#endif // __CUDA_ARCH__ >= 700
