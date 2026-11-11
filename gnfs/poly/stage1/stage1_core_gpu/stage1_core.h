/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: stage1_core.h 1112 2026-11-02 02:13:30Z jasonp_sf $
--------------------------------------------------------------------*/

#ifndef _STAGE1_CORE_H_
#define _STAGE1_CORE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* collisions kept per batch, slot 0 holding the counts. It was 1000:
   a batch can find tens of thousands, and every one past 999 was
   silently dropped (39 batches gave exactly 39 x 999 hits) */
#define FOUND_ARRAY_SIZE (1 << 20)
#define TRANSFORM_NUM_THREADS 256
#define MIN_COLLISION_LIST_SIZE 8192
#define MAX_COLLISION_LIST_SIZE (1 << 22)
#define RECOVER_MATCH_SIZE 32
#define MAX_BUCKET_BITS 11
#define MAX_BUCKET_KEYS 5632
#define PASS1_CACHE_KEYS 8192 /* 32 KB of shared memory */
#define MAX_SUBARRAYS 8190 /* must fit bucket_entry.subarray */

typedef struct {
	uint64 offset;
	uint32 p1;
	uint32 p2;
	uint32 which_qroot;
	uint32 pad;
} found_t;

/* a pair of key prefixes from bucket sort pass 2 that
   match. Both keys are in key array 'which_array' of the
   current batch and in bucket 'which_bin' (packed as
   which_array << 16 | which_bin), and each is in the
   subarray given by one half of 'subarrays'. The prefix
   words are as pass 1 wrote them, including the location
   bits. As with the found array, the first word of entry 0
   of the list holds the number of entries that follow it.
   The host sizes the list for each batch, with at least
   MIN_COLLISION_LIST_SIZE entries; if pass 2 finds more 
   collisions than fit, the recovery kernel stores the
   number found in p1 of entry 0 of the found array. If the
   overflow list inside pass 2 fills up, pass 2 adds the
   number of entries that did not fit to p2 of entry 0 */

typedef struct {
	uint32 which_array_bin;
	uint32 subarrays;
	uint32 prefix_word0;
	uint32 prefix_word1;
} collision_t;

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_H_ */
