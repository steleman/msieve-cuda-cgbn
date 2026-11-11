/* test harness for super-fast collision detection algorithm,
   thanks to R. Gerbicz */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h> 
#include <immintrin.h>

#if defined(WIN32) || defined(_WIN64)
	#define WIN32_LEAN_AND_MEAN

	#include <windows.h>
#else
	#include <fcntl.h>
	#include <unistd.h>
	#include <sys/resource.h>
#endif

#define MIN(a,b) ((a) < (b)? (a) : (b))
#define MAX(a,b) ((a) > (b)? (a) : (b))

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;

/* #define HAVE_PROF */
#ifdef HAVE_PROF
#define SHOW_PROF __attribute__((noinline))
#else
#define SHOW_PROF /* nothing */
#endif

/*---------------------------------------------------------------------*/
static void *
aligned_malloc(size_t len, uint32 align) {

	void *ptr, *aligned_ptr;
	unsigned long addr;

	ptr = malloc(len+align);

	 /* offset to next ALIGN-byte boundary */

	addr = (unsigned long)ptr;				
	addr = align - (addr & (align-1));
	aligned_ptr = (void *)((uint8 *)ptr + addr);

	*( (void **)aligned_ptr - 1 ) = ptr;
	return aligned_ptr;
}

static void
aligned_free(void *newptr) {

	void *ptr;

	if (newptr == NULL) 
		return;
	ptr = *( (void **)newptr - 1 );
	free(ptr);
}

static void * xmalloc(size_t len) {
	void *ptr = malloc(len);
	if (ptr == NULL) {
		printf("failed to allocate %u bytes\n", (uint32)len);
		exit(-1);
	}
	return ptr;
}

static void * xcalloc(size_t num, size_t len) {
	void *ptr = calloc(num, len);
	if (ptr == NULL) {
		printf("failed to calloc %u bytes\n", (uint32)(num * len));
		exit(-1);
	}
	return ptr;
}

static void * xrealloc(void *iptr, size_t len) {
	void *ptr = realloc(iptr, len);
	if (ptr == NULL) {
		printf("failed to reallocate %u bytes\n", (uint32)len);
		exit(-1);
	}
	return ptr;
}

double
get_cpu_time(void) {

#if defined(WIN32) || defined(_WIN64)
	FILETIME create_time = {0, 0};
	FILETIME exit_time = {0, 0};
	FILETIME kernel_time = {0, 0};
	FILETIME user_time = {0, 0};

	GetThreadTimes(GetCurrentThread(),
			&create_time,
			&exit_time,
			&kernel_time,
			&user_time);

	return ((uint64)user_time.dwHighDateTime << 32 | 
	               user_time.dwLowDateTime) / 10000000.0;
#else
	struct rusage r_usage;

	#if 0 /* use for linux 2.6.26+ */
	getrusage(RUSAGE_THREAD, &r_usage);
	#else
	getrusage(RUSAGE_SELF, &r_usage);
	#endif

	return ((uint64)r_usage.ru_utime.tv_sec * 1000000 +
	               r_usage.ru_utime.tv_usec) / 1000000.0;
#endif
}

static uint32 
get_rand(uint32 *rand_seed, uint32 *rand_carry) {
   
	/* A multiply-with-carry generator by George Marsaglia.
	   The period is about 2^63. */

	#define RAND_MULT 2131995753

	uint64 temp;

	temp = (uint64)(*rand_seed) * 
		       (uint64)RAND_MULT + 
		       (uint64)(*rand_carry);
	*rand_seed = (uint32)temp;
	*rand_carry = (uint32)(temp >> 32);
	return (uint32)temp;
}

/*------------------------------------------------------------------------*/
#define MAX_RADIX_BITS 8

#define BUCKET_BITS 8
#define BUCKET_SIZE (1 << BUCKET_BITS)
#define BUCKET_MASK (BUCKET_SIZE - 1)

typedef struct {
	uint64 key;
	uint32 data;
} match_t;

typedef struct {
	/* initial state */
	uint32 num_sort;
	uint32 num_sort_alloc;
	uint64 *sort_key;
	uint32 *sort_data;

	/* state for bucket sorting */
	uint32 num_bucket_alloc;
	uint32 * bucket_data;

	uint32 * bucket_len;
	uint32 * bucket_offset;
	uint32 * bucket_prefix;
	uint32 * bucket_next;

	/* state for pruning false positives */
	uint32 num_dup_alloc;
	uint32 * dup_data;

	uint32 * dup0;
	uint32 * dup1;
	uint32 * dup2;
	uint32 * bin_prefix;

	/* state for putative duplicate keys */
	uint32 num_survivors;
	uint32 num_survivors_alloc;
	uint64 * survivors;
	uint64 * sort_survivors;

	uint32 num_match_alloc;
	match_t * match;
} cpu_thread_data_t;

cpu_thread_data_t *
cpu_thread_data_init()
{
	cpu_thread_data_t *t = (cpu_thread_data_t *)calloc(1,
					sizeof(cpu_thread_data_t));

	t->num_survivors_alloc = 1000;
	t->survivors = (uint64 *)xmalloc(t->num_survivors_alloc *
					sizeof(uint64));
	t->sort_survivors = (uint64 *)xmalloc(t->num_survivors_alloc *
					sizeof(uint64));
	t->num_match_alloc = 1000;
	t->match = (match_t *)xmalloc(t->num_match_alloc *
					sizeof(match_t));
	return t;
}

void
cpu_thread_data_free(cpu_thread_data_t * t)
{
	free(t->sort_key);
	free(t->sort_data);
	if (t->bucket_data)
		aligned_free(t->bucket_data);
	if (t->dup_data)
		aligned_free(t->dup_data);
	free(t->survivors);
	free(t->sort_survivors);
	free(t->match);
	free(t);
}

static void 
grow_sort(cpu_thread_data_t * td, uint32 new_size)
{
	td->num_sort_alloc = new_size;
	td->sort_key = (uint64 *)xrealloc(td->sort_key, 
				new_size * sizeof(uint64));
	td->sort_data = (uint32 *)xrealloc(td->sort_data, 
				new_size * sizeof(uint32));
}

/*------------------------------------------------------------------------*/
static SHOW_PROF uint32 
bucket_sort(cpu_thread_data_t * td, uint32 radix_bits)
{
	/* perform a single radix sort pass of size
	   1<<radix_bits; this breaks each radix bin up
	   into buckets of size BUCKET_SIZE and 
	   allocates buckets on the fly, so keys are sorted
	   in a single pass */

	uint32 i;
	uint32 radix_size = 1 << radix_bits;
	uint32 radix_mask = radix_size - 1;
	uint32 num_sort = td->num_sort;
	uint32 num_buckets = radix_size + 1 + num_sort / BUCKET_SIZE;
	uint32 last_bucket;
	uint32 max_bin_size;
	uint64 * key = td->sort_key;
	uint32 * bucket_len;
	uint32 * bucket_offset;
	uint32 * bucket_prefix;
	uint32 * bucket_next;
	uint32 bucket_words;

	/* resize if necessary; allocate a single array and
	   concatenate all the bucket sort state to avoid
	   associative conflicts in L1 */

	bucket_words = 2 * radix_size + num_buckets * (BUCKET_SIZE + 1);
	if (bucket_words > td->num_bucket_alloc) {
		uint32 t = bucket_words * 3 / 2;
		td->num_bucket_alloc = t;
		aligned_free(td->bucket_data);
		td->bucket_data = (uint32 *)aligned_malloc(
					t * sizeof(uint32), 64);
	}
	bucket_len = td->bucket_len = td->bucket_data;
	bucket_offset = td->bucket_offset = bucket_len + radix_size;
	bucket_next = td->bucket_next = bucket_offset + radix_size;
	bucket_prefix = td->bucket_prefix = bucket_next + num_buckets;

	/* initialize; every radix bin gets one bucket. Also remember
	   the index of the last bucket allocated. Note that we only
	   store the bottom 32+radix_bits bits of keys */

	last_bucket = radix_size - 1;
	memset(bucket_len, 0, radix_size * sizeof(uint32));
	for (i = 0; i < radix_size; i++)
		bucket_offset[i] = i << BUCKET_BITS;

	/* move each key */

	for (i = 0; i < num_sort; i++) {
		uint64 k = key[i];
		uint32 bin = (uint32)(k & radix_mask);

		bucket_prefix[bucket_offset[bin]++] = (uint32)(k >> radix_bits);

		if ((bucket_offset[bin] & BUCKET_MASK) == 0) {

			/* bucket is full; allocate the next
			   one and extend the chain of buckets
			   for this bin. Also undo the increment
			   so the offset in the next bucket 
			   starts at zero */

			uint32 curr_bucket = (bucket_offset[bin] - 1) >> 
							BUCKET_BITS;
			bucket_next[curr_bucket] = ++last_bucket;
			bucket_offset[bin] = last_bucket << BUCKET_BITS;
			bucket_len[bin] += BUCKET_SIZE;
		}
	}

	/* account for partially filled bins and return the
	   largest bin size */

	max_bin_size = 0;
	for (i = 0; i < radix_size; i++) {
		bucket_len[i] += bucket_offset[i] & BUCKET_MASK;
		max_bin_size = MAX(max_bin_size, bucket_len[i]);
	}
	bucket_next[last_bucket] = 0;
	return max_bin_size;
}

/*------------------------------------------------------------------------*/
static const double d_extra_bits = 4.5;

static SHOW_PROF uint32
prune_dup_arrays(cpu_thread_data_t *td, uint32 match_bits, 
		uint32 dup_array_bits, uint32 num_prefix,
		uint32 estimated_dups)
{
	const uint32 max_iter = 20;
	uint32 i, j;
	uint32 *T = td->dup0;
	uint32 *T2 = td->dup1;
	uint32 *T3 = td->dup2;
	uint32 *bin_prefix = td->bin_prefix;
	uint32 num_survivors[max_iter];
	uint32 dup_shift = 0;
	uint32 dup_mask = (1 << dup_array_bits) - 1;

	/* Using hash table find possible repeated keys, if f(x) 
	   is not repeated then x is also not repeated.
           For a hash function use f(x)=floor(x/2^k) mod (2^l) 
	   for different k,l values.
	   Using a table of size 2^l we can get the possible 
	   repeated keys, in another table store the possible 
	   repeated hash values. In another pass extract the 
	   possible repeated keys.
	   In each ping-pong round we can reduce the size of 
	   the array (possible number of repeated keys). */

	for(i = 0; i < max_iter; i++) {
        
		uint32 *U, *U2, *U3;
		uint32 dup_shift2;
		uint32 dup_mask2;
		uint32 dup_array_size2;
		uint32 curr_num_prefix = 0;
		uint32 ilog2 = MAX(5,(int)(log(estimated_dups + 1) / 
				M_LN2 + d_extra_bits));

		ilog2 = MIN(ilog2, match_bits);
        
		if (i % 2 == 0) {
			/* hash data is in T array */
			U = T; U2 = T2; U3 = T3;
		}
		else {
			/* hash data is in T3 array */
			U = T3; U2 = T2; U3 = T;
		}
        
		dup_shift2 = dup_shift + 6;
		if (dup_shift2 + ilog2 > match_bits)
			dup_shift2 = 0;
        
		dup_mask2 = (1 << ilog2) - 1;
		dup_array_size2 = 1 << (ilog2 - 5);
		memset(U2, 0, dup_array_size2 * sizeof(uint32));
		memset(U3, 0, dup_array_size2 * sizeof(uint32));
        
		for (j = 0; j < num_prefix; j++) {
			/* look for set bit in previous dup array,
			   fill current dup array */
			uint32 t = bin_prefix[j];
			uint32 bit = (t >> dup_shift) & dup_mask;
			if (U[bit >> 5] & (1 << (bit & 31))) {
				uint32 bit2 = (t >> dup_shift2) & dup_mask2;
				if (U2[bit2 >> 5] & (1 << (bit2 & 31))) 
					U3[bit2 >> 5] |= 1 << (bit2 & 31); 
				else
					U2[bit2 >> 5] |= 1 << (bit2 & 31);
				bin_prefix[curr_num_prefix++] = t;
			}
		}
        
		num_prefix = num_survivors[i] = curr_num_prefix;

		/* fast paths */
		if (num_prefix == 0)
			return 0;
		else if (num_prefix == 2)
			return (bin_prefix[0] == bin_prefix[1]) ? 2 : 0;
		else if (i >= 3 && num_survivors[i-3] == num_survivors[i])
			break;

		dup_mask = dup_mask2;
		dup_shift = dup_shift2;
	}
	return num_prefix;
}

/*------------------------------------------------------------------------*/
static void
find_dup_one_bin(cpu_thread_data_t *td, 
		uint32 match_bits, uint32 dup_array_bits,
	        uint32 radix_bits, uint32 which_bin)
{
	uint32 dup_array_mask = (1 << dup_array_bits) - 1;
	uint32 dup_array_size = 1 << (dup_array_bits - 5);
	uint32 bucket = which_bin; /* first bucket at known position */
	uint32 bucket_off = bucket << BUCKET_BITS;
	uint32 *bucket_prefix = td->bucket_prefix;
	uint32 *bucket_next = td->bucket_next;
	uint32 max_bucket_off = td->bucket_offset[which_bin];
	uint32 num_dups = 0;

	uint32 *bin_prefix = td->bin_prefix;
	uint32 i = 0;
	uint32 num_survivors;
	uint32 *T = td->dup0;
	uint32 *T2 = td->dup1;

	/* initialize duplicate arrays */

	memset(T, 0, dup_array_size * sizeof(uint32));
	memset(T2, 0, dup_array_size * sizeof(uint32));
    
	/* proceed one bucket at a time, concatenating
	   all the buckets into bin_prefix and filling
	   the duplicate array T */

	while (1) {
		uint32 end = MIN(max_bucket_off, 
				bucket_off + BUCKET_SIZE);

		/* use two separate bit arrays to remember 
		   key groups with matching bottom bits */

		while (bucket_off < end) {
			uint32 curr = bucket_prefix[bucket_off++];
			uint32 bit = curr & dup_array_mask;
			bin_prefix[i++] = curr;
			if (T2[bit >> 5] & (1 << (bit & 31))) {
				T[bit >> 5] |= 1 << (bit & 31);
				num_dups++;
			}
			else {
				T2[bit >> 5] |= 1 << (bit & 31);
			}
		}
		/* find the next bucket for which_bin */
		if (bucket_off == max_bucket_off)
			break;
		bucket = bucket_next[bucket];
		if (bucket == 0)
			break;
		bucket_off = bucket << BUCKET_BITS;
	}

	/* initial bit array is written, iteratively prune the
	   list of false positives */

	if (num_dups == 0)
		return;

	num_survivors = prune_dup_arrays(td, match_bits, 
				dup_array_bits, i, 2 * num_dups);

	/* save any survivors */

	if (num_survivors > 0) {
		uint32 c = td->num_survivors;
		if (c + num_survivors >= td->num_survivors_alloc) {
			uint32 t = 2 * (c + num_survivors);
			td->num_survivors_alloc = t;
			td->survivors = (uint64 *)xrealloc(td->survivors,
							t * sizeof(uint64));
			td->sort_survivors = (uint64 *)xrealloc(
							td->sort_survivors,
							t * sizeof(uint64));
		}
		for (i = 0; i < num_survivors; i++) {
			td->survivors[c + i] = (uint64)bin_prefix[i] << 
						radix_bits | which_bin;
		}
		td->num_survivors += num_survivors;
	}
}

/*------------------------------------------------------------------------*/
static SHOW_PROF void 
find_dup_all(cpu_thread_data_t *td, uint32 match_bits,
		uint32 dup_array_bits)
{
	uint32 i;
	uint32 num_sort = td->num_sort;
	uint64 *key = td->sort_key;
	uint32 dup_array_mask = (1 << dup_array_bits) - 1;
	uint32 dup_array_size = 1 << (dup_array_bits - 5);
	uint32 num_dups = 0;

	uint32 *bin_prefix = td->bin_prefix;
	uint32 num_survivors;
	uint32 *T = td->dup0;
	uint32 *T2 = td->dup1;

	/* initialize duplicate arrays */

	memset(T, 0, dup_array_size * sizeof(uint32));
	memset(T2, 0, dup_array_size * sizeof(uint32));
    
	/* save prefix of each key directly */

	for (i = 0; i < num_sort; i++) {
		uint32 curr = (uint32)key[i];
		uint32 bit = curr & dup_array_mask;
		bin_prefix[i] = curr;
		if (T2[bit >> 5] & (1 << (bit & 31))) {
			T[bit >> 5] |= 1 << (bit & 31);
			num_dups++;
		}
		else {
			T2[bit >> 5] |= 1 << (bit & 31);
		}
	}

	/* initial bit array is written, iteratively prune the
	   list of false positives */

	if (num_dups == 0)
		return;

	num_survivors = prune_dup_arrays(td, match_bits, 
				dup_array_bits, num_sort,
				2 * num_dups);

	/* save any survivors */

	if (num_survivors > td->num_survivors_alloc) {
		uint32 t = 2 * num_survivors;
		td->num_survivors_alloc = t;
		td->survivors = (uint64 *)xrealloc(td->survivors,
						t * sizeof(uint64));
		td->sort_survivors = (uint64 *)xrealloc(
						td->sort_survivors,
						t * sizeof(uint64));
	}
	for (i = 0; i < num_survivors; i++)
		td->survivors[i] = bin_prefix[i];
	td->num_survivors = num_survivors;
}

/*------------------------------------------------------------------------*/
static int 
compare64(const void* a, const void* b) 
{
     uint64 a64 = *((uint64*)a);
     uint64 b64 = *((uint64*)b);
     
     if (a64 == b64)
	     return 0;
     else if (a64 < b64) 
	     return -1;
     return 1;
}

static int 
compare_match(const void* a, const void* b)
{
     match_t *a0 = (match_t *)a;
     match_t *b0 = (match_t *)b;
     
     if (a0->key == b0->key)
	     return 0;
     else if (a0->key < b0->key) 
	     return -1;
     return 1;
}

static SHOW_PROF uint32
finish_search(cpu_thread_data_t * td, uint32 match_bits)
{
	uint32 i, j;
	uint32 num_sort = td->num_sort;
	uint64 * sort_key = td->sort_key;
	uint32 * sort_data = td->sort_data;
	uint32 num_survivors = td->num_survivors;
	uint64 * survivors = td->survivors;
	uint64 * sort_survivors = td->sort_survivors;
	uint64 match_mask = ((uint64)1 << match_bits) - 1;
	uint32 num_match = 0;
	uint32 num_collision = 0;

	/* presence array state */

	uint32 p_bits;
	uint32 p_size;
	uint32 p_mask;
	uint32 p_alloc;
	uint32 * p_count;
	uint32 * p_array;

	/* find repeated keys and store without multiplicity */

	qsort(survivors, num_survivors, sizeof(uint64), compare64);
	for (i = 1, j = 0; i < num_survivors; i++) {
		if (survivors[i] == survivors[i-1] &&
		    (i == num_survivors - 1 || 
				survivors[i] != survivors[i+1]))
			survivors[j++] = survivors[i];
	}
	if (j == 0)
		return 0;
	num_survivors = j;

	/* list out entries for keys whose bottom match_bits
	   bits match one of the entries in survivors[] 
	   
	   With a small number of survivors we can just check
	   every key against every survivor. However, when a
	   large array is radix sorted we can get hundreds of
	   survivors in all bins combined, even if each bin
	   would only have one survivor. For these cases we 
	   need to radix sort the list of survivors to avoid
	   iterating through all of them */

	p_bits = MAX(5, (int)(log(num_survivors + 1) / 
				M_LN2 + d_extra_bits));
	p_size = 1 << (p_bits - 5);
	p_mask = (1 << p_bits) - 1;
	p_alloc = p_size + (1 << p_bits) + 1;
	if (p_alloc > td->num_dup_alloc) {
		uint32 t = 2 * p_alloc;
		td->num_dup_alloc = t;
		td->dup_data = (uint32 *)xrealloc(td->dup_data, 
					t * sizeof(uint32));
	}
	p_array = td->dup_data;
	p_count = td->dup_data + p_size;
	memset(p_array, 0, p_alloc * sizeof(uint32));

	/* increment the count in the *next* bucket for survivor i */

	for (i = 0; i < num_survivors; i++) {
		uint32 bit = (uint32)survivors[i] & p_mask;
		p_count[bit + 1]++;
		p_array[bit >> 5] |= 1 << (bit & 31);
	}

	/* prefix sum */

	for (i = 1; i < (1 << p_bits) + 1; i++)
		p_count[i] += p_count[i-1];

	/* scatter the survivor list */

	for (i = 0; i < num_survivors; i++) {
		uint32 bit = (uint32)survivors[i] & p_mask;
		sort_survivors[p_count[bit]++] = survivors[i];
	}

	/* iterate through keys */

	for (i = 0; i < num_sort; i++) {
		uint64 key = sort_key[i];
		uint32 bit = (uint32)key & p_mask;
		uint32 s_start, s_end;

		if (!(p_array[bit >> 5] & (1 << (bit & 31))))
			continue;

		s_start = (bit == 0) ? 0 : p_count[bit - 1];
		s_end = p_count[bit];

		for (j = s_start; j < s_end; j++) {
			if ((key & match_mask) == sort_survivors[j]) {
				if (num_match >= td->num_match_alloc) {
					uint32 t = 2 * num_match;
					td->num_match_alloc = t;
					td->match = (match_t *)xrealloc(
							td->match, 
							t * sizeof(match_t));
				}
				td->match[num_match].key = key;
				td->match[num_match].data = sort_data[i];
				num_match++;
				break;
			}
		}
	}

	qsort(td->match, num_match, sizeof(match_t), compare_match);
	for (i = 0; i < num_match - 1; i++) {
		for (j = i + 1; j < num_match && 
				td->match[i].key == td->match[j].key; j++) {
			num_collision++;
		}
	}

	return num_collision;
}

/*------------------------------------------------------------------------*/
static SHOW_PROF uint32 
collision_search(cpu_thread_data_t *td, uint32 key_bits)
{
	/* return the number of pairs of keys that are
	   identical in the input data. We assume the array
	   size is small to medium (less than maybe 1M 
	   keys for the largest problems) but key_bits is 
	   potentially very large (at least 32 bits, up 
	   to maybe 56-64 bits) */

	uint32 i; 
	uint32 max_bin_size;
	uint32 dup_array_bits;
	uint32 dup_array_size;
	uint32 match_bits;
	uint32 num_sort = td->num_sort;
	uint32 radix_bits = 0;
	uint32 dup_words;

	/* small problems use the input keys directly;
	   otherwise perform a single bucket sort pass 
	   to get future tables to approximately L1 
	   cache size. In either case, for large keys
	   we limit the size of internal data structures
	   by matching only the bottom (32+radix_bits)
	   bits of keys, and rely on the number of keys
	   to be small enough that false positive matches
	   are unlikely */

	if (num_sort < 55000) {
		max_bin_size = num_sort;
	}
	else {
		const uint32 target_dupsize = 5000;
		radix_bits = (uint32)(log((double)num_sort / 
					target_dupsize) / M_LN2 + 0.5);
		radix_bits = MIN(radix_bits, MAX_RADIX_BITS);
		max_bin_size = bucket_sort(td, radix_bits);
	}

	/* resize duplicate detection arrays if necessary,
	   and concatenate the arrays to avoid associative
	   conflicts in L1. 
	   
	   The duplicate detection arrays have 1-bit type 
	   so the actual memory allocated is much smaller */

	dup_array_bits = MAX(5,(int)((double)log(max_bin_size + 1) / 
				M_LN2 + d_extra_bits));
	match_bits = MIN(32, key_bits - radix_bits);
	dup_array_bits = MIN(dup_array_bits, match_bits);
	dup_array_size = 1 << (dup_array_bits - 5);
	dup_words = 3 * dup_array_size + max_bin_size;
	if (dup_words > td->num_dup_alloc) {
		uint32 t = dup_words * 3 / 2;
		td->num_dup_alloc = t;
		aligned_free(td->dup_data);
		td->dup_data = (uint32 *)aligned_malloc(
					t * sizeof(uint32), 64);
	}
	td->dup0 = td->dup_data;
	td->dup1 = td->dup_data + dup_array_size;
	td->dup2 = td->dup_data + 2 * dup_array_size;
	td->bin_prefix = td->dup_data + 3 * dup_array_size;
    
	/* for small problems, detect duplicates directly; 
	   otherwise radix sorted data proceeds one bin at a time */

	if (radix_bits) {
		td->num_survivors = 0;
		for (i = 0; i < (1 << radix_bits); i++) {
			if (td->bucket_len[i] == 0)
				continue;

			find_dup_one_bin(td, match_bits,
					dup_array_bits, radix_bits, i);
		}
	}
	else {
		find_dup_all(td, match_bits, dup_array_bits);
	}
    
	if (td->num_survivors == 0)
		return 0;
	return finish_search(td, radix_bits + match_bits);
}

/*------------------------------------------------------------------------*/
int main(int argc, char ** argv)
{
	uint32 i, j, k;
	uint32 key_bits = strtoul(argv[1], NULL, 10);
	uint32 num_sort = strtoul(argv[2], NULL, 10);
	uint32 num_runs = strtoul(argv[3], NULL, 10);
	double seconds = 0;
	double start = 0;
	uint32 seed1 = 0x11111;
	uint32 seed2 = 0x22222;
	uint32 num_match = 0;
	uint32 num_match_ref = 0;

	cpu_thread_data_t * ctx = cpu_thread_data_init();

	grow_sort(ctx, num_sort);

	for (i = 0; i < num_runs; i++) {
		for (j = 0; j < num_sort; j++) {
			uint64 key = (uint64)get_rand(&seed1, &seed2) << 32 |
					get_rand(&seed1, &seed2);
        
			ctx->sort_key[j] = key >> (64 - key_bits);
			ctx->sort_data[j] = get_rand(&seed1, &seed2);
		}
		ctx->num_sort = num_sort;

		start = get_cpu_time();
		num_match += collision_search(ctx, key_bits);
		seconds += get_cpu_time() - start;

		qsort(ctx->sort_key, num_sort, sizeof(uint64), compare64);
		for (j = 0; j < num_sort - 1; j++) {
			for (k = j + 1; k < num_sort &&
					ctx->sort_key[j] == ctx->sort_key[k]; k++) {
				num_match_ref++;
			}
		}
	}
	printf("found %u matches in %u runs of %u x %u-bit keys in %lf sec (%lf sec each)\n", 
			num_match, num_runs, num_sort, key_bits, 
			seconds, seconds / num_runs);
	printf("matches found should be %u\n", num_match_ref);
    
	cpu_thread_data_free(ctx);
	return 0;
}
