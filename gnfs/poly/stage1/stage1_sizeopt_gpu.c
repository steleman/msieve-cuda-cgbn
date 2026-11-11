/*--------------------------------------------------------------------
Copyright (c) 2026 Stefan Teleman <stefan.teleman@gmail.com>

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

Part of a fork of msieve (public domain, by Jason Papadopoulos); the
routines ported here from msieve keep their public-domain origin.
--------------------------------------------------------------------*/

/* Host driver for the GPU size optimization (sizeopt_core.cu).

   Per stage 1 batch, on the batch's stream: rebuild every collision in
   the device found array into a polynomial (so_collide), run the ten
   minimize() rounds of optimize_initial (so_minimize_kernel, then
   so_update for the integers), score projective roots and apply the
   stage 2 bound (so_finish), and gather the survivors (so_gather).
   Only the survivors, about one in 4000, come back over PCIe. */

#include "stage1_sizeopt_gpu.h"
#include <stage1_core_gpu/stage1_core.h>
#include <stage1_core_gpu/sizeopt_core.h>

#define SO_THREADS 128
#define SO_ROUNDS 10		/* 2 passes x at most 5 minimize() */
#define SO_NUM_WIDTHS 2

static const uint32_t so_widths[SO_NUM_WIDTHS] = { 512, 1024 };
static const uint32_t so_tpi[SO_NUM_WIDTHS] = { 8, 16 };

struct so_ctx {
	msieve_obj *obj;
	CUmodule module;
	uint32_t num_sm;

	gpu_launch_t minimize[2];	/* degree 4, 5 */
	uint32_t min_threads[2];		/* resident threads of each */
	gpu_launch_t collide[SO_NUM_WIDTHS];
	gpu_launch_t update[SO_NUM_WIDTHS];
	gpu_launch_t finish[SO_NUM_WIDTHS];
	gpu_launch_t gather[SO_NUM_WIDTHS];

	/* the current leading coefficient */
	uint32_t width;			/* index into so_widths */
	so_params_t prm;
	CUdeviceptr consts;

	/* per-candidate device arrays, for cap candidates */
	uint32_t cap;
	uint32_t cap_width;		/* width the arrays were sized for */
	uint32_t base;			/* slots collected so far */
	uint32_t target;			/* flush when this many are collected */
	stage1_sieve_data_t *d;		/* where survivors go */
	CUstream stream;
	CUdeviceptr big, state, list, surv, counters;

	/* survivors */
	uint32_t out_cap;
	CUdeviceptr out_big, out_state;
	uint32_t *h_out_big;
	so_state_t *h_out_state;

	mpz_t a[SO_MAX_DEG + 1];
	mpz_t lin[2];

	/* totals, logged by so_free */
	uint64_t num_found, num_cands, num_surv;
	uint64_t num_crap, num_expand_failed, num_line_iters, num_overflow;
};

/*------------------------------------------------------------------------*/
const char *
so_ptx_name(uint32 major, uint32 minor)
{
	/* the same choice as stage1_core_sm*.ptx */
	if (major == 5)
		return (minor < 2) ? "sizeopt_core_sm50.ptx" :
				     "sizeopt_core_sm52.ptx";
	if (major == 6)
		return (minor < 1) ? "sizeopt_core_sm60.ptx" :
				     "sizeopt_core_sm61.ptx";
	if (major == 7)
		return (minor < 5) ? "sizeopt_core_sm70.ptx" :
				     "sizeopt_core_sm75.ptx";
	if (major >= 12)
		return "sizeopt_core_sm120.ptx";
	if (major >= 10)
		return "sizeopt_core_sm100.ptx";
	if (major == 9)
		return "sizeopt_core_sm90.ptx";
	if (major >= 8) {
		if (minor < 6)
			return "sizeopt_core_sm80.ptx";
		if (minor < 9)
			return "sizeopt_core_sm86.ptx";
		return "sizeopt_core_sm89.ptx";
	}
	return NULL;
}

/*------------------------------------------------------------------------*/
so_ctx_t *
so_init(msieve_obj *obj, uint32 compute_major,
	uint32 compute_minor, uint32 num_sm)
{
	const char *name = so_ptx_name(compute_major, compute_minor);
	CUmodule module;

	if (name == NULL || cuModuleLoad(&module, name) != CUDA_SUCCESS) {
		logprintf(obj, "GPU size optimization off: cannot load %s\n",
				name ? name : "a module for this GPU");
		return NULL;
	}

	so_ctx_t *s = (so_ctx_t *)xcalloc(1, sizeof(so_ctx_t));
	s->obj = obj;
	s->module = module;
	s->num_sm = num_sm;

	gpu_launch_init(module, "so_minimize_4", s->minimize + 0);
	gpu_launch_init(module, "so_minimize_5", s->minimize + 1);

	/* the minimize kernels take work from a counter, so launch only
	   as many threads as are resident at once: more just queue up,
	   and with 1-2 blocks per SM of these large kernels the extra
	   blocks only cost time (measured) */
	for (uint32_t i = 0; i < 2; i++) {
		int blocks = 0;
		CUDA_TRY(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
				s->minimize[i].kernel_func, SO_THREADS, 0))
		s->min_threads[i] = MAX(1, blocks) * num_sm * SO_THREADS;
	}
	for (uint32_t i = 0; i < SO_NUM_WIDTHS; i++) {
		char buf[64];

		sprintf(buf, "so_collide_%u", so_widths[i]);
		gpu_launch_init(module, buf, s->collide + i);
		sprintf(buf, "so_update_%u", so_widths[i]);
		gpu_launch_init(module, buf, s->update + i);
		sprintf(buf, "so_finish_%u", so_widths[i]);
		gpu_launch_init(module, buf, s->finish + i);
		sprintf(buf, "so_gather_%u", so_widths[i]);
		gpu_launch_init(module, buf, s->gather + i);
	}

	CUDA_TRY(cuMemAlloc(&s->consts, SO_NUM_CONST *
			so_widths[SO_NUM_WIDTHS - 1] / 8))
	CUDA_TRY(cuMemAlloc(&s->counters, SO_NUM_STATS * sizeof(uint32_t)))

	for (uint32_t i = 0; i <= SO_MAX_DEG; i++)
		mpz_init(s->a[i]);
	mpz_init(s->lin[0]);
	mpz_init(s->lin[1]);

	logprintf(obj, "GPU size optimization on (%s)\n", name);
	return s;
}

/*------------------------------------------------------------------------*/
static void
so_free_arrays(so_ctx_t *s)
{
	if (s->cap) {
		CUDA_TRY(cuMemFree(s->big))
		CUDA_TRY(cuMemFree(s->state))
		CUDA_TRY(cuMemFree(s->list))
		CUDA_TRY(cuMemFree(s->surv))
		s->cap = 0;
	}
	if (s->out_cap) {
		CUDA_TRY(cuMemFree(s->out_big))
		CUDA_TRY(cuMemFree(s->out_state))
		free(s->h_out_big);
		free(s->h_out_state);
		s->out_cap = 0;
	}
}

void
so_free(so_ctx_t *s)
{
	if (s == NULL)
		return;

	logprintf(s->obj, "GPU size optimization: %" PRIu64 " collisions, "
			"%" PRIu64 " polynomials optimized, %" PRIu64
			" passed the stage 2 bound\n",
			s->num_found, s->num_cands, s->num_surv);
	if (s->num_crap || s->num_expand_failed || s->num_overflow)
		logprintf(s->obj, "GPU size optimization: %" PRIu64
			" failed the modular condition, %" PRIu64
			" failed to expand, %" PRIu64 " dropped\n",
			s->num_crap, s->num_expand_failed, s->num_overflow);

	so_free_arrays(s);
	CUDA_TRY(cuMemFree(s->consts))
	CUDA_TRY(cuMemFree(s->counters))
	CUDA_TRY(cuModuleUnload(s->module))
	for (uint32_t i = 0; i <= SO_MAX_DEG; i++)
		mpz_clear(s->a[i]);
	mpz_clear(s->lin[0]);
	mpz_clear(s->lin[1]);
	free(s);
}

/*------------------------------------------------------------------------*/
/* a nonnegative mpz as `words` little-endian 32-bit words */
static void
so_mpz_to_words(uint32_t *w, uint32_t words, const mpz_t x)
{
	size_t count = 0;

	memset(w, 0, words * sizeof(uint32_t));
	mpz_export(w, &count, -1, sizeof(uint32_t), 0, 0, x);
}

/* two's complement words to mpz */
static void
so_words_to_mpz(mpz_t x, const uint32_t *w, uint32_t words)
{
	if (w[words - 1] & 0x80000000) {
		uint32_t tmp[1024 / 32];
		uint32_t carry = 1;

		for (uint32_t i = 0; i < words; i++) {
			uint32_t v = ~w[i] + carry;
			carry = (carry && v == 0);
			tmp[i] = v;
		}
		mpz_import(x, words, -1, sizeof(uint32_t), 0, 0, tmp);
		mpz_neg(x, x);
	}
	else {
		mpz_import(x, words, -1, sizeof(uint32_t), 0, 0, w);
	}
}

/*------------------------------------------------------------------------*/
bool
so_set_coeff(so_ctx_t *s, const poly_search_t *poly, const poly_coeff_t *c)
{
	uint32_t need = (uint32_t)mpz_sizeinbase(c->trans_N, 2) + 16;

	so_flush(s);	/* the previous coefficient's candidates */

	if (c->degree != 4 && c->degree != 5)
		return false;

	uint32_t i;
	for (i = 0; i < SO_NUM_WIDTHS; i++) {
		if (need <= so_widths[i])
			break;
	}
	if (i == SO_NUM_WIDTHS)
		return false;
	s->width = i;
	uint32_t words = so_widths[i] / 32;

	/* the big constants */

	uint32_t buf[SO_NUM_CONST * 1024 / 32];
	mpz_t mod;
	mpz_init(mod);
	mpz_mul_ui(mod, c->high_coeff, c->degree);
	so_mpz_to_words(buf + SO_CONST_N * words, words, poly->N);
	so_mpz_to_words(buf + SO_CONST_TRANS_N * words, words, c->trans_N);
	so_mpz_to_words(buf + SO_CONST_TRANS_M0 * words, words, c->trans_m0);
	so_mpz_to_words(buf + SO_CONST_AD * words, words, c->high_coeff);
	so_mpz_to_words(buf + SO_CONST_MOD * words, words, mod);
	mpz_clear(mod);
	CUDA_TRY(cuMemcpyHtoD(s->consts, buf,
			SO_NUM_CONST * words * sizeof(uint32_t)))

	/* parameters; half_sieve and sizeof_q are set per batch */

	memset(&s->prm, 0, sizeof(so_params_t));
	s->prm.degree = c->degree;

	/* collect batches until a few times what the GPU runs at once:
	   ten rounds each wait for their slowest search, and small
	   batches (about 7000 collisions at 140 digits) left most of
	   the GPU idle */
	s->target = 8 * s->min_threads[c->degree - 4];
	s->prm.found_max = FOUND_ARRAY_SIZE - 1;
	s->prm.cmax_over_m0 = c->coeff_max / c->m0;
	s->prm.stage1_norm = poly->norm_max;

	/* loose here, exact (with the host's exp) on delivery */
	s->prm.sizeopt_bound = poly->sizeopt_norm * (1 + 1e-6);

	uint32_t prime = 0;
	for (uint32_t i = 0; i < SO_NUM_PRIMES; i++) {
		prime += prime_delta[i];
		s->prm.primes[i] = prime;
		s->prm.log_primes[i] = log((double)prime);
	}
	return true;
}

/*------------------------------------------------------------------------*/
static void
so_alloc(so_ctx_t *s, uint32_t num_found)
{
	uint32_t bytes = so_widths[s->width] / 8;

	if (num_found <= s->cap && s->cap_width >= s->width)
		return;

	so_free_arrays(s);
	s->cap = MAX(num_found, s->target);
	s->cap_width = s->width;

	CUDA_TRY(cuMemAlloc(&s->big, (size_t)s->cap * SO_NUM_BIG * bytes))
	CUDA_TRY(cuMemAlloc(&s->state, (size_t)s->cap * sizeof(so_state_t)))
	CUDA_TRY(cuMemAlloc(&s->list, (size_t)s->cap * sizeof(uint32_t)))
	CUDA_TRY(cuMemAlloc(&s->surv, (size_t)s->cap * sizeof(uint32_t)))
}

static void
so_alloc_out(so_ctx_t *s, uint32_t n)
{
	size_t bytes = SO_NUM_BIG * so_widths[SO_NUM_WIDTHS - 1] / 8;

	if (n <= s->out_cap)
		return;

	if (s->out_cap) {
		CUDA_TRY(cuMemFree(s->out_big))
		CUDA_TRY(cuMemFree(s->out_state))
		free(s->h_out_big);
		free(s->h_out_state);
	}
	s->out_cap = MAX(n, 1024);
	CUDA_TRY(cuMemAlloc(&s->out_big, s->out_cap * bytes))
	CUDA_TRY(cuMemAlloc(&s->out_state, s->out_cap * sizeof(so_state_t)))
	s->h_out_big = (uint32_t *)xmalloc(s->out_cap * bytes);
	s->h_out_state = (so_state_t *)xmalloc(s->out_cap *
					sizeof(so_state_t));
}

static void
so_launch(const gpu_launch_t *k, uint32_t threads_total, CUstream stream,
		void **args, uint32_t num_sm)
{
	uint32_t threads = MIN(SO_THREADS, k->threads_per_block);
	uint32_t blocks = (threads_total + threads - 1) / threads;

	blocks = MAX(1, MIN(blocks, 64 * num_sm));
	CUDA_TRY(cuLaunchKernel(k->kernel_func, blocks, 1, 1,
			threads, 1, 1, 0, stream, args, NULL))
}

/*------------------------------------------------------------------------*/
/* rebuild one stage 1 batch's collisions into candidate slots
   [base, base + num_found); the size optimization runs later, on
   many batches at once (so_flush) */

void
so_run_batch(so_ctx_t *s, stage1_sieve_data_t *d, CUstream stream,
		CUdeviceptr found, uint32 num_found,
		CUdeviceptr q, CUdeviceptr which_q, CUdeviceptr roots,
		uint32 sizeof_q, uint64 sieve_size)
{
	uint32_t w = s->width;

	if (num_found == 0)
		return;
	num_found = MIN(num_found, FOUND_ARRAY_SIZE - 1);

	s->d = d;
	s->stream = stream;
	if (s->base > 0 && s->base + num_found > s->target)
		so_flush(s);
	so_alloc(s, s->base + num_found);

	if (s->base == 0) {
		CUDA_TRY(cuMemsetD8Async(s->counters, 0,
				SO_NUM_STATS * sizeof(uint32_t), stream))
	}

	s->prm.sizeof_q = sizeof_q;
	s->prm.half_sieve = (int64_t)sieve_size / 2;
	s->prm.cap = s->cap;
	s->prm.slot_base = s->base;

	void *args[GPU_MAX_KERNEL_ARGS];
	args[0] = &found;
	args[1] = &q;
	args[2] = &which_q;
	args[3] = &roots;
	args[4] = &s->consts;
	args[5] = &s->big;
	args[6] = &s->state;
	args[7] = &s->list;
	args[8] = &s->counters;
	args[9] = &s->prm;
	so_launch(s->collide + w, num_found * so_tpi[w], stream, args,
			s->num_sm);

	s->base += num_found;
	s->num_found += num_found;
}

/*------------------------------------------------------------------------*/
/* size-optimize every collected candidate and pass the survivors on */

void
so_flush(so_ctx_t *s)
{
	CUstream stream = s->stream;
	uint32_t w = s->width;
	uint32_t words = so_widths[w] / 32;
	uint32_t slots = s->base;
	uint32_t deg = s->prm.degree;
	void *args[GPU_MAX_KERNEL_ARGS];

	if (slots == 0)
		return;
	s->base = 0;

	for (uint32_t i = 0; i < SO_ROUNDS; i++) {
		CUDA_TRY(cuMemsetD32Async(s->counters + SO_STAT_NEXT *
				sizeof(uint32_t), 0, 1, stream))
		args[0] = &s->state;
		args[1] = &s->list;
		args[2] = &s->counters;
		args[3] = &s->prm;
		so_launch(s->minimize + (deg - 4),
				MIN(slots, s->min_threads[deg - 4]),
				stream, args, s->num_sm);

		args[0] = &s->big;
		args[1] = &s->state;
		args[2] = &s->list;
		args[3] = &s->counters;
		args[4] = &s->prm;
		so_launch(s->update + w, slots * so_tpi[w], stream, args,
				s->num_sm);
	}

	args[0] = &s->big;
	args[1] = &s->state;
	args[2] = &s->list;
	args[3] = &s->surv;
	args[4] = &s->counters;
	args[5] = &s->prm;
	so_launch(s->finish + w, slots * so_tpi[w], stream, args, s->num_sm);

	uint32_t counters[SO_NUM_STATS];
	CUDA_TRY(cuMemcpyDtoHAsync(counters, s->counters,
			sizeof(counters), stream))
	CUDA_TRY(cuStreamSynchronize(stream))

	uint32_t n = counters[SO_STAT_SURVIVORS];
	s->num_cands += counters[SO_STAT_CANDS];
	s->num_surv += n;
	s->num_crap += counters[SO_STAT_CRAP];
	s->num_expand_failed += counters[SO_STAT_EXPAND_FAILED];
	s->num_line_iters += counters[SO_STAT_LINE_ITERS];

	/* what the CPU code prints for these */
	for (uint32_t i = 0; i < counters[SO_STAT_LINE_ITERS]; i++)
		printf(":");
	for (uint32_t i = 0; i < counters[SO_STAT_EXPAND_FAILED]; i++)
		fprintf(stderr, "expand failed\n");

	if (n == 0)
		return;

	so_alloc_out(s, n);
	args[0] = &s->big;
	args[1] = &s->state;
	args[2] = &s->surv;
	args[3] = &s->counters;
	args[4] = &s->out_big;
	args[5] = &s->out_state;
	so_launch(s->gather + w, n * SO_NUM_BIG * words, stream, args,
			s->num_sm);
	CUDA_TRY(cuMemcpyDtoHAsync(s->h_out_big, s->out_big,
			(size_t)n * SO_NUM_BIG * words * sizeof(uint32_t), stream))
	CUDA_TRY(cuMemcpyDtoHAsync(s->h_out_state, s->out_state,
			(size_t)n * sizeof(so_state_t), stream))
	CUDA_TRY(cuStreamSynchronize(stream))

	for (uint32_t i = 0; i < n; i++) {
		const uint32_t *b = s->h_out_big + (size_t)i * SO_NUM_BIG * words;
		const so_state_t *st = s->h_out_state + i;

		for (uint32_t j = 0; j <= deg; j++)
			so_words_to_mpz(s->a[j], b + (SO_BIG_A + j) * words,
					words);
		so_words_to_mpz(s->lin[0], b + SO_BIG_LIN0 * words, words);
		so_words_to_mpz(s->lin[1], b + SO_BIG_P * words, words);

		stage1_sizeopt_submit(s->d, deg, s->a, s->lin,
				st->pol_norm, st->alpha);
	}
}
