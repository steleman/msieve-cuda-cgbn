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

/* Large integer multiplication on the GPU for the NFS square root; see
   sqrt_gpu_mul.h and sqrtmul_core.cu. Ported from the author's
   CADO-NFS GPU square root (cado-nfs-gpu fork), MIT here. The square
   root runs several dependencies in threads (MSIEVE_SQRT_THREADS):
   each thread has its own stream and buffers, so their products run on
   the GPU at the same time; the module, the primes and the twiddle
   tables are shared. */

#include <common.h>
#include <cuda_xface.h>
#include <pthread.h>
#include <sys/time.h>
#include "sqrt_gpu_mul.h"
#include "sqrtmul_core.h"

typedef unsigned long long u64;
__extension__ typedef unsigned __int128 u128;

#define SM_MAXLG 30		/* 2^30 64-bit pieces: products of 2^36 bits */

/* the shared state */
static struct {
	int ok;
	CUcontext ctx;
	CUdevice dev;
	CUmodule module;
	CUfunction k_load, k_twiddle, k_dif, k_dit, k_pointwise, k_crt;
	sm_prime_t pr[SM_NP];
	CUdeviceptr tw_fwd[SM_MAXLG + 1], tw_inv[SM_MAXLG + 1];
	pthread_mutex_t lock;	/* the tables, the counters, the threads */
	size_t min_limbs;
	int check;
	double seconds;
	unsigned long calls;
	struct sm_thread *threads;	/* every thread's state, to free */
} S;

/* per thread: a stream, buffers */
typedef struct sm_thread {
	CUstream stream;
	CUdeviceptr d_a, d_A, d_B, d_out, d_w;
	size_t cap_N, cap_in, cap_hout;
	u64 *h_out;
	struct sm_thread *next;
} sm_thread_t;

static __thread sm_thread_t *sm_mine;

/*------------------------------------------------------------------*/
/* host arithmetic */

static u64
mulmod_h(u64 a, u64 b, u64 m)
{
	return (u64)((u128)a * b % m);
}

static u64
powmod_h(u64 a, u64 e, u64 m)
{
	u64 r = 1;

	for (; e; e >>= 1, a = mulmod_h(a, a, m))
		if (e & 1)
			r = mulmod_h(r, a, m);
	return r;
}

/* deterministic for n < 3.3e24 */
static int
is_prime_h(u64 n)
{
	static const u64 B[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
	u64 d;
	int s = 0;

	if (n < 2)
		return 0;
	for (int i = 0; i < 12; i++)
		if (n % B[i] == 0)
			return n == B[i];
	d = n - 1;
	while (!(d & 1)) {
		d >>= 1;
		s++;
	}
	for (int i = 0; i < 12; i++) {
		u64 x = powmod_h(B[i], d, n);
		int comp = 1;

		if (x == 1 || x == n - 1)
			continue;
		for (int k = 1; k < s && comp; k++) {
			x = mulmod_h(x, x, n);
			if (x == n - 1)
				comp = 0;
		}
		if (comp)
			return 0;
	}
	return 1;
}

static u64
to_mont_h(u64 x, u64 p)
{
	return (u64)(((u128)(x % p) << 64) % p);
}

static double
sm_now(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static const char *
sm_ptx_name(int32 major, int32 minor)
{
	if (major >= 12) return "sqrtmul_core_sm120.ptx";
	if (major >= 10) return "sqrtmul_core_sm100.ptx";
	if (major == 9) return "sqrtmul_core_sm90.ptx";
	if (major == 8)
		return (minor < 6) ? "sqrtmul_core_sm80.ptx" :
			(minor < 9) ? "sqrtmul_core_sm86.ptx" :
				      "sqrtmul_core_sm89.ptx";
	if (major == 7)
		return (minor < 5) ? "sqrtmul_core_sm70.ptx" :
				     "sqrtmul_core_sm75.ptx";
	if (major == 6)
		return (minor < 1) ? "sqrtmul_core_sm60.ptx" :
				     "sqrtmul_core_sm61.ptx";
	return NULL;
}

/*------------------------------------------------------------------*/
int
sqrt_gpu_start(msieve_obj *obj)
{
	const char *e = getenv("MSIEVE_SQRT_GPU");
	gpu_config_t config;
	uint32 devs[MAX_GPU], lr = 0, ls = 1;
	gpu_info_t *info;
	const char *name;
	sm_crt_t C;
	u64 p1, p2, p3;
	u128 pp;
	int k;

	memset(&S, 0, sizeof(S));
	if (e != NULL && strcmp(e, "0") == 0)
		return 0;
	if (GMP_LIMB_BITS != 64)
		return 0;
	S.min_limbs = 2048;
	if ((e = getenv("MSIEVE_SQRT_GPU_MIN")) != NULL && atol(e) > 0)
		S.min_limbs = (size_t)atol(e);
	S.check = ((e = getenv("MSIEVE_SQRT_GPU_CHECK")) != NULL &&
			atoi(e) != 0);

	gpu_init(&config);
	if (config.num_gpu <= 0)
		return 0;
#ifdef HAVE_MPI
	lr = obj->mpi_local_rank;
	ls = obj->mpi_local_size;
#endif
	if (gpu_select(&config, obj->which_gpu, lr, ls, devs, NULL) == 0)
		return 0;
	info = config.info + devs[0];
	name = sm_ptx_name(info->compute_version_major,
			info->compute_version_minor);
	if (name == NULL)
		return 0;

	S.dev = info->device_handle;
	CUDA_TRY(cuDevicePrimaryCtxRetain(&S.ctx, S.dev))
	CUDA_TRY(cuCtxPushCurrent(S.ctx))
	if (cuModuleLoad(&S.module, name) != CUDA_SUCCESS) {
		logprintf(obj, "square root: cannot load %s, GMP only\n", name);
		goto fail;
	}
	{
		CUdeviceptr ap;
		size_t asz;
		uint32 abi[2] = { 0, 0 };

		if (cuModuleGetGlobal(&ap, &asz, S.module, "sm_abi") !=
				CUDA_SUCCESS || asz != sizeof(abi) ||
		    cuMemcpyDtoH(abi, ap, sizeof(abi)) != CUDA_SUCCESS ||
		    abi[0] != SM_ABI || abi[1] != sizeof(sm_crt_t)) {
			logprintf(obj, "square root: %s does not match this "
				"msieve (an older file in the current "
				"directory? copy the .ptx files of this build);"
				" GMP only\n", name);
			CUDA_TRY(cuModuleUnload(S.module))
			goto fail;
		}
	}
	CUDA_TRY(cuModuleGetFunction(&S.k_load, S.module, "sm_load"))
	CUDA_TRY(cuModuleGetFunction(&S.k_twiddle, S.module, "sm_twiddle"))
	CUDA_TRY(cuModuleGetFunction(&S.k_dif, S.module, "sm_dif"))
	CUDA_TRY(cuModuleGetFunction(&S.k_dit, S.module, "sm_dit"))
	CUDA_TRY(cuModuleGetFunction(&S.k_pointwise, S.module, "sm_pointwise"))
	CUDA_TRY(cuModuleGetFunction(&S.k_crt, S.module, "sm_crt"))

	/* the three largest primes c 2^40 + 1 below 2^62 */
	k = 0;
	for (u64 c = (((u64)1 << 62) - 1) >> SM_LOG2_MAX; k < SM_NP; c--) {
		const u64 p = (c << SM_LOG2_MAX) + 1;
		sm_prime_t *P;
		u64 inv = p, g;
		u128 R;

		if (!is_prime_h(p))
			continue;
		P = S.pr + k++;
		P->p = p;
		for (int i = 0; i < 6; i++)
			inv *= 2 - p * inv;
		P->minv = -inv;
		R = ((u128)1 << 64) % p;
		P->r2 = (u64)(R * R % p);
		/* a non-residue g: g^c has order 2^40 */
		for (g = 3; powmod_h(g, (p - 1) / 2, p) == 1; g++)
			;
		P->w0 = powmod_h(g, c, p);
	}
	p1 = S.pr[0].p;
	p2 = S.pr[1].p;
	p3 = S.pr[2].p;
	C.p1_inv_mod_p2R = to_mont_h(powmod_h(p1 % p2, p2 - 2, p2), p2);
	C.p1p2_inv_mod_p3R = to_mont_h(powmod_h(mulmod_h(p1 % p3, p2 % p3,
				p3), p3 - 2, p3), p3);
	C.p1_mod_p3R = to_mont_h(p1 % p3, p3);
	pp = (u128)p1 * p2;
	C.p1p2[0] = (u64)pp;
	C.p1p2[1] = (u64)(pp >> 64);
	{
		CUdeviceptr ptr;
		size_t sz;

		CUDA_TRY(cuModuleGetGlobal(&ptr, &sz, S.module, "c_pr"))
		CUDA_TRY(cuMemcpyHtoD(ptr, S.pr, sizeof(S.pr)))
		CUDA_TRY(cuModuleGetGlobal(&ptr, &sz, S.module, "c_crt"))
		CUDA_TRY(cuMemcpyHtoD(ptr, &C, sizeof(C)))
	}
	pthread_mutex_init(&S.lock, NULL);
	CUDA_TRY(cuCtxPopCurrent(NULL))
	S.ok = 1;
	logprintf(obj, "square root: products of %zu limbs and more on GPU %u "
			"(%s)%s\n", S.min_limbs, devs[0], info->name,
			S.check ? ", each checked with GMP" : "");
	return 1;

fail:
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuDevicePrimaryCtxRelease(S.dev))
	return 0;
}

/*------------------------------------------------------------------*/
void
sqrt_gpu_finish(msieve_obj *obj)
{
	if (!S.ok)
		return;
	logprintf(obj, "square root: %lu products on the GPU, %.1f s\n",
			S.calls, S.seconds);
	CUDA_TRY(cuCtxPushCurrent(S.ctx))
	while (S.threads != NULL) {
		sm_thread_t *t = S.threads;

		S.threads = t->next;
		if (t->d_a) cuMemFree(t->d_a);
		if (t->d_A) cuMemFree(t->d_A);
		if (t->d_B) cuMemFree(t->d_B);
		if (t->d_out) cuMemFree(t->d_out);
		if (t->d_w) cuMemFree(t->d_w);
		if (t->h_out) cuMemFreeHost(t->h_out);
		cuStreamDestroy(t->stream);
		free(t);
	}
	for (int lg = 0; lg <= SM_MAXLG; lg++) {
		if (S.tw_fwd[lg]) cuMemFree(S.tw_fwd[lg]);
		if (S.tw_inv[lg]) cuMemFree(S.tw_inv[lg]);
	}
	CUDA_TRY(cuModuleUnload(S.module))
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuDevicePrimaryCtxRelease(S.dev))
	pthread_mutex_destroy(&S.lock);
	S.ok = 0;
}

/*------------------------------------------------------------------*/
static sm_thread_t *
sm_thread(void)
{
	sm_thread_t *t = sm_mine;

	if (t != NULL)
		return t;
	t = (sm_thread_t *)xcalloc(1, sizeof(sm_thread_t));
	CUDA_TRY(cuStreamCreate(&t->stream, CU_STREAM_NON_BLOCKING))
	CUDA_TRY(cuMemAlloc(&t->d_w, SM_NP * sizeof(u64)))
	pthread_mutex_lock(&S.lock);
	t->next = S.threads;
	S.threads = t;
	pthread_mutex_unlock(&S.lock);
	sm_mine = t;
	return t;
}

static uint32
sm_blocks(size_t n, uint32 cap)
{
	size_t b = (n + 255) / 256;

	return (uint32)(b == 0 ? 1 : MIN(b, cap));
}

/* the twiddle tables for N = 2^lg (made once, under the lock) */
static void
sm_twiddles(sm_thread_t *t, int lg, CUdeviceptr *fwd, CUdeviceptr *inv)
{
	pthread_mutex_lock(&S.lock);
	if (S.tw_fwd[lg] == 0) {
		const u64 half = ((u64)1 << lg) / 2;
		u64 w[SM_NP], wi[SM_NP];
		CUdeviceptr f, iv;

		CUDA_TRY(cuMemAlloc(&f, SM_NP * half * sizeof(u64) + 8))
		CUDA_TRY(cuMemAlloc(&iv, SM_NP * half * sizeof(u64) + 8))
		for (int j = 0; j < SM_NP; j++) {
			const u64 p = S.pr[j].p;
			const u64 x = powmod_h(S.pr[j].w0,
					(u64)1 << (SM_LOG2_MAX - lg), p);

			w[j] = to_mont_h(x, p);
			wi[j] = to_mont_h(powmod_h(x, p - 2, p), p);
		}
		if (half) {
			CUdeviceptr tab[2] = { f, iv };
			u64 *src[2] = { w, wi };

			for (int k = 0; k < 2; k++) {
				void *args[3] = { &tab[k], (void *)&half,
						  &t->d_w };

				CUDA_TRY(cuMemcpyHtoDAsync(t->d_w, src[k],
						sizeof(w), t->stream))
				CUDA_TRY(cuLaunchKernel(S.k_twiddle, 256, SM_NP,
						1, 256, 1, 1, 0, t->stream,
						args, NULL))
				CUDA_TRY(cuStreamSynchronize(t->stream))
			}
		}
		S.tw_fwd[lg] = f;
		S.tw_inv[lg] = iv;
	}
	*fwd = S.tw_fwd[lg];
	*inv = S.tw_inv[lg];
	pthread_mutex_unlock(&S.lock);
}

/* forward transform of a[0 .. n) into A (N = 2^lg) */
static void
sm_transform(sm_thread_t *t, CUdeviceptr A, CUdeviceptr a, u64 n, int lg,
		CUdeviceptr T)
{
	u64 N = (u64)1 << lg;
	uint32 blocks = sm_blocks(N / 2, 4096);
	void *largs[4] = { &A, &a, &n, &N };

	CUDA_TRY(cuLaunchKernel(S.k_load, blocks, SM_NP, 1, 256, 1, 1, 0,
			t->stream, largs, NULL))
	for (u64 h = N / 2; h >= 1; h /= 2) {
		void *args[4] = { &A, &N, &h, &T };

		CUDA_TRY(cuLaunchKernel(S.k_dif, blocks, SM_NP, 1, 256, 1, 1,
				0, t->stream, args, NULL))
	}
}

/* r[0 .. na+nb) = a * b, 64-bit limbs, least significant first */
static void
sm_mul(u64 *r, const u64 *a, size_t na, const u64 *b, size_t nb)
{
	const double t0 = sm_now();
	sm_thread_t *t;
	const u64 nr = na + nb;
	u64 N;
	int lg = 1;
	const int square = (a == b && na == nb);
	CUdeviceptr Tf, Ti, dB;
	sm_ninv_t ninv;
	u128 acc = 0;

	CUDA_TRY(cuCtxSetCurrent(S.ctx))
	t = sm_thread();
	while (((u64)1 << lg) < nr)
		lg++;
	N = (u64)1 << lg;
	sm_twiddles(t, lg, &Tf, &Ti);

	if (N > t->cap_N) {
		if (t->d_A) cuMemFree(t->d_A);
		if (t->d_B) cuMemFree(t->d_B);
		if (t->d_out) cuMemFree(t->d_out);
		t->cap_N = N;
		CUDA_TRY(cuMemAlloc(&t->d_A, SM_NP * N * sizeof(u64)))
		CUDA_TRY(cuMemAlloc(&t->d_B, SM_NP * N * sizeof(u64)))
		CUDA_TRY(cuMemAlloc(&t->d_out, 3 * N * sizeof(u64)))
	}
	if (nr > t->cap_in) {
		if (t->d_a) cuMemFree(t->d_a);
		t->cap_in = nr;
		CUDA_TRY(cuMemAlloc(&t->d_a, nr * sizeof(u64)))
	}
	if (3 * nr > t->cap_hout) {
		if (t->h_out) cuMemFreeHost(t->h_out);
		t->cap_hout = 3 * nr;
		CUDA_TRY(cuMemAllocHost((void **)&t->h_out,
				t->cap_hout * sizeof(u64)))
	}
	for (int j = 0; j < SM_NP; j++)
		ninv.v[j] = powmod_h(N % S.pr[j].p, S.pr[j].p - 2, S.pr[j].p);

	CUDA_TRY(cuMemcpyHtoDAsync(t->d_a, a, na * sizeof(u64), t->stream))
	if (!square)
		CUDA_TRY(cuMemcpyHtoDAsync(t->d_a + na * sizeof(u64), b,
				nb * sizeof(u64), t->stream))
	sm_transform(t, t->d_A, t->d_a, na, lg, Tf);
	if (square) {
		dB = t->d_A;
	}
	else {
		sm_transform(t, t->d_B, t->d_a + na * sizeof(u64), nb, lg, Tf);
		dB = t->d_B;
	}
	{
		void *args[3] = { &t->d_A, &dB, &N };

		CUDA_TRY(cuLaunchKernel(S.k_pointwise, sm_blocks(N, 4096),
				SM_NP, 1, 256, 1, 1, 0, t->stream, args, NULL))
	}
	for (u64 h = 1; h <= N / 2; h *= 2) {
		void *args[4] = { &t->d_A, &N, &h, &Ti };

		CUDA_TRY(cuLaunchKernel(S.k_dit, sm_blocks(N / 2, 4096),
				SM_NP, 1, 256, 1, 1, 0, t->stream, args, NULL))
	}
	{
		u64 nrr = nr;
		void *args[5] = { &t->d_out, &t->d_A, &N, &nrr, &ninv };

		CUDA_TRY(cuLaunchKernel(S.k_crt, sm_blocks(nr, 8192), 1, 1,
				256, 1, 1, 0, t->stream, args, NULL))
	}
	CUDA_TRY(cuMemcpyDtoHAsync(t->h_out, t->d_out, 3 * nr * sizeof(u64),
			t->stream))
	CUDA_TRY(cuStreamSynchronize(t->stream))

	/* r = sum of out[i] (3 limbs) * 2^(64 i) */
	for (u64 i = 0; i < nr; i++) {
		acc += t->h_out[3 * i];
		if (i >= 1)
			acc += t->h_out[3 * (i - 1) + 1];
		if (i >= 2)
			acc += t->h_out[3 * (i - 2) + 2];
		r[i] = (u64)acc;
		acc >>= 64;
	}
	/* the product has nr limbs: nothing may be left over */
	acc += t->h_out[3 * (nr - 1) + 1];
	if (nr >= 2)
		acc += t->h_out[3 * (nr - 2) + 2];
	acc += (u128)t->h_out[3 * (nr - 1) + 2] << 64;
	if (acc != 0) {
		printf("square root GPU: product overflows %llu limbs\n",
				(unsigned long long)nr);
		exit(-1);
	}
	pthread_mutex_lock(&S.lock);
	S.calls++;
	S.seconds += sm_now() - t0;
	pthread_mutex_unlock(&S.lock);
}

/*------------------------------------------------------------------*/
void
sqrt_mpz_mul(mpz_t r, const mpz_t a, const mpz_t b)
{
	size_t na = mpz_size(a), nb = mpz_size(b);
	mpz_t t;
	mp_limb_t *rp;

	if (!S.ok || na < S.min_limbs || nb < S.min_limbs) {
		mpz_mul(r, a, b);
		return;
	}
	mpz_init2(t, (mp_bitcnt_t)(na + nb) * 64);
	rp = mpz_limbs_write(t, (mp_size_t)(na + nb));
	sm_mul((u64 *)rp, (const u64 *)mpz_limbs_read(a), na,
			(const u64 *)mpz_limbs_read(b), nb);
	mpz_limbs_finish(t, (mp_size_t)(na + nb));
	if (mpz_sgn(a) * mpz_sgn(b) < 0)
		mpz_neg(t, t);
	if (S.check) {
		mpz_t u;

		mpz_init(u);
		mpz_mul(u, a, b);
		if (mpz_cmp(u, t) != 0) {
			printf("square root GPU: product of %zu x %zu limbs "
				"differs from GMP's\n", na, nb);
			exit(-1);
		}
		mpz_clear(u);
	}
	mpz_swap(r, t);
	mpz_clear(t);
}
