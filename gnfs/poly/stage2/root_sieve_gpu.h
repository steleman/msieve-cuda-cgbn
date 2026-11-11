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

Part of a fork of msieve (public domain, by Jason Papadopoulos).
--------------------------------------------------------------------*/

/* Root optimization on the GPU.

   Polynomials submitted here are root-optimized by msieve's code with
   the batched root sieve (root_sieve_batch.c), each in a coroutine on
   one of the engine's threads. Its optimize_basic() calls and line
   sieves come here many at a time; each thread sends its coroutines'
   requests to the GPU as one batch (so_basic_<degree> and ls_sieve in
   sizeopt_core.cu) and resumes them with the results. The results are
   those of the batched root sieve on the CPU, bit for bit (both built
   without floating-point contraction). */

#ifndef _ROOT_SIEVE_GPU_H_
#define _ROOT_SIEVE_GPU_H_

#include <poly_skew.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ro_gpu ro_gpu_t;

/* proto: a set-up poly_rootopt_t whose parameters (N, degree, bounds,
   callback) every root optimization uses. NULL if the GPU or its
   module is not available, or the degree is not 4-6 (logged) */
ro_gpu_t * ro_gpu_init(msieve_obj *obj, poly_rootopt_t *proto);

/* queue one polynomial (the arguments of poly_rootopt_run); blocks
   while the queue is full. Thread safe */
void ro_gpu_submit(ro_gpu_t *g, mpz_t *alg_coeffs, mpz_t *rat_coeffs,
			double sizeopt_norm, double projective_alpha);

/* wait for every queued polynomial, then free everything */
void ro_gpu_finish(ro_gpu_t *g);

#ifdef __cplusplus
}
#endif

#endif /* !_ROOT_SIEVE_GPU_H_ */
