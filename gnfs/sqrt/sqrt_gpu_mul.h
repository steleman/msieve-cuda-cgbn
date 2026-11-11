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

/* The NFS square root's large multiplications on the GPU
   (sqrt_gpu_mul.c, kernels in sqrtmul_core.cu): the products of the
   relations' product tree and of the Newton iteration, up to tens of
   millions of bits. At 120 digits they are only part of the square
   root's time (reading the relations, the many small polynomial
   products and the reductions stay on the CPU): 33.7 -> 29.1 s.
   sqrt_mpz_mul is mpz_mul, with the
   products whose operands both have at least MSIEVE_SQRT_GPU_MIN limbs
   (2048) done on the GPU when one is there; the result is the same
   integer. Without CUDA it is mpz_mul.

   MSIEVE_SQRT_GPU=0: GMP only. MSIEVE_SQRT_GPU_CHECK=1: every GPU
   product is compared with mpz_mul's (slow; tests). */

#ifndef _GNFS_SQRT_SQRT_GPU_MUL_H_
#define _GNFS_SQRT_SQRT_GPU_MUL_H_

#include <msieve.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CUDA
/* before the square root's threads start; 0 if the GPU is not used */
int sqrt_gpu_start(msieve_obj *obj);

/* after they ended: the counts into the log, the GPU memory freed */
void sqrt_gpu_finish(msieve_obj *obj);

void sqrt_mpz_mul(mpz_t r, const mpz_t a, const mpz_t b);
#else
#define sqrt_gpu_start(obj) 0
#define sqrt_gpu_finish(obj) do {} while (0)
#define sqrt_mpz_mul(r, a, b) mpz_mul(r, a, b)
#endif

#ifdef __cplusplus
}
#endif

#endif
