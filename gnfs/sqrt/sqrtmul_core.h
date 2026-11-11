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

/* shared by sqrtmul_core.cu (the kernels) and sqrt_gpu_mul.c (host) */

#ifndef _GNFS_SQRT_SQRTMUL_CORE_H_
#define _GNFS_SQRT_SQRTMUL_CORE_H_

#define SM_NP 3			/* primes */
#define SM_LOG2_MAX 40		/* the primes are c 2^40 + 1 */

/* the interface between the host and the PTX: the PTX holds
   sm_abi = { SM_ABI, sizeof(sm_crt_t) }; raise SM_ABI when a kernel's
   arguments or these structures change */
#define SM_ABI 1

typedef struct {
	unsigned long long p, minv, r2;	/* modulus, -p^-1 mod 2^64,
					   2^128 mod p */
	unsigned long long w0;		/* a root of unity of order 2^40 */
} sm_prime_t;

typedef struct {
	unsigned long long p1_inv_mod_p2R;	/* (p1^-1 mod p2) R mod p2 */
	unsigned long long p1p2_inv_mod_p3R;	/* ((p1 p2)^-1 mod p3) R */
	unsigned long long p1_mod_p3R;		/* (p1 mod p3) R mod p3 */
	unsigned long long p1p2[2];		/* p1 p2, two limbs */
} sm_crt_t;

typedef struct {
	unsigned long long v[SM_NP];		/* N^-1 mod p_j (plain) */
} sm_ninv_t;

#endif
