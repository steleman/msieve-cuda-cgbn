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

/* host side of the GPU size optimization (sizeopt_core.cu) */

#ifndef _STAGE1_SIZEOPT_GPU_H_
#define _STAGE1_SIZEOPT_GPU_H_

#include <stdbool.h>
#include <stage1.h>
#include <cuda_xface.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct so_ctx so_ctx_t;

/* the sizeopt_core_sm*.ptx module for a compute capability (NULL if
   none); it also holds the root optimization's kernels */
const char * so_ptx_name(uint32 compute_major, uint32 compute_minor);

/* load the module for this GPU (in the current context); NULL if the
   PTX file is missing, after logging why */
so_ctx_t * so_init(msieve_obj *obj, uint32 compute_major,
			uint32 compute_minor, uint32 num_sm);

void so_free(so_ctx_t *s);

/* per leading coefficient: returns true if the GPU handles it (degree 4
   or 5, and a CGBN width big enough for its integers) */
bool so_set_coeff(so_ctx_t *s, const poly_search_t *poly,
			const poly_coeff_t *c);

/* rebuild the collisions in the device found array (count in its
   slot 0) into candidates; the size optimization runs on many
   batches at once, when enough are collected or at so_flush() */
void so_run_batch(so_ctx_t *s, stage1_sieve_data_t *d, CUstream stream,
		CUdeviceptr found, uint32 num_found,
		CUdeviceptr q, CUdeviceptr which_q, CUdeviceptr roots,
		uint32 sizeof_q, uint64 sieve_size);

/* size-optimize the collected candidates and hand each polynomial
   that passes the stage 2 bound to stage1_sizeopt_submit(); call at
   the end of each leading coefficient's search */
void so_flush(so_ctx_t *s);

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_SIZEOPT_GPU_H_ */
