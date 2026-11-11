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

/* The block Lanczos iteration on a GPU (kernels: lanczos_core.cu).
   The matrix and every Lanczos vector live in device memory; a vector
   handle (vv_alloc) is a device address. With MPI, each process runs
   its block of the matrix on its own GPU (local rank r of a node takes
   its devices as gpu_select gives them), and the grid operations of
   lanczos_matmul.c go through host buffers (la_gpu_host_buf). The
   dispatcher (lanczos_dispatch.c) chooses this backend or the CPU's
   per matrix. */

#include <cuda_xface.h>
#include "../lanczos.h"
#include "lanczos_gpu.h"
#ifdef HAVE_MPI
#include <mpi-ext.h>	/* MPIX_CUDA_AWARE_SUPPORT, if the MPI has it */
#endif

typedef struct {
	CUcontext ctx;			/* the device's primary context */
	CUdevice dev;
	uint32 cuda_mpi;		/* MPI takes device buffers */
	CUmodule module;
	CUfunction spmv_rows, dense_rows, spmv_cols;
	CUfunction vxor, vmask, mul_nxb, mul_bxn;
	uint32 num_sm;

	uint32 nrows, ncols, num_dense, dw;
	uint64 nnz;
	CUdeviceptr row_off, row_col;	/* by rows: b = A x */
	CUdeviceptr col_off, col_row;	/* by columns: b = A^T x */
	CUdeviceptr bits;		/* dense rows, dw words a column */
	CUdeviceptr d_X, d_out;		/* VBITS x VBITS operands */

	v_t *host_buf[3];		/* pinned, for MPI */
	size_t host_cap[3];
	size_t dev_bytes;
} gpudata_t;

static gpudata_t *la_g;

static const char *
la_ptx_name(int32 major, int32 minor)
{
	if (major >= 12)
		return "lanczos_core_sm120.ptx";
	if (major >= 10)
		return "lanczos_core_sm100.ptx";
	if (major == 9)
		return "lanczos_core_sm90.ptx";
	if (major == 8)
		return (minor < 6) ? "lanczos_core_sm80.ptx" :
			(minor < 9) ? "lanczos_core_sm86.ptx" :
				      "lanczos_core_sm89.ptx";
	if (major == 7)
		return (minor < 5) ? "lanczos_core_sm70.ptx" :
				     "lanczos_core_sm75.ptx";
	if (major == 6)
		return (minor < 1) ? "lanczos_core_sm60.ptx" :
				     "lanczos_core_sm61.ptx";
	return NULL;
}

static uint32
la_grid(uint32 n, uint32 per_block)
{
	uint32 g = (n + per_block - 1) / per_block;

	return MAX(1, MIN(g, la_g->num_sm * 16));
}

static void
la_launch(CUfunction f, uint32 grid, uint32 block, uint32 smem,
		void **args)
{
	CUDA_TRY(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, 0,
				args, NULL))
}

/*------------------------------------------------------------------*/
/* the GPU for this process, if the matrix fits on it. Returns 0 to
   use the CPU */

uint32
gpu_matrix_extra_init(msieve_obj *obj, packed_matrix_t *p)
{
	gpu_config_t config;
	uint32 devs[MAX_GPU], ndev, lr = 0, ls = 1;
	gpu_info_t *info;
	const char *name;
	gpudata_t *g;
	la_col_t *A = p->unpacked_cols;
	uint32 i, nrows = p->nrows, ncols = p->ncols;
	uint32 nd = p->num_dense_rows, dw = (nd + 31) / 32;
	uint64 nnz = 0;
	size_t need, mfree = 0, mtotal = 0;
	CUcontext ctx;

	if (A == NULL)
		return 0;
	gpu_init(&config);
#ifdef HAVE_MPI
	lr = obj->mpi_local_rank;
	ls = obj->mpi_local_size;
#endif
	ndev = gpu_select(&config, obj->which_gpu, lr, ls, devs, NULL);
	if (ndev == 0)
		return 0;
	info = config.info + devs[0];
	name = la_ptx_name(info->compute_version_major,
			info->compute_version_minor);
	if (name == NULL)
		return 0;

	for (i = 0; i < ncols; i++)
		nnz += A[i].weight;

	/* the primary context: the one a CUDA-aware MPI (UCX) works in */
	CUDA_TRY(cuDevicePrimaryCtxRetain(&ctx, info->device_handle))
	CUDA_TRY(cuCtxPushCurrent(ctx))

	/* the matrix twice, its bitmaps, and the iteration's vectors
	   (7 of them plus a double-size scratch) */
	need = (size_t)nnz * 2 * sizeof(uint32) +
		(size_t)(nrows + ncols + 2) * sizeof(uint64) +
		(size_t)ncols * dw * sizeof(uint32) +
		(size_t)9 * MAX(nrows, ncols) * sizeof(v_t);
	CUDA_TRY(cuMemGetInfo(&mfree, &mtotal))
	if (need > mfree / 10 * 9) {
		logprintf(obj, "linear algebra on the CPU: the matrix needs "
				"%.0f MB, GPU %u has %.0f MB free\n",
				need / 1048576.0, devs[0], mfree / 1048576.0);
		CUDA_TRY(cuCtxPopCurrent(NULL))
		CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
		return 0;
	}

	g = (gpudata_t *)xcalloc(1, sizeof(gpudata_t));
	g->ctx = ctx;
	g->dev = info->device_handle;
	if (cuModuleLoad(&g->module, name) != CUDA_SUCCESS) {
		logprintf(obj, "linear algebra on the CPU: cannot load %s\n",
				name);
		CUDA_TRY(cuCtxPopCurrent(NULL))
		CUDA_TRY(cuDevicePrimaryCtxRelease(info->device_handle))
		free(g);
		return 0;
	}
	CUDA_TRY(cuModuleGetFunction(&g->spmv_rows, g->module, "la_spmv_rows"))
	CUDA_TRY(cuModuleGetFunction(&g->dense_rows, g->module, "la_dense_rows"))
	CUDA_TRY(cuModuleGetFunction(&g->spmv_cols, g->module, "la_spmv_cols"))
	CUDA_TRY(cuModuleGetFunction(&g->vxor, g->module, "la_xor"))
	CUDA_TRY(cuModuleGetFunction(&g->vmask, g->module, "la_mask"))
	CUDA_TRY(cuModuleGetFunction(&g->mul_nxb, g->module, "la_mul_nxb"))
	CUDA_TRY(cuModuleGetFunction(&g->mul_bxn, g->module, "la_mul_bxn"))
	g->num_sm = (uint32)info->num_compute_units;
	g->nrows = nrows;
	g->ncols = ncols;
	g->num_dense = nd;
	g->dw = dw;
	g->nnz = nnz;

	/* by columns: the la_col_t entries as they are */
	{
		uint64 *off = (uint64 *)xmalloc((ncols + 1) * sizeof(uint64));
		uint32 *row = (uint32 *)xmalloc(MAX(1, nnz) * sizeof(uint32));
		uint32 *bits = (uint32 *)xcalloc(MAX(1, (size_t)ncols * dw),
						sizeof(uint32));
		uint64 *roff = (uint64 *)xcalloc(nrows + 1, sizeof(uint64));
		uint32 *rcol = (uint32 *)xmalloc(MAX(1, nnz) * sizeof(uint32));
		uint64 k = 0;

		for (i = 0; i < ncols; i++) {
			la_col_t *c = A + i;

			off[i] = k;
			memcpy(row + k, c->data, c->weight * sizeof(uint32));
			k += c->weight;
			if (dw)
				memcpy(bits + (size_t)i * dw,
					c->data + c->weight,
					dw * sizeof(uint32));
			for (uint32 j = 0; j < c->weight; j++)
				roff[c->data[j] + 1]++;
		}
		off[ncols] = k;

		/* by rows, columns ascending */
		for (i = 0; i < nrows; i++)
			roff[i + 1] += roff[i];
		{
			uint64 *pos = (uint64 *)xmalloc(nrows *
							sizeof(uint64));

			memcpy(pos, roff, nrows * sizeof(uint64));
			for (i = 0; i < ncols; i++) {
				la_col_t *c = A + i;

				for (uint32 j = 0; j < c->weight; j++)
					rcol[pos[c->data[j]]++] = i;
			}
			free(pos);
		}

		CUDA_TRY(cuMemAlloc(&g->col_off, (ncols + 1) * sizeof(uint64)))
		CUDA_TRY(cuMemAlloc(&g->col_row, MAX(1, nnz) * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&g->row_off, (nrows + 1) * sizeof(uint64)))
		CUDA_TRY(cuMemAlloc(&g->row_col, MAX(1, nnz) * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&g->bits, MAX(1, (size_t)ncols * dw) *
						sizeof(uint32)))
		CUDA_TRY(cuMemcpyHtoD(g->col_off, off,
				(ncols + 1) * sizeof(uint64)))
		CUDA_TRY(cuMemcpyHtoD(g->col_row, row, nnz * sizeof(uint32)))
		CUDA_TRY(cuMemcpyHtoD(g->row_off, roff,
				(nrows + 1) * sizeof(uint64)))
		CUDA_TRY(cuMemcpyHtoD(g->row_col, rcol, nnz * sizeof(uint32)))
		if (dw)
			CUDA_TRY(cuMemcpyHtoD(g->bits, bits, (size_t)ncols *
						dw * sizeof(uint32)))
		free(off);
		free(row);
		free(bits);
		free(roff);
		free(rcol);
	}
	CUDA_TRY(cuMemAlloc(&g->d_X, VBITS * sizeof(v_t)))
	CUDA_TRY(cuMemAlloc(&g->d_out, VBITS * sizeof(v_t)))
	g->dev_bytes = need;

	/* as the CPU backend does once it has packed the matrix: the
	   column entries are not needed any more */
	for (i = 0; i < ncols; i++) {
		free(A[i].data);
		A[i].data = NULL;
	}
	p->unpacked_cols = NULL;
	p->extra = g;
	la_g = g;

	logprintf(obj, "linear algebra on GPU %u (%s): %u x %u, %" PRIu64
			" nonzeros, %u dense rows, %.0f MB\n", devs[0],
			info->name, nrows, ncols, nnz, nd, need / 1048576.0);

#ifdef HAVE_MPI
	/* the MPI grid operations take the device vectors directly if
	   the MPI is CUDA-aware (MSIEVE_LA_CUDA_MPI=0: through host
	   buffers anyway) */
	{
		const char *e = getenv("MSIEVE_LA_CUDA_MPI");

		g->cuda_mpi = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
		g->cuda_mpi = (MPIX_Query_cuda_support() == 1);
#endif
		if (e != NULL && strcmp(e, "0") == 0)
			g->cuda_mpi = 0;
		if (p->mpi_size > 1)
			logprintf(obj, "linear algebra MPI: %s\n",
				g->cuda_mpi ? "GPU buffers passed to MPI "
					"(CUDA-aware)" :
					"through host buffers");
	}

	/* or rings: NCCL, or the MPI test transport */
	la_ring_init(obj, p);
#endif
	return 1;
}

/*------------------------------------------------------------------*/
void
gpu_matrix_extra_free(packed_matrix_t *p)
{
	gpudata_t *g = (gpudata_t *)p->extra;

	if (g == NULL)
		return;
#ifdef HAVE_MPI
	la_ring_free();
#endif
	CUDA_TRY(cuMemFree(g->col_off))
	CUDA_TRY(cuMemFree(g->col_row))
	CUDA_TRY(cuMemFree(g->row_off))
	CUDA_TRY(cuMemFree(g->row_col))
	CUDA_TRY(cuMemFree(g->bits))
	CUDA_TRY(cuMemFree(g->d_X))
	CUDA_TRY(cuMemFree(g->d_out))
	for (uint32 i = 0; i < 3; i++) {
		if (g->host_buf[i] != NULL)
			CUDA_TRY(cuMemFreeHost(g->host_buf[i]))
	}
	CUDA_TRY(cuModuleUnload(g->module))
	CUDA_TRY(cuCtxPopCurrent(NULL))
	CUDA_TRY(cuDevicePrimaryCtxRelease(g->dev))
	free(g);
	p->extra = NULL;
	la_g = NULL;
}

size_t
gpu_packed_matrix_sizeof(packed_matrix_t *p)
{
	gpudata_t *g = (gpudata_t *)p->extra;

	return g->dev_bytes;
}

/* the MPI grid operations may use the device vectors directly */
uint32
la_gpu_cuda_mpi(void)
{
	return la_g != NULL && la_g->cuda_mpi;
}

/* this process's GPU, for the NCCL check (one GPU per process) */
void
la_gpu_uuid(uint8 out[16])
{
	CUuuid u;

	CUDA_TRY(cuDeviceGetUuid(&u, la_g->dev))
	memcpy(out, u.bytes, 16);
}

/* the GPU's work so far finished (before MPI reads device memory: a
   CUDA-aware MPI copies on its own streams) */
void
la_gpu_sync(void)
{
	CUDA_TRY(cuCtxSynchronize())
}

/* a pinned host buffer of n vectors, for the MPI grid operations */
v_t *
la_gpu_host_buf(uint32 which, size_t n)
{
	gpudata_t *g = la_g;

	if (n > g->host_cap[which]) {
		if (g->host_buf[which] != NULL)
			CUDA_TRY(cuMemFreeHost(g->host_buf[which]))
		g->host_cap[which] = n + n / 8;
		CUDA_TRY(cuMemAllocHost((void **)&g->host_buf[which],
				g->host_cap[which] * sizeof(v_t)))
	}
	return g->host_buf[which];
}

/*------------------------------------------------------------------*/
void
gpu_mul_core(packed_matrix_t *p, void *x, void *b)
{
	gpudata_t *g = (gpudata_t *)p->extra;
	CUdeviceptr dx = (CUdeviceptr)x, db = (CUdeviceptr)b;

	{
		void *args[5] = { &g->row_off, &g->row_col, &dx, &db,
				  &g->nrows };

		la_launch(g->spmv_rows, (g->nrows + 7) / 8, 256, 0, args);
	}
	if (g->num_dense) {
		void *args[6] = { &g->bits, &g->dw, &g->num_dense, &dx, &db,
				  &g->ncols };

		la_launch(g->dense_rows, g->num_sm * 8, 256, 0, args);
	}
}

void
gpu_mul_trans_core(packed_matrix_t *p, void *x, void *b)
{
	gpudata_t *g = (gpudata_t *)p->extra;
	CUdeviceptr dx = (CUdeviceptr)x, db = (CUdeviceptr)b;
	void *args[8] = { &g->col_off, &g->col_row, &g->bits, &g->dw,
			  &g->num_dense, &dx, &db, &g->ncols };

	la_launch(g->spmv_cols, (g->ncols + 7) / 8, 256,
			g->num_dense * sizeof(v_t), args);
}

/*------------------------------------------------------------------*/
void *
gpu_vv_alloc(uint32 n, void *extra)
{
	CUdeviceptr d;

	(void)extra;
	CUDA_TRY(cuMemAlloc(&d, MAX(1, (size_t)n) * sizeof(v_t)))
	return (void *)d;
}

void
gpu_vv_free(void *v)
{
	if (v != NULL)
		CUDA_TRY(cuMemFree((CUdeviceptr)v))
}

void
gpu_vv_copyin(void *dest, v_t *src, uint32 n)
{
	if (n)
		CUDA_TRY(cuMemcpyHtoD((CUdeviceptr)dest, src,
					(size_t)n * sizeof(v_t)))
}

void
gpu_vv_copy(void *dest, void *src, uint32 n)
{
	if (n && dest != src)
		CUDA_TRY(cuMemcpyDtoD((CUdeviceptr)dest, (CUdeviceptr)src,
					(size_t)n * sizeof(v_t)))
}

void
gpu_vv_copyout(v_t *dest, void *src, uint32 n)
{
	if (n)
		CUDA_TRY(cuMemcpyDtoH(dest, (CUdeviceptr)src,
					(size_t)n * sizeof(v_t)))
}

void
gpu_vv_clear(void *v, uint32 n)
{
	if (n)
		CUDA_TRY(cuMemsetD8((CUdeviceptr)v, 0,
					(size_t)n * sizeof(v_t)))
}

void
gpu_vv_xor(void *dest, void *src, uint32 n)
{
	CUdeviceptr d = (CUdeviceptr)dest, s = (CUdeviceptr)src;
	void *args[3] = { &d, &s, &n };

	if (n)
		la_launch(la_g->vxor, la_grid(n, 256), 256, 0, args);
}

void
gpu_vv_mask(void *v, v_t mask, uint32 n)
{
	CUdeviceptr d = (CUdeviceptr)v;
	void *args[3] = { &d, &mask, &n };

	if (n)
		la_launch(la_g->vmask, la_grid(n, 256), 256, 0, args);
}

void
gpu_vv_mul_NxB_BxB_acc(packed_matrix_t *A, void *v, v_t *x,
			void *y, uint32 n)
{
	gpudata_t *g = (gpudata_t *)A->extra;
	CUdeviceptr dv = (CUdeviceptr)v, dy = (CUdeviceptr)y;
	void *args[4] = { &dv, &g->d_X, &dy, &n };

	CUDA_TRY(cuMemcpyHtoD(g->d_X, x, VBITS * sizeof(v_t)))
	if (n)
		la_launch(g->mul_nxb, la_grid(n, 256), 256, 0, args);
}

void
gpu_vv_mul_BxN_NxB(packed_matrix_t *A, void *x, void *y,
			v_t *xy, uint32 n)
{
	gpudata_t *g = (gpudata_t *)A->extra;
	CUdeviceptr dx = (CUdeviceptr)x, dy = (CUdeviceptr)y;
	void *args[4] = { &dx, &dy, &g->d_out, &n };

	CUDA_TRY(cuMemsetD8(g->d_out, 0, VBITS * sizeof(v_t)))
	if (n)
		la_launch(g->mul_bxn, la_grid(n, 256), 256, 0, args);
	CUDA_TRY(cuMemcpyDtoH(xy, g->d_out, VBITS * sizeof(v_t)))

#ifdef HAVE_MPI
	{
		/* as the CPU backend: combine across the MPI row, then
		   the MPI column */
		v_t xytmp[VBITS];

		global_xor(xy, xytmp, VBITS, A->mpi_ncols,
				A->mpi_la_col_rank, A->mpi_la_row_grid);
		global_xor(xytmp, xy, VBITS, A->mpi_nrows,
				A->mpi_la_row_rank, A->mpi_la_col_grid);
	}
#endif
}
