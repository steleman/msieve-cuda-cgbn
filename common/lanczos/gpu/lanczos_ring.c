/*--------------------------------------------------------------------
MIT License, Copyright (c) 2026 Stefan Teleman (full text:
common/lanczos/gpu/lanczos_gpu.c)
--------------------------------------------------------------------*/

/* The GPU linear algebra's MPI grid operations as rings over a
   transport (MPI builds):

   - NCCL (HAVE_NCCL, MSIEVE_LA_NCCL=1): the device vectors, ncclSend /
     ncclRecv and ncclAllGather, the XORs as GPU kernels, all ordered on
     the default stream with no host synchronization. NCCL has no XOR
     reduction, so the XOR reduce-scatter is a ring of sends and
     receives.
   - MPI on host buffers (MSIEVE_LA_RING_TEST=1, for testing the ring
     logic: NCCL refuses two processes on one GPU, so the NCCL transport
     cannot run on a single-GPU machine).

   The chunks are those of matmul_util.c: of total vectors over n
   processes, chunk k starts at k * (total / n), the last one takes the
   remainder, and process r owns chunk r. */

#include <cuda_xface.h>
#include "../lanczos.h"
#include "lanczos_gpu.h"

#ifdef HAVE_MPI

#ifdef HAVE_NCCL
/* some NCCL headers (2.29) declare f() for f(void) */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include <nccl.h>
#pragma GCC diagnostic pop
#endif

typedef struct ring_t ring_t;

struct ring_t {
	uint32 rank, size;
	/* send s (ns vectors) to rank + 1 while receiving r (nr) from
	   rank - 1 */
	void (*xchg)(ring_t *t, v_t *s, uint32 ns, v_t *r, uint32 nr);
	void (*vxor)(v_t *d, v_t *s, uint32 n);
	void (*vcopy)(v_t *d, v_t *s, uint32 n);
	MPI_Comm comm;
#ifdef HAVE_NCCL
	ncclComm_t nccl;
#endif
};

static uint32 r_off(uint32 total, uint32 n, uint32 k) {
	return k * (total / n);
}

static uint32 r_size(uint32 total, uint32 n, uint32 k) {
	return total / n + ((k == n - 1) ? total % n : 0);
}

/* chunk r of buf (all n chunks laid out) to everyone */
static void
ring_allgather_inplace(ring_t *t, v_t *buf, uint32 total)
{
	uint32 n = t->size, r = t->rank;

	for (uint32 i = 0; i + 1 < n; i++) {
		uint32 si = (r + n - i) % n;
		uint32 ri = (r + 2 * n - i - 1) % n;

		t->xchg(t, buf + r_off(total, n, si), r_size(total, n, si),
			buf + r_off(total, n, ri), r_size(total, n, ri));
	}
}

/* XOR the n processes' work[] (clobbered) chunk by chunk: afterwards
   chunk r of work[] is the XOR of everyone's chunk r. tmp holds a
   chunk */
static void
ring_reduce_scatter_xor(ring_t *t, v_t *work, v_t *tmp, uint32 total)
{
	uint32 n = t->size, r = t->rank;

	for (uint32 i = 0; i + 1 < n; i++) {
		uint32 si = (r + 2 * n - i - 1) % n;
		uint32 ri = (r + 2 * n - i - 2) % n;

		t->xchg(t, work + r_off(total, n, si), r_size(total, n, si),
			tmp, r_size(total, n, ri));
		t->vxor(work + r_off(total, n, ri), tmp, r_size(total, n, ri));
	}
}

/* global_allgather: send = this process's chunk */
static void
ring_allgather(ring_t *t, v_t *send, v_t *recv, uint32 total)
{
	uint32 r = t->rank;

	t->vcopy(recv + r_off(total, t->size, r), send,
		 r_size(total, t->size, r));
	ring_allgather_inplace(t, recv, total);
}

/* global_xor: recv = XOR of everyone's send */
static void
ring_xor(ring_t *t, v_t *send, v_t *recv, v_t *tmp, uint32 total)
{
	t->vcopy(recv, send, total);
	if (t->size == 1)
		return;
	ring_reduce_scatter_xor(t, recv, tmp, total);
	ring_allgather_inplace(t, recv, total);
}

/* global_xor_scatter: recv = chunk r of the XOR of everyone's send
   (send is clobbered) */
static void
ring_xor_scatter(ring_t *t, v_t *send, v_t *recv, v_t *tmp, uint32 total)
{
	uint32 r = t->rank;

	ring_reduce_scatter_xor(t, send, tmp, total);
	t->vcopy(recv, send + r_off(total, t->size, r),
		 r_size(total, t->size, r));
}

/*------------------------------------------------------------------*/
/* the MPI transport on host buffers (testing) */

static void
mpi_xchg(ring_t *t, v_t *s, uint32 ns, v_t *r, uint32 nr)
{
	uint32 next = (t->rank + 1) % t->size;
	uint32 prev = (t->rank + t->size - 1) % t->size;

	MPI_TRY(MPI_Sendrecv(s, (int)(VWORDS * ns), MPI_LONG_LONG, (int)next,
			 91, r, (int)(VWORDS * nr), MPI_LONG_LONG, (int)prev,
			 91, t->comm, MPI_STATUS_IGNORE))
}

static void
host_xor(v_t *d, v_t *s, uint32 n)
{
	for (uint32 i = 0; i < n; i++)
		d[i] = v_xor(d[i], s[i]);
}

static void
host_copy(v_t *d, v_t *s, uint32 n)
{
	memmove(d, s, (size_t)n * sizeof(v_t));
}

/*------------------------------------------------------------------*/
#ifdef HAVE_NCCL
/* the NCCL transport: device buffers, stream-ordered on stream 0 */

#define NCCL_TRY(x) { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { \
	printf("NCCL error %s at %s:%d\n", ncclGetErrorString(e_), \
		__FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } }

static void
nccl_xchg(ring_t *t, v_t *s, uint32 ns, v_t *r, uint32 nr)
{
	uint32 next = (t->rank + 1) % t->size;
	uint32 prev = (t->rank + t->size - 1) % t->size;

	NCCL_TRY(ncclGroupStart())
	NCCL_TRY(ncclSend(s, (size_t)VWORDS * ns, ncclUint64, (int)next,
			t->nccl, 0))
	NCCL_TRY(ncclRecv(r, (size_t)VWORDS * nr, ncclUint64, (int)prev,
			t->nccl, 0))
	NCCL_TRY(ncclGroupEnd())
}

static void
dev_xor(v_t *d, v_t *s, uint32 n)
{
	gpu_vv_xor(d, s, n);		/* a kernel on stream 0 */
}

static void
dev_copy(v_t *d, v_t *s, uint32 n)
{
	if (n && d != s)
		CUDA_TRY(cuMemcpyDtoDAsync((CUdeviceptr)d, (CUdeviceptr)s,
				(size_t)n * sizeof(v_t), 0))
}

/* the NCCL all-gather where the chunks allow it: equal chunks through
   ncclAllGather, the last chunk's remainder broadcast from its owner */
static void
nccl_allgather(ring_t *t, v_t *send, v_t *recv, uint32 total)
{
	uint32 n = t->size, chunk = total / n, rem = total % n;

	NCCL_TRY(ncclGroupStart())
	NCCL_TRY(ncclAllGather(send, recv, (size_t)VWORDS * chunk,
			ncclUint64, t->nccl, 0))
	if (rem)
		NCCL_TRY(ncclBroadcast(send + chunk, recv + (size_t)n * chunk,
				(size_t)VWORDS * rem, ncclUint64, (int)(n - 1),
				t->nccl, 0))
	NCCL_TRY(ncclGroupEnd())
}
#endif

/*------------------------------------------------------------------*/
/* the rings of the GPU linear algebra: one along the MPI row and one
   down the MPI column, as matmul_util.c's comms */

static ring_t ring_row, ring_col;	/* row_grid, col_grid */
static uint32 ring_mode;		/* 0 off, 1 NCCL, 2 MPI test */
#ifdef HAVE_NCCL
static v_t *ring_tmp_dev;		/* NCCL: a chunk on the GPU */
#endif

static void
ring_set(ring_t *t, MPI_Comm comm, uint32 rank, uint32 size, uint32 mode)
{
	memset(t, 0, sizeof(*t));
	t->comm = comm;
	t->rank = rank;
	t->size = size;
	if (mode == 2) {
		t->xchg = mpi_xchg;
		t->vxor = host_xor;
		t->vcopy = host_copy;
	}
#ifdef HAVE_NCCL
	else {
		t->xchg = nccl_xchg;
		t->vxor = dev_xor;
		t->vcopy = dev_copy;
	}
#endif
}

#ifdef HAVE_NCCL
/* NCCL refuses two processes of a communicator on one GPU, and a
   failing ncclCommInitRank can leave the others waiting: check first,
   with MPI, that the ring's processes have different GPUs (every
   process gets the same answer) */
static uint32
ring_gpus_distinct(ring_t *t)
{
	uint8 mine[16], *all;
	uint32 ok = 1;

	la_gpu_uuid(mine);
	all = (uint8 *)xmalloc(16 * t->size);
	MPI_TRY(MPI_Allgather(mine, 16, MPI_BYTE, all, 16, MPI_BYTE,
				t->comm))
	for (uint32 i = 0; i < t->size && ok; i++)
		for (uint32 j = i + 1; j < t->size && ok; j++)
			ok = (memcmp(all + 16 * i, all + 16 * j, 16) != 0);
	free(all);
	return ok;
}

/* an NCCL communicator for an MPI one (its rank 0 makes the id);
   returns 0 on failure, with nothing created */
static uint32
nccl_comm_for(ring_t *t)
{
	ncclUniqueId id;
	ncclResult_t e = ncclSuccess;

	if (t->rank == 0)
		e = ncclGetUniqueId(&id);
	MPI_TRY(MPI_Bcast(&e, sizeof(e), MPI_BYTE, 0, t->comm))
	if (e != ncclSuccess)
		return 0;
	MPI_TRY(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, t->comm))
	return ncclCommInitRank(&t->nccl, (int)t->size, id, (int)t->rank) ==
			ncclSuccess;
}
#endif

/* called by every process of the grid, from gpu_matrix_extra_init:
   MSIEVE_LA_NCCL=1 for NCCL (all processes agree, or all fall back),
   MSIEVE_LA_RING_TEST=1 for the MPI test transport. Returns the mode */
uint32
la_ring_init(msieve_obj *obj, packed_matrix_t *p)
{
	const char *en = getenv("MSIEVE_LA_NCCL");
	const char *et = getenv("MSIEVE_LA_RING_TEST");
	uint32 want = 0;

	ring_mode = 0;
	if (p->mpi_size <= 1)
		return 0;
	if (et != NULL && strcmp(et, "1") == 0)
		want = 2;
#ifdef HAVE_NCCL
	else if (en != NULL && strcmp(en, "1") == 0)
		want = 1;
#else
	(void)en;
#endif
	if (want == 0)
		return 0;

	ring_set(&ring_row, p->mpi_la_row_grid, p->mpi_la_col_rank,
			p->mpi_ncols, want);
	ring_set(&ring_col, p->mpi_la_col_grid, p->mpi_la_row_rank,
			p->mpi_nrows, want);

#ifdef HAVE_NCCL
	if (want == 1) {
		int ok_row = 1, ok_col = 1, ok, all, distinct;

		/* one GPU per process in each ring, decided by all */
		distinct = (ring_row.size <= 1 || ring_gpus_distinct(&ring_row)) &&
			   (ring_col.size <= 1 || ring_gpus_distinct(&ring_col));
		MPI_TRY(MPI_Allreduce(&distinct, &all, 1, MPI_INT, MPI_MIN,
				obj->mpi_la_grid))
		if (!all) {
			logprintf(obj, "linear algebra: NCCL not used (it "
				"needs one GPU per process; GPUs are shared "
				"here), MPI instead\n");
			return 0;
		}
		if (ring_row.size > 1)
			ok_row = nccl_comm_for(&ring_row);
		if (ring_col.size > 1)
			ok_col = nccl_comm_for(&ring_col);
		ok = ok_row && ok_col;
		MPI_TRY(MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN,
				obj->mpi_la_grid))
		if (!all) {
			if (ok_row && ring_row.size > 1)
				ncclCommDestroy(ring_row.nccl);
			if (ok_col && ring_col.size > 1)
				ncclCommDestroy(ring_col.nccl);
			logprintf(obj, "linear algebra: NCCL not usable "
				"(NCCL needs one GPU per process), MPI "
				"instead\n");
			return 0;
		}
		ring_tmp_dev = (v_t *)gpu_vv_alloc(MAX(p->nrows, p->ncols) /
				MIN(ring_row.size, ring_col.size) + 2 *
				MAX(ring_row.size, ring_col.size) + 64, NULL);
	}
#endif
	ring_mode = want;
	logprintf(obj, "linear algebra MPI: grid operations by %s\n",
		(want == 1) ? "NCCL" : "MPI rings on host buffers (test)");
	return want;
}

void
la_ring_free(void)
{
#ifdef HAVE_NCCL
	if (ring_mode == 1) {
		if (ring_row.size > 1)
			ncclCommDestroy(ring_row.nccl);
		if (ring_col.size > 1)
			ncclCommDestroy(ring_col.nccl);
		gpu_vv_free(ring_tmp_dev);
		ring_tmp_dev = NULL;
	}
#endif
	ring_mode = 0;
}

uint32
la_ring_mode(void)
{
	return ring_mode;
}

/* gpu_mpi_mul with the rings. NCCL: x, b, scratch are device vectors;
   test mode: host buffers h0 (nsubcols), h1, h2 (MAX(nrows, ncols)),
   staged by the caller */
void
la_ring_mul(packed_matrix_t *A, void *x, void *b, void *scratch,
		uint32 sym)
{
	size_t big = MAX(A->ncols, A->nrows);

	if (ring_mode == 2) {
		v_t *h0 = la_gpu_host_buf(0, big);
		v_t *h1 = la_gpu_host_buf(1, big);
		v_t *h2 = la_gpu_host_buf(2, big);
		v_t *s2 = (v_t *)scratch + big;

		vv_copyout(h0, x, A->nsubcols);
		ring_allgather(&ring_col, h0, h1, A->ncols);
		vv_copyin(scratch, h1, A->ncols);
		mul_core(A, scratch, s2);
		vv_copyout(h1, s2, A->nrows);
		ring_xor(&ring_row, h1, h2, h0, A->nrows);
		vv_copyin(scratch, h2, A->nrows);
		if (!sym)
			return;
		mul_trans_core(A, scratch, s2);
		vv_copyout(h1, s2, A->ncols);
		ring_xor_scatter(&ring_col, h1, h0, h2, A->ncols);
		vv_copyin(b, h0, A->nsubcols);
		return;
	}

#ifdef HAVE_NCCL
	{
		v_t *s2 = (v_t *)scratch + big;

		if (ring_col.size > 1)
			nccl_allgather(&ring_col, (v_t *)x, (v_t *)scratch,
					A->ncols);
		else
			dev_copy((v_t *)scratch, (v_t *)x, A->ncols);
		mul_core(A, scratch, s2);
		ring_xor(&ring_row, s2, (v_t *)scratch, ring_tmp_dev,
				A->nrows);
		if (!sym)
			return;
		mul_trans_core(A, scratch, s2);
		if (ring_col.size > 1)
			ring_xor_scatter(&ring_col, s2, (v_t *)b,
					ring_tmp_dev, A->ncols);
		else
			dev_copy((v_t *)b, s2, A->ncols);
	}
#endif
}

#endif /* HAVE_MPI */
