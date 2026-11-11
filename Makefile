# --------------------------------------------------------------------
# This source distribution is placed in the public domain by its author,
# Jason Papadopoulos. You may use it for any purpose, free of charge,
# without having to notify anyone. I disclaim any responsibility for any
# errors.
# 
# Optionally, please be nice and tell me if you find this source to be
# useful. Again optionally, if you add to the functionality present here
# please consider making those additions public too, so that others may 
# benefit from your work.	
#
#  $Id: Makefile 1100 2026-10-10 16:13:45Z jasonp_sf $
# --------------------------------------------------------------------

# override from command line
WIN = 0
WIN64 = 0
VBITS = 64

# gcc with basic optimization (-march flag could
# get overridden by architecture-specific builds)
CC = gcc
# standard C11 plus POSIX/GNU extensions (pthreads, dlopen, ...).
# Note that -std=c11 also means -ffp-contract=off (gcc's GNU modes
# fuse a*b+c into FMA instructions), so size optimization results
# can differ slightly from a gnu17 build. -Woverlength-strings is
# off because the objective function in optimize_deg6.c is one
# 5024-byte string, longer than the 4095 bytes C99 compilers must
# support
STD_FLAGS = -std=c11 -D_GNU_SOURCE -D_DEFAULT_SOURCE
WARN_FLAGS = -Wall -W -Wpedantic -Wstrict-prototypes -Wold-style-definition \
	     -Wmissing-prototypes -Wno-overlength-strings
OPT_FLAGS = -O3 -fomit-frame-pointer -march=native \
	    -D_FILE_OFFSET_BITS=64 -DNDEBUG -D_LARGEFILE64_SOURCE -DVBITS=$(VBITS)

# OPT_FLAGS = -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -DVBITS=$(VBITS)
# OPT_FLAGS = -O3 -march=native -g -pg \
# 	    -D_FILE_OFFSET_BITS=64 -DNDEBUG -D_LARGEFILE64_SOURCE -DVBITS=$(VBITS)

# use := instead of = so we only run the following once
SVN_VERSION := $(shell svnversion .)
ifeq ($(SVN_VERSION),)
	SVN_VERSION := unknown
endif

GPU_LATTICE ?= 1

CFLAGS = $(STD_FLAGS) $(OPT_FLAGS) $(MACHINE_FLAGS) $(WARN_FLAGS) \
	 	-DMSIEVE_SVN_VERSION="\"$(SVN_VERSION)\"" \
		-I. -Iaprcl -Iinclude -Ignfs -Ignfs/poly -Ignfs/poly/stage1

# tweak the compile flags

ifneq ($(CUDA),) # ---------------------------------------------------------------

NVCC_FLAGS = -Wno-deprecated-gpu-targets -O3 -lineinfo -ptx \
	     -DCUB_IGNORE_DEPRECATED_CPP_DIALECT

ifeq ($(WIN),1)  # ---------- CUDA on windows or Msys2 ---------
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"

ifeq ($(WIN64),1)
	CUDA_LIBS = "$(CUDA_ROOT)/lib/x64/cuda.lib"
else
	CUDA_LIBS = "$(CUDA_ROOT)/lib/win32/cuda.lib"
endif

else        # ----------- CUDA on linux ------------------------
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	CUDA_LIBS = $(CUDA_ROOT)/lib64/stubs/libcuda.so
endif       # --------------------------------------------------

	CFLAGS += -I"$(CUDA_ROOT)/include" -Icub -DHAVE_CUDA
	# GPU_LATTICE=1 (the default): the root sieve's lattice choice on
	# the GPU too (lattice_core.h); GPU_LATTICE=0: on the CPU
	ifeq ($(GPU_LATTICE),1)
		CFLAGS += -DRS_GPU_LATTICE
	endif
	LIBS += $(CUDA_LIBS)

ifeq ($(CUDA),12)
	GPU_OBJS = \
		stage1_core_sm60.ptx \
		stage1_core_sm61.ptx \
		stage1_core_sm70.ptx \
		stage1_core_sm75.ptx \
		sizeopt_core_sm60.ptx \
		sizeopt_core_sm61.ptx \
		sizeopt_core_sm70.ptx \
		sizeopt_core_sm75.ptx \
		lanczos_core_sm60.ptx \
		lanczos_core_sm61.ptx \
		lanczos_core_sm70.ptx \
		lanczos_core_sm75.ptx \
		sieve_core_sm60.ptx \
		lasieve_core_sm60.ptx \
		sqrtmul_core_sm60.ptx \
		sieve_core_sm61.ptx \
		lasieve_core_sm61.ptx \
		sqrtmul_core_sm61.ptx \
		sieve_core_sm70.ptx \
		lasieve_core_sm70.ptx \
		sqrtmul_core_sm70.ptx \
		sieve_core_sm75.ptx \
		lasieve_core_sm75.ptx \
		sqrtmul_core_sm75.ptx \
		stage1_core_sm80.ptx \
		stage1_core_sm86.ptx \
		stage1_core_sm89.ptx \
		stage1_core_sm90.ptx \
		stage1_core_sm100.ptx \
		stage1_core_sm120.ptx \
		sizeopt_core_sm80.ptx \
		sizeopt_core_sm86.ptx \
		sizeopt_core_sm89.ptx \
		sizeopt_core_sm90.ptx \
		sizeopt_core_sm100.ptx \
		sizeopt_core_sm120.ptx \
		lanczos_core_sm80.ptx \
		lanczos_core_sm86.ptx \
		lanczos_core_sm89.ptx \
		lanczos_core_sm90.ptx \
		lanczos_core_sm100.ptx \
		lanczos_core_sm120.ptx \
		sieve_core_sm80.ptx \
		sieve_core_sm86.ptx \
		sieve_core_sm89.ptx \
		sieve_core_sm90.ptx \
		sieve_core_sm100.ptx \
		sieve_core_sm120.ptx \
		lasieve_core_sm80.ptx \
		lasieve_core_sm86.ptx \
		lasieve_core_sm89.ptx \
		lasieve_core_sm90.ptx \
		lasieve_core_sm100.ptx \
		lasieve_core_sm120.ptx \
		sqrtmul_core_sm80.ptx \
		sqrtmul_core_sm86.ptx \
		sqrtmul_core_sm89.ptx \
		sqrtmul_core_sm90.ptx \
		sqrtmul_core_sm100.ptx \
		sqrtmul_core_sm120.ptx \
		cub/built
	# CUDA 12.8 and later also build for sm_80 ... sm_120: the loaders ask
	# for those files on compute 8.0 and newer, and without them stage 1
	# stopped on a modern card
	GPU_SM = 600,610,700,750,800,860,890,900,1000,1200
endif
ifeq ($(CUDA),13)
	GPU_OBJS = \
		stage1_core_sm80.ptx \
		stage1_core_sm86.ptx \
		stage1_core_sm89.ptx \
		stage1_core_sm90.ptx \
		stage1_core_sm100.ptx \
		stage1_core_sm120.ptx \
		sizeopt_core_sm80.ptx \
		sizeopt_core_sm86.ptx \
		sizeopt_core_sm89.ptx \
		sizeopt_core_sm90.ptx \
		sizeopt_core_sm100.ptx \
		sizeopt_core_sm120.ptx \
		lanczos_core_sm80.ptx \
		lanczos_core_sm86.ptx \
		lanczos_core_sm89.ptx \
		lanczos_core_sm90.ptx \
		lanczos_core_sm100.ptx \
		lanczos_core_sm120.ptx \
		sieve_core_sm80.ptx \
		lasieve_core_sm80.ptx \
		sqrtmul_core_sm80.ptx \
		sieve_core_sm86.ptx \
		lasieve_core_sm86.ptx \
		sqrtmul_core_sm86.ptx \
		sieve_core_sm89.ptx \
		lasieve_core_sm89.ptx \
		sqrtmul_core_sm89.ptx \
		sieve_core_sm90.ptx \
		lasieve_core_sm90.ptx \
		sqrtmul_core_sm90.ptx \
		sieve_core_sm100.ptx \
		lasieve_core_sm100.ptx \
		sqrtmul_core_sm100.ptx \
		sieve_core_sm120.ptx \
		lasieve_core_sm120.ptx \
		sqrtmul_core_sm120.ptx \
		cub/built
	GPU_SM = 800,860,890,900,1000,1200
endif

endif # ---------------------------------------------------------------------------

ifeq ($(ECM),1)
	CFLAGS += -DHAVE_GMP_ECM
	LIBS += -lecm
endif
ifeq ($(WIN),1)

else
	LIBS += -ldl
endif
ifeq ($(MPI),1)
	CC = mpicc
	CFLAGS += -DHAVE_MPI
endif
# NCCL=1 (with CUDA and MPI): the GPU linear algebra's grid operations
# can use NCCL (MSIEVE_LA_NCCL=1 at run time). NCCL_HOME: its include/
# and lib/ (default: the system's)
ifeq ($(NCCL),1)
	CFLAGS += -DHAVE_NCCL
ifneq ($(NCCL_HOME),)
	CFLAGS += -I$(NCCL_HOME)/include
	LIBS += -L$(NCCL_HOME)/lib -Wl,-rpath,$(NCCL_HOME)/lib
endif
	LIBS += -lnccl
endif
ifeq ($(BOINC),1)
	# fill in as appropriate
	BOINC_INC_DIR = .
	BOINC_LIB_DIR = .
	CFLAGS += -I$(BOINC_INC_DIR) -DHAVE_BOINC
	LIBS += -L$(BOINC_LIB_DIR) -lboinc_api -lboinc
endif
ifeq ($(NO_ZLIB),1)
	CFLAGS += -DNO_ZLIB
else
	LIBS += -lz
endif


# Note to MinGW users: the library does not use pthreads calls in
# win32 or win64, so it's safe to pull libpthread into the link line.
# Of course this does mean you have to install the minGW pthreads bundle...

LIBS += -lgmp -lm -lpthread

#---------------------------------- Generic file lists -------------------

COMMON_HDR = \
	aprcl/mpz_aprcl32.h \
	common/lanczos/lanczos.h \
	common/filter/filter.h \
	common/filter/filter_priv.h \
	common/filter/merge_util.h \
	include/batch_factor.h \
	include/common.h \
	include/cuda_xface.h \
	include/dd.h \
	include/ddcomplex.h \
	include/gmp_xface.h \
	include/integrate.h \
	include/msieve.h \
	include/mp.h \
	include/polyroot.h \
	include/thread.h \
	include/util.h

COMMON_GPU_HDR = \
	common/lanczos/gpu/lanczos_gpu.h \
	common/lanczos/cpu/lanczos_cpu.h

COMMON_NOGPU_HDR = \
	common/lanczos/cpu/lanczos_cpu.h

COMMON_SRCS = \
	aprcl/mpz_aprcl32.c \
	common/filter/clique.c \
	common/filter/filter.c \
	common/filter/merge.c \
	common/filter/merge_post.c \
	common/filter/merge_pre.c \
	common/filter/merge_util.c \
	common/filter/singleton.c \
	common/lanczos/lanczos.c \
	common/lanczos/lanczos_io.c \
	common/lanczos/lanczos_matmul.c \
	common/lanczos/lanczos_pre.c \
	common/lanczos/matmul_util.c \
	common/lanczos/cpu/lanczos_matmul0.c \
	common/lanczos/cpu/lanczos_matmul1.c \
	common/lanczos/cpu/lanczos_matmul2.c \
	common/lanczos/cpu/lanczos_vv.c \
	common/smallfact/gmp_ecm.c \
	common/smallfact/smallfact.c \
	common/smallfact/squfof.c \
	common/smallfact/tinyqs.c \
	common/batch_factor.c \
	common/cuda_xface.c \
	common/dickman.c \
	common/driver.c \
	common/expr_eval.c \
	common/hashtable.c \
	common/integrate.c \
	common/minimize.c \
	common/mp.c \
	common/polyroot.c \
	common/prime_delta.c \
	common/prime_sieve.c \
	common/savefile.c \
	common/strtoll.c \
	common/thread.c \
	common/util.c

# GPU builds: the block Lanczos iteration on the GPU too, chosen per
# matrix by lanczos_dispatch.c (kernels: lanczos_core_sm*.ptx)
COMMON_GPU_SRCS = \
	common/lanczos/gpu/lanczos_gpu.c \
	common/lanczos/gpu/lanczos_ring.c \
	common/lanczos/lanczos_dispatch.c

COMMON_NOGPU_SRCS =

ifneq ($(CUDA),)
	COMMON_SRCS += $(COMMON_GPU_SRCS)
	COMMON_HDR += $(COMMON_GPU_HDR)
else
	COMMON_SRCS += $(COMMON_NOGPU_SRCS)
	COMMON_HDR += $(COMMON_NOGPU_HDR)
endif

COMMON_OBJS = $(COMMON_SRCS:.c=.o)
COMMON_GPU_OBJS = $(COMMON_GPU_SRCS:.c=.o)
COMMON_NOGPU_OBJS = $(COMMON_NOGPU_SRCS:.c=.o)

#---------------------------------- QS file lists -------------------------

QS_HDR = mpqs/mpqs.h

QS_SRCS = \
	mpqs/gf2.c \
	mpqs/mpqs.c \
	mpqs/poly.c \
	mpqs/relation.c \
	mpqs/sieve.c \
	mpqs/sieve_core.c \
	mpqs/sqrt.c

QS_OBJS = \
	mpqs/gf2.qo \
	mpqs/mpqs.qo \
	mpqs/poly.qo \
	mpqs/relation.qo \
	mpqs/sieve.qo \
	mpqs/sqrt.qo \
	mpqs/sieve_core_generic_32k.qo \
	mpqs/sieve_core_generic_64k.qo

#---------------------------------- NFS file lists -------------------------

NFS_HDR = \
	gnfs/filter/filter.h \
	gnfs/filter/relpar.h \
	gnfs/relstore.h \
	gnfs/poly/poly.h \
	gnfs/poly/poly_skew.h \
	gnfs/poly/stage1/stage1.h \
	gnfs/poly/stage2/stage2.h \
	gnfs/poly/stage2/line_sieve_core.h \
	gnfs/poly/stage2/murphy_core.h \
	gnfs/poly/stage2/alpha_core.h \
	gnfs/poly/stage2/lattice_core.h \
	gnfs/sieve/sieve.h \
	gnfs/sieve/sieve_line_int.h \
	gnfs/sieve/sieve_core.h \
	gnfs/sieve/lasieve_core.h \
	gnfs/sqrt/sqrt.h \
	gnfs/gnfs.h

NFS_GPU_HDR = \
	gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu \
	gnfs/poly/stage1/stage1_core_gpu/cuda_intrinsics.h \
	gnfs/poly/stage1/stage1_core_gpu/problem_setup.h \
	gnfs/poly/stage1/stage1_core_gpu/stage1_core.h \
	gnfs/poly/stage1/stage1_core_gpu/transpose.h \
	gnfs/poly/stage1/stage1_core_gpu/sizeopt_core.h \
	gnfs/poly/stage1/stage1_sizeopt_gpu.h \
	gnfs/poly/stage2/root_sieve_gpu.h \
	gnfs/sqrt/sqrt_gpu_mul.h \
	gnfs/sqrt/sqrtmul_core.h

NFS_NOGPU_HDR = \
	gnfs/poly/stage1/cpu_intrinsics.h

NFS_SRCS = \
	gnfs/poly/poly.c \
	gnfs/poly/poly_param.c \
	gnfs/poly/poly_skew.c \
	gnfs/poly/polyutil.c \
	gnfs/poly/root_score.c \
	gnfs/poly/size_score.c \
	gnfs/poly/stage1/stage1.c \
	gnfs/poly/stage1/stage1_roots.c \
	gnfs/poly/stage2/optimize.c \
	gnfs/poly/stage2/optimize_deg6.c \
	gnfs/poly/stage2/root_sieve.c \
	gnfs/poly/stage2/root_sieve_deg45_x.c \
	gnfs/poly/stage2/root_sieve_deg5_xy.c \
	gnfs/poly/stage2/root_sieve_deg6_x.c \
	gnfs/poly/stage2/root_sieve_deg6_xy.c \
	gnfs/poly/stage2/root_sieve_deg6_xyz.c \
	gnfs/poly/stage2/root_sieve_line.c \
	gnfs/poly/stage2/root_sieve_batch.c \
	gnfs/poly/stage2/root_sieve_util.c \
	gnfs/poly/stage2/stage2.c \
	gnfs/filter/duplicate.c \
	gnfs/filter/filter.c \
	gnfs/filter/singleton.c \
	gnfs/filter/relpar.c \
	gnfs/relstore.c \
	gnfs/sieve/sieve_line.c \
	gnfs/sieve/sieve_util.c \
	gnfs/sqrt/sqrt.c \
	gnfs/sqrt/sqrt_a.c \
	gnfs/fb.c \
	gnfs/ffpoly.c \
	gnfs/gf2.c \
	gnfs/gnfs.c \
	gnfs/relation.c

NFS_OBJS = $(NFS_SRCS:.c=.no)

NFS_GPU_SRCS = \
	gnfs/poly/stage1/stage1_sieve_gpu.c \
	gnfs/poly/stage1/stage1_sizeopt_gpu.c \
	gnfs/poly/stage2/root_sieve_gpu.c \
	gnfs/sieve/sieve_line_gpu.c \
	gnfs/sieve/sieve_lattice_gpu.c \
	gnfs/sqrt/sqrt_gpu_mul.c

NFS_GPU_OBJS = $(NFS_GPU_SRCS:.c=.no)

NFS_NOGPU_SRCS = \
	gnfs/poly/stage1/stage1_sieve_cpu.c

NFS_NOGPU_OBJS = $(NFS_NOGPU_SRCS:.c=.no)

ifneq ($(CUDA),)
	NFS_HDR += $(NFS_GPU_HDR)
	NFS_SRCS += $(NFS_GPU_SRCS)
	NFS_OBJS += $(NFS_GPU_OBJS)
else
	NFS_HDR += $(NFS_NOGPU_HDR)
	NFS_SRCS += $(NFS_NOGPU_SRCS)
	NFS_OBJS += $(NFS_NOGPU_OBJS)
	GPU_OBJS =
endif

#---------------------------------- make targets -------------------------

help:
	@echo "to build:"
	@echo "make all"
	@echo "add 'WIN=1 if building on windows"
	@echo "add 'WIN64=1 if building on 64-bit windows"
	@echo "add 'ECM=1' if GMP-ECM is available (enables ECM)"
	@echo "add 'CUDA=X' for GPU support (X = CUDA version, 12 or 13)"
	@echo "add 'MPI=1' for parallel processing using MPI"
	@echo "add 'BOINC=1' to add BOINC wrapper"
	@echo "add 'NO_ZLIB=1' if you don't have zlib"
	@echo "add 'VBITS=X' for linear algebra with X-bit vectors (64, 128, 256)"

all: $(COMMON_OBJS) $(QS_OBJS) $(NFS_OBJS) $(GPU_OBJS)
	rm -f libmsieve.a
	ar r libmsieve.a $(COMMON_OBJS) $(QS_OBJS) $(NFS_OBJS)
	ranlib libmsieve.a
	$(CC) $(CFLAGS) demo.c -o msieve $(LDFLAGS) \
			libmsieve.a $(LIBS)

clean:
	cd cub && make clean WIN=$(WIN) WIN64=$(WIN64) && cd ..
	rm -f msieve msieve.exe libmsieve.a $(COMMON_OBJS) $(QS_OBJS) \
		$(NFS_OBJS) $(NFS_GPU_OBJS) $(NFS_NOGPU_OBJS) *.ptx \
		$(COMMON_GPU_OBJS) $(COMMON_NOGPU_OBJS)

#----------------------------------------- build rules ----------------------

# common file build rules

%.o: %.c $(COMMON_HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# QS build rules

mpqs/sieve_core_generic_32k.qo: mpqs/sieve_core.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -DBLOCK_KB=32 -DHAS_SSE2 \
		-DROUTINE_NAME=qs_core_sieve_generic_32k \
		-c -o $@ mpqs/sieve_core.c

mpqs/sieve_core_generic_64k.qo: mpqs/sieve_core.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -DBLOCK_KB=64 -DHAS_SSE2 \
		-DROUTINE_NAME=qs_core_sieve_generic_64k \
		-c -o $@ mpqs/sieve_core.c

%.qo: %.c $(COMMON_HDR) $(QS_HDR)
	$(CC) $(CFLAGS) -c -o $@ $<

# NFS build rules

%.no: %.c $(COMMON_HDR) $(NFS_HDR)
	$(CC) $(CFLAGS) -Ignfs -c -o $@ $<

# GPU build rules

stage1_core_sm60.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_60 -Wno-deprecated-gpu-targets -O3 -lineinfo -ptx -restrict -o $@ $<

stage1_core_sm61.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_61 -o $@ $<

stage1_core_sm70.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_70 -o $@ $<

stage1_core_sm75.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_75 -o $@ $<

stage1_core_sm80.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_80 -o $@ $<

stage1_core_sm86.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_86 -o $@ $<

stage1_core_sm89.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_89 -o $@ $<

stage1_core_sm90.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_90 -o $@ $<

stage1_core_sm100.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_100 -o $@ $<

stage1_core_sm120.ptx: $(NFS_GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) -arch sm_120 -o $@ $<

# GPU size optimization. Needs CGBN (CGBN_INCLUDE, default
# /usr/local/include). -fmad=false: the norm minimization must round
# like the CPU code, which is built without floating-point contraction
SIZEOPT_SRC = gnfs/poly/stage1/stage1_core_gpu/sizeopt_core.cu
SIZEOPT_HDR = $(SIZEOPT_SRC) \
	gnfs/poly/stage1/stage1_core_gpu/sizeopt_core.h \
	gnfs/poly/stage2/line_sieve_core.h \
	gnfs/poly/stage2/murphy_core.h \
	gnfs/poly/stage2/alpha_core.h \
	gnfs/poly/stage2/lattice_core.h \
	gnfs/poly/stage1/stage1_core_gpu/stage1_core.h \
	gnfs/poly/stage1/stage1_core_gpu/cuda_intrinsics.h
CGBN_INCLUDE ?= /usr/local/include
SIZEOPT_FLAGS = $(NVCC_FLAGS) -fmad=false -I$(CGBN_INCLUDE)

sizeopt_core_sm%.ptx: $(SIZEOPT_HDR)
	$(NVCC) $(SIZEOPT_FLAGS) -arch sm_$* -o $@ $(SIZEOPT_SRC)

# the GPU line siever's kernels
SIEVE_SRC = gnfs/sieve/sieve_core.cu

sieve_core_sm%.ptx: $(SIEVE_SRC) gnfs/sieve/sieve_core.h
	$(NVCC) $(NVCC_FLAGS) -Ignfs/sieve -arch sm_$* -o $@ $(SIEVE_SRC)

# the NFS square root's large multiplications
SQRTMUL_SRC = gnfs/sqrt/sqrtmul_core.cu

sqrtmul_core_sm%.ptx: $(SQRTMUL_SRC) gnfs/sqrt/sqrtmul_core.h
	$(NVCC) $(NVCC_FLAGS) -Ignfs/sqrt -arch sm_$* -o $@ $(SQRTMUL_SRC)

LASIEVE_SRC = gnfs/sieve/lasieve_core.cu

lasieve_core_sm%.ptx: $(LASIEVE_SRC) gnfs/sieve/lasieve_core.h
	$(NVCC) $(NVCC_FLAGS) -Ignfs/sieve -I$(CGBN_INCLUDE) -arch sm_$* -o $@ $(LASIEVE_SRC)

# the GPU linear algebra kernels, for this build's VBITS
LANCZOS_SRC = common/lanczos/gpu/lanczos_core.cu

lanczos_core_sm%.ptx: $(LANCZOS_SRC)
	$(NVCC) $(NVCC_FLAGS) -DVBITS=$(VBITS) -arch sm_$* -o $@ $(LANCZOS_SRC)

cub/built:
	cd cub && make WIN=$(WIN) WIN64=$(WIN64) sm=$(GPU_SM) && cd ..
