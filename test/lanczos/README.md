# Lanczos tests (GPU and CPU backends)

There is no NFS relation data here, so these tests use synthetic matrices with an NFS-like row
profile and check every dependency independently.

- `matgen <name> <nrows> <excess cols> <weight> <skew> <seed>` writes `<name>.mat` (and, linked
  against an MPI `libmsieve.a`, `<name>.mat.idx`) with `dump_matrix()`.
- `run.sh <matrix> <name> <env...>` runs `msieve -nc2 skip_matbuild=1` on it (any `.fb` works: the
  linear algebra does not use the polynomial), then `depcheck`. `MPIRUN`, `BIN`, `NCARGS`,
  `THREADS` select an MPI run, e.g. `MPIRUN="mpirun --bind-to none -np 4" BIN=_mpi
  NCARGS="mpi_nrows=2 mpi_ncols=2"`.
- `depcheck <name>` checks that every dependency in `<name>.dep` combines the columns of
  `<name>.mat` into zero.
- `optest <nrows> <dense rows> [ncols]` compares every GPU backend operation with the CPU
  backend's and with a host reference.

Build (from the msieve directory, after `make all CUDA=13`):

    gcc -O2 -Iinclude -o matgen test/lanczos/matgen.c libmsieve.a -lgmp -lz -lm -lpthread -ldl \
        /usr/local/cuda/lib64/stubs/libcuda.so
    gcc -O2 -o depcheck test/lanczos/depcheck.c
    gcc -O2 -DVBITS=64 -DHAVE_CUDA -I. -Iinclude -I/usr/local/cuda/include -o optest \
        test/lanczos/optest.c libmsieve.a -lgmp -lz -lm -lpthread -ldl \
        /usr/local/cuda/lib64/stubs/libcuda.so
