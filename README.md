# msieve with GPU polynomial selection (sm_120, CGBN)

A fork of [msieve](https://sourceforge.net/projects/msieve/) (SVN trunk r1113) that makes its NFS
polynomial selection fast on current NVIDIA GPUs: the stage 1 collision search is fixed and sped up
**115×** (on one leading coefficient, compared with the CPU-only build), stage 2 runs on every CPU,
**size optimization runs on the GPU**, with its integer arithmetic in
[CGBN](https://github.com/steleman/nvidia-cgbn) and results bit-identical to the CPU code, and
**root optimization runs on the GPU** too, as a batched version of msieve's root sieve that finds
the same polynomials (1.91× faster than 32 CPU threads at 155 digits), with its root properties
computed on the GPU in CGBN and its Murphy E optimization on the GPU on data-centre cards.
The **linear algebra** (block Lanczos) and msieve's **line siever** run on the GPU as well, the
latter with exactly the CPU siever's relations (18× faster on a c95), and a new **GPU lattice
siever** (msieve has none), the default for `-ns` in GPU builds, sieves that c95 in 43.7 s against
9.2 hours for the GPU line siever; `./msieve -n` factors it in 1 min 8 s from its factor base, and
plain `./msieve` (the NFS by default in GPU builds) in 2 min 43 s from scratch, against 47 min with
msieve's quadratic sieve. Everything else (MPQS,
filtering, square root) is msieve's own, apart from the build changes below. The original documentation is in `Readme`, `Readme.nfs` and `Readme.qs`;
the design and the measurements are in **`msieve-gpu-design.pdf`** (source `msieve-gpu-design.tex`).

Measured on an NVIDIA RTX PRO 4500 Blackwell (sm_120, 32 GB, CUDA 13.2) with an Intel i9-13900K
(32 threads), on a 120-digit semiprime unless noted.

## What changed

| Area | Change | Effect |
| --- | --- | --- |
| Lost hits | the GPU found array held 1000 collisions per batch and dropped every one past 999, silently; now 2²⁰, with a warning on overflow | all hits kept (as shipped: 38,961 = 39 × 999 of 2.59 M) |
| sm_120 | `stage1_core_sm120.ptx`, loaded for compute 12.x; `cub/` built for it | Blackwell gets its own code |
| Root arithmetic (stage 1) | 64-bit and Montgomery root lifting, incremental Garner CRT, special-q roots lifted on the GPU | see the stage 1 table |
| Special-q producers | 4 threads enumerate special-q and fill a ring of batches the GPU drains (`MSIEVE_Q_THREADS`) | stage 1 no longer waits on the CPU |
| Stage 2 threads | size and root optimization on one thread per CPU, each with its own state (`MSIEVE_STAGE2_THREADS`) | 809.5 s → 35.6 s on one coefficient |
| GPU size optimization | collisions are rebuilt, size-optimized and scored on the GPU; only the ~1 in 4000 that pass the size bound come back (`MSIEVE_GPU_SIZEOPT`) | faster than 32 CPU threads, 1 CPU core used |
| Uninitialized `x_base` | `sieve_x_t.x_base` (root sieve) was never `mpz_init`'ed; GMP coped until its limbs were read | initialized |
| Dropped work at exit | the stage 2 pool was freed with work queued, losing the last hits; it is drained first | the last coefficient's polynomials arrive |
| Batched root sieve | msieve's root sieve reorganized so each polynomial has hundreds of independent `optimize_basic()` calls and line sieves at a time: all norm bounds' line sizes at once, the x-line sizes of the y-blocks in three rounds started from interpolated neighbours, lines and candidates scored together | a few hundred GPU round trips per polynomial instead of 47,000 |
| GPU root optimization | the batched root sieve runs in coroutines on CPU threads; its `optimize_basic()` calls and line sieves go to the GPU in batches, and an adaptive share of the calls stays on the CPU (`MSIEVE_GPU_ROOTOPT`) | 155 digits: 195.6 s → 107.0 s, same polynomials |
| Root sieve setup on the GPU | `init_sieve()`: coefficients, P and d reduced mod the 52 prime powers below 1000 in CGBN, the ~6,400 roots listed in the CPU's order and kept on the GPU for the line sieve, one copy to the host (`MSIEVE_GPU_RSSETUP`) | bit-identical; no roots sent with line batches; c155 ~5% faster |
| Lattice choice on the GPU (default; `GPU_LATTICE=0` for the CPU) | the xy and x lattice sieves and the x-line heap on the GPU, on the resident roots, which then never go to the host | bit-identical; 0.1% of the CPU time, ~4% slower on the RTX PRO 4500 (one more round trip per choice); the default because the target is an RTX PRO 6000 Blackwell or larger |
| Kept polynomials in CGBN | rotation and translation of the polynomials kept (`rf_final`), their Murphy E inputs from the size kernel, rotations as int64 through the heap: no GMP between the GPU steps | bit-identical; same speed here (111.2 vs 110.4 s, noise), 0.5 GB less memory |
| Per-run lattice numbers without GMP | degrees 4 and 5: y_base and the lattice factors as 64-bit words, the 128-bit lattice size formed by `ls_sieve` in CGBN, `mpz` only for CPU fallbacks | bit-identical (also the CPU fallbacks and `GPU_LATTICE=0`) |
| Degree 6 lattices without GMP | plane and x-lattice constants as 64/128-bit words; its x-lines go to the GPU as parts, numbers formed by `ls_sieve` in CGBN (before, most did not fit 64-bit coordinates and were sieved on the CPU: 55,802 lines on the GPU against 1,435); the batched CPU sieve takes parts lines too (numbers with GMP); GMP is left for the CPU fallbacks and the callback | GPU = batched CPU on 16 degree-6 RSA-1024 polynomials, and degrees 4/5 unchanged |
| Lattice CRT on the GPU | each line's coordinates (the lattice CRT, x_base, y) formed in CGBN from its parts, and its candidates made there, instead of with GMP on the host; their `optimize_basic()` on the GPU too (`MSIEVE_GPU_LINECAND`, on by default) | bit-identical; same speed here; sieve kernel 128 → 64 registers |
| Root properties on the GPU | the alpha of every rotation in a root heap (~200 per polynomial): rotated coefficients formed and reduced mod the 303 primes below 2000 in CGBN, root counting mod p and lifting mod p^k ported from `ffpoly.c` / `root_score.c` (`MSIEVE_GPU_ALPHA`) | bit-identical to the CPU; c155: 107.8 → 102.6 s |
| Bernstein filter on the GPU | the size score of every rotation that passes alpha: the coefficients as double-double straight from CGBN, roots by a port of msieve's Jenkins-Traub root finder, the integral spread over a warp (`MSIEVE_GPU_BERNSTEIN`) | within 1 ulp of the CPU (48% bit-identical), the same filter decisions; host GMP only for the ~9 of ~200 rotations per polynomial that pass |
| Murphy E on the GPU | the Murphy E optimization of each root heap's candidates, one warp per candidate, the integral's points spread over the lanes (`MSIEVE_GPU_MURPHY`) | E within 6e-9 of the CPU's; kernel alone 1.2× 32 CPU threads here, on by default (for the target hardware; on this card `MSIEVE_GPU_MURPHY=0` is faster); then the Bernstein kernel writes the survivors' inputs on the GPU and nothing goes through the host |
| C11 | builds as `-std=c11` with `-Wpedantic` and prototype warnings, zero warnings (CUDA, CPU-only and ECM builds) | also fixes a use-after-`realloc` in the CPU stage 1 |

**Stage 1**, one leading coefficient, every variant producing the same 2,588,149 distinct hits
(`Q_ROOTS_MIN=1`, the full search):

| Build | Time |
| --- | ---: |
| CPU-only msieve (one thread) | 778.6 s |
| GPU, hit cap fixed, full search | 65.8 s |
| + 64-bit, then Montgomery root lifting | 36.4 → 30.0 s |
| + root lifting on the GPU | 18.5 s |
| + incremental CRT | 12.5 s |
| + 4 special-q producer threads | 8.3 s |
| + hits echoed only with `-v` | **6.8 s** |

**Size optimization**, same leading coefficients, CPU path with 32 threads against the GPU path;
the size-optimized polynomials are identical:

| Input | 32 CPU threads | GPU (+1 core) | Polynomials passing |
| --- | ---: | ---: | --- |
| 120 digits, one coefficient (1.78 M collisions) | 20.5 s | **15.6 s** | 235 = 235 |
| 140 digits, one coefficient (0.90 M, 1024-bit) | 14.8 s | **12.1 s** | 155 = 155 |
| 95 digits, degree 4, coefficients 1000–3000 | 1.6 s | **1.5 s** | 2,064 = 2,064 |

The GPU kernel is limited by double-precision throughput (this GPU: 332 GFLOP/s without FMA,
1/64 of single precision); a data-centre GPU with full-rate FP64 would be several times faster.

**End to end** (stage 1 + size + root optimization, 30 minutes each, every root-optimized score
recorded, before the GPU size optimization): the fixed builds found 2–3× as many size-optimized
candidates as the unmodified program and the best Murphy E (2.939e-10 with msieve's default
search, 2.872e-10 with the full search, 2.786e-10 unmodified). msieve's default (special-q with at
least *degree* roots) is kept.

**Root optimization**, all 6,273 polynomials of one 155-digit leading coefficient (default E bound),
msieve's code on 32 CPU threads against the GPU engine:

| | Time | Saved (E above the bound) | Best E |
| --- | ---: | ---: | ---: |
| msieve, 32 CPU threads | 195.6 s | 49 | 2.510e-12 |
| GPU, `optimize_basic()` calls only | 169.6 s | 50 (the 49 + 1) | 2.510e-12 |
| + line sieve on the GPU, buffers, fair job shares, adaptive CPU share | 107.0 s | 50 (the 49 + 1) | 2.510e-12 |
| + root properties on the GPU (CGBN) | **102.6 s** | 50 (the 49 + 1) | 2.510e-12 |
| (+ Murphy E on the GPU, this card) | 170–183 s | 50 | 2.510e-12 |
| (step 1: msieve's root sieve unchanged, calls one at a time; 64 polynomials) | 449.6 s vs 11.6 s | identical | |

msieve's root sieve makes ~47,000 `optimize_basic()` calls per polynomial *one after the other*
(each line-size search starts from the previous line's result), and one call takes ~5 ms on a GPU
thread against ~20 µs on a CPU core, so running it unchanged on the GPU was 39× slower. The batched
root sieve runs the same searches with different starting points, so a polynomial asks for
~300 calls at a time; at full load the GPU then does 3.3 M calls/s (this card; 32 CPU threads:
3.0 M/s), bit-identical to the CPU. What changes: a line's ends can move within the search
tolerance, and ties between equal sieve scores go to the lower position. Every scored polynomial
recorded (`min_evalue=1e-12`): the best E is the original's at 95, 120, 140 and 155 digits and
97–100% of the scored polynomials are identical. On this card the GPU is now the limit (its
double precision is 1/64 of single), which is why Murphy E is faster on the CPU here: its kernel
alone does 1,512 candidates/s against 1,263 for 32 CPU threads, but adding it to a saturated GPU
slows the whole run. It is on by default anyway, as are the line candidates' `optimize_basic()` on
the GPU: the defaults are for the target hardware (an RTX PRO 6000 Blackwell or larger), not this
card. They used to follow the FP64 ratio (on only at ≥ 1/4 of single precision), but the RTX PRO
6000 Blackwell is 1/64 too, so the ratio would have kept them off there. On a small card set
`MSIEVE_GPU_MURPHY=0 MSIEVE_GPU_LINECAND=0`: on this one a c155 coefficient takes 168 s with the
defaults (5.6 GB, 207% CPU) against ~111 s with both off (8.3 GB, ~1750% CPU), same results. With few polynomials in flight (one 120-digit coefficient
is 235) GPU and CPU take about the same time. Degree 6 (above ~200 digits) batches everything too: its z-blocks' y-line searches in strided
rounds with interpolated starts (as degree 5's x-lines), and the x-line searches of all planes of
a z-lattice at once (`sieve_x_run_deg6_batch`; same starts as the original, so the same results).
Tested on 16 size-optimized degree-6 polynomials of RSA-1024 (309 digits, `polydegree=6`): GPU and
batched CPU identical; against the unbatched searches 3,183 of 3,200 polynomials the same, the top
200 E values equal. GPU round trips 8.0 million → 8,346, time 1,061 s → 417 s (CPU 466 s). Design document §7.

## Build

```sh
export PATH=/usr/local/cuda-13.2/bin:$PATH
make all CUDA=13                 # GPU build: stage 1 and size optimization on the GPU
make all CUDA=13 ECM=1           # with GMP-ECM
make all CUDA=13 GPU_LATTICE=0   # the root sieve's lattice choice on the CPU
make all                         # CPU only
```

Needs GMP, zlib, pthreads, the CUDA toolkit (12 or 13) and, for the GPU size optimization, the
CGBN headers (`CGBN_INCLUDE`, default `/usr/local/include`; the fork at
github.com/steleman/nvidia-cgbn, which adds signed arithmetic). The build writes
`stage1_core_sm*.ptx` and `sizeopt_core_sm*.ptx` (sm_80, 86, 89, 90, 100, 120 with CUDA 13; H100 and
B200 used to get the sm_80 code).
**msieve loads them from the current directory**, so run it where they are, or copy them next to
the data. The first run after a build spends a few seconds while the driver compiles the PTX; it
caches the result.

## Run

```sh
./msieve -v -np "min_coeff=X max_coeff=Y" <N>     # stage 1 + size + root optimization
./msieve -v -np1 -nps <N>                          # stage 1 + size optimization only
./msieve -g 0 ...                                  # choose the GPU
```

The usual options are in `Readme.nfs`. Settings added by this fork (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MSIEVE_STAGE2_THREADS` | online CPUs | threads for size and root optimization |
| `MSIEVE_GPU_SIZEOPT` | on | `0` keeps size optimization on the CPU |
| `MSIEVE_Q_THREADS` | 4 | special-q producer threads; `0` = the original serial loop |
| `MSIEVE_CPU_LIFT` | off | `1` lifts special-q roots on the CPU |
| `MSIEVE_OLD_CRT` | off | `1` uses the original root combination |
| `MSIEVE_SIZEOPT_LOG` | — | file: log every size-optimization survivor and the hit count (measurement) |
| `MSIEVE_GPU_ROOTOPT` | on | `0`: root optimization on the CPU, msieve's own code |
| `MSIEVE_ROOTOPT_THREADS`, `MSIEVE_ROOTOPT_SLOTS` | CPUs, 256 | threads (in all, divided among the GPUs) and polynomials per thread of the GPU root optimization |
| `MSIEVE_SIMULATE_GPUS` | — | testing: n logical GPUs on the physical one(s) |
| `MSIEVE_ROOTOPT_BATCHED` | GPU: on, CPU: off | `0` in the GPU engine: msieve's root sieve (calls one at a time, slow); `1` on the CPU: the batched root sieve there (same results as the GPU) |
| `MSIEVE_GPU_LINESIEVE` | on | `0`: the batched line sieve on the CPU |
| `MSIEVE_GPU_RSSETUP` | on | `0`: the root sieve setup on the CPU |
| `MSIEVE_GPU_LA`, `MSIEVE_GPU_LA_MIN` | on, 100000 | `0`: linear algebra on the CPU; the smallest matrix (columns) put on the GPU |
| `MSIEVE_LA_CUDA_MPI` | on if the MPI is CUDA-aware | `0`: the GPU linear algebra's MPI traffic through host buffers |
| `MSIEVE_LA_NCCL` | off | `1` (`NCCL=1` builds): the grid operations through NCCL (one GPU per process; untested, see above) |
| `MSIEVE_LA_RING_TEST` | off | `1`: the NCCL rings through MPI on host buffers (testing) |
| `MSIEVE_GPU_LINECAND` | on | `1`/`0`: line candidates' `optimize_basic()` on the GPU / with the CPU-shared batches |
| `MSIEVE_GPU_ALPHA` | on | `0`: root properties (alpha) on the CPU |
| `MSIEVE_GPU_BERNSTEIN` | on | `0`: the Bernstein size score on the CPU |
| `MSIEVE_GPU_MURPHY` | on | `1`/`0`: Murphy E optimization on the GPU / CPU |
| `MSIEVE_ROOTOPT_CPU_SHARE` | auto | share of each batch's `optimize_basic()` calls (and Murphy E candidates) done on the CPU; auto adapts it so threads do not wait for the GPU |
| `MSIEVE_ROOTOPT_STATS` | — | set: per-thread statistics of the GPU root optimization |
| `MSIEVE_FILTER_THREADS` | online CPUs | threads that parse relations in the NFS filtering, matrix build and square root |
| `MSIEVE_SQRT_THREADS` | 4 (8 with 32 CPUs or more) | NFS square-root dependencies tried at once |
| `MSIEVE_SQRT_GPU`, `MSIEVE_SQRT_GPU_MIN` | on, 2048 | GPU builds: the square root's products whose operands both have at least this many 64-bit limbs on the GPU (NTT); `0`: GMP only |
| `MSIEVE_SQRT_GPU_CHECK` | off | `1`: every GPU product of the square root compared with GMP's (slow; tests) |
| `MSIEVE_POLY_TIME_SCALE` | 0.0078125 (1/128) | GPU builds: factor on msieve's polynomial selection time limits (`poly_deadline=` overrides; 0.25, then 1/32 until Nov 10) |
| `MSIEVE_ROOTOPT_TOPK` | 48 (GPU builds), 0 | root-optimize a size-optimized polynomial only if its size score is among the K best so far; `0`: all |
| `MSIEVE_SIZEOPT_LOG` | — | `<file>` (measurement): every size optimization survivor (`S`, `P` lines) and every root optimization result (`R c_d Y1 E size alpha`) |
| `MSIEVE_POLY_KEEP_E` | 0.25 | polynomials are kept down to this fraction of msieve's minimum E (the best is used even when none reaches it) |
| `MSIEVE_DEFAULT_NFS`, `MSIEVE_NFS_MIN_DIGITS` | on, 80 | GPU builds: without an NFS option, the NFS for inputs of at least this many digits when a CUDA device is there; `0`: MPQS as before |
| `MSIEVE_RELSTORE`, `MSIEVE_RELSTORE_MB` | on, a quarter of physical memory | `0`: no in-memory copy of the run's relations (everything reads the savefile); its memory limit |

Compile-time: `MACHINE_FLAGS=-DQ_ROOTS_MIN=1` searches special-q with any number of roots (msieve's
GPU default is the degree). Stage 1 hits are printed on the screen only with `-v`.

## Correctness

Every change was checked by comparing outputs, not just timing:

- stage 1: the sorted, distinct hit set of one leading coefficient, against the CPU-only build and
  every earlier GPU variant;
- size optimization: every surviving polynomial (coefficients, alpha, norm), CPU path against GPU
  path, at 95, 120 and 140 digits; and 1, 8, 16 and 31 stage 2 threads against one;
- root optimization: every scored polynomial with 1 and 8 threads; the batched root sieve on the
  CPU against the GPU engine (identical at 95, 120, 140 and 155 digits) and against msieve's code
  (best E, polynomials saved); its line-size searches against `compute_line_size()` (0 differences in
  ~7,000).

Floating-point results are bit-identical only between builds **without fused multiply-adds**:
`-std=c11` turns them off in GCC, and the GPU code is built with `-fmad=false`. GCC's old default
(GNU C, contraction on) changes a few size-optimization results (232 instead of 235 survivors on
the 120-digit coefficient).

## More than one GPU

Polynomial selection uses every GPU of the machine by default; `-g <num>` restricts it to one
(library callers that pass a device number keep one GPU; `MSIEVE_ALL_GPUS` asks for all).
`CUDA_VISIBLE_DEVICES` restricts them as usual.

- **Stage 1** (collision search and the GPU size optimization): thread t runs on GPU t mod n,
  each with its own context, one to four threads per GPU (`-t`; at least one per GPU, four was the
  limit in all). Each GPU's memory budget (30%, or `gpu_mem_mb`) is shared by its threads.
- **Root optimization**: one engine per GPU (context, kernels, tables, threads), all taking
  polynomials from one queue; a polynomial stays on the GPU whose thread took it. The threads
  (`MSIEVE_ROOTOPT_THREADS`, default the online CPUs) are divided among the GPUs. The log gives the
  polynomials per GPU.
- **Testing on one card**: `MSIEVE_SIMULATE_GPUS=<n>` runs n logical GPUs on the physical one(s)
  (lane i on device i mod devices). It exercises every multi-GPU path but cannot show a speedup.
- `gpu_init()` recorded at most 4 devices and wrote past its table with more; it now records up
  to 64.

Results are identical with 1, 2 and 3 simulated GPUs and with `-g 0` (stage 1 and size optimization
survivors at 95, 120 and 140 digits; root optimization at 95–155 digits and degree 6). A speedup
on several real cards is untested (single-GPU machine).

### Several nodes (MPI)

Build with `make all CUDA=13 MPI=1` and run with `mpirun --bind-to none -np <processes> ./msieve
-np ...` (`--bind-to none`, or OpenMPI pins each process's threads to one core). Polynomial
selection then runs on every process:

- **Stage 1**: process r of R searches every R-th of the leading coefficients the a_d sieve
  accepts (k mod R = r), so all processes search the same low, best region and a deadline cuts
  them alike. `-nps` and `-npr` split their input file by line the same way (each process reads
  the whole file, so it must be on every node).
- **GPUs and CPUs**: a node's processes divide its GPUs (local rank r of L takes devices r, r + L,
  ...; with fewer GPUs than processes they share, and stage 1 divides the card's memory budget
  among them) and its CPUs (the default stage 2 and root optimization thread counts divide by L).
  `-g <num>` puts every process of the node on that GPU.
- **Output**: rank 0 writes the usual files, rank r > 0 writes `<name>.m|ms|p.mpiNN` (as the logs,
  `<log>.mpiNN`). At the end rank 0 collects them over MPI (no shared file system needed) into its
  `.m`, `.ms` or `.p`, removes them, takes every process's best polynomial (sent exactly) and
  writes the `.fb`.

Tested on this machine with 1–4 processes: `-np` on a c95 coefficient range (stage 1 and size
optimization survivors, `.p` and `.fb` identical to a run without MPI), also with `-g 0` and with
2 simulated GPUs per process; `-npr` at 120 and 155 digits (identical `.p` and `.fb`); `-np1` and
`-nps` (identical `.m` and `.ms`). Several real nodes are untested.

### Analysis (Nov 2026)

SLI does not matter for this: CUDA sees each card as its own device, SLI is a
graphics feature, and NVLink (peer-to-peer copies between cards) only helps work that exchanges
data between GPUs.

| Part | Before (one GPU) | Across GPUs |
| --- | --- | --- |
| Stage 1 (collision search) | up to 4 threads, each with its own context, all on device `-g` | independent work per leading coefficient: threads spread over the devices (thread t on device t mod n, up to 4 per device); near-linear, no traffic between GPUs |
| Size optimization | on stage 1's contexts | follows stage 1 |
| Root optimization | one engine, one context, 32 threads × 256 coroutine slots, one shared queue | independent polynomials whose data (roots, sieve tables) stay on their GPU: one engine per device behind the same queue (`ro_gpu_submit` / `ro_gpu_finish` stay as they are); near-linear |
| Several nodes | linear algebra has MPI (CPU) | polynomial selection splits by leading-coefficient range per rank, then per GPU as above |
| Linear algebra | CPU (Block Lanczos, MPI) | done: GPU backend, one MPI process per GPU on msieve's grid, CUDA-aware MPI where available (see "Linear algebra on the GPU") |

One polynomial is not worth splitting over GPUs: its work is chains of dependent steps whose
batches are small already, and a split would add transfers between the cards. Testing needs no
second card: the engines can map n logical GPUs onto the device(s) as `i % ndev` (as
`PRIMEFACTORS_SIMULATE_GPUS` does in the primefactors programs), which runs every multi-GPU path
but shows no speedup.

## Linear algebra on the GPU

GPU builds run the block Lanczos iteration (`-nc2`, and QS's matrix step) on the GPU: the matrix
and every Lanczos vector stay in device memory. The matrix is stored twice, by rows for A·x and by
columns for Aᵀ·x, with a warp per row or column, and its dense rows as bitmaps; the 64×64 block
products use nibble tables in shared memory (`common/lanczos/gpu/lanczos_core.cu`). The CPU backend
stays: `lanczos_dispatch.c` chooses per matrix, the GPU unless `MSIEVE_GPU_LA=0`, the matrix has
fewer than `MSIEVE_GPU_LA_MIN` columns (default 100,000), or it does not fit in the GPU's free
memory (about 8 bytes per nonzero and 72 per column).

cuSPARSE was measured first and is no replacement: it has no GF(2) arithmetic. Its two workable
routes, int8 SpMM with parity and the deprecated custom-operator `cusparseSpMMOp` (XOR on `double`
bit patterns), were 8-12x slower than the custom XOR kernel on a 4M x 4M, 240M-nonzero matrix
(29.5-39.6 ms against 3.4-3.8 ms per product).

**Several GPUs and nodes**: one MPI process per GPU, on msieve's existing MPI grid
(`mpirun --bind-to none -np <P> ./msieve -nc2 "mpi_nrows=<r> mpi_ncols=<c>"`); each process runs its
block of the matrix on its GPU (a node's processes divide its GPUs). With a CUDA-aware MPI the
per-iteration grid operations pass the device vectors to MPI directly (UCX moves them, over NVLink
where there is one), their local XORs run on the GPU, and the GPU is synchronized before each MPI
call; otherwise, or with `MSIEVE_LA_CUDA_MPI=0`, they go through pinned host buffers. The log says
which. The rare result gathers and checkpoint scatters always go through host copies: with HPC-X,
a gather mixing the root's device piece with a host result gave wrong data.

**NCCL** (build option `NCCL=1`, with `CUDA` and `MPI`; `NCCL_HOME` for a non-system NCCL; at run
time `MSIEVE_LA_NCCL=1`, off by default): the per-iteration grid operations go through NCCL on the
device vectors, ordered on one stream with no host synchronization. NCCL has no XOR reduction, so
the all-gather is `ncclAllGather` (plus a broadcast of the last chunk's remainder) and the XOR
reductions are rings of `ncclSend`/`ncclRecv` with the GPU XOR kernel (`lanczos_ring.c`). NCCL needs
one GPU per process: the processes of each ring first compare their GPUs' UUIDs over MPI and all
fall back to MPI if any GPU is shared (a failing `ncclCommInitRank` left the others waiting).
**Untested with NCCL itself**: this machine has one GPU, NCCL refuses two processes on it, and the
card has no MIG. What is tested: the ring algorithms through an MPI transport on host buffers
(`MSIEVE_LA_RING_TEST=1`; grids 1×2, 2×1, 2×2, 1×3, 3×1 and 2×3, every dependency correct), the
fallback (1×2 and 2×2 correct, with Fedora's MPI + NCCL 2.30 and with HPC-X + the HPC SDK's NCCL
2.29), and every build.

Building against a CUDA-aware MPI, e.g. HPC-X from the NVIDIA HPC SDK (Fedora's OpenMPI is not
CUDA-aware):

    source /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/hpcx-init.sh
    hpcx_load
    OMPI_CC=gcc make all CUDA=13 MPI=1    # HPC-X's mpicc would use nvc

Measured with all processes on this one card (so the gain is only from avoiding host copies;
separate GPUs with NVLink are the intended case): 600k matrix, 1×2 grid, 36.4 s CUDA-aware
against 40.6 s through host buffers; on the 200k matrix the messages are too small for it (11.2
against 8.4 s). Every grid (1×2, 2×1, 2×2, 1×3) correct in both modes.

Synthetic NFS-like matrices (60 nonzeros per column, rows skewed to the low indices), RTX PRO 4500,
every dependency checked (`test/lanczos`):

| Matrix | GPU | CPU, 16 threads |
| --- | ---: | ---: |
| 200k × 200k, 12M nonzeros | 3.6 s | 56.4 s (1 thread: below the CPU's threading size) |
| 600k × 600k, 36M nonzeros | 15.8 s | 116.0 s |
| 2M × 2M, 120M nonzeros | 157.5 s (5.0 ms per iteration) | — |

Each sparse product runs at about 760 GB/s on this card's 896 GB/s; the RTX PRO 6000 Blackwell
(1.8 TB/s) and data-centre GPUs should scale with their memory bandwidth. VBITS 128 and 256 are
correct too but not faster (16.3 and 17.1 s at 600k). Also checked: MPI grids 1×2, 2×1, 2×2, 1×3
(all processes sharing this one card), checkpoint on SIGINT and `-ncr` resume, QS on CPU and GPU.
Two things fixed on the way: the checkpoint interval was calibrated with `time()` in whole
seconds, and the GPU finishes the calibration interval in under one, so it checkpointed every
10,000 dimensions (now it waits for 5 s of timing: once per hour, as meant); and `make clean`
without `CUDA=` left the GPU-only objects behind.

## Line sieving on the GPU

GPU builds run msieve's line siever (`-ns`) on the GPU and find exactly the candidates and relations
that the CPU siever (`gnfs/sieve/sieve_line.c`) finds. The GPU does the sieving
(`gnfs/sieve/sieve_core.cu`) in batches of up to 16 lines (values of b) that share a logarithm base:

- `ls_starts` computes each factor-base entry's first position on each line, with the CPU's skip
  rules and its compression of even lines;
- per 2²⁰-position chunk, `ls_bucket` sorts the hits of primes above 16,384 into buckets per segment;
- `ls_segment` sieves 16,384 positions per block in shared memory (both sides in one 32-bit word):
  primes below 256 and the prime powers by position, the others by entry with atomics, then the
  buckets. It computes the cutoffs the way `sieve_line.c` does (from the exact bit size of the
  norms, a 384-bit evaluation at each window's ends, with the same 2/3-large-prime choice), applies
  the gcd test, and resieves the primes ≥ 1000 for each candidate.

Host threads (`gnfs/sieve/sieve_line_gpu.c`) then trial-divide the small primes, divide out the
resieved ones, and check the cofactors as `do_one_factoring()` does. The cofactors then go to
msieve's batch factoring. While the host factors one chunk, the GPU sieves the next. The batch
factoring's remainder tree (`common/batch_factor.c`) now runs on several threads, in the CPU siever
too (`MSIEVE_BATCH_THREADS`, default: online CPUs).

Settings:

| Setting | Effect |
| --- | --- |
| `MSIEVE_GPU_SIEVE=0` | the CPU siever |
| `MSIEVE_SIEVE_THREADS` | host factoring threads (default: online CPUs) |
| `MSIEVE_BATCH_THREADS` | batch-factoring threads, both sievers (default: online CPUs) |
| `MSIEVE_SIEVE_DUMP=<file>` | both sievers write every candidate (`b a nlp_r nlp_a`), to compare them |

The GPU siever falls back to the CPU, with a log line, for:

- polynomials of degree above 8;
- coefficients of 255 bits or more;
- norms that could exceed 382 bits;
- a GPU with less than 74 KB of shared memory per block.

If one line's chunk has more candidates than the buffer holds (2²¹), it stops with an error. At b = 1
of the c95, one chunk had about 80,000.

**Identical to the CPU siever** on a c95 (the polynomial from `-np "min_coeff=1000 max_coeff=3000"`).
Relations are compared with the primes of each side sorted:

| b range | Candidates | Relations |
| --- | --- | --- |
| 1000–1009 | 295,216, identical | 1,788, identical |
| 1–300 | 40,464,202, identical | 345,088, identical |
| 1000–2000 | — | 133,558, identical |

**Speed**: wall time, RTX PRO 4500 Blackwell, 32 CPU threads:

| b range | CPU siever | CPU siever, threaded batch factoring | GPU siever |
| --- | ---: | ---: | ---: |
| 1000–1009 | — | — | 0.4 s |
| 1–300 | 107.0 s | 97.8 s | **5.8 s** (18×) |
| 1000–2000 | b 1000–1921 in 120 s (stopped) | — | **4.8 s** |

CPU profile before the port: `fill_one_block` (sieving) 42%, resieving 21%, trial division and GMP
about 30%.

Each step's gain, GPU siever, wall time:

| Version | b 1–300 | b 1000–2000 |
| --- | ---: | ---: |
| first version: candidates factored per batch, serial batch factoring | 12.3 s | 10.2 s |
| threaded batch factoring | 7.3 s | 7.8 s |
| GPU sieves the next chunk while the host factors | **5.8 s** | **4.8 s** |

Breakdown from the siever's log line (seconds inside the siever, without startup):

| Version | b range | Total | Waiting for the GPU | Host factoring | Batch factoring |
| --- | --- | ---: | ---: | ---: | ---: |
| before threading the batch | 1–300 | 11.8 | 0.8 | 4.0 | 6.7 |
| before threading the batch | 1000–2000 | 9.7 | 2.5 | 3.8 | 3.1 |
| final | 1–300 | 5.2 | 0.2 | 3.2 | 1.4 |
| final | 1000–2000 | 4.2 | 0.4 | 2.6 | 0.8 |

So on these short ranges the host limits the speed: host factoring of 40 million candidates
(b 1–300) or 22 million (b 1000–2000) is about 60% of the time, and the GPU is mostly idle. Over a
whole factorization it is the other way round (below).

**The whole c95** (`-ns "1,100000000"`, the siever stops when it has the relations filtering
wants), then `-nc` for filtering, the linear algebra (on the GPU) and the square root:

| Step | Result |
| --- | --- |
| Sieving | b 1 to 15,005,348, **6,308,499 relations in 33,210 s** (9 h 14 min) |
| Siever's breakdown | waiting for the GPU 21,951 s, host factoring of 14.18 billion candidates 10,737 s, batch factoring 717 s |
| Relations checked independently | 6,308,499 valid, 0 invalid (each listed prime divides its norm, the rest has only primes < 256) |
| Filtering | 0 duplicates, 6,472,801 unique relations (with the free relations), matrix 428,944 × 429,170 |
| Linear algebra (GPU) | 35.4 M nonzeros, 6,784 Lanczos iterations, 33 dependencies |
| Square root | p47 88950672208081723397483294793131868682627115413 × p48 303880173004512093573696089949868722431828017591 |
| Postprocessing | 120 s |

Over the whole run the GPU is the limit, not the host: the candidates per line fall as b grows,
while each line costs the GPU about the same, so two thirds of the time is waiting for the GPU
(2.2 ms per line on average). The lattice siever below gets the relations for this number in 43.7 s.
This run predates the fix to msieve's batch factoring (lattice section), which could drop a few
relations per run.

**Several GPUs or nodes**: one GPU per process (the first one `gpu_select` gives it). As with
msieve's CPU line siever, several processes, GPUs or machines get separate `-ns` ranges.

## Lattice sieving on the GPU

msieve has no lattice siever of its own. GPU builds add one, and it is what `-ns` runs there:
`-ns "q0,q1"` sieves special-q lattices on the algebraic side for q from q0 to q1 (above the factor
base; without a range, from the factor base's end, or from where the last run stopped, kept in
`<savefile>.lattice`). `-ns "line ..."` runs the GPU line siever instead (the range is then one of
b), `MSIEVE_GPU_SIEVE=0` msieve's CPU line siever; if the lattice siever cannot run (no usable GPU),
the line siever runs. It stops, like the line siever, when it has the relations that filtering wants, and
writes them to the savefile in msieve's format, so `-nc` (filtering, linear algebra, square root)
runs on them unchanged. Each root r of f mod q is one lattice, Gauss-reduced with the polynomial's
skew: a = a0·i + a1·j, b = b0·i + b1·j, −I/2 ≤ i < I/2, 0 ≤ j < J, with I = 4096 and J = 2048 by
default. The sieve value is one 32-bit word per position, the rational log sum in the low 16 bits
and the algebraic one in the high 16.

The GPU (`gnfs/sieve/lasieve_core.cu`) sieves a batch of lattices (64 here, sized to free memory):

- `la_roots`: each factor-base entry's root in each lattice, ρ = −v/u mod p with
  u = a0 − R·b0, v = a1 − R·b1 (projective roots: u = b0, v = b1). An entry with u ≡ 0 divides
  the norm at every position of the rows j ≡ 0 (mod p); it is marked and handled per row;
- `la_rowlogs`: for every row of every lattice, the logs of those u ≡ 0 entries whose p divides j;
- `la_large`: the primes p ≥ I with the Franke–Kleinjung walk (a reduced basis per prime, one step
  per hit), adding into the lattice's sieve in global memory. One launch per lattice keeps the
  atomic adds in L2;
- `la_tile`: a block per 4 rows in shared memory. It adds the primes 32 ≤ p < I and sets each
  run of 32 positions' thresholds from the norms there (the 2/3-large-prime choice as msieve's),
  lowered by the row's u ≡ 0 logs. The primes below 32 are not sieved; their expected contribution
  lowers the thresholds too. It then applies the gcd test, stores each candidate in a list and a
  GPU hash table, and resieves the small primes ≥ 256;
- `la_large` again, resieving: each hit on a candidate (a bitmap check, then the hash) adds the
  prime to that candidate's own list;
- `la_filter`: the rational side's cofactor test on the exact 128-bit norm, which removes two
  thirds of the candidates;
- `la_cofactor` (CGBN, 256 bits, 4 threads per candidate): the rest of the host's tests. The
  algebraic norm by Horner's rule (two's complement modulo 2²⁵⁶, exact as the host bounds the norms
  of each batch below 2²⁵³), divided by q, the listed primes, the row's prime and the primes below
  256 (in 12 groups whose products fit 32 bits); msieve's algebraic large-prime bounds; and the
  base-2 Fermat test on both cofactors (at 128 bits). Only 2.8% of the candidates past `la_filter`
  are copied to the host, and nearly all of them become relations.

Host threads (`gnfs/sieve/sieve_lattice_gpu.c`) divide out q, the listed primes, the u ≡ 0 prime of
the candidate's row (j < 2¹³ has at most one prime factor ≥ 256, `la_jprime`) and the primes below
256, check the cofactors as the line siever does (large-prime bounds, a Fermat test) and send the
rest to msieve's batch factoring. They factor batch k while the GPU sieves batch k+1 (two pinned
buffers, one host job at a time), and batch factoring runs in a background thread on a second
relation batch (its relations and the host threads' reach the savefile under one lock). The stop
test adds the relations the batched cofactors not run yet should give, at the rate seen so far,
then drains everything and checks the count. Relations that two special-q both find are removed by
msieve's filtering.

Settings:

| Setting | Effect |
| --- | --- |
| `MSIEVE_LATTICE_LOGI` | log₂ I (default 12; J = I/2) |
| `MSIEVE_LATTICE_TINY` | primes below it are not sieved (default 32) |
| `MSIEVE_LATTICE_FILTER=0` | no rational test on the GPU (every candidate to the host) |
| `MSIEVE_LATTICE_COFACTOR=0` | no algebraic and Fermat tests on the GPU (`la_cofactor`) |
| `MSIEVE_LATTICE_BATCH` | lattices per batch |
| `MSIEVE_LATTICE_CHECK=1` | compares the GPU's sieve sums of the first lattice with a direct computation on the host |
| `MSIEVE_LATTICE_VERIFY=1` | checks every cofactor for a factor-base prime ≥ 256 the GPU did not report (there must be none) |
| `MSIEVE_BATCH_DUMP=<file>` | writes the cofactors sent to batch factoring, for `test/sieve/batchreplay.c` |
| `MSIEVE_SIEVE_THREADS`, `MSIEVE_BATCH_THREADS` | as for the line siever |

**Three bugs that lost relations** (found after the first version of this section, which reported
25,468 relations on the test range and 6,374,921 for the c95; fixed in the same commit as this text):

| Bug | Effect | Fix |
| --- | --- | --- |
| Entries with u ≡ 0 were neither sieved nor resieved | their prime stayed in the cofactor; batch factoring (primes above the factor base) cannot split it: 105 such cofactors on the test range (110 relations more after the fix), 30,127 relations on the c95 | `la_rowlogs` adds their logs per row; `la_filter` and the host divide out j's prime |
| At i = 0 the large primes with ρ = 0 hit the whole column and the Franke–Kleinjung reduction cannot walk them | the candidate at (j = 1, i = 0) kept them: 6 on the c95 | that position goes to the host untested, which scans the factor base for it |
| msieve's batch factoring (`common/batch_factor.c`, upstream code) skipped the relations of a remainder-tree leaf whose remainder was zero | a zero remainder means the leaf's cofactors divide the prime product, i.e. they are all smooth: exactly the relations wanted. Whether a leaf hits zero depends on which relations the batch puts together, which in the sievers depends on thread timing: 7–9 relations in about one run in six | the leaf is checked anyway (`gcd(0, n) = n` marks the whole cofactor smooth) |

The third bug affects every msieve siever that batches (the GPU line siever and msieve's CPU line
siever too). How it was found: in a run that lost relations every count matched a normal run, the
lost relations all came from batch factoring and clustered in neighbouring lattices, and
`MSIEVE_LATTICE_VERIFY` reported no missed prime, so the GPU had reported everything.
`MSIEVE_BATCH_DUMP=<file>` writes the cofactors a run sends to batch factoring, and
`test/sieve/batchreplay.c` feeds them through msieve's batch factoring in shuffled orders: 2 of
20 orders found 3,624,662 relations instead of 3,624,670, and instrumenting the remainder tree
showed 2 zero-remainder leaves with exactly those 8 relations. With the fix all 20 orders give
3,624,670, the same relations. `MSIEVE_LATTICE_VERIFY` found the first two bugs directly
(`MSIEVE_LATTICE_CHECK` had skipped the same u ≡ 0 entries as the GPU, so it passed).

One hardening on the way, not a fix of any observed loss: `la_tile` looks up hash keys while other
blocks still insert theirs, so those lookups read from L2 (`__ldcg`); a plain load may return a
line the SM cached before another SM filled a slot. `la_large`'s lookups, in a later launch, keep
plain loads (`__ldcg` everywhere cost 12.6 s of GPU time on the c95).

**Correctness.** `MSIEVE_LATTICE_CHECK`: 0 of 7,096 sampled sieve sums differ from the host's
direct sums (q 2,200,031), the u ≡ 0 rows included. `MSIEVE_LATTICE_VERIFY`: 0 cofactors with an
unreported factor-base prime, on the test range and on the c95. Every relation was checked
independently (`test/sieve/relcheck.py`: each listed prime divides its side's norm, and what is
left has only primes below 256): 25,768 of 25,768 on q 2,200,000–2,201,000 and 6,405,054 of
6,405,054 for the whole c95. 40 runs of the test range (20 plain, 10 under CPU load, 10 under nsys)
all gave 25,768 relations, and 8 runs of the whole c95 all gave 6,405,054.

**The whole c95**, `./msieve -n` with the c95's factor base (every stage at its default), on an
otherwise idle machine:

| Step | Result |
| --- | --- |
| Sieving | special-q 2,120,430–2,384,071, 18,038 lattices, **6,344,609 relations in 42.6 s** (other runs 43.0 s) |
| Siever's log line | GPU 38.6 s; 7,217,394 candidates factored on the host alongside the GPU, batch factoring 28.4 s; waiting for the host 0.1 s |
| Candidates | 776,648,279 found, 256,003,536 past `la_filter`, 7,217,394 past `la_cofactor` (170.8 million failed its algebraic bounds, 78.0 million its Fermat test); 2,754,137 relations direct, 4,463,189 cofactors to batch factoring |
| Filtering | 104,888 duplicates, 6,398,252 unique relations, matrix 126,032 × 126,259 |
| Linear algebra (GPU) | 10.3 M nonzeros, 1,994 Lanczos iterations, 33 dependencies |
| Square root | p47 × p48, the same factors as the line siever's run |
| **Total** | **111.4 s** (sieving 42.6 s, postprocessing about 68 s); **67.8 s** with the postprocessing below (sieving 43.1 s, postprocessing about 24 s) |

All 6,344,609 relations of the same sieving run with `-ns` are valid (`test/sieve/relcheck.py`).
Without `-n`, msieve factors a c95 with its quadratic sieve (MPQS, CPU): 46 min 40 s.

**Each step's gain on the whole c95** (sieving; the siever's log line):

| Version | Sieving | GPU | Host | Waiting for the host | Candidates to the host |
| --- | ---: | ---: | ---: | ---: | ---: |
| the three fixes (00e63ab; 8 runs) | 105.0 s | 34.7 s | 68.4 s, batch factoring 31.5 s | — | 258,450,297 |
| + `la_cofactor` (algebraic and Fermat tests on the GPU), 8 threads per candidate | 93.0–94.2 s | 47.1 s | 45 s, batch factoring 32 s | — | 7,286,121 |
| + 4 threads per candidate, Fermat at 128 bits, unrolled trial division, host overlapped with the GPU | 71.8–72.3 s | 39.6 s | — | 30.7 s | 7,309,655 |
| + batch factoring in the background, the stop estimate | **43.7 s** | 38.9 s | — | 0.1 s | 7,217,394 |

The overlapped versions sieve 18,038–18,272 lattices instead of 18,212 (where they stop differs);
thread count did not matter in the third row (16, 24, 31 threads: 71.1–71.5 s): batch factoring
held the host job up, so the GPU waited for it. Kernel time of `la_cofactor` per 72 lattices
(nsys): 43.7 ms with 8 threads per candidate and the Fermat test at 256 bits, 36.1–37.1 ms with 4
and 128 bits (the Fermat test was 19 ms of the first); `la_filter` 1.8 → 4.0 ms when its primes
came from a constant-memory table, below 1.7 ms written out as constants.

**The two sievers** on the c95, RTX PRO 4500 Blackwell, 32 CPU threads:

| | Line siever (GPU) | Lattice siever (GPU) |
| --- | ---: | ---: |
| Sieving | 33,210 s | **43.7 s** (760×) |
| Relations | 6,308,499 | 6,344,609 |
| Duplicates | 0 | 104,888 |
| Matrix | 428,944 × 429,170 | 126,032 × 126,259 |
| Lanczos iterations | 6,784 | 1,994 |
| Postprocessing | 120 s | about 24 s |
| Total | 9 h 16 min | **1 min 8 s** (`./msieve -n`) |

The lattice relations also give a matrix 3.4 times smaller, from about as many relations.

**Postprocessing.** After the sieving work above it was the larger part (about 68 s). Reading
the gzipped savefile is not what costs: on the same relations 62.6 s from the gzipped file, 59.5 s
from uncompressed text. msieve's filtering reads and parses every relation in several passes, on
one thread: duplicate removal's first pass checks each relation against its norms, singleton
removal's first pass computes each large ideal's root (`find_large_ideals`, 38% of a profile), and
the matrix and every square-root dependency read their relations again. Two changes:

- **All CPUs** (`gnfs/filter/relpar.{c,h}`): a pass adds a block of 16,384 lines; they are parsed
  (and their ideals found) on all threads, each with its own copy of the polynomials
  (`nfs_read_relation` keeps scratch values in them), then the pass takes the results in order, so
  its output is unchanged. Duplicate pass 1, singleton removal's initial pass and the cycle reader
  (matrix, square root) use it. `MSIEVE_FILTER_THREADS` (default: online CPUs).
- **In memory** (`gnfs/relstore.{c,h}`): when `factor_gnfs` starts a new savefile, every relation
  written to it (all sievers, and the free relations filtering adds) is also copied into memory as
  text, under the lock that orders the savefile, so the sievers do no more work. The first time
  filtering asks, they are parsed once on all CPUs (primality tested, as duplicate pass 1 does), and
  every later pass reads the parsed relations instead of the file. The savefile is written as
  before; another process (`-nc`), a restarted savefile, `MSIEVE_RELSTORE=0` or a store over
  `MSIEVE_RELSTORE_MB` (default: a quarter of physical memory) read the file as before.

The c95's postprocessing (`./msieve -n`; times from the log's timestamps):

| Version | Duplicates, pass 1 | Pass 2 | Singletons, first pass | In-memory singletons | Matrix and Lanczos | Square root, per dependency | Up to the square root |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| msieve's, one thread, gzipped savefile | 22 s | 2 s | 18 s | 6 s | 8 s | 6 s | 56 s |
| relations parsed on all CPUs | 8 s | 2 s | 8 s | 5 s | 8 s | 5.5 s | 31 s |
| relations in memory | **4 s** | <1 s | **5 s** | 5 s | 6 s | 4 s | **20 s** |

`./msieve -n` on the c95: 111.4 s before, **67.8 s** after (sieving 43.1 s; the square root needed
one dependency). Each dependency has about an even chance of giving the factors, so the square root
took 1–5 dependencies in these runs, 4–6 s each.

**The square root, several dependencies at once.** `nfs_find_factors` now tries
`MSIEVE_SQRT_THREADS` dependencies (default 4; 8 with 32 CPUs or more) at a time: their relations are read in one pass
(`nfs_read_dependencies`: the union of the dependencies' relations, read once from the store or the
savefile, a copy per dependency), then each runs on its own thread (its own copies of the monic
polynomial and of `msieve_obj`, for the random seeds), and the results are taken in dependency
order with the serial loop's stopping rule. On the c95 (`-nc3`, three sets of dependencies from
three linear algebra runs):

| Dependency 1 | 1 at a time | 4 at a time |
| --- | ---: | ---: |
| gives the factors (2 of the 3 sets) | 6.2 s | 6.5 s |
| does not (1 set: 2 needed) | 12.2 s | 6.5 s |

Four at a time take as long as one, and need a second round only when all four fail (1 in 16).
`./msieve -n`: 68.0 s in two runs, square root 4 s. Checks: the duplicates, ideals, every matrix size and the cycle
weights are the same with the store as with a file-path `-nc` on the same savefile (only the
dependency the square root picks differs, as the linear algebra is random), and the file path gives
the same results as before.

Two ways of filling the store were measured and dropped: parsing each relation in the siever (inside
the batch factoring's output lock it serialized the batch threads; outside it, contention for the
savefile lock cost 8 s of sieving), and listing the primes below 256 in the relations so that the
parse needs no trial division (8 s more sieving).

**Each step's gain**, q 2,200,000–2,201,000 (72 lattices; the siever's log line, without startup;
the first two rows were measured with the line siever's run on the same machine):

| Version | Relations | Total | GPU | Host factoring | Batch factoring | Candidates to the host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first: the host sorts the resieved hits and joins them to the candidates | 25,468 | 4.3 s | 1.7 s | 2.4 s | 0.2 s | 2,235,499 |
| hits into each candidate's list through a GPU hash table, rational test on the GPU | 25,468 | 1.4 s | 0.3 s | 0.1 s | 0.2 s | 815,082 |
| one lattice per `la_large` launch, primes below 32 not sieved, idle machine | 25,658 | 0.7–0.8 s | 0.1 s | 0.1 s | 0.2 s | 1,023,712 |
| the three fixes | **25,768** | **0.7 s** | 0.1 s | 0.1–0.2 s | 0.2 s | 1,029,718 |

The first full run (18,672 lattices, sharing the machine with the line siever's run) took 838 s,
668 s of it host factoring.

Kernel time per 72 lattices (nsys, idle GPU, the fixed version), by the bound below which primes
are not sieved:

| Primes not sieved | `la_tile` | `la_large` (both passes) | `la_roots` + `la_rowlogs` + `la_filter` | Candidates | Relations |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 147.3 ms | 74.4 ms | 3.4 ms | 2,548,519 | 26,574 |
| below 16 | 62.3 ms | 76.2 ms | 3.7 ms | 3,193,573 | 26,448 |
| below 32 (default) | 53.6 ms | 76.5 ms | 3.6 ms | 3,129,055 | 25,768 |
| below 64 | 48.2 ms | 75.8 ms | 3.6 ms | 3,079,208 | 25,001 |

Sieving every prime finds 3% more relations than the default but costs 94 ms more `la_tile` time
per 72 lattices, about 24 s more GPU time for the c95 (sieving took 105 s then); a bound of 16 finds
2.6% more for 9 ms. Before the fixes the default found 25,658 relations with `la_tile` at 51.9 ms.
With all lattices of a batch in one `la_large` launch, `la_large` took 416.6 ms (measured with the
line siever on the GPU too), against 113.8 ms with one lattice per launch under the same conditions:
the batch's sieve arrays did not fit in L2, so the atomic adds went to DRAM.

**In-memory singleton passes on all CPUs.** msieve repeats them after each clique removal; each
pass now marks, on all threads, the relations with an ideal of count 1 and decrements their ideals'
counts atomically, and the survivors are compacted in order at the end. The relations left (every
ideal in at least two of them) do not depend on the order of removal, and a stale count only puts a
removal off to the next pass, so the result is the same (the filtering log is identical apart from
the pass counts). The six rounds on the c95: 1.15–1.22 s → 0.39 s; `./msieve -n` 67.2 s. The rest
of that stage, clique removal and the merge (about 1 s each, heap-based), stays sequential.

**The NFS by default (GPU builds).** msieve used its quadratic sieve (MPQS, CPU) unless given an
NFS option. GPU builds now run the whole NFS without one for inputs of 80 digits or more (85 at first, below) when a CUDA
device is there (`MSIEVE_NFS_MIN_DIGITS`; `MSIEVE_DEFAULT_NFS=0`: MPQS as before). From scratch
(polynomial selection included), RTX PRO 4500 and 32 CPU threads, balanced semiprimes:

| Digits | MPQS (`./msieve` before) | NFS (`./msieve` now) | Of it, polynomial selection |
| --- | ---: | ---: | --- |
| 80 | 132 s | — (msieve has no NFS polynomial parameters below 80 digits; this one fell just under) | — |
| 85 | 350 s | **227 s** (with `-n`) | 159 s: none reached the minimum E in 59 s, one retry |
| 90 | 981 s | **395 s** | 339 s, one retry |
| 95 | 2,800 s | **413 s** | 345 s |

Two things in polynomial selection had to change first; the first measurements of the NFS from
scratch were 165 s with no polynomial (c85), 1,324 s (c90) and 1,130 s (c95):

- **The time limit.** It counted only stage 1's own CPU and GPU time, which barely grows while stage 1
  waits for root optimization, and with the GPU it waits most of the time: every size-optimized
  polynomial passed the stage 2 bound at these sizes, about 1.2 million at 90 digits, and root
  optimization took them all (c90: the 166 s limit ran 20 minutes; c95: 333 s ran 18 minutes). Stage 1
  now also stops at the limit's wall-clock end, and past it size-optimized polynomials are not
  root-optimized any more (the log says how many: 9,000–21,000 in these runs). A first round now ends
  near its limit (59 s → 67 s, 166 s → 176 s, 333 s → 345 s).
- **No polynomial.** msieve keeps only polynomials above a minimum Murphy E; when none reaches it the
  factorization stops (the c85). It now searches again with a quarter of the minimum in half the
  time, up to three times. A retry ends 60–80 s past its limit (the root optimizations still under
  way are dearer with the lower minimum).

The rest of these runs: sieving 33.7–43.6 s, filtering, linear algebra and square root 25–35 s.

**Shorter time limits.** msieve's limits were set for polynomial selection on the CPU and were most
of a run. Shorter limits (`poly_deadline=`), from scratch:

| Input | Limit | Total | Polynomial E | Sieving |
| --- | ---: | ---: | --- | ---: |
| c95 | 333 s (msieve's) | 413 s | 2.968e-08 | 43.6 s |
| c95 | 167 s | 248 s | 2.745e-08 | 43.0 s |
| c95 | 83 s | 163 s | 2.745e-08 | 43.1 s |
| c95 | 42 s | 124 s | 2.745e-08 | 43.1 s |
| c90 | 166 s (msieve's) | 395 s (one retry) | 4.645e-08 | 33.7 s |
| c90 | 83 s | 294 s (one retry) | 4.645e-08 | 33.4 s |
| c90 | 42 s | 254 s (one retry) | 4.645e-08 | 33.3 s |
| c90 | 21 s | 236 s (one retry) | 4.592e-08 | 39.5 s |

At 90 digits the retry (about 140 s whatever the limit) was then most of the time. So polynomials
are now kept down to a quarter of the minimum E (`MSIEVE_POLY_KEEP_E`), with msieve's Bernstein bound
unchanged so no more Murphy E work is done: the best one found is used when none reaches the minimum.
GPU builds also took a quarter of msieve's limits by default (`MSIEVE_POLY_TIME_SCALE`; 1/32 since
Nov 10, see "Shorter polynomial selection" below). With both,
from scratch, `./msieve` without options:

| Digits | MPQS (before) | NFS, msieve's limits (first fixes) | **NFS now** (a quarter) | An eighth |
| --- | ---: | ---: | ---: | ---: |
| 85 | 350 s | 227 s | **93.4 s** (limit 15 s, E 9.31e-08) | — |
| 90 | 981 s | 395 s | **110.6 s** (42 s, E 4.723e-08) | 89.6 s (E 4.645e-08) |
| 95 | 2,800 s | 413 s | **163.4 s** (83 s, E 2.745e-08) | 124.2 s (E 2.745e-08) |

No retries, all factors correct, sieving 33.6–43.0 s. An eighth found the same polynomials here, but
two numbers are a small sample and the sieving's share grows with the input, so the default is a
quarter.

**Larger inputs**, from scratch, `./msieve` (a quarter of msieve's limits) against msieve's full
limits (`MSIEVE_POLY_TIME_SCALE=1`):

| Input | Limit | Selection | Polynomial E | Sieving | Total |
| --- | ---: | ---: | --- | ---: | ---: |
| c100, a quarter | 174 s | 187 s | 1.269e-08 | 71.2 s | **291 s** |
| c100, msieve's | 695 s | 704 s | 1.269e-08 (the same polynomial) | 71.2 s | 808 s |
| c105, a quarter | 331 s | 343 s | 6.339e-09 | 116.7 s | **516 s** |
| c105, msieve's | 1,322 s | 1,230 s | 6.649e-09 | 118.4 s | 1,386 s |
| c110, a quarter (degree 5) | 675 s | 883 s | 1.101e-09 | 283.4 s | **1,229 s** (1,022.7 s with the fixes below) |

All factored correctly, with one sieving round each (c100: matrix 176,168 columns, c105 212,804,
c110 297,952; Lanczos on the GPU 3–6 s). The full limit found the same polynomial at 100 digits and a
slightly higher E at 105 that sieved no faster. At 110 digits msieve chose degree 5; its first round
ended after 264 s (degree 5's stage 1 counts four GPU threads' time against the limit) with nothing
past the Bernstein bound, and the retry found the polynomial. That run also kept 485,371 polynomials
above a quarter of the minimum E and printed and wrote each; now those below the minimum are only kept
when they are the best so far (the `.p` file as before: a c85 8 polynomials instead of 492).

**Degree 5 and the end of the limit.** Three more fixes, from the c110: stage 1 compared its four GPU
threads' time added up with the limit (now their average); degree 5's Bernstein bound (1/6.6 of the
minimum E at 110 digits, against degree 4's 1/30000) let nothing through in a whole first round, so
it is now derived from the lower minimum that is kept, as the retry's was; and at the limit the GPU
root optimization still finished every job it had started, 306 s of them for degree 5, so near the
limit it now starts a job only if it can end in time (each thread's average job time) and drops the
queued jobs past it. The retry also wrote its lower minimum's polynomials to the `.p` file
(486,577); reporting now uses msieve's minimum throughout. From scratch:

| Digits | Limit | Selection | Total before | **Total now** | Polynomial E |
| --- | ---: | ---: | ---: | ---: | --- |
| 85 | 15 s | 17 s | 93.4 s | **85.9 s** | 9.31e-08 |
| 95 | 83 s | 85 s | 163.4 s | **152.6 s** | 2.745e-08 |
| 110 (degree 5) | 675 s | 687 s | 1,229 s (a retry) | **1,022.7 s** | 1.104e-09 |

No retries, the same polynomials, factors correct, 6–12 polynomials in each `.p` file.

**The crossover below 85 digits.** msieve's polynomial parameters start at 80 digits, and an
80-digit input (79.99 there) found none, so its NFS stopped; up to 5 digits below the table now use
its first entry (the E-value cutoff still follows the input). From scratch, `-n` against MPQS:

| Digits | MPQS | **NFS** | Polynomial E | Sieving |
| --- | ---: | ---: | --- | ---: |
| 80 | 132.4 s | **72.8 s** | 1.755e-07 | 34.2 s |
| 81 | 186.5 s | **87.8 s** | 1.623e-07 | 46.8 s |
| 82 | 265.8 s | **76.8 s** | 1.381e-07 | 36.5 s |
| 83 | 267.9 s | **83.4 s** | 1.157e-07 | 39.4 s |
| 84 | 382.4 s | **82.3 s** | 1.164e-07 | 38.7 s |

The NFS wins at every size, so the crossover is below 80 digits, where msieve's NFS does not go
(`MIN_NFS_BITS`, about 79.5 digits). The default now starts at 80 digits.

**Below 80 digits**, with a test build whose NFS minimum is lowered (`-DMIN_NFS_BITS=230`; msieve's
is 264 bits, about 79.5 digits), NFS (`-n`) against MPQS:

| Digits (bits) | MPQS | NFS |
| --- | ---: | --- |
| 75 (247) | **49.1 s** | stops at once: more than 5 digits below the polynomial parameter table |
| 76 (250) | **54.2 s** | 313 s, no factor: the matrix had more rows than columns (40,285 × 39,799) |
| 77 (254) | **69.4 s** | 78.8 s |
| 78 (257) | 98.2 s | **80.6 s** |
| 79 (260) | 109.3 s | **82.4 s** |

The crossover is near 78 digits, but the NFS saves only 17–27 s at 78–79 digits and msieve's
parameters (tuned from 80 digits) gather too few relations below that, so the default stays at 80
digits and msieve's minimum is unchanged (`MIN_NFS_BITS` can now be set at build time).

**Above 110 digits**, from scratch (degree 5, `./msieve` with a quarter of msieve's limits; c115
also with the full limit):

| Input | Limit | Selection | Polynomial E | Sieving | Filtering, LA, sqrt | Total | Peak RSS |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| c115, a quarter | 1,228 s | 20.6 min | 6.432e-10 | 403.3 s | 1.3 min | **28.5 min** | 6.6 GB |
| c115, msieve's | 4,913 s | 80.0 min | 6.432e-10 (the same) | 403.5 s | 1.3 min | 90.0 min | 7.4 GB |
| c120, a quarter | 2,397 s | 40.1 min | 3.555e-10 | 654.8 s | 1.7 min | **52.7 min** | 7.2 GB |

All factored correctly (p58 × p58, p60 × p60), one sieving round each, no retries. c115's matrix had
370,886 columns (5,867 Lanczos iterations on the GPU), c120's 517,766 (8,189). The full limit at 115
digits found the same polynomial as a quarter, so a quarter holds there too; polynomial selection is
now most of a run (76% at 120 digits), its limit doubling every 5 digits against about 1.6 times for
sieving.

**CADO-NFS on the same c120, CPU only** (32 threads, its default parameters, unmodified build):

| Stage | CADO-NFS (CPU) | This fork (1 GPU) |
| --- | ---: | ---: |
| Polynomial selection | **2.6 min** | 40.1 min |
| Sieving | 19.8 min (16,770 CPU-s, 16.2 M relations) | **10.9 min** (655 s, 12.1 M relations) |
| Filtering | 0.8 min | 0.5 min |
| Linear algebra | 4.6 min (5,305 CPU-s; matrix 635,213 rows, 107/row) | **0.6 min** (517,766 columns) |
| Characters, square root | 0.6 min | 0.2 min |
| **Total** | **28.6 min** | 52.7 min |

On this machine one RTX PRO 4500 did the sieving of 25.6 CPU threads of CADO-NFS's `las` and the linear
algebra of about 150 threads of its `bwc`; msieve's polynomial selection is what loses the run.

**Shorter polynomial selection (Nov 10).** At 120 digits polynomial selection was 76% of a run, and
CADO-NFS's 2.6-minute CPU selection found a slightly better polynomial than this fork's 40 minutes. Scored
on one scale (CADO-NFS's `score`, its c120 bounds and area), the c120 with less time:

| Time limit (of msieve's) | Selection | Murphy E (CADO-NFS scale) |
| --- | ---: | ---: |
| 1/32 | 6.4 min | 3.691e-06 |
| 1/16 | 11.7 min | 3.691e-06 (the same polynomial) |
| 1/8 | 20.1 min | 3.751e-06 |
| 1/4 (the default until now) | 40.1 min | 3.751e-06 (the same polynomial) |
| CADO-NFS (CPU, 32 threads) | 2.6 min | 3.794e-06 |

msieve root-optimizes every size-optimized polynomial that passes the stage 2 bound (22,696 at 1/4),
which keeps the GPU's queue full; CADO-NFS root-optimizes only the 48 best by size (`nrkeep`).
`MSIEVE_ROOTOPT_TOPK=<K>` does the same online: a polynomial is root-optimized if its size score is
among the K best so far. But measured (a new `R` line in `MSIEVE_SIZEOPT_LOG` gives every root
optimization's E, matched to its survivor by the leading and rational coefficients, which root
optimization keeps), the size score hardly predicts the final E among the survivors: Spearman -0.09
over 7,917 c120 polynomials, the best one ranked 1,456 of 10,389 by size, and the top 1% by size holds
none of the top 1% by E. So the filter is a random subset in effect. With K = 48 at 1/8 the freed GPU
produced 2.4 times the survivors (27,420 against 11,517) but no better polynomial (3.691e-06); K = 1000
kept the best (3.751e-06, 3,850 root optimizations against 9,333). What the filter does do is end a
short selection at its limit (1/32: 308 s instead of 384 s, the same polynomial).

So GPU builds now take **1/32 of msieve's limits** (`MSIEVE_POLY_TIME_SCALE`) and root-optimize the
**48 best by size so far** (`MSIEVE_ROOTOPT_TOPK`). From scratch, all factored correctly:

| Input | Total, a quarter | **Total now** | Selection | Sieving (before → now) | Polynomial E (before → now) |
| --- | ---: | ---: | ---: | --- | --- |
| c85 | 85.9 s | **76 s** | 4 s | 34–43 → 43.6 s | 9.31e-08 → 8.865e-08 |
| c95 | 152.6 s | **84 s** | 12 s | 43.0 → 46.1 s | 2.745e-08 → 2.433e-08 |
| c100 | 291 s | **129 s** | 25 s | 71.2 → 74.2 s | 1.269e-08 → 1.251e-08 |
| c105 | 516 s | **195 s** | 43 s | 116.7 → 114.9 s | 6.339e-09 → 6.327e-09 |
| c110 | 1,023 s | **437 s** | 90 s | 283.4 → 283.4 s | 1.104e-09 → 1.096e-09 |
| c115 | 28.5 min | **11.1 min** | 161 s | 403.3 → 432.4 s | 6.432e-10 → 6.428e-10 |
| c120 | 52.7 min | **18.2 min** | 5.1 min | 654.8 → 682.3 s | 3.555e-10 → 3.459e-10 |

The weaker polynomials cost 0–7% more sieving against 1.1–2.9 times less in total. The c120 now
takes 18.2 min against CADO-NFS's 28.6 min on 32 CPU threads (its polynomial selection 2.6 min, this
fork's 5.1). Measured on this RTX PRO 4500; nothing here was run on an RTX PRO 6000 or B200.

**32-bit arithmetic in the lattice siever (Nov 10).** With polynomial selection short, sieving was 63%
of the c120 and GPU-bound. Its kernels used 64-bit integers throughout, and the GPU emulates 64-bit
division and modulo. Now in 32 bits where the values fit: `la_large`'s Franke–Kleinjung reduction and
walk (p < 2^30; the basis entries are bounded by p, and the j steps are clamped to J, which ends the walk
the same way), `la_small`'s (r j + I/2) mod p (below I × J), and `la_roots`'s inverse mod p. On 8,946 c120
lattices (nsys): `la_large` 9,752 → 8,028 ms, `la_tile` 4,706 → 2,653 ms, `la_roots` 448 → 336 ms, GPU
time 15.9 → 12.0 s; relations identical (with each relation's primes sorted: their order follows the
atomics), `MSIEVE_LATTICE_CHECK` and `MSIEVE_LATTICE_VERIFY` clean. Full c120 from scratch: sieving
682.3 → **516.3 s**, total 18.2 → **15.4 min** (the same 378,572 lattices and 17,250,971 candidates).

`la_large` is now two thirds of the sieving's GPU time. Its sieve pass makes 10.38 M hits per lattice
(counted) in 0.69 ms, about 17 G atomics/s, against 38 G/s for random atomic adds into a 32 MB buffer
(56 G/s into 8 MB, 10.5 G/s into 64 MB) in a microbenchmark; the walk alone takes 0.07 ms. Two sieve
arrays reused while in L2 (instead of one per lattice of a batch) and other grid sizes did not help;
finding the limit needs Nsight Compute's counters (root on this machine:
`runs/prof120/ncu_la_large.sh`).

Nsight Compute on the sieve pass (0.69 ms a lattice): nothing saturated (L2 throughput 22%, DRAM 27%,
issue slots 6%, no eligible warp 89% of the time), achieved occupancy 55% (66.7% theoretical, 50
registers), 45 cycles of long-scoreboard stall per issued instruction, half of the stall samples on the
`MOV` that reloads the address register of the next atomic. The adds were generic `atom.add` (the sieve
pointer comes through `la_batch_t`, not provably global). Each warp-wide atomic carries 18.6 of 32 lanes
on average (10.37 M sectors in 556,471 requests): the lanes' walks end at different times. Tried, relations
identical, no gain: `red.global.add` instead of the atomic (8,074 ms against 8,028), the walk unrolled with
four adds in flight per thread in distinct registers (8,189 ms; four separate calls had reused one register
pair, 8,382 ms). So the limit is not the atomics' return or a register. A per-warp queue in shared
memory, so that every atomic add carries 32 lanes, was slower too (8,833 ms).

**One sieve array, kept in L2.** What the counters did show: 10% of the atomic sectors missed L2. A batch
had a 32 MB array per lattice (2 GB for 64), each sieved by its own `la_large` launch and all read later by
one `la_tile` launch, so every launch started on an array that was not in L2 (the microbenchmark: 20 G
atomics/s on cold 32 MB arrays, 38 G/s warm). Now one array: lattice k is sieved by `la_large`, then
`la_tile` reads its tiles and clears them, and lattice k + 1 reuses the array while it is in L2 (the
check mode, which reads the sums back, keeps an array per lattice). The first try alternated two arrays,
64 MB, the whole L2, so each was evicted before its turn and nothing changed. On the 8,946 lattices:
`la_large` 8,028 → 6,367 ms, GPU time 12.0 → 9.9 s, relations identical, both check modes clean. Full
c120 from scratch: sieving 516.3 → **429.2 s**, total 15.4 → **13.95 min** (CADO-NFS on 32 CPU threads:
28.6 min).

**1/128 of the time limits (Nov 10).** With sieving faster, polynomial selection was again a large share
(the c120: 5.1 of 13.95 min). At 1/64 and 1/128 of msieve's limits the c120 found the same polynomial as
at 1/32 (82 s instead of 5.1 min; CADO-NFS score 3.6908e-06 throughout). From scratch with the current
siever, all factored correctly:

| Input | Selection (1/32 → 1/128) | Polynomial E (1/32 → 1/128) | Sieving at 1/128 | **Total at 1/128** |
| --- | --- | --- | ---: | ---: |
| c85 | 4 → 3 s | 8.865e-08 → 8.897e-08 | 38.8 s | **68 s** |
| c95 | 12 → 5 s | 2.433e-08 → 2.468e-08 | 34.8 s | **66 s** |
| c100 | 25 → 7 s | the same | 52.0 s | **90 s** |
| c105 | 43 → 13 s | the same | 78.2 s | **128 s** |
| c110 | 90 → 28 s | 1.096e-09 → 1.056e-09 | 186.2 s | **302 s** (335 s at 1/32, sieving 181.3 s) |
| c115 | 161 → 44 s | the same | 268.3 s | **388 s** |
| c120 | 308 → 81 s | the same | 429.3 s | **610 s** |

The lattice siever also checks that the `lasieve_core_sm*.ptx` it loads (from the current directory)
matches the binary (`la_abi`: an interface number and `sizeof(la_batch_t)`); an older file had made an
illegal memory access, and now gives a message and the line siever instead.

GPU builds now default to 1/128. The c120 takes 10.2 min from scratch (52.7 min with a quarter of the
limits and the earlier siever; CADO-NFS on 32 CPU threads 28.6 min); polynomial selection is 13% of it.

**`la_tile` (Nov 10).** Nsight Compute on `la_tile` (per lattice): theoretical occupancy 33% (a 64 KB
tile, one block an SM), a fifth of the stall samples on loading the tile a word at a time, and 18 M shared
bank conflicts a lattice in the candidate scan (each thread's run of 32 positions, 32 words from its
neighbour's: every lane on the same bank). Now the tile moves 16 bytes at a time and each lane starts its
run at its own offset (relations identical; `la_tile` 2,611 → 1,985 ms on the test range), and tiles are
8,192 positions, three blocks an SM (`LA_TILE_POS`; 1,985 → 1,465 ms; 4,096 positions: 2,545 ms). With
half the run, each thread's thresholds follow the norms more closely: 328,760 relations instead of
328,657 on the test range (878 new, 775 not found), all valid by `test/sieve/relcheck.py`, both check
modes clean. Full c120 from scratch: sieving 429.3 → **380.2 s**, total 10:10 → **9:49** (CADO-NFS on 32
CPU threads: 28.6 min).

**The square root: 8 dependencies at once, large products on the GPU (Nov 10).** With 32 CPUs or more,
8 dependencies are tried at once (it was 4): a round of 8 took 33.8 s against about 32 s for 4, the same
on average, but 4 all fail one time in 16 and need a second round (the c120 run that hit it: 64 → 34 s).
The square root's products of large integers (`mpz_mul` in `gnfs/sqrt/`, through `sqrt_mpz_mul`) go to
the GPU when both operands have at least 2,048 limbs: a number theoretic transform modulo three primes
c·2^40 + 1, CRT, carries on the host: the same integer as GMP's (ported from this author's GPU square root
for CADO-NFS; driver API and PTX like msieve's other GPU code, with the same PTX check as the lattice
siever). `MSIEVE_SQRT_GPU_CHECK=1` compared all 38,688 products of a c120 square root (threshold 512) with
GMP: 0 differences. But the products are only part of the time; the c120 (`-nc3`, 8 dependencies):

| Phase | GMP | GPU products |
| --- | ---: | ---: |
| reading the dependencies' relations | 8 s | 8 s |
| the product of the relations | 11 s | 9 s |
| Newton iteration, to the first answer | 14 s | 11 s |
| square root | 33.7 s | 29.1 s (thresholds 512 to 32,768 limbs: 29.0–30.7 s) |

The rest is the many small polynomial products and reductions on the CPU threads. Larger inputs should
gain more, their large products being a larger share (not measured).

**Free relations on all CPUs (Nov 10).** Filtering looks for free relations (primes p modulo which both
polynomials split into linear factors) one prime at a time: 6 s of the c120's 37 s of filtering.
`poly_get_zeros` with `count_only` only reads the polynomials, so the candidates are now tested on all
CPUs (`MSIEVE_FILTER_THREADS`) and the relations written in order, as before: the same 61,945 free
relations, the same matrix (517,393 cycles, the same weight); filtering of the c120's relations 45.5 →
38.2 s.

**c130 and c140 (Nov 10)**, from scratch with everything above (`./msieve`, defaults), RTX PRO 4500 and
32 CPU threads; both factored correctly (p65 × p65, p70 × p70):

| Stage | c120 | c130 | c140 |
| --- | ---: | ---: | ---: |
| Polynomial selection (1/128 of msieve's limits) | 1.4 min | 5.2 min | 19.5 min |
| Sieving | 380 s | 1,515 s | 4,437 s |
| Filtering | ~31 s | 61 s | 148 s |
| Linear algebra (matrix columns) | 34 s (517 k) | 144 s (1.32 M) | 578 s (2.96 M) |
| Square root | ~29 s | 85 s | 233 s |
| **Total** | **9.8 min** | **35.4 min** | **109.6 min** |
| Peak memory | 7.2 GB | 7.5 GB | 13.2 GB |

- The c140 is the number the CADO-NFS GPU fork factors in 75.9 min on the same machine (its sieving 58 min,
  polynomial selection 8.8 min, linear algebra 3.7 min): at 140 digits that fork is the faster one, and its
  lead is in sieving and linear algebra.
- Polynomial quality holds at 1/128: scored with CADO-NFS's Murphy E (its c140 bounds and area), msieve's c140
  polynomial is 1.181e-06 against CADO-NFS's 1.196e-06 (1.3% lower). But selection now takes 19.5 min at
  140 digits (msieve's limits grow about 4 times every 10 digits); whether less would do there was not
  measured.
- The square root's GPU products (`-nc3`, same conditions): c130 109 → 90 s, c140 296 → 245 s, 17% at both
  (c120: 14%).

### RSA-2048: an extrapolated estimate

**This is an extrapolation, an order of magnitude at best.** RSA-2048 (617 digits, 2048 bits) cannot be
factored with this fork, CADO-NFS, or any existing hardware; the numbers below say how far out of reach
it is. The method: the number field sieve's heuristic cost
L(N) = exp((64/9)^(1/3) (ln N)^(1/3) (ln ln N)^(2/3)) (the o(1) term ignored), scaled from two
reference points.

- **The record**: RSA-250 (829 bits), factored by the CADO-NFS team in 2020 with about 2,700
  CPU-core-years (Intel Xeon Gold 6130, 2.1 GHz; taken here as about 2,450 sieving and 250 linear
  algebra), **with no GPU**, and a matrix of about 405 million rows.
- **This machine**: the c120 above, CADO-NFS on 32 CPU threads against this fork on one RTX PRO 4500,
  which gives the conversion (sieving: 1 GPU ≈ 25.6 CPU threads; linear algebra: 1 GPU ≈ 150 threads).
  An i9-13900K hardware thread is taken as one Xeon core; that is uncertain within a factor of two or so.

| Quantity | Estimate |
| --- | --- |
| L(RSA-2048) / L(RSA-250) | 2.3 × 10¹¹ |
| L(RSA-2048) / L(c120) | 1.0 × 10¹⁸ |
| RSA-250 on RTX PRO 4500-class GPUs (at the c120 conversion) | about 97 GPU-years (sieving 96, linear algebra 1.6) |
| RSA-2048, CPU | about 6 × 10¹⁴ core-years |
| RSA-2048, RTX PRO 4500-class GPUs, scaled from RSA-250 | about 2.3 × 10¹³ GPU-years |
| RSA-2048, the same, scaled from this fork's own c120 (sieving + linear algebra) | about 2.3 × 10¹³ GPU-years |
| with 10⁴ / 10⁶ / 10⁸ GPUs, perfectly parallel | 2 × 10⁹ / 2 × 10⁷ / 2 × 10⁵ years |
| matrix dimension (√L scaling, from RSA-250 / from the c120) | 2 × 10¹⁴ to 6.5 × 10¹⁴ rows |
| matrix memory (about 110 entries per row, 4 bytes each) | about 0.09 to 0.3 exabytes |
| 96 GB GPUs just to hold the matrix | about 10⁶ |
| relations (√L scaling, from RSA-250's ~9 × 10⁹ / the c120's 1.6 × 10⁷) | about 4 × 10¹⁵ to 1.7 × 10¹⁶ |

The two time estimates agree, but that only says the c120 conversion is consistent with the record,
not that L's asymptotic form holds over another 1,200 bits: the √L rule for the matrix already misses
the c120 by a factor of 3.3 (scaled down from RSA-250 it predicts 191,000 rows, against 635,213). The
conclusion does not depend on that error: RSA-2048 costs about 10¹¹ times RSA-250, which itself took
thousands of core-years on a large cluster; with a hundred million GPUs it would still take some 10⁵
years, and its matrix alone would need about a million 96 GB GPUs.

**Open points**: whether less than 1/128 of the polynomial limits would do at 140 digits and above
(19.5 min of the c140's 110), and whether a predictor better than the size score exists for choosing what
to root-optimize;
the MPQS/NFS crossover below 85 digits (msieve's NFS parameters start at 80); the bound of 16, untested on whole runs; sizes other than the c95 untested
(I = 4096 and msieve's line-siever parameters throughout); msieve still uses its quadratic sieve
without `-n`, for any size.

## License

msieve is public domain (see the header of each file), and so are the changes this fork makes to
msieve's own files. The files this fork adds are under the **MIT license**, Copyright (c) 2026
Stefan Teleman (the text is in each file's header): `gnfs/poly/stage1/stage1_sizeopt_gpu.{c,h}`,
`gnfs/poly/stage1/stage1_core_gpu/sizeopt_core.{cu,h}`, `gnfs/poly/stage2/root_sieve_gpu.{c,h}`,
`gnfs/poly/stage2/root_sieve_batch.c`, `gnfs/poly/stage2/line_sieve_core.h`,
`gnfs/poly/stage2/murphy_core.h`, `gnfs/poly/stage2/alpha_core.h`,
`gnfs/poly/stage2/lattice_core.h`, `common/lanczos/gpu/lanczos_core.cu`,
`common/lanczos/gpu/lanczos_gpu.{c,h}`, `common/lanczos/lanczos_dispatch.c`, `test/lanczos/*`,
`test/sieve/relcheck.py`, `test/sieve/batchreplay.c`, `gnfs/filter/relpar.{c,h}`, `gnfs/relstore.{c,h}`,
`gnfs/sieve/sieve_core.{cu,h}`, `gnfs/sieve/sieve_line_gpu.c`, `gnfs/sieve/lasieve_core.{cu,h}`,
`gnfs/sieve/sieve_lattice_gpu.c`, `gnfs/sqrt/sqrt_gpu_mul.{c,h}` and `gnfs/sqrt/sqrtmul_core.{cu,h}`
(ported from the author's own GPU square root for CADO-NFS), as well as this
README and the design document. The kernels in `sizeopt_core.cu` and the batched root sieve port
msieve's own public-domain routines (`minimize()`, `optimize_initial()`, `pol_expand()`,
`compute_line_size()`, the line sieve, `analyze_poly_murphy()`, `dickman()`, the double-exponential
integrator, `analyze_poly_roots()`, `poly_get_zeros()` and others). `gnfs/sieve/sieve_line_int.h` holds
`sieve_line.c`'s structures, moved there, and stays public domain. CGBN is MIT-licensed.
