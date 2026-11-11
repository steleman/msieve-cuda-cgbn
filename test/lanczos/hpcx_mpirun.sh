#!/bin/bash
# mpirun from HPC-X (CUDA-aware OpenMPI of the NVIDIA HPC SDK); adjust the path
source /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/hpcx-init.sh
hpcx_load
exec mpirun "$@"
