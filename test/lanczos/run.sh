#!/bin/bash
# MIT License, Copyright (c) 2026 Stefan Teleman (full text: common/lanczos/gpu/lanczos_gpu.c)
# run.sh <matrix> <name> <env...>: msieve -nc2 on <matrix>.mat, checked with depcheck
mat=$1; name=$2; shift 2
N=$(head -1 t.fb | cut -d' ' -f2)
rm -f $name.*; ln -sf $mat.mat $name.mat; [ -f $mat.mat.idx ] && ln -sf $mat.mat.idx $name.mat.idx
[ -f $mat.cyc ] && ln -sf $mat.cyc $name.cyc
/usr/bin/time -f "$name: %e s wall, %M KB" env "$@" ${MPIRUN} ./msieve${BIN} -nf t.fb -s $name -l $name.log -nc2 "skip_matbuild=1 $NCARGS" -t ${THREADS:-16} $N > $name.out 2> $name.time
tail -1 $name.time
cat $name.log* 2>/dev/null | grep -h "linear algebra on\|lanczos halted\|recovered\|error\|BLanczosTime\|memory use" | sed 's/.*2026  //' | sort -u
./depcheck $name
