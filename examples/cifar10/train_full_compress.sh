#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver.prototxt 

$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver_compress.prototxt 
	--snapshot=$MODEL/model/cifar10_full_iter_100000.solverstate $@
