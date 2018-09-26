#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver.prototxt $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver_lr1.prototxt \
    --snapshot=$MODEL/model/cifar10_full_iter_60000.solverstate $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver_lr2.prototxt \
    --snapshot=$MDOEL/model/cifar10_full_iter_65000.solverstate $@
