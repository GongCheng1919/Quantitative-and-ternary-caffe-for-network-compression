#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
$TOOLS/caffe train \
    --solver=$MODEL/cifar10_full_solver.prototxt $@
