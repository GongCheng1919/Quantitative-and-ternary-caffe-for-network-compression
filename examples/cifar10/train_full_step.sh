#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
./build/tools/caffe train \
    --solver=./examples/cifar10/cifar10_full_solver_compress_fc.prototxt \
    --snapshot=./examples/cifar10/model/cifar10_full_iter_100000.solverstate 

./build/tools/caffe train \
    --solver=./examples/cifar10/cifar10_full_solver_compress_conv3.prototxt \
    --snapshot=./examples/cifar10/model/cifar10_full_compress_iter_110000.solverstate 

./build/tools/caffe train \
    --solver=./examples/cifar10/cifar10_full_solver_compress_conv2.prototxt \
    --snapshot=./examples/cifar10/model/cifar10_full_compress_iter_120000.solverstate 

./build/tools/caffe train \
    --solver=./examples/cifar10/cifar10_full_solver_compress_conv1.prototxt \
    --snapshot=./examples/cifar10/model/cifar10_full_compress_iter_130000.solverstate $@
