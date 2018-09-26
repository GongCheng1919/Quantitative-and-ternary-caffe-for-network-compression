#!/usr/bin/env sh
set -e
LOG=examples/mnist/logs/log_fc_compress.log
./build/tools/caffe train --solver=examples/mnist/lenet_fc_compress_solver.prototxt $@ 2>&1 | tee $LOG
