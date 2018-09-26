#!/usr/bin/env sh
set -e
LOG=examples/mnist/logs/log_tn.log
./build/tools/caffe train --solver=examples/mnist/lenet_tn_solver.prototxt $@ 2>&1 | tee $LOG
