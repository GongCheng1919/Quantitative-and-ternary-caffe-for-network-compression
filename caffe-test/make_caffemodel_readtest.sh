g++ -o caffemodel_readtest caffemodel_readtest.cpp -I ../include -D CPU_ONLY -I ../.build_realse/src/ -L ../build/lib -lcaffe -lglog -lboost_system -lprotobuf
./caffemodel_readtest ../examples/mnist/model/lenet_iter_10000.caffemodel
