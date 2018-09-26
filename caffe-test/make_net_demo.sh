g++ -o net_demo net_demo.cpp -I ../include -D CPU_ONLY -I ../.build_realse/src/ -L ../build/lib -lcaffe -lglog -lboost_system -lprotobuf
./net_demo ../examples/mnist/lenet.prototxt
