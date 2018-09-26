export LD_LIBRARY_PATH=/home/gongcheng/lab/caffe/build/lib:$LD_LIBRARY_PATH 
g++ -o RprotoW RprotoW.cpp -I ../include -D CPU_ONLY -I ../.build_realse/src/ -L ../build/lib -lcaffe -lglog -lboost_system -lprotobuf
./RprotoW ../examples/mnist/model/lenet_iter_10000.caffemodel ./lenet_iter_10000.txt
