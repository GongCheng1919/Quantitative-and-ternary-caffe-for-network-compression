#include <iostream>
#include <vector>
#include <caffe/net.hpp>
#include <caffe/layer.hpp>
using namespace caffe;
using namespace std;
int main(int argn,char* argv[]){
	std::string proto(argv[1]);
	Net<float> nn(proto, caffe::TEST);
	vector<string> bn = nn.blob_names();
	for (int i = 0; i < bn.size(); i++){
		cout << "Blob #" << i << ":" << bn[i] << endl;
	}
	return 0;
}