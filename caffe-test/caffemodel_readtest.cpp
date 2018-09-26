#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
  #include "../include/caffe/proto/caffe.pb.h"
 
  using namespace std;
  using namespace caffe;
 
 
  int main(int argc, char* argv[]) 
  { 
 
   caffe::NetParameter msg; 
 
   fstream input(argv[1], ios::in | ios::binary); 
   if (!msg.ParseFromIstream(&input)) 
   { 
     cerr << "Failed to parse address book." << endl; 
     return -1; 
   } 
 
   ::google::protobuf::RepeatedPtrField< LayerParameter >* layer = msg.mutable_layer();
   ::google::protobuf::RepeatedPtrField< LayerParameter >::iterator it = layer->begin();
   for (; it != layer->end(); ++it)
   {
     cout << it->name() << endl;
     cout << it->type() << endl;
     cout << it->convolution_param().weight_filler() << endl;
   } 
 
   return 0;
  }
