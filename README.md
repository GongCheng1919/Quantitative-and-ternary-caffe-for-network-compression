# Quantitative-and-ternary-caffe-for-network-compression
Quantitative and ternary caffe for network compression.

# Compiler
You need to change the configuration file (Makeconfig) according to your platform.
# Use
Once the compilation is complete, you can add the specified compression parameters to each layer to indicate the compression operations the layer needs to perform:

### weights_compress: 
 Optional parameter: ""(default),"Ternary","Ternary_Quantize","Quantize"
### weights_caompress_param: 
 Optional parameter:  
 
 delta: Threshold of the step function (for ternary)
 
 alpha: Scaling factor (for ternary)
 
 fixedpos: Fixed position (for quantize)
 
 maxbits: Store the maximum number of quantized integers (for quantize)
 
 ### activations_compress: 
 Optional parameter: Same as weights_compress
 ### activations_compress_param:
 Optional parameter: Same as weights_compress_param
