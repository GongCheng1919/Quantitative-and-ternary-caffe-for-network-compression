#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {
/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();
	  //std::cout<<"Layer "<<param.name()<<" has "<< param.weights_compress_size() <<" params and has" <<blobs_.size()<<" blobs and "<<layer_param_.param_size()<<std::endl;
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }
	  if(param.activations_compress_param_size()>0){
		  LOG(INFO)<<"activations_compress_param info ("<<param.activations_compress_param_size()<<")\n"
					<<"delta="<<param.activations_compress_param(0).delta()
					<<",alpha="<<param.activations_compress_param(0).alpha()
					<<",fixedpos="<<param.activations_compress_param(0).fixedpos()
					<<",maxbits="<<param.activations_compress_param(0).maxbits();
	  }
	  int max_weights_compress_num=(layer_param_.param_size()>=layer_param_.weights_compress_size()?layer_param_.param_size():layer_param_.weights_compress_size());
	  if(max_weights_compress_num>0){
		  //GC add weights_compress_
		weights_compress_.resize(max_weights_compress_num,"");
		for (int i = 0; i < max_weights_compress_num; ++i){
			//GC add weights_compress_
		  if(i<layer_param_.weights_compress_size() && (layer_param_.weights_compress(i)=="Ternary"
		  ||layer_param_.weights_compress(i)=="Quantize"
		  ||layer_param_.weights_compress(i)=="Ternary_Quantize")){
			  weights_compress_[i]=layer_param_.weights_compress(i);
			  std::cout<<"Layer "<<param.name()<<"> params "<<i<<": USE "<<weights_compress_[i]<<" compression weight"<<std::endl;
		  }else{
			std::cout<<"Layer "<<param.name()<<"> params "<<i<<": USE full precision weight"<<std::endl;
			weights_compress_[i]="";
		  }
		}
	  }
	  //add activations_compress_
	  int activations_compress_num=layer_param_.activations_compress_size();
	  if(activations_compress_num>0){
		  //GC add weights_compress_
		activations_compress_.resize(activations_compress_num,"");
		for (int i = 0; i < activations_compress_num; ++i){
			//GC add weights_compress_
		  if((layer_param_.activations_compress(i)=="Ternary"
		  ||layer_param_.activations_compress(i)=="Quantize"
		  ||layer_param_.activations_compress(i)=="Ternary_Quantize")){
			  activations_compress_[i]=layer_param_.activations_compress(i);
			  std::cout<<"Layer "<<param.name()<<"> activetion "<<i<<": USE "<<activations_compress_[i]<<" compression weight"<<std::endl;
		  }else{
			std::cout<<"Layer "<<param.name()<<"> activetion "<<i<<": USE full precision weight"<<std::endl;
			activations_compress_[i]="";
		  }
		}
	  }
    }
  virtual ~Layer() {}

  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }
  LayerParameter* mutable_layer_param() { return &layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }
	/*
  压缩方法
  */
  void CompressLayerWeights(){
	//std::cout<<"blobs_ size is "<<blobs_.size()<<" for Layer "<<layer_param_.name()<<std::endl;
	//LOG(INFO)<<"weights Compress in "<<layer_param_.name()<<" Phase="<<phase_<<"("<<caffe::TRAIN<<","<<caffe::TEST<<")"
	//<<" weights_compress_.size()="<<weights_compress_.size()
	//<<" layer_param_weights_compress_size()"<<layer_param_.weights_compress_size()
	//<<" blobs_.size="<<blobs_.size();
	for(int i=0;i<blobs_.size();i++){
		//std::cout<<" weights_compress["<<i<<"]="<<weights_compress_[i]<<std::endl;
		bool has_compress_param=true;
		if(i<weights_compress_.size() &&weights_compress_[i]=="Ternary"){
			//std::cout<<"weights_compress_param_size "<<layer_param_.weights_compress_param_size()<<" for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.weights_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved weights compression configrations for "<< weights_compress_[i]<<" (" << i+1<<">"<<layer_param_.weights_compress_param_size()<<")";
				}
				has_compress_param=false;
				layer_param_.add_weights_compress_param();
				
			}
			//std::cout<<"start ternary compress\n";
			//std::cout<<"weights_compress_param_size "<<layer_param_.weights_compress_param_size()<<" for Layer "<<layer_param_.name()<<std::endl;
			blobs_[i]->ternarize_data(phase_,false,layer_param_.weights_compress_param(i),"weights");
			if(phase_==TRAIN || !has_compress_param){
				layer_param_.mutable_weights_compress_param(i)->set_delta(blobs_[i]->get_delta());
				layer_param_.mutable_weights_compress_param(i)->set_alpha(blobs_[i]->get_alpha()/pow(2,blobs_[i]->get_fixedpos()));
				layer_param_.mutable_weights_compress_param(i)->set_fixedpos(blobs_[i]->get_fixedpos());
				layer_param_.mutable_weights_compress_param(i)->set_maxbits(blobs_[i]->get_maxbits());
								//std::cout<<"Layer "<<layer_param_.name()<<" "<<layer_param_.weights_compress_param(i).delta()<<" | "<<
					//layer_param_.weights_compress_param(i).alpha()<<" | "<<
					//layer_param_.weights_compress_param(i).fixedpos()<<" | "<<
					//layer_param_.weights_compress_param(i).maxbits()<<" | "<<std::endl;
				//std::cout<<"end ternary compress\n";
			}
		}
		if(i<weights_compress_.size() &&weights_compress_[i]=="Ternary_Quantize"){
			//std::cout<<weights_compress_[i]<<" weights_compress for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.weights_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved weights compression configrations for "<< weights_compress_[i]<<" (" << i+1<<">"<<layer_param_.weights_compress_param_size()<<")";
				}
				has_compress_param=false;
				layer_param_.add_weights_compress_param();
				
			}
			//std::cout<<"start ternary quantize compress\n";
			blobs_[i]->ternarize_data(phase_,true,layer_param_.weights_compress_param(i),"weights");
			if(phase_==TRAIN||!has_compress_param){
				layer_param_.mutable_weights_compress_param(i)->set_delta(blobs_[i]->get_delta());
				layer_param_.mutable_weights_compress_param(i)->set_alpha(blobs_[i]->get_alpha()/pow(2,blobs_[i]->get_fixedpos()));
				layer_param_.mutable_weights_compress_param(i)->set_fixedpos(blobs_[i]->get_fixedpos());
				layer_param_.mutable_weights_compress_param(i)->set_maxbits(blobs_[i]->get_maxbits());
								//std::cout<<"Layer "<<layer_param_.name()<<" "<<layer_param_.weights_compress_param(i).delta()<<" | "<<
					//layer_param_.weights_compress_param(i).alpha()<<" | "<<
					//layer_param_.weights_compress_param(i).fixedpos()<<" | "<<
					//layer_param_.weights_compress_param(i).maxbits()<<" | "<<std::endl;
				//std::cout<<"end ternary quantize compress\n";
			}
		}
		if(i<weights_compress_.size() &&weights_compress_[i]=="Quantize"){
			//std::cout<<weights_compress_[i]<<" weights_compress for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.weights_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved weights compression configrations for "<< weights_compress_[i]<<" (" << i+1<<">"<<layer_param_.weights_compress_param_size()<<")";
				}
				has_compress_param=false;
				layer_param_.add_weights_compress_param();
				
			}
			blobs_[i]->quantize_data(phase_,layer_param_.weights_compress_param(i),"weights");
			if(phase_==TRAIN || !has_compress_param){
				layer_param_.mutable_weights_compress_param(i)->set_delta(blobs_[i]->get_delta());
				layer_param_.mutable_weights_compress_param(i)->set_alpha(blobs_[i]->get_alpha()/pow(2,blobs_[i]->get_fixedpos()));
				layer_param_.mutable_weights_compress_param(i)->set_fixedpos(blobs_[i]->get_fixedpos());
				layer_param_.mutable_weights_compress_param(i)->set_maxbits(blobs_[i]->get_maxbits());
			}
		}
	}
  }
  /*
  切换计算权值为压缩权值
  */
  void ExchangeCompressWeights(){
	for(int i=0;i<blobs_.size();i++){
		if(i<weights_compress_.size() && (weights_compress_[i]=="Ternary"
		||weights_compress_[i]=="Ternary_Quantize")){
			//std::cout<<"Exchange "<<weights_compress_[i]<<" weights_compress weights instead of full weights for Layer "<<layer_param_.name()<<std::endl;
			blobs_[i]->exchange_data_ternary(true);
		}
		if(i<weights_compress_.size() && weights_compress_[i]=="Quantize"){
			//std::cout<<"Exchange "<<weights_compress_[i]<<" weights_compress weights instead of full weights for Layer "<<layer_param_.name()<<std::endl;
			blobs_[i]->exchange_data_quantize(true);
		}
	}
  }
  /*
  切换计算权值为全精度权值
  */
  void ExchangeFullWeights(){
	for(int i=0;i<blobs_.size();i++){
		blobs_[i]->exchange_data_ternary(false);
		blobs_[i]->exchange_data_quantize(false);
	}
  }

  //压缩激活值
  //note:激活值压缩必须是在前向传播之后，即获取了输出值，其用压缩值替换前向输出和反向计算梯度
  void CompressLayerActivations(const vector<Blob<Dtype>*>& top){
	for(int i=0;i<top.size();i++){
		bool has_compress_param=true;
		if(i<activations_compress_.size() &&activations_compress_[i]=="Ternary"){
			//std::cout<<weights_compress_[i]<<" weights_compress for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.activations_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved activations compression configrations for "<< activations_compress_[i]<<" (" << i+1<<">"<<layer_param_.activations_compress_param_size()<<") "
					<<" and set the default param for Test!";
				}
				has_compress_param=false;
				layer_param_.add_activations_compress_param();
				//layer_param_.add_activations_compress_param();
			}
			top[i]->ternarize_data(phase_,false,layer_param_.activations_compress_param(i),"activations");
			if(phase_==TRAIN || !has_compress_param){
				layer_param_.mutable_activations_compress_param(i)->set_delta(top[i]->get_delta());
				layer_param_.mutable_activations_compress_param(i)->set_alpha(top[i]->get_alpha()/pow(2,top[i]->get_fixedpos()));
				layer_param_.mutable_activations_compress_param(i)->set_fixedpos(top[i]->get_fixedpos());
				layer_param_.mutable_activations_compress_param(i)->set_maxbits(top[i]->get_maxbits());
			}
		}
		if(i<activations_compress_.size() &&activations_compress_[i]=="Ternary_Quantize"){
			//std::cout<<weights_compress_[i]<<" weights_compress for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.activations_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved activations compression configrations for "<< activations_compress_[i]<<" (" << i+1<<">"<<layer_param_.activations_compress_param_size()<<")"
					<<" and set the default param for Test!";
				}
				has_compress_param=false;
				layer_param_.add_activations_compress_param();
				
			}
			top[i]->ternarize_data(phase_,true,layer_param_.activations_compress_param(i),"activations");
			if(phase_==TRAIN || !has_compress_param){
				layer_param_.mutable_activations_compress_param(i)->set_delta(top[i]->get_delta());
				layer_param_.mutable_activations_compress_param(i)->set_alpha(top[i]->get_alpha()/pow(2,top[i]->get_fixedpos()));
				layer_param_.mutable_activations_compress_param(i)->set_fixedpos(top[i]->get_fixedpos());
				layer_param_.mutable_activations_compress_param(i)->set_maxbits(top[i]->get_maxbits());
			}
		}
		if(i<activations_compress_.size() &&activations_compress_[i]=="Quantize"){
			//std::cout<<weights_compress_[i]<<" weights_compress for Layer "<<layer_param_.name()<<std::endl;
			if(i>=layer_param_.activations_compress_param_size()){
				if(phase_==TEST){
					LOG(INFO) << "Cannot Find saved activations compression configrations for "<< activations_compress_[i]<<" (" << i+1<<">"<<layer_param_.activations_compress_param_size()<<")"
					<<" and set the default param for Test!";
				}
				has_compress_param=false;
				layer_param_.add_activations_compress_param();
		
			}
			top[i]->quantize_data(phase_,layer_param_.activations_compress_param(i),"activations");
			if(phase_==TRAIN || !has_compress_param){
				layer_param_.mutable_activations_compress_param(i)->set_delta(top[i]->get_delta());
				layer_param_.mutable_activations_compress_param(i)->set_alpha(top[i]->get_alpha()/pow(2,top[i]->get_fixedpos()));
				layer_param_.mutable_activations_compress_param(i)->set_fixedpos(top[i]->get_fixedpos());
				layer_param_.mutable_activations_compress_param(i)->set_maxbits(top[i]->get_maxbits());
			}
		}
	}
  }
  /*
  切换计算权值为压缩权值
  */
  void ExchangeCompressActivations(const vector<Blob<Dtype>*>& top){
	for(int i=0;i<top.size();i++){
		if(i<activations_compress_.size() && (activations_compress_[i]=="Ternary"
		||activations_compress_[i]=="Ternary_Quantize")){
			//std::cout<<"Exchange "<<weights_compress_[i]<<" weights_compress weights instead of full weights for Layer "<<layer_param_.name()<<std::endl;
			top[i]->exchange_data_ternary(true);
		}
		if(i<activations_compress_.size() && activations_compress_[i]=="Quantize"){
			//std::cout<<"Exchange "<<weights_compress_[i]<<" weights_compress weights instead of full weights for Layer "<<layer_param_.name()<<std::endl;
			top[i]->exchange_data_quantize(true);
		}
	}
  }
  /*
  切换计算权值为全精度权值
  */
  void ExchangeFullActivations(const vector<Blob<Dtype>*>& top){
	for(int i=0;i<top.size();i++){
		top[i]->exchange_data_ternary(false);
		top[i]->exchange_data_quantize(false);
	}
  }


 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> weights_compress_;
  vector<string> activations_compress_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;

  /*三值化参数，即delta和alpha*/


  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//LOG(INFO)<<" Get into layer "<<layer_param_.name()<<"'s Forward!";
	  //在Forward之前，对权值做三值化:GC
	CompressLayerWeights();
	  //切换使用三值权值代替全精度权值进行前向传播
	ExchangeCompressWeights();
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
  //GC
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
	//切换使用全精度权值
	ExchangeFullWeights();
	//对激活值做压缩
	CompressLayerActivations(top);
	//切换激活值为压缩表示
	ExchangeCompressActivations(top);
	if(phase_==TEST){
		static Dtype topbit=0,bottombit=1000000;
		Dtype ttopbit=topbit,tbottombit=bottombit;
		//计算top中绝对最大值和绝对最小值。
		for (int i=0;i<top.size();i++){
			//找到最大值
			const Dtype* temp= top[i]->cpu_data();
			for(int j=0;j<top[i]->count();j++){
				Dtype bitwidth=temp[j]>0?temp[j]:-temp[j];
				if(bitwidth>ttopbit){
					ttopbit=bitwidth;
				}
				if(bitwidth<tbottombit){
					tbottombit=bitwidth;
				}
			}
		}
		if(ttopbit>topbit){
			topbit=ttopbit;
			std::cout<<layer_param_.name()<<" Top abs max is "<<topbit<<std::endl;
		}
		if(tbottombit<bottombit){
			bottombit=tbottombit;
			std::cout<<layer_param_.name()<<" Top abs min is "<<bottombit<<std::endl;
		}
	}
  return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//切换使用三值权值代替全精度权值进行反向传播
	ExchangeCompressWeights();
	//切换激活值为压缩表示
	ExchangeCompressActivations(top);
	
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
    //切换使用全精度权值进行权值更新
	ExchangeFullWeights();
	  //切换使用全精度激活值进行权值更新：事实上并没有必要，
	  //因为已经使用了压缩top计算反向梯度，但是为了后续批次能够依次以全精度->压缩值的流程，可以考虑打开
	ExchangeFullActivations(top);
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  //如果需要三值化，则对最新的参数三值化，然后写入文本
  CompressLayerWeights();
  //如果有权值需要三值化，则只存储三值权值
  //ExchangeCompressWeights();
  //还原，保存全精度，同时也保存压缩的权值
  ExchangeFullWeights();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
  //切换使用全精度权值进行训练
  //ExchangeFullWeights();
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
