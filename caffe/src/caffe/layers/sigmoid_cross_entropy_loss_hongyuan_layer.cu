#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_hongyuan_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdio.h>
namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads, const int num, 
          const Dtype* input_data, const Dtype* target, const Dtype* has_mask, const Dtype* coco_cats, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    const int index = i / 81.0 / 28.0 / 28.0;
    const int coco_index = i / (index + 1.0) / 28.0 / 28.0;
    //LOG(INFO) << "Doing forward now";
    const int coco_value = static_cast<int>(coco_cats[index]);
    const int mask_value = static_cast<int>(has_mask[index]);
    //printf("%d\n", coco_index);
    if (mask_value == 0 || coco_value == 0 || coco_index != coco_value) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;
      //if (target_value == 1){
      //    printf("%f\n", loss[i]);
      //}
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count, const int num, 
    const int ignore_label, const Dtype* target, const Dtype* has_mask, const Dtype* coco_cats, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    const int index = i / 81.0 / 28.0 / 28.0;
    const int coco_index = i / (index + 1.0) / 28.0 / 28.0;
    const int coco_value = static_cast<int>(coco_cats[index]);
    const int mask_value = static_cast<int>(has_mask[index]);
    if (mask_value == 0 || coco_value == 0 || coco_index != coco_value) {
      diff[i] = 0;
    }
  }
}


template <typename Dtype>
void SigmoidCrossEntropyLossHongyuanLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  const Dtype* has_mask = bottom[2]->gpu_data();
  const Dtype* coco_cats = bottom[3]->gpu_data();
  
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, num, input_data, target, has_mask, coco_cats, loss_data,
      has_ignore_label_, ignore_label_, count_data);
  // Only launch another CUDA kernel if we actually need the valid count.
  //caffe_gpu_asum(count, count_data, &valid_count);
//if (normalization_ == LossParameter_NormalizationMode_VALID &&
  //    has_ignore_label_) {
  caffe_gpu_asum(count, count_data, &valid_count);
  //} else {
  //  valid_count = count;
  //}
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
  printf("%f\n", loss / normalizer_);
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

template <typename Dtype>
void SigmoidCrossEntropyLossHongyuanLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    const Dtype* has_mask = bottom[2]->gpu_data();
    const Dtype* coco_cats = bottom[3]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
      // NOLINT_NEXT_LINE(whitespace/operators)
    //if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, num, ignore_label_, target, has_mask, coco_cats, bottom_diff);
    //}
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    //printf("loss is %f\n", loss_weight);

    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossHongyuanLayer);

}  // namespace caffe
