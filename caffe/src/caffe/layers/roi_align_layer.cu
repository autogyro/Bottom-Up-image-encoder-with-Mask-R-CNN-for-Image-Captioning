// ------------------------------------------------------------------
// Project: Mask R-CNN
// File: ROIAlignLayer
// Adopted from roi_pooling_layer.cu (written by Ross Grischik)
// Author: Jasjeet Dhaliwal
// ------------------------------------------------------------------

#include "caffe/mask_rcnn_layers.hpp"
#include <cfloat>
using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::fabs;
using std::cout;

namespace caffe
{

template <typename T>
  inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <>
inline __device__
double gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename T>
__device__ T bilinear_interpolate(
      const T* bottom_data,
      const int height,
      const int width,
      T y,
      T x,
      const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      // empty
      return 0;
    }

    if (y <= 0) {
      y = 0;
    }
    if (x <= 0) {
      x = 0;
    }

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
      y_high = y_low = height - 1;
      y = (T)y_low;
    } else {
      y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
      x_high = x_low = width - 1;
      x = (T)x_low;
    } else {
      x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
  }

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype *bottom_data,
                                const Dtype spatial_scale, const int channels, const int height,
                                const int width, const int pooled_height, const int pooled_width,
                                const Dtype *bottom_rois, Dtype *top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int argmax_index = index * 4;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    //Util Values
    Dtype zero = 0.0, one = 1.0;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w + 1.0, one);
    Dtype roi_height = max(roi_end_h - roi_start_h + 1.0, one);
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    const Dtype* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    const int sampling_ratio = static_cast<int>(spatial_scale);

    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);// e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // Define an empty pooling region to be zero

    const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
    Dtype output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
          const Dtype y = roi_start_h + ph * bin_size_h +
              static_cast<Dtype>(iy + .5f) * bin_size_h /
                  static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const Dtype x = roi_start_w + pw * bin_size_w +
                static_cast<Dtype>(ix + .5f) * bin_size_w /
                    static_cast<Dtype>(roi_bin_grid_w);

            Dtype val = bilinear_interpolate(
                offset_bottom_data, height, width, y, x, index);
            output_val += val;
          }
        }
        output_val /= count;

        top_data[index] = output_val;
    }
    // int maxidx[4];
    // Dtype maxmult[4];
    // //int bottom_offset =  (roi_batch_ind * channels + c) * height * width ;
    // //bottom_data += (roi_batch_ind * channels + c) * height * width;
    // /* Normalization function - normalizes values between -1 and 1.
    // a = -1, b = 1
    // y = f(x) = [[(b - a) (x - roi_start_h)] / [roi_end_h - roi_start_h]] + a
    // x = f^{-1}(y) = [[(f(x) - a)(roi_end_h - roi_end_h)] / (b - a)] + roi_start_h
    // Normalized coordinates of 4 regularly sampled points in the ROI:
    // sn_1 = (-0.5,-0.5)
    // sn_2 = (-0.5,0.5)
    // sn_3 = (0.5,-0.5)
    // sn_4 = (0.5,0.5)
    // // Debugging purposes
    // Dtype x_pos = (((0.5 + 1)*(roi_end_w - roi_start_w))/2.0) + roi_start_w;
    // Dtype x_neg = (((-0.5 + 1)*(roi_end_w - roi_start_w))/2.0) + roi_start_w;
    // Dtype y_pos = (((0.5 + 1)*(roi_end_h - roi_start_h))/2.0) + roi_start_h;
    // Dtype y_neg = (((-0.5 + 1)*(roi_end_h - roi_start_h))/2.0) + roi_start_h;
    // Dtype samples[2] = {x_neg, y_neg, x_neg, y_pos,
    //                     x_pos, y_neg, x_pos, y_pos};
    // */
    //
    // Dtype samples_n[8] = {-0.5, -0.5, -0.5, 0.5,
    //                       0.5, -0.5, 0.5, 0.5};
    // //Holds interpolated values for each sample point
    // Dtype bisampled[4];
    // int counter = 0;
    // Dtype x_smp_n = -2.0, y_smp_n = -2.0, h_idx_n = -2.0, w_idx_n = -2.0;
    // //Bilinearly Interpolate 4 sampled values
    // for (int smp = 0; smp < sizeof(samples_n) / sizeof(*samples_n); smp += 2)
    // {
    //   x_smp_n = samples_n[smp];
    //   y_smp_n = samples_n[smp + 1];
    //
    //   bisampled[smp / 2] = 0.0;
    //   int b_index[4] = {-1, -1, -1, -1}; // -1,-1,-1,-1};
    //   //int b_index_curr[4] = {-1,-1,-1,-1};
    //   Dtype multiplier[4] = {Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)};
    //   //Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)};
    //   counter = 0;
    //   //ceil(hstart)
    //   //floor(hend)
    //   for (int h_idx = ceil(hstart); h_idx <= floor(hend) && h_idx <= height && h_idx >= 0; ++h_idx)
    //   {
    //     for (int w_idx = ceil(wstart); w_idx <= floor(wend) && w_idx <= width && w_idx >= 0; ++w_idx)
    //     {
    //       if (counter < 4)
    //       {
    //         b_index[counter] = ((((roi_batch_ind * channels) + c) * height) + h_idx) * width + w_idx;
    //         //    b_index_curr[counter]= h_idx*width + w_idx;
    //         //Normalize width and height to lie between -1 and 1
    //         h_idx_n = static_cast<Dtype>((static_cast<Dtype>(2) * (static_cast<Dtype>(h_idx) - roi_start_h) / (roi_end_h - roi_start_h)) - 1);
    //         w_idx_n = static_cast<Dtype>((static_cast<Dtype>(2) * (static_cast<Dtype>(w_idx) - roi_start_w) / (roi_end_w - roi_start_w)) - 1);
    //         h_idx_n = min(max(h_idx_n, static_cast<Dtype>(-1.0)), one);
    //         w_idx_n = min(max(w_idx_n, static_cast<Dtype>(-1.0)), one);
    //         multiplier[counter] = max(zero, static_cast<Dtype>(1 - fabs(x_smp_n - w_idx_n))) * max(zero, static_cast<Dtype>(1 - fabs(y_smp_n - h_idx_n)));
    //         //bisampled[smp/2] += multiplier[counter];
    //         bisampled[smp / 2] += bottom_data[b_index[counter]] * multiplier[counter];
    //         ++counter;
    //       }
    //       else
    //       {
    //         goto stop;
    //       }
    //     } //w
    //   }   //h
    // stop:
    //   if (bisampled[smp / 2] > maxvalue)
    //   {
    //     maxvalue = bisampled[smp / 2];
    //     //Using two loops to comply with c++ convention
    //     for (int i = 0; i < 4; ++i)
    //     {
    //       maxidx[i] = b_index[i];
    //       maxmult[i] = multiplier[i];
    //     }
    //   }
    // } //smp
    //Store value in the top blob


}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
{
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *bottom_rois = bottom[1]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  int *argmax_idx = max_pts_.mutable_gpu_data();
  Dtype *argmax_mult = max_mult_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  //Change CAFFE_CUDA_NUM_THREADS to 64
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void ROIAlignBackward(const int nthreads, const T *top_diff,
                                  const int num_rois, const T spatial_scale,
                                 const int channels, const int height, const int width,
                                 const int pooled_height, const int pooled_width, T *bottom_diff,
                                 const T *bottom_rois)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];
    const int sampling_ratio = static_cast<int>(spatial_scale);
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          gpu_atomic_add(
              static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          gpu_atomic_add(
              static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          gpu_atomic_add(
              static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        } // if
      } // ix
    } // iy
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                        const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
  if (!propagate_down[0])
  {
    return;
  }
  const Dtype *bottom_rois = bottom[1]->gpu_data();
  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int *argmax_idx = max_pts_.gpu_data();
  const Dtype *argmax_mult = max_mult_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  // CAFFE_CUDA_NUM_THREADS replaced with 64
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), 16>>>(
      count, top_diff, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

} // namespace caffe
