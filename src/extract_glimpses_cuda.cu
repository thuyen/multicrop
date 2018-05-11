#include <cuda.h>
#include <cuda_runtime.h>
#include "ATen/ATen.h"

// for cuda::type<scalar_t>;
#include "ATen/cuda/CUDATypeConversion.cuh"
// line 107, 303, 316 for lambda syntax

#include "utils.h"

template <typename T>
__global__ void Crop2DFKernel(
    const int numels,
    const T* image,
    const int16_t * fixs,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int w = i % pooled_width;
    int h = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n =  i / pooled_width / pooled_height / channels;

    const int16_t* pos = fixs + 2*n;
    int row = pos[0] - (pooled_height/2 - h)*stride;
    int col = pos[1] - (pooled_width/2  - w)*stride;

    if (row < 0) row = 0;
    if (row >= height) row = height - 1;

    if (col < 0) col = 0;
    if (col >= width) col = width - 1;

    int j = c * height * width + row * width + col;


    top_data[i] = image[j];
  }
}


template <typename T>
__global__ void Crop2DLKernel(
    const int numels,
    const T* image,
    const int16_t * fixs,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int n =  i / channels / pooled_width / pooled_height;

    const int16_t* pos = fixs + 2*n;
    int row = pos[0] - (pooled_height/2 - h)*stride;
    int col = pos[1] - (pooled_width/2  - w)*stride;

    if (row < 0) row = 0;
    if (row >= height) row = height - 1;

    if (col < 0) col = 0;
    if (col >= width) col = width - 1;

    int j = row * width* channels + col * channels + c;

    top_data[i] = image[j];
  }
}

at::Tensor crop2d_gpu(
    const at::Tensor &X, // 3d image hwc
    const at::Tensor &R, // boxes
    int pooled_height, int pooled_width,
    int stride=1, bool first=false
    ) {

  at::Tensor output;
  int channels, off=0;

  if (X.dim() == 2) {
    channels = 1;
    off = 1;
    output = X.type().zeros(
        {R.size(0), pooled_height, pooled_width});
  } else if (first) {
    channels = X.size(0);
    output = X.type().zeros(
        {R.size(0), channels, pooled_height, pooled_width});
  } else {
    channels = X.size(2);
    output = X.type().zeros(
        {R.size(0), pooled_height, pooled_width, channels});
  }

  const int output_size = output.numel();
  const int threads = 1024;
  const int blocks = (output_size + threads - 1) / threads;

  if (first) {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop2d_cuda", [&] {
        using cuda_scalar_t = at::cuda::type<scalar_t>;
        Crop2DFKernel<cuda_scalar_t>
          <<<blocks, threads>>>(
              output_size,
              X.data<cuda_scalar_t>(),
              R.data<int16_t>(),
              X.size(1-off),
              X.size(2-off),
              channels,
              pooled_height,
              pooled_width,
              stride,
              output.data<cuda_scalar_t>());
          });
  } else {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop2d_cuda", [&] {
        using cuda_scalar_t = at::cuda::type<scalar_t>;
        Crop2DLKernel<cuda_scalar_t>
          <<<blocks, threads>>>(
              output_size,
              X.data<cuda_scalar_t>(),
              R.data<int16_t>(),
              X.size(0),
              X.size(1),
              channels,
              pooled_height,
              pooled_width,
              stride,
              output.data<cuda_scalar_t>());
          });
  }
  return output;
}


template <typename T>
__global__ void Crop3DFKernel(
    const int numels,
    const T* image,
    const int16_t * fixs,
    const int length,
    const int height,
    const int width,
    const int channels,
    const int pooled_length,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int w = i % pooled_width;
    int h = (i / pooled_width) % pooled_height;
    int l = (i / pooled_width / pooled_height) % pooled_length;
    int c = (i / pooled_width / pooled_height / pooled_length) % channels;
    int n = i / pooled_width / pooled_height / pooled_length / channels;

    const int16_t * pos = fixs + 3*n;
    int len = pos[0] - (pooled_length/2 - l)*stride;
    int row = pos[1] - (pooled_height/2 - h)*stride;
    int col = pos[2] - (pooled_width/2  - w)*stride;

    if (len < 0) len = 0;
    if (len >= length) len = length - 1;

    if (row < 0) row = 0;
    if (row >= height) row = height - 1;

    if (col < 0) col = 0;
    if (col >= width) col = width - 1;


    int j = c * length * height * width + len * height * width + row * width + col;

    top_data[i] = image[j];
  }
}

template <typename T>
__global__ void Crop3DLKernel(
    const int numels,
    const T* image,
    const int16_t * fixs,
    const int length,
    const int height,
    const int width,
    const int channels,
    const int pooled_length,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int l = (i / channels / pooled_width / pooled_height) % pooled_length;
    int n =  i / channels / pooled_width / pooled_height / pooled_length;

    const int16_t * pos = fixs + 3*n;
    int len = pos[0] - (pooled_length/2 - l)*stride;
    int row = pos[1] - (pooled_height/2 - h)*stride;
    int col = pos[2] - (pooled_width/2  - w)*stride;

    if (len < 0) len = 0;
    if (len >= length) len = length - 1;

    if (row < 0) row = 0;
    if (row >= height) row = height - 1;

    if (col < 0) col = 0;
    if (col >= width) col = width - 1;


    int j = len*height*width*channels + row * width* channels + col * channels + c;

    top_data[i] = image[j];
  }
}


at::Tensor crop3d_gpu(
    const at::Tensor &X, // 4d image thwc
    const at::Tensor &R, // boxes
    int pooled_length, int pooled_height, int pooled_width,
    int stride=1, bool first=false
    ) {

  at::Tensor output;
  int channels, off=0;
  if (X.dim() == 3) {
    channels = 1;
    off = 1;
    output = X.type().zeros(
        {R.size(0), pooled_length, pooled_height, pooled_width});
  } else if (first) {
    channels = X.size(0);
    output = X.type().zeros(
        {R.size(0), channels, pooled_length, pooled_height, pooled_width});
  } else {
    channels = X.size(3);
    output = X.type().zeros(
        {R.size(0), pooled_length, pooled_height, pooled_width, channels});
  }

  const int output_size = output.numel();
  const int threads = 1024;
  const int blocks = (output_size + threads - 1) / threads;

  if (first) {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop3d_cuda", [&] {
        using cuda_scalar_t = at::cuda::type<scalar_t>;
        Crop3DFKernel<cuda_scalar_t>
          <<<blocks, threads>>>(
              output_size,
              X.data<cuda_scalar_t>(),
              R.data<int16_t>(),
              X.size(1-off),
              X.size(2-off),
              X.size(3-off),
              channels,
              pooled_length,
              pooled_height,
              pooled_width,
              stride,
              output.data<cuda_scalar_t>());
          });
  } else {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop3d_cuda", [&] {
        using cuda_scalar_t = at::cuda::type<scalar_t>;
        Crop3DLKernel<cuda_scalar_t>
          <<<blocks, threads>>>(
              output_size,
              X.data<cuda_scalar_t>(),
              R.data<int16_t>(),
              X.size(0),
              X.size(1),
              X.size(2),
              channels,
              pooled_length,
              pooled_height,
              pooled_width,
              stride,
              output.data<cuda_scalar_t>());
          });
  }

  return output;
}

//void crop2d_gpu(
//    const at::Tensor &X, // 3d image hwc
//    const at::Tensor &R, // boxes
//    int pooled_height, int pooled_width,
//    int stride=1, bool first=false
//    ) {
//  //cuda::type<float> x;
//  AT_DISPATCH_ALL_TYPES(
//      X.type(), "crop2d", [&](){
//      using cuda_scalar_t = at::cuda::type<scalar_t>;
//      //scalar_t s = 1;
//      });
//}
//
//at::Tensor crop2d_gpu(
//    const at::Tensor &X, // 3d image hwc
//    const at::Tensor &R, // boxes
//    int pooled_height, int pooled_width,
//    int stride=1, bool first=false
//    ) {
//  return AT_DISPATCH_ALL_TYPES(
//      X.type(), "crop2d", [&]() -> at::Tensor {
//      return _crop2d_gpu<at::cuda::type<scalar_t>>(
//          X, R, pooled_height, pooled_width, stride, first);
//      });
//}
//
//at::Tensor crop3d_gpu(
//    const at::Tensor &X, // 4d image thwc
//    const at::Tensor &R, // boxes
//    int pooled_length, int pooled_height, int pooled_width,
//    int stride=1, bool first=false
//    ) {
//  return AT_DISPATCH_ALL_TYPES(X.type(), "crop3d", [&]() -> at::Tensor {
//      return _crop3d_gpu<at::cuda::type<scalar_t>>(
//          X, R, pooled_length, pooled_height, pooled_width, stride, first);
//      });
//}
