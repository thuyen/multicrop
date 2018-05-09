#include <torch/torch.h>
//#include <ATen/ATen.h>

template <typename T>
void Crop2D(
    const int numels,
    const T* image,
    const T* fixs,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  int i;
#pragma omp parallel for
  for (i = 0; i < numels; ++i) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int n =  i / channels / pooled_width / pooled_height;

    const T* pos = fixs + 2*n;
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

at::Tensor crop2d(
    const at::Tensor &X, // 3d image hwc
    const at::Tensor &R, // boxes
    int pooled_height, int pooled_width, int stride=1
    ) {

  at::Tensor output;
  int channels = 1;

  if (X.dim() == 2) {
    output = X.type().zeros(
        {R.size(0), pooled_height, pooled_width});
  } else {
    channels = X.size(2);
    output = X.type().zeros(
        {R.size(0), pooled_height, pooled_width, channels});
  }


  Crop2D<int16_t>(
      output.numel(),
      X.data<int16_t>(),
      R.data<int16_t>(),
      X.size(0),
      X.size(1),
      channels,
      pooled_height,
      pooled_width,
      stride,
      output.data<int16_t>());
  return output;
}


template <typename T>
void Crop3D(
    const int numels,
    const T* image,
    const T* fixs,
    const int length,
    const int height,
    const int width,
    const int channels,
    const int pooled_length,
    const int pooled_height,
    const int pooled_width,
    const int stride,
    T* top_data) {
  int i;
#pragma omp parallel for
  for (i = 0; i < numels; ++i) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int l = (i / channels / pooled_width / pooled_height) % pooled_length;
    int n =  i / channels / pooled_width / pooled_height / pooled_length;

    const T* pos = fixs + 3*n;
    int len = pos[0] - (pooled_length/2 - l)*stride;
    int row = pos[1] - (pooled_height/2 - h)*stride;
    int col = pos[2] - (pooled_width/2  - w)*stride;

    if (len < 0) len = 0;
    if (len >= length) col = length - 1;

    if (row < 0) row = 0;
    if (row >= height) row = height - 1;

    if (col < 0) col = 0;
    if (col >= width) col = width - 1;


    int j = len*height*width*channels + row * width* channels + col * channels + c;

    top_data[i] = image[j];
  }
}

at::Tensor crop3d(
    const at::Tensor &X, // 4d image thwc
    const at::Tensor &R, // boxes
    int pooled_length, int pooled_height, int pooled_width,
    int stride=1
    ) {

  at::Tensor output;
  int channels = 1;
  if (X.dim() == 3) {
    output = X.type().zeros(
        {R.size(0), pooled_length, pooled_height, pooled_width});
  } else {
    channels = X.size(3);
    output = X.type().zeros(
        {R.size(0), pooled_length, pooled_height, pooled_width, channels});
  }


  Crop3D<int16_t>(
      output.numel(),
      X.data<int16_t>(),
      R.data<int16_t>(),
      X.size(0),
      X.size(1),
      X.size(2),
      channels,
      pooled_length,
      pooled_height,
      pooled_width,
      stride,
      output.data<int16_t>());
  return output;
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(multicrop, m) {
  m.def("crop2d", &crop2d, "crop2d");
  m.def("crop3d", &crop3d, "crop3d");
}
