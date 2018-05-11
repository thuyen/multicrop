//#include <ATen/ATen.h>
//
#include <torch/torch.h>

template <typename T>
void Crop2DF(
    const int numels,
    const T* image,
    const int16_t* fixs,
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
void Crop2DL(
    const int numels,
    const T* image,
    const int16_t* fixs,
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

at::Tensor crop2d_cpu(
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


  if (first) {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop2d", [&] {
        Crop2DF<scalar_t>(
            output.numel(),
            X.data<scalar_t>(),
            R.data<int16_t>(),
            X.size(1-off),
            X.size(2-off),
            channels,
            pooled_height,
            pooled_width,
            stride,
            output.data<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop2d", [&] {
        Crop2DL<scalar_t>(
            output.numel(),
            X.data<scalar_t>(),
            R.data<int16_t>(),
            X.size(0),
            X.size(1),
            channels,
            pooled_height,
            pooled_width,
            stride,
            output.data<scalar_t>());
    });
  }
  return output;
}

at::Tensor crop2d(
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


  if (first) {
    Crop2DF<int16_t>(
        output.numel(),
        X.data<int16_t>(),
        R.data<int16_t>(),
        X.size(1-off),
        X.size(2-off),
        channels,
        pooled_height,
        pooled_width,
        stride,
        output.data<int16_t>());
  } else {
    Crop2DL<int16_t>(
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
  }
  return output;
}


template <typename T>
void Crop3DF(
    const int numels,
    const T* image,
    const int16_t* fixs,
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
    int w = i % pooled_width;
    int h = (i / pooled_width) % pooled_height;
    int l = (i / pooled_width / pooled_height) % pooled_length;
    int c = (i / pooled_width / pooled_height / pooled_length) % channels;
    int n = i / pooled_width / pooled_height / pooled_length / channels;

    const int16_t* pos = fixs + 3*n;
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
void Crop3DL(
    const int numels,
    const T* image,
    const int16_t* fixs,
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

    const int16_t* pos = fixs + 3*n;
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

at::Tensor crop3d_cpu(
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

  if (first) {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop3d", [&] {
        Crop3DF<scalar_t>(
            output.numel(),
            X.data<scalar_t>(),
            R.data<int16_t>(),
            X.size(1-off),
            X.size(2-off),
            X.size(3-off),
            channels,
            pooled_length,
            pooled_height,
            pooled_width,
            stride,
            output.data<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES(X.type(), "crop3d", [&] {
        Crop3DL<scalar_t>(
            output.numel(),
            X.data<scalar_t>(),
            R.data<int16_t>(),
            X.size(0),
            X.size(1),
            X.size(2),
            channels,
            pooled_length,
            pooled_height,
            pooled_width,
            stride,
            output.data<scalar_t>());
    });
  }
  return output;
}

at::Tensor crop3d(
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

  if (first) {
    Crop3DF<int16_t>(
        output.numel(),
        X.data<int16_t>(),
        R.data<int16_t>(),
        X.size(1-off),
        X.size(2-off),
        X.size(3-off),
        channels,
        pooled_length,
        pooled_height,
        pooled_width,
        stride,
        output.data<int16_t>());
  } else {
    Crop3DL<int16_t>(
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
  }

  return output;
}

