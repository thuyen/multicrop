#include <torch/torch.h>

at::Tensor crop2d_cpu(
    const at::Tensor &X, // 3d image hwc
    const at::Tensor &R, // boxes
    int pooled_height, int pooled_width,
    int stride=1, bool first=false
    );

at::Tensor crop3d_cpu(
    const at::Tensor &X, // 4d image thwc
    const at::Tensor &R, // boxes
    int pooled_length, int pooled_height, int pooled_width,
    int stride=1, bool first=false
    );

at::Tensor crop2d(
    const at::Tensor &X, // 3d image hwc
    const at::Tensor &R, // boxes
    int pooled_height, int pooled_width,
    int stride=1, bool first=false
    );

at::Tensor crop3d(
    const at::Tensor &X, // 4d image thwc
    const at::Tensor &R, // boxes
    int pooled_length, int pooled_height, int pooled_width,
    int stride=1, bool first=false
    );

