#include <torch/torch.h>
#include "extract_glimpses.h"

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(multicrop, m) {
  m.def("crop2d", &crop2d, "crop2d");
  m.def("crop3d", &crop3d, "crop3d");
}
