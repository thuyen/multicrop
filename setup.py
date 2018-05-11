# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = []

#https://stackoverflow.com/questions/45600866/add-c-function-to-existing-python-module-with-pybind11

if torch.cuda.is_available():
    extension = CUDAExtension(
        name='multicrop',
        sources = [
            'src/gpu_ops.cpp',
            'src/extract_glimpses_cuda.cu',
            'src/extract_glimpses.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp'],
                            'nvcc': ['-O2']})
else:
    extension = CppExtension(
        name='multicrop',
        sources = [
            'src/cpu_ops.cpp',
            'src/extract_glimpses.cpp',
        ],
        extra_compile_args={'cxx': ['-g', '-fopenmp']})

ext_modules.append(extension)


setup(
    name='multicrop',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
