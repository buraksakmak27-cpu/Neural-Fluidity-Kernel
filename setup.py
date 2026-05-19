import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Retrieve CUDA path if configured
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

# Check if nvcc is executable or CUDA is present
has_cuda = cuda_home is not None or os.system("where nvcc >nul 2>nul") == 0

if not has_cuda:
    print("=" * 80)
    print("WARNING: CUDA compiler (nvcc) or CUDA Toolkit path was not detected.")
    print("AOT compilation will be skipped. You can compile this extension JIT")
    print("at runtime inside a CUDA-enabled Python environment using:")
    print("  demo_cuda.py")
    print("=" * 80)

setup(
    name='fluid_linear_cuda',
    ext_modules=[
        CUDAExtension(
            name='fluid_linear_cuda',
            sources=[
                'neural_fluidity/cuda/bindings.cpp',
                'neural_fluidity/cuda/fluid_linear.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ] if has_cuda else [],
    cmdclass={
        'build_ext': BuildExtension
    } if has_cuda else {}
)
