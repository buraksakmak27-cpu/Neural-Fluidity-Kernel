import os
import sys
from setuptools import setup, find_packages

# Read the README.md content safely for the long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""

# Check for PyTorch and CUDA availability to decide if we compile the AOT CUDA extension.
# Note: Users can still JIT-compile the extension on-the-fly if AOT compilation is skipped.
ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # Check if CUDA is available for AOT compilation
    # Check CUDA_HOME or CUDA_PATH env variables, or check if nvcc is on the path.
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    has_nvcc = (os.system("where nvcc >nul 2>nul") == 0) if sys.platform == 'win32' else (os.system("which nvcc >/dev/null 2>&1") == 0)
    has_cuda = (cuda_home is not None) or has_nvcc or (torch.cuda.is_available())
    
    if has_cuda:
        ext_modules = [
            CUDAExtension(
                name='neural_fluidity.fluid_linear_cuda',
                sources=[
                    'neural_fluidity/cuda/bindings.cpp',
                    'neural_fluidity/cuda/fluid_linear.cu',
                ],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3']
                }
            )
        ]
        cmdclass = {'build_ext': BuildExtension}
    else:
        print("[Neural Fluidity setup] CUDA was not detected. AOT compilation is skipped, "
              "but package will install successfully. JIT compiler or CPU fallback will be used.")
except ImportError:
    print("[Neural Fluidity setup] PyTorch is not pre-installed. AOT CUDA compilation is skipped. "
          "PyTorch will be installed via install_requires.")

setup(
    name='neural-fluidity',
    version='0.1.0',
    description='A hardware-accelerated library for Neural Fluidity Kernel, focusing on dynamic SVD low-rank updates inside PyTorch backward passes to prevent catastrophic forgetting.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Google Deepmind Team & Ali',
    author_email='ali@example.com',
    url='https://github.com/buraksakmak27-cpu/Neural-Fluidity-Kernel',
    project_urls={
        'Source': 'https://github.com/buraksakmak27-cpu/Neural-Fluidity-Kernel',
        'Tracker': 'https://github.com/buraksakmak27-cpu/Neural-Fluidity-Kernel/issues',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks']),
    include_package_data=True,
    package_data={
        'neural_fluidity': [
            'cuda/bindings.cpp',
            'cuda/fluid_linear.cu',
        ],
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.9',
)
