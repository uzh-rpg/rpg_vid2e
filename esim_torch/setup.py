from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='esim_torch',
    package_dir={'':'src'},
    packages=['esim_torch'],
    ext_modules=[
        CUDAExtension(name='esim_cuda',
                      sources=[
                      'src/esim_torch/esim_cuda_kernel.cu',
                      ],
                     # extra_compile_args={
                     #'cxx': ['-g'],
                     #'nvcc': ['-arch=sm_60', '-O3', '-use_fast_math']
                     #}
                     )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
