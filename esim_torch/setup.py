from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='esim_cuda',
    ext_modules=[
        CUDAExtension(name='esim_cuda',
                      sources=[
                      'esim_cuda_kernel.cu',
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
