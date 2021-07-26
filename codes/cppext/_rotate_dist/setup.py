from setuptools import setup
from torch.utils import cpp_extension as cpp

name = 'rotate_dist_cppext'

setup(
    name=name,
    ext_modules=[
        cpp.CUDAExtension(name, ["rotate_dist.cpp", "cuda/rotate_dist.cu"],
                          extra_compile_args={"cxx": [], "nvcc": ["-O3"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)
