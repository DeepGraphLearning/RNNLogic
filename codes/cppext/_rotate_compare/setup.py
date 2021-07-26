from setuptools import setup
from torch.utils import cpp_extension as cpp

name = 'rotate_compare_cppext'

setup(
    name=name,
    ext_modules=[
        cpp.CUDAExtension(name, ["rotate_compare.cpp", "cuda/rotate_compare.cu"],
                          extra_compile_args={"cxx": [], "nvcc": []})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)
