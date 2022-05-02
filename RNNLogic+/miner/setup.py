from setuptools import setup
from torch.utils import cpp_extension as cpp

setup(
    name='rnnlogic_ext',
    ext_modules=[
        cpp.CppExtension('rnnlogic_ext', ["rnnlogic.cpp", "pyrnnlogic.cpp"],
                         extra_compile_args={"cxx": ["-O3"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)
