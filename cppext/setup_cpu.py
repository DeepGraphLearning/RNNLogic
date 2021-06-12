from setuptools import setup
from torch.utils import cpp_extension as cpp

setup(
    name='groundings_cppext',
    ext_modules=[
        cpp.CppExtension('groundings_cppext', ["groundings.cpp"],
                         extra_compile_args={"cxx": ["-O3"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)

setup(
    name='rule_sample_cppext',
    ext_modules=[
        cpp.CppExtension('rule_sample_cppext', ["rule_sample.cpp"],
                         extra_compile_args={"cxx": ["-O3", "-std=c++17"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)

setup(
    name='matching_cppext',
    ext_modules=[
        cpp.CppExtension('matching_cppext', ["matching.cpp"],
                         extra_compile_args={"cxx": ["-O3"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)
