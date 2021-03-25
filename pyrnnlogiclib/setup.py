from distutils.core import setup, Extension

# define the extension module
pyrnnlogic = Extension('pyrnnlogic',
                       sources=['rnnlogic.cpp', 'pyrnnlogic.cpp'],
                       extra_compile_args=['-lm -O3 -ffast-math'])

# run the setup
setup(ext_modules=[pyrnnlogic])
