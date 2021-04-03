from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform

extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = ['-lGLEW', '-lglut']

if platform.system() == 'Darwin':
  extra_link_args.append('-F/System/Library/Frameworks')
  extra_compile_args.append('-stdlib=libc++')
  extra_link_args.append('-stdlib=libc++')
else:
  extra_link_args.append('-lGL')
  extra_link_args.append('-lGLU')
  extra_link_args.append('-fopenmp')
  extra_compile_args.append('-fopenmp')

setup(
  name="pyrender",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('pyrender',
      ['pyrender.pyx',
       'offscreen.cpp',
      ],
      language='c++',
      include_dirs=[np.get_include(),],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    )
  ]
)


