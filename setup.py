
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)

# THIS IS THE ORIGINAL CODE
# from distutils.core import setup
# from distutils.extension import Extension
#
# from Cython.Build import cythonize
# import numpy
#
# extensions = [
#   Extension('im2col_cython', ['im2col_cython.pyx'],
#             include_dirs = [numpy.get_include()]
#   ),
# ]
#
# setup(
#     ext_modules = cythonize(extensions),
# )
