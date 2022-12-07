# -*- coding: utf-8 -*-

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "lines",
        ["lines.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='lines',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
