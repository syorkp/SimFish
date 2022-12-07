# -*- coding: utf-8 -*-

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "sector_sum",
        ["sector_sum.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='sector_sum',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
