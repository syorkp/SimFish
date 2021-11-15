# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("sector_sum.pyx"),
    include_dirs=[numpy.get_include()]
)
