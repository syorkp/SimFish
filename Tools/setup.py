# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:31:27 2020

@author: azylb
"""
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ray_cast.pyx"),
    include_dirs=[numpy.get_include()]
)
