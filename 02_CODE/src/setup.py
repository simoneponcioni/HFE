import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("cython/material_mapping.pyx"),
    include_dirs=[numpy.get_include()],
)
