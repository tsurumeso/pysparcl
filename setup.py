from Cython.Distutils import build_ext
from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension('pysparcl.core',
              sources=['pysparcl/core.pyx'],
              include_dirs=[numpy.get_include()])
]

setup(
    name='pysparcl',
    version='1.0.0',
    author='tsurumeso',
    license='GPL-2.0 License',
    packages=['pysparcl'],
    ext_modules=ext_modules,
)
