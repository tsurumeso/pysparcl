import numpy
from Cython.Distutils import build_ext
from setuptools import setup, Extension


ext_modules = [
    Extension(
        'pysparcl.internal',
        sources=['pysparcl/internal.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='pysparcl',
    version='1.2.0',
    author='tsurumeso',
    license='GPL-2.0 License',
    packages=['pysparcl'],
    ext_modules=ext_modules,
)
