import numpy
from setuptools import Extension
from setuptools import setup


ext_modules = [
    Extension(
        'pysparcl.distfun',
        sources=['pysparcl/distfun.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='pysparcl',
    version='1.4.1',
    author='tsurumeso',
    license='GPL-2.0 License',
    packages=['pysparcl'],
    ext_modules=ext_modules,
)
