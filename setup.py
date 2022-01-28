from setuptools import dist
from setuptools import Extension
from setuptools import setup

dist.Distribution().fetch_build_eggs([
    'cython',
    'numpy'
])

import numpy  # NOQA


ext_modules = [
    Extension(
        'pysparcl.distfun',
        sources=['pysparcl/distfun.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='pysparcl',
    version='1.4.2',
    author='tsurumeso',
    license='GPL-2.0 License',
    packages=['pysparcl'],
    install_requires=[
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
    ext_modules=ext_modules,
)
