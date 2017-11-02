import os
import sys
import imp
import numpy

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

try:
    __doc__ = open('README.md').read()
except IOError:
    pass

__file__ = './'
ROOT            = 'paths'
LOCATION        = os.path.abspath(os.path.dirname(__file__))

NAME            = "paths"
VERSION         = "0.1"
AUTHOR          = "Michael Habeck"
EMAIL           = "mhabeck@gwdg.de"
URL             = "http://www.uni-goettingen.de/de/444206.html"
SUMMARY         = "Model evidence from nonequilibrium simulations"
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'scipy', 'csb']

setup(
    name=NAME,
    packages=find_packages(exclude=('tests',)),
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=SUMMARY,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    cmdclass={'build_ext': build_ext},
    ## ext_modules=cythonize("cg/*.pyx"), 
    include_dirs = [numpy.get_include()],
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
    )

