from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults()['binding'] = True
get_directive_defaults()['linetrace'] = True


extensions = [
    Extension("weights", 
             ["sph2sed/weights.pyx"],
             define_macros=[('CYTHON_TRACE', '1')])
]

setup(
    name='sph2sed',
    version='0.1',
    description='SPH 2 SED',
    author='Christopher Lovell',
    author_email='c.lovell@sussex.ac.uk',
    packages=['sph2sed'],
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    include_dirs = [np.get_include()],
)

