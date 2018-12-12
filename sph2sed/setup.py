from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults()['binding'] = True
get_directive_defaults()['linetrace'] = True

extensions = [
    Extension("weights", ["weights.pyx"], 
        define_macros=[('CYTHON_TRACE', '1')])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs = [np.get_include()],
)

