from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("weights", ["weights.pyx"]),
#    Extension("test_module", ["test_module.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs = [np.get_include()],
)

