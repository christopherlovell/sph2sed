from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

extensions = [
    Extension("sph2sed/weights", ["sph2sed/weights.pyx"]),
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

