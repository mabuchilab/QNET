from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np


ext_modules = [
    Extension("qnet.misc.kerr_cysolve",
              ["qnet/misc/src/kerr_cysolve.pyx"],
              include_dirs=[np.get_include()],
              extra_link_args=['-lm']),
]

setup(
    name='QNET',
    version='1.0',


    cmdclass={'build_ext': build_ext},
    py_modules=['qnet'],
    ext_modules=ext_modules, requires=['sympy', 'Cython', 'numpy', 'qutip']
)
