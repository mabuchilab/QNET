from distutils.core import setup
# from distutils.extension import Extension
#
# from Cython.Distutils import build_ext
# import numpy as np
#
#
# ext_modules = [
#     Extension("qnet.misc.kerr_cysolve",
#               ["qnet/misc/src/kerr_cysolve.pyx"],
#               include_dirs=[np.get_include()],
#               extra_link_args=['-lm']),
# ]

version = "1.1"


setup(
    name='QNET',
    version=version,
    description="""Tools for symbolically analyzing quantum feedback networks.""",
    scripts=["bin/parse_qhdl.py"],
    author="Nikolas Tezak",
    author_email="nikolas.tezak@gmail.com",
    url="http://mabuchilab.github.io/QNET/",
    # cmdclass={'build_ext': build_ext},
    py_modules=['qnet'],
    # ext_modules=ext_modules,
    requires=[
        'sympy',
        # 'Cython',
        'numpy',
        'qutip',
    ]
)
