# coding=utf-8
import sys
from distutils.core import setup
from pkgutil import walk_packages
import qnet
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

version = qnet.__version__


def find_packages(path=".", prefix=""):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name

packages = list(find_packages(qnet.__path__, qnet.__name__))

setup(
    name='QNET',
    version=version,
    description="Tools for symbolically analyzing quantum feedback networks.",
    scripts=["bin/parse_qhdl.py"],
    author="Nikolas Tezak",
    author_email="nikolas.tezak@gmail.com",
    url="http://github.com/mabuchilab/QNET",
    # cmdclass={'build_ext': build_ext},
    packages=packages,
    # ext_modules=ext_modules,
    install_requires=[
        'sympy',
        'ply',
        'six',
        'numpy',
        'qutip>=3.0.1',
        'pyx==0.12.1' if sys.version_info < (3, 0) else 'pyx>=0.13',
    ],
    dependency_links=[
        ("http://downloads.sourceforge.net/project/pyx/pyx/0.12.1/"
         + "PyX-0.12.1.tar.gz?r=http%3A%2F%2Fsourceforge.net"
         + "%2Fprojects%2Fpyx%2Ffiles%2Fpyx%2F"
         + "0.12.1%2F&ts=1407271744&use_mirror=kent")
        if sys.version_info < (3, 0) else (
            "http://downloads.sourceforge.net/project/pyx/pyx/0.13/"
            + "PyX-0.13.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects"
            + "%2Fpyx%2Ffiles%2F&ts=1423687678&use_mirror=iweb"
            )
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
