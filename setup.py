import os
import sys
from distutils.core import setup
from setuptools import find_packages

def get_version(filename):
    """Extract the package version"""
    with open(filename) as in_fh:
        for line in in_fh:
            if line.startswith('__version__'):
                return line.split('=')[1].strip()[1:-1]
    raise ValueError("Cannot extract version from %s" % filename)


version = get_version('qnet/__init__.py')

setup(
    name='QNET',
    version=version,
    description="Tools for symbolically analyzing quantum feedback networks.",
    author="Nikolas Tezak, Michael Goerz",
    author_email="mail@michaelgoerz.net",
    url="http://github.com/mabuchilab/QNET",
    # cmdclass={'build_ext': build_ext},
    packages=find_packages(exclude=["tests"]),
    # ext_modules=ext_modules,
    install_requires=[
        'matplotlib', 'sympy', 'six', 'numpy', 'attrs',
    ],
    extras_require={
        'dev': ['click', 'pytest>=3.3.0', 'sphinx', 'sphinx-autobuild',
                'sphinx_rtd_theme', 'better-apidoc', 'nose',
                'coverage', 'pytest-cov', 'pytest-xdist'],
        'simulation': ['qutip>=3.0.1'],
        'circuit_visualization': 'pyx>0.14',
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)

