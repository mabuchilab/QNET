#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages


def get_version(filename):
    """Extract the package version"""
    with open(filename) as in_fh:
        for line in in_fh:
            if line.startswith('__version__'):
                return line.split('=')[1].strip()[1:-1]
    raise ValueError("Cannot extract version from %s" % filename)


with open('README.rst') as readme_file:
    readme = readme_file.read()

try:
    with open('HISTORY.rst') as history_file:
        history = history_file.read()
except OSError:
    history = ''

requirements = ['sympy<1.2', 'scipy', 'numpy', 'attrs', 'uniseg']

dev_requirements = [
    'coverage', 'coveralls', 'pytest', 'pytest-cov', 'pytest-xdist', 'twine', 'pep8',
    'flake8', 'wheel', 'sphinx', 'sphinx-autobuild', 'sphinx_rtd_theme',
    'sphinx-autodoc-typehints', 'ipython']
dev_requirements.append('better-apidoc>=0.2.0')


version = get_version(os.path.join('.', 'src', 'qnet', '__init__.py'))

setup(
    name='QNET',
    version=version,
    description="Computer algebra package for quantum mechanics and "
    "photonic quantum networks",
    author="Nikolas Tezak, Michael Goerz",
    author_email='mail@michaelgoerz.net',
    url='https://github.com/mabuchilab/QNET',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'simulation': ['qutip>=3.0.1'],
        'visualization': ['matplotlib', 'pyx>0.14'],
    },
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=[
        'qnet', 'computer algebra', 'symbolic algebra', 'science',
        'quantum computing', 'quantum mechanics', 'quantum optics',
        'quantum networks', 'circuits', 'SLH', 'qutip', 'sympy'],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
)
