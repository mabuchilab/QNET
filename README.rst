====
QNET
====

.. image:: https://img.shields.io/badge/github-mabuchilab/QNET-blue.svg
   :alt: Source code on Github
   :target: https://github.com/mabuchilab/QNET
.. image:: https://img.shields.io/pypi/v/QNET.svg
   :alt: QNET on the Python Package Index
   :target: https://pypi.python.org/pypi/QNET
.. image:: https://badges.gitter.im/mabuchilab/QNET.svg
   :alt: Join the chat at https://gitter.im/mabuchilab/QNET
   :target: https://gitter.im/mabuchilab/QNET?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. image:: https://img.shields.io/travis/mabuchilab/QNET.svg
   :alt: Travis Continuous Integration
   :target: https://travis-ci.org/mabuchilab/QNET
.. image:: https://ci.appveyor.com/api/projects/status/x6sh1ko8eivt5xdh?svg=true
   :alt: Appveyor Continuous Integration
   :target: https://ci.appveyor.com/project/goerz/qnet
.. image:: https://img.shields.io/coveralls/github/mabuchilab/QNET/develop.svg
   :alt: Coveralls
   :target: https://coveralls.io/github/mabuchilab/QNET?branch=develop
.. image:: https://readthedocs.org/projects/qnet/badge/?version=latest
   :alt: Documentation Status
   :target: https://qnet.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :alt: MIT License
   :target: https://opensource.org/licenses/MIT


Computer algebra package for quantum mechanics and photonic quantum networks

Development of QNET happens on `Github`_. You can read the full documentation at `ReadTheDocs`_.

.. _ReadTheDocs: https://qnet.readthedocs.io/en/latest/

Features
--------

* Extensible computer algebra system for quantum operators, quantum states, super operators
* Building on SymPy_ for scalar symbolic algebra
* Implementation of Gough and James' SLH algebra for photonic quantum circuits
* Designed for use within the Jupyter_ notebook
* Publication-ready, configurable rendering of mathematical formulas
* Conversion to QuTiP_ objects for numerical simulation

Note that version 2.0 of QNET is a major redesign. See :ref:`History <history>` for details.


Dependencies
------------

* Python_ version 3.5 or higher. The last version of QNET to support Python 2 is 1.4.3.
* The SymPy_ symbolic algebra Python package to implement symbolic 'scalar' algebra, i.e., the coefficients of state, operator or super-operator expressions can be symbolic SymPy expressions as well as pure python numbers.
* The NumPy_ package for numerical calculations
* Optional: QuTiP_ python package as an extremely useful, efficient and full featured numerical backend. Operator expressions where all symbolic scalar parameters have been replaced by numeric ones, can be converted to (sparse) numeric matrix representations, which are then used to solve for the system dynamics using the tools provided by QuTiP.
* Optional: The PyX_ python package for visualizing circuit expressions as box/flow diagrams. This requires a LaTeX installation on your system. On Linux/Macos and Windows `TeX Live`_ and MiKTeX_ are recommended, respectively.

A convenient way of obtaining Python as well as some of the packages listed here (SymPy, SciPy, NumPy) is to download Anaconda_ Python Distribution, which is free for academic use.
A highly recommended way of working with QNET and QuTiP_, or scientific python codes in general is through the excellent IPython_ command-line shell, or the very polished browser-based Jupyter_ notebook interface.

.. _Python: http://www.python.org
.. _QNET: http://mabuchilab.github.com/QNET/
.. _SymPy: http://SymPy.org/
.. _QuTiP: http://code.google.com/p/qutip/
.. _PyX: http://pyx.sourceforge.net/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _IPython: http://ipython.org/
.. _Jupyter: http://jupyter.org
.. _TeX Live: https://www.tug.org/texlive/
.. _MiKTeX: https://miktex.org


Installation
------------
To install the latest released version of QNET, run this command in your terminal:

.. code-block:: console

    $ pip install qnet

This is the preferred method to install QNET, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


To install the latest development version of QNET from `Github`_.

.. code-block:: console

    $ pip install git+https://github.com/mabuchilab/qnet.git@develop#egg=qnet

.. _Github: https://github.com/mabuchilab/qnet

Usage
-----

To use QNET in a project::

    import qnet
