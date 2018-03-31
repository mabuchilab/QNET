==================
Installation/Setup
==================

Dependencies
------------

In addition to these core components, the software uses the following existing software packages:

1. Python_ version 3.3 or higher. Python 2 is no longer supported.
2. The SymPy_ symbolic algebra Python package to implement symbolic 'scalar' algebra, i.e. the coefficients of state, operator or super-operator expressions can be symbolic SymPy expressions as well as pure python numbers.
3. The QuTiP_ python package as an extremely useful, efficient and full featured numerical backend. Operator expressions where all symbolic scalar parameters have been replaced by numeric ones, can be converted to (sparse) numeric matrix representations, which are then used to solve for the system dynamics using the tools provided by QuTiP.
4. The PyX_ python package for visualizing circuit expressions as box/flow diagrams.
5. The SciPy_ and NumPy_ packages (needed for QuTiP but also by the ``qnet.algebra`` package)

A convenient way of obtaining Python as well as some of the packages listed here (SymPy, SciPy, NumPy, PLY) is to download Anaconda_ Python Distribution, which is for academic use.
A highly recommended way of working with QNET and QuTiP and just scientific python codes in action is to use the excellent IPython_ shell which comes both with a command-line interface as well as a very polished browser-based notebook interface.

.. _Python: http://www.python.org
.. _QHDL: http://rsta.royalsocietypublishing.org/content/370/1979/5270.abstract
.. _QNET: http://mabuchilab.github.com/QNET/
.. _SymPy: http://SymPy.org/
.. _QuTiP: http://code.google.com/p/qutip/
.. _PyX: http://pyx.sourceforge.net/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _IPython: http://ipython.org/

Installation/Configuration
--------------------------

To install QNET you need a working Python installation as well as `pip <https://pip.pypa.io/en/latest/installing.html>`_
which comes pre-installed with Anaconda.

Run::

    pip install QNET
