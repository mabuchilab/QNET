QNET
====

.. image:: https://travis-ci.org/mabuchilab/QNET.svg
    :target: https://travis-ci.org/mabuchilab/QNET
    :alt: Build Status

.. image:: https://readthedocs.org/projects/qnet/badge/?version=latest
    :target: http://qnet.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

The QNET package is a set of tools created and compiled to aid in the design and analysis of photonic circuit models.
Our proposed Quantum Hardware Description Language (cf. QHDL_) serves to describe a circuit topology and specification of a larger entity in terms of parametrizable subcomponents.
This is strongly analogous to the specification of electric circuitry using the structural description elements of VHDL or Verilog.
The physical systems that can be modeled within the framework include quantum optical experiments that can be described as nodes with internal degrees of freedom such as interacting quantum harmonic oscillators and/or N-level quantum systems that,
in turn are coupled to a finite number of external bosonic quantum fields.

To get help installing and using QNET, please read this README, visit our `homepage <http://mabuchilab.github.com/QNET>`_ which includes the `official documentation <https://qnet.readthedocs.org/en/latest/>`_, sign up to our `mailing list <http://groups.google.com/group/qnet-user>`_,
or consult and perhaps contribute to our `wiki <https://github.com/mabuchilab/QNET/wiki>`_.

In particular, check out the `Roadmap <https://github.com/mabuchilab/QNET/wiki/Roadmap>`_.
In the near future, it will be possible to use QNET together with `Cirq <https://github.com/ntezak/Cirq>`_ which
allows to edit circuits graphically from within the IPython_ notebook.


Contents
--------

The package consists of the following components:

1. A symbolic computer algebra package ``qnet.algebra`` for Hilbert Space quantum mechanical operators, the Gough-James circuit algebra and also an algebra for Hilbert space *Ket*-states and *Super-operators* which themselves act on operators.
2. The QHDL_ language definition and parser ``qnet.qhdl`` including a front-end located at ``bin/parse_qhdl.py`` that can convert a QHDL-file into a circuit component library file.
3. A library of existing primitive or composite circuit components ``qnet.circuit_components`` that can be embedded into a new circuit definition.


.. _Dependencies:

Dependencies
------------

In addition to these core components, the software uses the following existing software packages:

0. Python_ version 2.6 or higher. QNET is still officially a Python 2 package, but migration to Python 3 should not be too hard to achieve.
1. The gEDA_ toolsuite for its visual tool ``gschem`` for the creation of circuits end exporting these to QHDL ``gnetlist``. We have created device symbols for our primitive circuit components to be used with ``gschem`` and we have included our own ``gnetlist`` plugin for exporting to QHDL.
2. The SymPy_ symbolic algebra Python package to implement symbolic 'scalar' algebra, i.e. the coefficients of state, operator or super-operator expressions can be symbolic SymPy expressions as well as pure python numbers.
3. The QuTiP_ python package as an extremely useful, efficient and full featured numerical backend. Operator expressions where all symbolic scalar parameters have been replaced by numeric ones, can be converted to (sparse) numeric matrix representations, which are then used to solve for the system dynamics using the tools provided by QuTiP.
4. The PyX_ python package for visualizing circuit expressions as box/flow diagrams.
5. The SciPy_ and NumPy_ packages (needed for QuTiP but also by the ``qnet.algebra`` package)
6. The PLY_ python package as a dependency of our Python Lex/Yacc based QHDL parser.

A convenient way of obtaining Python as well as some of the packages listed here (SymPy, SciPy, NumPy, PLY) is to download the Enthought_ Python Distribution (EPD) or Anaconda_ which are both free for academic use.
A highly recommended way of working with QNET and QuTiP and just scientific python codes in action is to use the excellent IPython_ shell which comes both with a command-line interface as well as a very polished browser-based notebook interface.

.. _Python: http://www.python.org
.. _gEDA: http://www.gpleda.org
.. _QHDL: http://rsta.royalsocietypublishing.org/content/370/1979/5270.abstract
.. _QNET: http://mabuchilab.github.com/QNET/
.. _SymPy: http://SymPy.org/
.. _QuTiP: http://code.google.com/p/qutip/
.. _PyX: http://pyx.sourceforge.net/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/
.. _PLY: http://www.dabeaz.com/ply/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _IPython: http://ipython.org/

Installation/Configuration
--------------------------

To install QNET you need a working Python installation as well as `pip <https://pip.pypa.io/en/latest/installing.html>`_
which comes pre-installed with both the Enthought Python distribution and Anaconda.
If you have already installed PyX_ just run:
Run::

    pip install QNET

If you still need to install PyX_, run::

    pip install --process-dependency-links QNET

gEDA
----

Setting up gEDA/gschem/gnetlist is a bit more involved. To do this, you should download the QNET package manually from github.
If you are using Linux or OSX, geda is available via common package managers such as `port` and `homebrew` on OSX or
apt for Linux.

To configure gEDA to include our special quantum circuit component symbols you will need to copy the following configuration files from the ``QNET/gEDA_support/config`` sub-directory to the ``$HOME/.gEDA`` directory:

- ``~/.gEDA/gafrc``
- ``~/.gEDA/gschemrc``

Then install the QHDL netlister plugin within gEDA by creating a symbolic link (or copy the file there)
In the shell cd into the QNET directory and then run
::

    ln -s gEDA_support/gnet-qhdl.scm  /path/to/gEDA_resources_folder/scheme/gnet-qhdl.scm

**Note that you should replace "/path/to/gEDA_resources_folder" with the full path to the gEDA resources directory!**

in my case that path is given by ``/opt/local/share/gEDA``, but in general simply look for the gEDA-directory that contains the file named ``system-gafrc``.

Using QNET in practice
----------------------

A possible full workflow using QNET is thus:

I. Use ``gschem`` (of gEDA) to graphically design a circuit model.
II. Export the schematic to QHDL using ``gnetlist`` (also part of gEDA)
III. Parse the QHDL-circuit definition file into a Python circuit library component using the parser front-end ``bin/parse_qhdl.py``.
IV. Analyze the model analytically using our symbolic algebra and/or numerically using QuTiP.

This package is still work in progress and as it is developed by a single developer, documentation and comprehensive testing code is still somewhat lacking.
Any contributions, bug reports and general feedback from end-users would be highly appreciated. If you have found a bug, it would be extremely helpful if you could try to write a minimal code example that reproduces the bug.
Feature requests will definitely be considered. Higher priority will be given to things that many people ask for and that can be implemented efficiently.

To learn of how to carry out each of these steps, we recommend looking at the provided examples and reading the relevant sections in the QNET manual.
Also, if you want to implement and add your own primitive device models, please consult the QNET manual.

Acknowledgements
----------------

`Hideo Mabuchi <mailto:hmabuchi@stanford.edu>`_ had the initial idea for a software package that could exploit the Gough-James SLH formalism to generate an overall open quantum system model for a quantum feedback network based solely on its topology and the component models in analytic form.
The actual QNET package was then planned and implemented by `Nikolas Tezak <mailto:ntezak@stanford.edu>`_. In the Fall of 2015 `Michael Goerz <mailto:goerz@stanford.edu>`_ joined as a main developer.

In its current form, QNET comprises
functionality [#additionalFeatures]_ that goes well beyond what would be necessary to achieve the original goal, but which has proven to be immensely useful.
In addition to the authors of the software packages listed under Dependencies_ that QNET relies on, we would like to acknowledge the following people's direct support to QNET which included their vision, ideas, examples, bug reports and feedback.

- Michael Armen
- Armand Niederberger
- Joe Kerckhoff
- Dmitri Pavlichin
- Gopal Sarma
- Ryan Hamerly
- Michael Hush

Work on QNET was directly supported by DARPA-MTO under Award No. N66001-11-1-4106. Nikolas Tezak is also supported by a Simons Foundation Math+X fellowship as well as a Stanford Graduate Fellowship.

.. [#additionalFeatures] E.g., all algebras except the operator algebra are not strictly necessary to achieve just the original objective.

License
-------

QNET is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

QNET is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with QNET.  If not, see `this page <http://www.gnu.org/licenses/>`_.

Copyright (C) 2012-2016, Nikolas Tezak


