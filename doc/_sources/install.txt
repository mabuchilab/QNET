==================
Installation/Setup
==================

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

A convenient way of obtaining Python as well as some of the packages listed here (SymPy, SciPy, NumPy, PLY) is to download the Enthought_ Python Distribution (EPD) which is free for academic use.
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
.. _Enthought: http://www.enthought.com/
.. _IPython: http://ipython.org/

Installation/Configuration
--------------------------

Copy the QNET folder to any location you'd like. We will tell python how to find it by setting the ``$PYTHONPATH`` environment variable to include the QNET directory's path.
Append the following to your local bash configuration ``$HOME/.bashrc`` or something equivalent:

::

    export QNET=/path/to/cloned/repository
    export PYTHONPATH=$QNET:$PYTHONPATH

**Note that you should replace "/path/to/cloned/repository" with the full path to the cloned QNET directory!**
On my personal laptop that path is given by ``/Users/nikolas/Projects/QNET``, but you can place QNET anywhere you'd like.

On Windows a similar procedure should exist. Environment variable can generally be set via the windows control panels.
It should be sufficient to set just the `PYTHONPATH` environment variable.


To configure gEDA to include our special quantum circuit component symbols you will need to copy the following configuration files from the ``$QNET/gEDA_support/config`` directory to the ``$HOME/.gEDA`` directory:

- ``~/.gEDA/gafrc``
- ``~/.gEDA/gschemrc``

Then install the QHDL netlister plugin within gEDA by creating a symbolic link

::

    ln -s $QNET/gEDA_support/gnet-qhdl.scm  /path/to/gEDA_resources_folder/scheme/gnet-qhdl.scm

**Note that you should replace "/path/to/gEDA_resources_folder" with the full path to the gEDA resources directory!**

in my case that path is given by ``/opt/local/share/gEDA``, but in general simply look for the gEDA-directory that contains the file named ``system-gafrc``.
