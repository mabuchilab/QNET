Install
=======

In addition to the core python packages, the QNET pacakge draws on and extends the following existing packages:

1) ``gEDA`` for graphic creation of circuits end exporting these to QHDL (using our own plugin)
2) ``SymPy`` for the symbolic 'scalar' algebra, i.e. the number coefficients of operator expressions can be symbolic as well
3) ``QuTiP`` as the numerical backend, operator expressions where all scalar parameters have been replaced by numeric ones,
   can be converted to (sparse) numeric matrix representations, which are then used to solve for the system dynamics using the tools provided by QuTiP


Howto set up QNET
-----------------

What you'll need:

1) A Python installation with the following additional packages installed:

    - scipy, numpy, matplotlib, ply, sympy
    - PyX       http://pyx.sourceforge.net/
    - QuTiP     http://code.google.com/p/qutip/

   The easiest way to achieve this is to download the (optimized) Enthought python distribution (EPD)
   which is free for academic use:
   http://www.enthought.com/products/epd.php

   But you will still need to install PyX and QuTiP manually.

2) GIT and a user account on GitHub https://github.com
   I will then give you access to our repository

3) The open source ``gEDA`` toolsuite which allows for schematic capture of photonic circuits

   http://www.gpleda.org/

   On many linux distributions there should exists packages, on OSX I recommend using the package managers
   MacPorts, Fink or Brew


4) Once, you have cloned the GIT repository into a local directory, you should set up some environment variables for your shell::

    export QNET=/path/to/cloned/repository
    export PYTHONPATH=$QNET

5) Configure ``gEDA`` to include our special quantum circuit component symbols. To do this you will need to copy the configuration files from the ``$QNET/gEDA_support/config`` directory to ``$HOME/.gEDA``::

    ~/.gEDA/gafrc
    ~/.gEDA/gschemrc


6) Install the QHDL netlister plugin within ``gEDA`` by creating a symbolic link::

        ln -s $QNET/gEDA_support/gnet-qhdl.scm  /path/to/gEDA_resources_folder/scheme/gnet-qhdl.scm

   in my case ``/path/to/gEDA_resources_folder == /opt/local/share/gEDA``, simply look for the folder that contains the file named ``system-gafrc``.

7) At this point you have set up everything you need to create circuits with ``gschem`` and export them to our QHDL-format using gnetlist. To get started with this, just read the attached tutorial I wrote for our group's internal blog.

8) Test that everything is installed and working, e.g. run the enhanced python shell 'ipython' and do::

       import qnet.algebra.circuit_algebra as ca

   if this does not fail, it should work.
