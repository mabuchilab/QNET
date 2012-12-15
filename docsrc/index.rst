.. qnet documentation master file, created by
   sphinx-quickstart on Fri Sep  7 18:34:40 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QNET's documentation!
================================

QNET is a package to help designing, modeling and analyzing quantum optical circuits using the input/output formalism.
Our proposed Quantum Hardware Description Language (QHDL) serves to describe a circuit topology and specification of a larger
entity in terms of parametrizable subcomponents. To compute a Markovian open system model using either the SLH formalism or the ABCD formalism,
QNET includes a QHDL parser which converts a QHDL circuit specification into a symbolic representation of the system model
that can be manipulated and analyzed using a custom computer algebraic package :py:mod:`qnet.algebra`.
In addition to support for symbolic Hilbert space operators :py:mod:`qnet.algebra.operator_algebra` and a
fully symbolic implementation of the Gough/James circuit algebra :py:mod:`qnet.algebra.circuit_algebra` the algebra package
also includes symbolic Hilbert space state algebra as well as an algebra of superoperators.

The full workflow to define and investigate a particular circuit model is:

I) use ``gschem`` (of ``gEDA``) to model a circuit
II) export the schematic to QHDL using gnetlist (of ``gEDA``)
III) parse the QHDL-circuit file into a Python circuit library component
IV) Analyze the model analytically using our symbolic algebra and/or numerically using QuTiP.

Obviously this package is still very much work in progress and as it is developed by a single developer, documentation and comprehensive testing code is still lacking. Any contributions and feedback from end-users would thus be highly appreciated.



Contents:

.. toctree::
   :maxdepth: 4

   install
   schematic_capture
   netlisting
   parsing_qhdl
   model_analysis_simulation
   qnet


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

