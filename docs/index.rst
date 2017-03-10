Welcome to QNET's documentation!
================================

The QNET package is a set of tools to aid in the design and analysis of photonic circuit models, but it features a flexible symbolic algebra module that can be applied in a more general setting.
Our proposed Quantum Hardware Description Language [QHDL]_ serves to describe a circuit topology and specification of a larger entity in terms of parametrizable subcomponents.
By design this is analogous to the specification of electric circuitry using the structural description elements of VHDL or Verilog.

The physical systems that can be modeled within the framework include quantum optical experiments that can be described as nodes with internal degrees of freedom such as interacting quantum harmonic oscillators and/or N-level quantum systems that,
in turn are coupled to a finite number of bosonic quantum fields. Furthermore, the formalism applies also to superconducting microwave circuit (Circuit QED) systems.

For a rigorous introduction to the underlying mathematical physics we refer to the original treatment of Gough and James [GoughJames08]_, [GoughJames09]_ and the references given therein.


The main components of this package are:

1. A symbolic computer algebra package :mod:`qnet.algebra` for Hilbert Space quantum mechanical operators, the Gough-James circuit algebra and also an algebra for Hilbert space states and Super-operators.
2. The QHDL language definition and parser :mod:`qnet.qhdl` including a front-end located at ``bin/parse_qhdl.py`` that can convert a QHDL-file into a circuit component library file.
3. A library of existing primitive or composite circuit components :mod:`qnet.circuit_components` that can be embedded into a new circuit definition.


In practice one might want to use these to:

I. Define and specify your basic circuit component model and create a library file, :doc:`circuit_component`
II. Use ``gschem`` (of gEDA) to graphically design a circuit model, :doc:`schematic_capture`
III. Export the schematic to QHDL using ``gnetlist`` (also part of gEDA) or directly write a QHDL file, :doc:`netlisting`
IV. Parse the QHDL-circuit definition file into a Python circuit library component using the parser front-end ``bin/parse_qhdl.py``, :doc:`parsing_qhdl`
V. Analyze the model analytically using our symbolic algebra and/or numerically using QuTiP, :doc:`symbolic_algebra`, :doc:`model_analysis_simulation`

This package is still work in progress and as it is currently being developed by a single developer (interested in `helping? <mailto:goerz@stanford.edu>`_), documentation and comprehensive testing code are still somewhat lacking.
Any contributions, bug reports and general feedback from end-users would be highly appreciated. If you have found a bug, it would be extremely helpful if you could try to write a minimal code example that reproduces the bug.
Feature requests will definitely be considered. Higher priority will be given to things that many people ask for and that can be implemented efficiently.

To learn of how to carry out each of these steps, we recommend looking at the provided examples and reading the relevant sections in the QNET manual.
Also, if you want to implement and add your own primitive device models, please consult the QNET manual.


Contents:

.. toctree::
   :maxdepth: 2

   install
   symbolic_algebra
   circuit_rules
   circuit_component
   schematic_capture
   netlisting
   parsing_qhdl
   model_analysis_simulation
   references


API
===

.. toctree::
   :maxdepth: 1

   API of the qnet package <API/qnet>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



