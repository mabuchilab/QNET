"""
This module contains all defined *primitive* circuit component definitions as
well as the compiled circuit definitions
that are automatically created via the ``$QNET/bin/parse_qhdl.py`` script.
For some examples on how to create your own circuit definition file, check out
the source code to

    * :mod:`qnet.circuit_components.single_sided_jaynes_cummings_cc`
    * :mod:`qnet.circuit_components.three_port_opo_cc`
    * :mod:`qnet.circuit_components.kerr_cavity_cc`

The module :py:mod:`qnet.circuit_components.component` features some base
classes for component definitions and the module
:py:mod:`qnet.circuit_components.library`
features some utility functions to help manage the circuit component
definitions.
"""
