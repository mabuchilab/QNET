#!/usr/bin/env python
# encoding: utf-8
"""
This module contains all defined *primitive* circuit component definitions as well as the compiled circuit definitions
that are automatically created via the ``$QNET/bin/parse_qhdl.py`` script.
For some examples on how to create your own circuit definition file, check out the source code to

    * :py:module:``single_sided_jaynes_cummings_cc``
    * :py:module:``three_port_opo_cc``
    * :py:module:``kerr_cavity_cc``

The module :py:module:``component`` features some base classes for component definitions and the module :py:module:``library``
features some utility functions to help manage the circuit component definitions.
"""
