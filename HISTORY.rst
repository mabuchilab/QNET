=======
History
=======

The original 1.0 relase of QNET centered around an implementation of the Quantum Hardware Description Language (QHDL_) that serves to describe a circuit topology and specification of a larger entity in terms of parametrizable subcomponents.
This is strongly analogous to the specification of electric circuitry using the structural description elements of VHDL or Verilog.

Version 2.0 of QNET shifts the focus of the package to provide a broad symbolic algebra package for quantum mechanics, and the implementation of the SLH circuit algebra. Support of QHDL_ was removed from QNET, with the intention of re-implementing it in a separate QHDL package, that works on top of QNET. The split was made because the two aspects of the original QNET package serves two different audiences: The basic algebraic tools are will be used by theorists or for numerical models, while QHDL_, the definition of circuit components, or the use of the gEDA_ ``gschem`` tool are primarily of interest for experimentalists. By developing these two aspects in different packages, we hope the better address the particular needs of each user group.

If you are currently using QHDL through QNET 1.0, you should not upgrade to QNET 2.0. Also, QNET 2.0 drops support for Python 2.

QNET uses `Semantic Versioning`_.

.. _Semantic Versioning: https://semver.org
.. _QHDL: http://rsta.royalsocietypublishing.org/content/370/1979/5270.abstract
.. _gEDA: http://www.gpleda.org


1.0.0
-----

* initial release

2.0.0
-----

* major restructuring
* drop Python 2 support
* remove support for parsing the quantum-hardware-description-language (QHDL)
  and the circuit component library. QNET now provides only the fundamental
  algebraic tools. The QHDL functionality will be extended in a separate future
  QHDL package
* a new printing system
