Parsing QHDL
============

Given a QHDL-file ``my_circuit.qhdl`` which contains an entity named ``MyEntity`` (Note again the CamelCaseConvention for entity names!), we have two options for the final python circuit model file:

    1. We can compile it to an output in the local directory.
       To do this run in the shell::

            $QNET/bin/parse_qhdl.py -f my_circuit.qhdl -l

    2. We can compile it and install it within the module :py:mod:`qnet.circuit_components`.
       To do this run in the shell::

            $QNET/bin/parse_qhdl.py -f my_circuit.qhdl -L

In either case the output file will be named based on a CamelCase to lower_case_with_underscore convention with a ``_cc`` suffix to the name.
I.e., for the above example ``MyEntity`` will become ``my_entity_cc.py``.
In the case of entity names with multiple subsequent capital letters such as ``PseudoNAND``
the convention is to only add underlines before the first and the last of the capitalized group,
i.e. the output would be written to ``pseudo_nand_cc.py``.