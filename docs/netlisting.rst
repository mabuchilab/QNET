.. _netlisting:

==========
Netlisting
==========


Using ``gnetlist``
==================


Given a well-formed ``gschem`` circuit specification file we can use the ``gnetlist`` tool
that comes with the ``gEDA`` suite to export it to a QHDL-file.

Using the command-line, if the ``.sch`` schematic file is located at the path ``my_dir/my_schematic.sch``,
and you wish to produce a QHDL file at the location ``my_other_dir/my_netlist.qhdl``, run the following command::

    gnetlist -g qhdl my_dir/my_schematic.sch -o my_other_dir/my_netlist.qhdl

It is generally a very good idea to inspect the produced QHDL file code and verify that it looks like it
should before trying to compile it into a python ``circuit_component`` library file.


.. _qhdl_syntax:

The QHDL Syntax
===============

A QHDL file consists of two basic parts:

    1. An ``entity`` declaration, which should be thought of as defining the external interface of the specified circuit.
       I.e., it defines global input and output ports as well as parameters for the overall model.

    2. A corresponding ``architecture`` declaration, that, in turn consists of two parts:

        A) The architecture head defines what *types* of components can appear in the circuit. I.e., for each ``component``
           declaration in the architecture head, there can exist multiple *instances* of that component type in the circuit.
           The head also defines the internal ``signal`` lines of the circuit.

        B) The architecture body declares what instances of which component type exists in the circuit, how its ports
           are mapped to the internal signals or entity ports, and how its internal parameters relate to the entity parameters.
           In QHDL, each signal may only connect exactly two ports, where one of three cases is true:

                        I. It connects an entity input with a component instance input
                        II. It connects an entity output with a component instance output
                        III. It connects a component output with a component input


Before showing some examples of QHDL files, we present the general QHDL syntax in somewhat abstract form.
Here, square brackets ``[optional]`` denote optional keywords/syntax and the ellipses ``...`` denote repetition:

.. code-block:: qhdl

    -- this is a comment

    -- entity definition
    -- this serves as the external interface to the circuit, specifying inputs and outputs
    -- as well as parameters of the model
    entity my_entity is
        [generic ( var1: generic_type [:= default_var1]] [; var2: generic_type [...] ...]);]
        port (i_1,i_2,...i_n:in fieldmode; o_1,o_2,...o_n:out fieldmode);
    end entity my_entity;

    -- architecture definition
    -- this is the actual implementation of the entity in terms of subcomponents
    architecture my_architecture of my_entity is
        -- architecture head
        -- each type of subcomponent, i.e. its ports and its parameters are defined here similarly
        -- to the entity definition above
        component my_component
            [generic ( var3: generic_type [:= default_var3]] [; var4: generic_type [...] ...]);]
            port (p1,p2,...pm:in fieldmode; q1,q2,...qm:out fieldmode);
        end component my_component;

        [component my_second_component
            [generic ( var5: generic_type [:= default_var5]] [; var6: generic_type [...] ...]);]
            port (p1,p2,...pr:in fieldmode; q1,q2,...qr:out fieldmode);

        end component my_second_component;

        ...

        ]

        -- internal signals to connect component instances
        [signal s_1,s_2,s_3,...s_m fieldmode;]



    begin
        -- architecture body
        -- here the actual component instances are defined and their ports are mapped to signals
        -- or to global (i.e. entity-) ports
        -- furthermore, global (entity-) parameters are mapped to component instance parameters.

        COMPONENT_INSTANCE_ID1: my_component
            [generic map(var1 => var3, var1 => var4);]
            port map (i_1, i_2, ... i_m, s_1, s_2, ...s_m);

        [COMPONENT_INSTANCE_ID2: my_component
            [generic map(var1 => var3, var1 => var4);]
            port map (s_1, s_2, ... s_m, o_1, o_2, ...o_m);

        COMPONENT_INSTANCE_ID3: my_second_component
            [generic map (...);]
            port map (...);
        ...
            ]

     end architecture my_architecture;


where ``generic_type`` is one of ``int``, ``real``, or ``complex``.

QHDL-Example files:
-------------------

A Mach-Zehnder-circuit
^^^^^^^^^^^^^^^^^^^^^^

This toy-circuit realizes a Mach-Zehnder interferometer.

.. figure:: _static/MachZehnder.png
    :width: 800 px

.. literalinclude:: ../examples/qhdl/MachZehnder.qhdl
    :language: qhdl

A Pseudo-NAND-gate
^^^^^^^^^^^^^^^^^^

This circuit consists of a Kerr-nonlinear cavity, a few beamsplitters and a bias input amplitude to realize a NAND-gate for the inputs A and B.
For details see [Mabuchi11]_.

.. figure:: _static/PseudoNAND.png
    :width: 800 px

    The ``gschem`` schematic from which the QHDL file below was automatically created.



.. literalinclude:: ../examples/qhdl/PseudoNAND.qhdl
    :language: qhdl


A Pseudo-NAND-Latch
^^^^^^^^^^^^^^^^^^^

This circuit consists of two subcomponents that each act almost (i.e., for all relevant input conditions) like a NAND logic gate
in a symmetric feedback conditions. As is known from electrical circuits this arrangement allows the fabrication of a bi-stable
system with memory or state from two systems that have a one-to-one input output behavior. See also [Mabuchi11]_

.. literalinclude:: ../examples/qhdl/PseudoNANDLatch.qhdl
    :language: qhdl

