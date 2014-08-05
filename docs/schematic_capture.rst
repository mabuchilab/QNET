Schematic Capture
=================

Here we explain how to create photonic circuits visually using ``gschem``


1) From the 'Add' menu select 'Component' to open the component symbol library.

2) Layout components on the grid

3) Double-click the component symbols to edit the properties of each component instance.
   Be sure to set a unique instance identifier ``refdes``-attribute.

   If a component symbol has an attribute named ``params``, its value should be understood as a list of the form:
   ``param1_name:param1_type[:default_value1];param2_name:param2_type[:default_value2];...`` where the default values are optional.
   To assign a value to a component param, add an attribute of the param name and set the value either to a corresponding numerical value
   or to a parameter name of the overall circuit.

4) For all input and output ports of the circuit that are part of its external interface add dedicated input and output pad objects.
   Assign names to them (``refdes``-attribute) that correspond to their port names and assign sequence numbers to them, numbering the inputs as
   ``i1, i2, ...`` and the outputs as ``o1, o2, ...``

5) Draw all internal signals to connect component ports with each other and with port objects.

6) Add a ``params``-attribute to the whole circuit specifying all model parameters similarly to above.

7) Add a ``module-name``-attribute to the whole circuit to specify its entity name. Please use ``CamelCaseConventions``
   for naming your circuit, because it will ultimately be the name of a Python class.

As an example, consider `this screencast for creating a Pseudo-NAND-Latch <http://www.youtube.com/watch?v=B8ZLnDjvWM4>`_.