Circuit Component Definition
============================

The best way to get started on defining one's own circuit component definition is to look at the examples
provided in the component library :py:mod:`qnet.circuit_components`.
Every circuit component object is a python class definition that derives off the class :py:class:`qnet.circuit_components.component.Component`.
The subclass must necessarily overwrite the following class attributes of this ``Component`` class:

    * ``CDIM`` needs to be set to the full number (``int``) of input or output noises, i.e.,
      the row dimension of the coupling vector :math:`\mathbf{L}` or the scattering matrix :math:`\mathbf{S}` of the corresponding :math:`(\mathbf{S},\mathbf{L},H)` model.

    * ``PORTSIN`` needs to be set to a list of port labels for the relevant input ports of the component, i.e., those that could be connected to other components.
      The number of entries can be smaller or equal than ``CDIM``.

    * ``PORTSOUT`` needs to be set to a list of port labels for the relevant output ports of the component, i.e., those that could be connected to other components.
      The number of entries can be smaller or equal than ``CDIM``.

    * If your model depends on parameters you should specify this both via the ``_params`` attribute and
      by adding a class attribute with the name of the parameter and a default value that is either numeric or symbolic.
      Checkout some of the existing modules such as :py:mod:`qnet.circuit_components.single_sided_opo_cc` to see how these parameters should be set.

    * If your model has internal quantum degrees of freedom, you need to implement the ``_space`` property.
      If your model has a single quantum degree of freedom such as an empty cavity or an OPO,
      just follow the example of :py:mod:`qnet.circuit_components.single_sided_opo_cc` (click on 'source' to see the source-code).
      If your model's space will be a tensor product of several degrees of freedom, follow the example of
      :py:mod:`qnet.circuit_components.single_sided_jaynes_cummings_cc`, which defines Hilbert space properties
      for the different degrees of freedom and has the ``_space`` property return a tensor product of them.

      In general, it is important to properly assign a unique name and namespace to all internal degrees of freedom to
      rule out ambiguities when your final circuit includes more than one instance of your model.

    * Optionally, you may overwrite the ``name`` attribute to change the default name of your component.

Most importantly, the subclass must implement a ``_toSLH(self):`` method.
Doing this requires some knowledge of how to use the operator algebra :py:mod:`qnet.algebra.operator_algebra`.
For a component model with multiple input/output ports **with no direct scattering between some ports**,
i.e., the scattering matrix :math:`\mathbf{S}` is (block-) diagonal
we allow for a formalism to define this substructure on the circuit-symbolic level by not just defining a component model,
but also models for the irreducible subblocks of your component.
This leads to two alternative ways of defining the circuit components:

    1.  Simple case, creating a symbolically *irreducible* circuit model, this is probably what you should go with:

        This suffices if the purpose of defining the component is only to derive the final quantum equations of motion for an overall system, i.e.,
        no analysis should be carried out on the level of the circuit algebra but
        only on the level of the underlying operator algebra of the full circuit's :math:`(\mathbf{S},\mathbf{L},H)` model.

        Subclassing the ``Component`` class takes care of implementing the class constructor ``__init__`` and this should
        not be overwritten unless you are sure what you are doing. The pre-defined constructor takes care of handling
        the flexible specification of model parameters as well as the name and namespace via its arguments.
        I.e., for a model named ``MyModel`` whose ``_parameters`` attribute is given by ``['kappa', 'gamma']``,
        one can either specify all or just some of the parameters as named arguments. The rest get replaced by the default values.
        Consider the following code examples::

            MyModel(name = "M")
            # -> MyModel(name = "M", namespace = "", kappa = MyModel.kappa, gamma = MyModel.gamma)

            MyModel(name = "M", kappa = 1)
            # -> MyModel(name = "M", namespace = "", kappa = 1, gamma = MyModel.gamma)

            MyModel(kappa = 1)
            # -> MyModel(name = MyModel.name, namespace = "", kappa = 1, gamma = MyModel.gamma)

        The model parameters passed to the constructor are subsequently accessible to the object's methods as instance attributes.
        I.e., within the ``_toSLH(self)``-method of the above example one would access the value of the ``kappa`` parameter as ``self.kappa``.


    2.  Complex case, create a symbolically *reducible* circuit model:

        In this case you will need to define subcomponent model for each irreducible block of your model.
        We will not discuss this advanced method here, but instead refer to the following modules as examples:

        * :py:mod:`qnet.circuit_components.relay_cc`
        * :py:mod:`qnet.circuit_components.single_sided_jaynes_cummings_cc`
        * :py:mod:`qnet.circuit_components.double_sided_opo_cc`


A simple example
----------------

As an example we will now define a simple (symbolically irreducible) version of the single sided Jaynes-Cummings model.
The model is given by:

.. math::
        S & = \mathbf{1}_2 \\
        L & = \begin{pmatrix} \sqrt{\kappa}a \\ \sqrt{\gamma} \sigma_- \end{pmatrix} \\
        H & = \Delta_f a^\dagger a + \Delta_a \sigma_+ \sigma_- + ig\left(\sigma_+ a - \sigma_- a^\dagger \right)

Then, we can define the corresponding component class as::

    from sympy import symbols, I, sqrt
    from qnet.algebra.circuit_algebra import Create, LocalSigma, SLH, Destroy, local_space, Matrix, identity_matrix

    class SingleSidedJaynesCummings(Component):

        CDIM = 2

        name = "Q"

        kappa = symbols('kappa', real = True)       # decay of cavity mode through cavity mirror
        gamma = symbols('gamma', real = True)       # decay rate into transverse modes
        g = symbols('g', real = True)               # coupling between cavity mode and two-level-system
        Delta_a = symbols('Delta_a', real = True)   # detuning between the external driving field and the atom
        Delta_f = symbols('Delta_f', real = True)   # detuning between the external driving field and the cavity
        FOCK_DIM = 20                               # default truncated Fock-space dimension

        _parameters = ['kappa', 'gamma', 'g', 'Delta_a', 'Delta_f', 'FOCK_DIM']

        PORTSIN = ['In1', 'VacIn']
        PORTSOUT = ['Out1', 'UOut']

        @property
        def fock_space(self):
            """The cavity mode's Hilbert space."""
            return local_space("f", make_namespace_string(self.namespace, self.name), dimension = self.FOCK_DIM)

        @property
        def tls_space(self):
            """The two-level-atom's Hilbert space."""
            return local_space("a", make_namespace_string(self.namespace, self.name), basis = ('h', 'g'))

        @property
        def _space(self):
            return self.fock_space * self.tls_space


        def _toSLH(self):
            a = Destroy(self.fock_space)
            sigma = LocalSigma(self.tls_space, 'g', 'h')
            H = self.Delta_f * a.dag() * a + self.Delta_a * sigma.dag() * sigma \
                    + I * self.g * (sigma.dag() * a - sigma * a.dag())
            L1 = sqrt(self.kappa) * a
            L2 = sqrt(self.gamma) * sigma
            L = Matrix([[L1],
                        [L2]])
            S = identity_matrix(2)
            return SLH(S, L, H)



Creating custom component symbols for ``gschem``
------------------------------------------------

Creating symbols in gschem is similar to the schematic capture process itself:

1. Using the different graphical objects (lines, boxes, arcs, text) create the symbol as you see fit.

2. Add pins for the symbols inputs and outputs. Define their ``pintype`` (``in`` or ``out``) and their ``pinnumber``
   (which can be text or a number) according to the port names.
   Finally, define their ``pinseq`` attributes to match the order of the list in the python component definition,
   so for the above example, one would need 4 pins, two inputs, two outputs with the following properties:

    - ``pintype=in, pinnumber=In1, pinseq=i1``
    - ``pintype=in, pinnumber=VacIn, pinseq=i2``
    - ``pintype=out, pinnumber=Out1, pinseq=o1``
    - ``pintype=out, pinnumber=UOut, pinseq=o2``

3. Define the parameters the model depends on, by adding a ``params`` attribute to the top level circuit.
   For the example above the correct param string would be::

    kappa:real;gamma:real;g:real;Delta_a:real;Delta_f:real;FOCK_DIM:int:20

4. Add the name of the component by setting the ``device`` top-level-attribute, in this example to ``SingleSidedJaynesCummings``

5. Specify the default name by adding a ``refdes`` attribute that is equal to the default name plus an appended question mark (e.g. ``Q?``).
   When designing a circuit, this helps to quickly identify unnamed subcomponents.

The result could look something like this:

.. figure:: _static/single_sided_jaynes_cummings.png
    :width: 800 px
