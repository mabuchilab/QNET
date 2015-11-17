import re
from textwrap import dedent
from collections import OrderedDict

from qnet.algebra.abstract_algebra import prod
from qnet.algebra.hilbert_space_algebra import TrivialSpace
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, set_union, TrivialSpace, LocalSigma,
    ScalarTimesOperator, OperatorPlus, OperatorTimes
)
import sympy


def local_ops(expr):
    """Given a QNET symbolic expression, return a set of "atomic" operators
    occuring in that expression (instances of
    qnet.algebra.operator_algebra.Operator). The set is "atomic" in the sense
    that the operators are not algebraic combinations of other operators."""
    if isinstance(expr, Operator.scalar_types + (str,)):
        return set()
    elif isinstance(expr, (LocalOperator, IdentityOperator.__class__)):
        return set([expr])
    elif isinstance(expr, Circuit):
        slh = expr.toSLH()
        Ls = slh.L.matrix.flatten().tolist()
        H = slh.H
        return set_union(*tuple(map(local_ops, Ls))) | local_ops(H)
    elif isinstance(expr, Operation):
        return set_union(*tuple(map(local_ops, expr.operands)))
    else:
        raise TypeError(str(expr))


class QSDOperator(object):
    """Encapsulation of a QSD (symbolic) Operator, containing all information
    required to instantiate that operator and to use it in C++ code
    expressions.

    Examples:

        >>> A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
        >>> Ad0 = QSDOperator('Operator', 'Ad0', '= A0.hc()')

    """

    _known_types = ['AnnihilationOperator', 'TransitionOperator',
                    'IdentityOperator', 'Operator']

    def __init__(self, qsd_type, name, instantiator):
        self._type = None
        self._name = None
        self._instantiator = None
        self.qsd_type = qsd_type.strip()
        self.name = name.strip()
        self.instantiator = instantiator.strip()

    def __key(self):
        return (self._type, self._name, self._instantiator)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return "{cls}{args}".format(cls=self.__class__.__name__,
                                    args=str(self.__key()))

    @property
    def qsd_type(self):
        """QSD object type, i.e., name of the C++ class. See
        ``_known_types`` class attribute for allowed type names
        """
        return self._type

    @qsd_type.setter
    def qsd_type(self, value):
        if not value in self._known_types:
            raise ValueError("Type '%s' must be one of %s"
                              % (value, self._known_types))
        self._type = value

    @property
    def name(self):
        """The name of the operator object. Must be a valid C++ variable
        name.
        """
        return self._name

    @name.setter
    def name(self, value):
        if not re.match(r'\w[\w\d]+', value):
            raise ValueError("Name '%s' is not a valid C++ variable name"
                             % value)
        self._name = value

    @property
    def instantiator(self):
        """String that instantiates the operator object. This
        must either be the constructur arguments of the operator's QSD class,
        or a C++ expression (starting with an equal sign) that initializes the
        object
        """
        return self._instantiator.strip() # strip out leading space for ' = ..'

    @instantiator.setter
    def instantiator(self, value):
        if value[0] not in ['=', '(']:
            raise ValueError(("Instantiator '%s' does not start with '=' "
                              "(assignement instantiation) or '"
                              "(' (constructor instantiation)") % value)
        if (value.startswith('(') and not value.endswith(')')):
            raise ValueError("Instantiator '%s' does not end with ')'" % value)
        if value.startswith('='):
            # ensure that '=' is surrounded by spaces
            value = ' = ' + value[1:].strip()
        self._instantiator = value

    @property
    def instantiation(self):
        """Complete line of C++ code that instantiates the operator

        Example:

            >>> A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
            >>> print(A0.instantiation)
            AnnihilationOperator A0(0);
        """
        return '{_type} {_name}{_instantiator};'.format(**self.__dict__)

    def __len__(self):
        return 3

    def __iter__(self):
        """Allows to convert ``QSDOperator`` into a tuple.

        Example:

            >>> A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
            >>> qsd_type, name, instantiator = A0
        """
        return iter((self.qsd_type, self.name, self.instantiator))

    def __str__(self):
        """The string representation of an operator is simply its name

        Example:

            >>> A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
            >>> assert(str(A0) == str(A0.name))
        """
        return str(self.name)



class QSDCodeGen(object):
    """Object that allows to generate QSD programs for QNET expressions"""
    template = dedent(r'''
    #include "Complex.h"
    #include "ACG.h"
    #include "CmplxRan.h"
    #include "State.h"
    #include "Operator.h"
    #include "FieldOp.h"
    #include "SpinOp.h"
    #include "Traject.h"

    int main()
    {{
    // Primary Operators
    {OPERATORBASIS}
      /*AnnihilationOperator A1(0);  // 1st freedom*/
      /*NumberOperator N1(0);*/
      /*IdentityOperator Id1(0);*/
      /*AnnihilationOperator A2(1);  // 2nd freedom*/
      /*NumberOperator N2(1);*/
      /*IdentityOperator Id2(1);*/
      /*SigmaPlus Sp(2);             // 3rd freedom*/
      /*IdentityOperator Id3(2);*/
      /*Operator Sm = Sp.hc();       // Hermitian conjugate*/
      /*Operator Ac1 = A1.hc();*/
      /*Operator Ac2 = A2.hc();*/


    // Hamiltonian
    {PARAMETERS}
      /*double E = 20.0;           */
      /*double chi = 0.4;      */
      /*double omega = -0.7;       */
      /*double eta = 0.001;*/
      /*Complex I(0.0,1.0);*/
      /*double gamma1 = 1.0;       */
      /*double gamma2 = 1.0;       */
      /*double kappa = 0.1;        */



    {HAMILTONIAN}
      /*Operator H = (E*I)*(Ac1-A1)*/
                 /*+ (0.5*chi*I)*(Ac1*Ac1*A2 - A1*A1*Ac2)*/
                 /*+ omega*Sp*Sm + (eta*I)*(A2*Sp-Ac2*Sm);*/
    // Lindblad operators
    {NLINDBLADS}
      /*const int nL = 3;*/
    {LINDBLADS}
      /*Operator L[nL]={{
      sqrt(2*gamma1)*A1,
      sqrt(2*gamma2)*A2,
      sqrt(2*kappa)*Sm
      }};*/

    // Initial state
    {INITIAL_STATE}
      /*State phi1(50,FIELD);       // see paper Section 4.2*/
      /*State phi2(50,FIELD);*/
      /*State phi3(2,SPIN);*/
      /*State stateList[3] = {{phi1,phi2,phi3}};*/
      /*State psiIni(3,stateList);*/

    // Trajectory
    {TRAJECTORYPARAMS}
      /*double dt = 0.01;    // basic time step                            */
      /*int numdts = 100;    // time interval between outputs = numdts*dt  */
      /*int numsteps = 5;    // total integration time = numsteps*numdts*dt*/
      /*int nOfMovingFreedoms = 2;*/
      /*double epsilon = 0.01;     // cutoff probability*/
      /*int nPad = 2;              // pad size*/
      /*ACG gen(38388389);         // random number generator with seed*/
      /*ComplexNormal rndm(&gen);  // Complex Gaussian random numbers*/
      /*AdaptiveStep stepper(psiIni, H, nL, L);       // see paper Section 5*/
    // Output
    {OBSERVABLES}
      /*const int nOfOut = 3;*/
      /*Operator outlist[nOfOut]={{ Sp*A2*Sm*Sp, Sm*Sp*A2*Sm, A2 }};*/
      /*char *flist[nOfOut]={{"X1.out","X2.out","A2.out"}};*/
      /*int pipe[] = {{1,5,9,11}}; // controls standard output (see `onespin.cc')*/
      int pipe[4] = {{1,2,3,4}};

    // Simulate one trajectory (for several trajectories see `onespin.cc')
      Trajectory traj(psiIni, dt, stepper, &rndm);  // see paper Section 5
      traj.plotExp( nOfOut, outlist, flist, pipe, numdts, numsteps,
                    nOfMovingFreedoms, epsilon, nPad );
    }}''').strip()

    def __init__(self, circuit, num_vals=None):
        self.circuit = circuit.toSLH()
        self.num_vals = {}
        self.syms = set(circuit.all_symbols())

        # Set of qnet.algebra.operator_algebra.Operator, all "atomic"
        # operators required in the code generation
        self._local_ops = local_ops(self.circuit)

        self._full_space = self.circuit.space
        self._local_spaces = self._full_space.local_factors()
        self._hilbert_space_index = {space: index
                                    for (index, space)
                                    in enumerate(self._local_spaces)}

        # List of qnet.algebra.operator_algebra.Operator instances
        self._observables = []
        # ... and the output file for each observable
        self._outfiles = []
        # Both of these attributes are managed via the add_observable method

        # Mapping QNET Operator => QSDOperator for every operator in
        # self._local_ops. This is kept as an ordered dictionary, such that the
        # instantiation of an operator may refer to a previously defined
        # operator
        self._qsd_ops = OrderedDict()
        self._update_qsd_ops(self._local_ops)

        if num_vals is not None:
            self.num_vals.update(num_vals)

    def _update_qsd_ops(self, operators):
        """Update self._qsd_ops to that every operator in operators is mapped
        to an appropriate QSDOperator. The names of the QSDOperators are chosen
        automatically based on the operator type and the index of the Hilbert
        space they act in. For a Hilbert space index k, the operators names are
        chosen as follows:

            IdentityOperator => Id
            Create => A{k}
            Destroy => Ad{k}
            LocalSigma |i><j| => S{k}_{i}_{j}

        :param operators: iterable (list, set, ...) of operators for which to
            define QSDOpertors. These operators must be "atomic", i.e. they
            must not be an algebraic combination of other operators
        :type operators: [qnet.operator_algebra.Operator, ...]
        """
        self._qsd_ops[IdentityOperator] = QSDOperator(
                qsd_type='Operator',
                name="Id",
                instantiator='= '+'*'.join(
                            ["Id{k}".format(k=k)
                            for k in range(len(self._hilbert_space_index))]))
        # In order to achieve stable output, we go through the operators in an
        # arbitrary, but well-defined order (sorting them according to their
        # string representation)
        for op in sorted(operators, key=str):
            if isinstance(op, IdentityOperator.__class__):
                continue
            elif isinstance(op, (Create, Destroy)):
                a = Destroy(op.space)
                k = self._hilbert_space_index[op.space]
                self._qsd_ops[a] = QSDOperator(
                    qsd_type='AnnihilationOperator',
                    name="A{k}".format(k=k),
                    instantiator=('(%d)'%k))
                ad = a.dag()
                self._qsd_ops[ad] = QSDOperator(
                    qsd_type='Operator',
                    name="Ad{k}".format(k=k),
                    instantiator=('= A{k}.hc()'.format(k=k)))
            elif isinstance(op, LocalSigma):
                k = self._hilbert_space_index[op.space]
                i, j = op.operands[1:]
                self._qsd_ops[op] = QSDOperator(
                    qsd_type='TransitionOperator',
                    name="S{k}_{i}_{j}".format(k=k,i=i,j=j),
                    instantiator='({kij})'.format(
                                 kij=','.join([str(n) for n in (k, i, j)])))
            else:
                raise TypeError(str(op))


    def generate_code(self):
        """Return C++ program that corresponds to the circuit as a multiline
        string"""
        return self.template.format(
                OPERATORBASIS=self._operator_basis_lines(),
                PARAMETERS=self._parameters_lines(),
                HAMILTONIAN=self._hamiltonian_lines(),
                NLINDBLADS='const int nL = {nL:d};'\
                           .format(nL=self.circuit.cdim),
                LINDBLADS=self._lindblads_lines(),
                INITIAL_STATE ='',
                TRAJECTORYPARAMS ='',
                OBSERVABLES ='',
                )

    def write(self, filename):
        """Write C++ program that corresponds to the circuit"""
        with open(filename, 'w') as out_fh:
            out_fh.write(self.generate_code())

    def __str__(self):
        return self.generate_code()


    def _operator_basis_lines(self):
        """Return a multiline string of C++ code that defines and initializes
        all operators in the system"""
        lines = [] # order of lines is important, see below
        for k in range(len(self._local_spaces)):
            lines.append("IdentityOperator Id{k}({k});".format(k=k))
        for op in self._qsd_ops:
            # We assume that self._qsd_ops is an OrderedDict, so that
            # instantiations may refer to earlier operators
            lines.append(self._qsd_ops[op].instantiation)
        return "\n".join(lines)

    def _parameters_lines(self):
        lines = set() # parameter definitions may be in any order
        lines.add("Complex I(0.0,1.0);")
        for s in list(self.syms):
            try:
                val = self.num_vals[s]
            except KeyError:
                raise KeyError(("There is no value for symbol %s in "
                                "self.num_vals") % s)
            if s.is_real is True:
                if val.imag != 0:
                    raise ValueError(("Value %s for %s is complex, but "
                                     "should be real") % (val, s))
                lines.add("double {!s} = {:g};"
                          .format(s, val.real))
            else:
                lines.add("Complex {!s}({:g},{:g});"
                          .format(s, val.real, val.imag))
        return "\n".join(sorted(lines))

    def _operator_str(self, op):
        """For a given instance of ``qnet.operator_algebra.Operator``,
        recursively generate the C++ expression that will instantiate the
        operator.
        """
        if isinstance(op, LocalOperator):
            return str(self._qsd_ops[op])
        elif isinstance(op, ScalarTimesOperator):
            return "({}) * ({})".format(self._scalar_str(op.coeff),
                    self._operator_str(op.term))
        elif isinstance(op, OperatorPlus):
            return "({})".format(" + ".join([self._operator_str(o)
                for o in op.operands]))
        elif isinstance(op, OperatorTimes):
            return "({})".format(" * ".join([self._operator_str(o)
                for o in op.operands]))
        elif op is IdentityOperator:
            return str(self._qsd_ops[op])
        else:
            raise TypeError(str(op))

    def _scalar_str(self, sc):
        if isinstance(sc, sympy.Basic):
            return str(sc)
        return "{:g}".format(sc)

    def _hamiltonian_lines(self):
        H = self.circuit.H
        return "Operator H = {};".format(self._operator_str(H))

    def _lindblads_lines(self):
        L_op_lines = []
        for L_op in self.circuit.L.matrix.flatten():
            L_op_lines.append(self._operator_str(L_op))
        return "Operator L[nL]={\n" + ",\n".join(L_op_lines) + "\n}"


