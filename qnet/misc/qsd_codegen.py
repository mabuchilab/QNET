import re
import random
import sys
import os
import logging
import struct
from textwrap import dedent
from collections import OrderedDict
from functools import partial

from qnet.algebra.abstract_algebra import prod
from qnet.algebra.hilbert_space_algebra import TrivialSpace, BasisNotSetError
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, set_union, TrivialSpace, LocalSigma,
    ScalarTimesOperator, OperatorPlus, OperatorTimes
)
from qnet.algebra.state_algebra import (
    Ket, LocalKet, BasisKet, CoherentStateKet, TensorKet
)
import sympy

# max unsigned int in C/C++ when compiled the same way as python
UNSIGNED_MAXINT = 2 ** (struct.Struct('I').size * 8 - 1) - 1


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


def find_kets(expr, cls=LocalKet):
    """Given a QNET Ket instance, return the set of LocalKet instances
    contained in it"""
    if isinstance(expr, Operator.scalar_types + (str,)):
        return set()
    elif isinstance(expr, cls):
        return set([expr])
    elif isinstance(expr, Operation):
        finder = partial(find_kets, cls=cls)
        return set_union(*tuple(map(finder, expr.operands)))
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


class QSDCodeGenError(Exception):
    """Exception raised for missing data in QSDCodeGen instance"""
    pass


class QSDCodeGen(object):
    """Object that allows to generate QSD programs for QNET expressions"""

    _known_steppers = ['Order4Step', 'AdaptiveStep', 'AdaptiveJump',
                       'AdaptiveOrthoJump']

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

    // Hamiltonian
    {PARAMETERS}

    {HAMILTONIAN}

    // Lindblad operators
    {LINDBLADS}

    // Observables
    {OBSERVABLES}

    // Initial state
    {INITIAL_STATE}
      /*State phi1(50,FIELD);       // see paper Section 4.2*/
      /*State phi2(50,FIELD);*/
      /*State phi3(2,SPIN);*/
      /*State stateList[3] = {{phi1,phi2,phi3}};*/
      /*State psiIni(3,stateList);*/

    // Trajectory
    {TRAJECTORY}
    }}''').strip()

    def __init__(self, circuit, num_vals=None):
        self.circuit = circuit.toSLH()
        self.num_vals = {}
        self.syms = set(circuit.all_symbols())
        self._psi_initial = None
        self._traj_params = {}
        self._moving_params = {}
        self._rnd_seed = None

        # Set of qnet.algebra.operator_algebra.Operator, all "atomic"
        # operators required in the code generation
        self._local_ops = local_ops(self.circuit)
        # The add_observable method may later extend this set

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
        # The add_observable method may later extend this mapping

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

    def add_observable(self, op, filename):
        """Register an operators as an observable, together with a filename
        in which the expectation values and standard deviations of the operator
        will be written.

        :param op: Observable
        :type op: qnet.algebra.operator_algebra.Operator
        :param filename: Name of file to which to write the "plot" of
            (averaged) expectation values for the observable.
        :type filename: str
        """
        op_local_ops = local_ops(op)
        self._local_ops.update(op_local_ops)
        self._update_qsd_ops(op_local_ops)
        self.syms.update(op.all_symbols())
        self._observables.append(op)
        self._outfiles.append(filename)

    def set_moving_basis(self, move_dofs, delta=1e-4, width=2, move_eps=1e-4):
        """Activate the use of the the moving basis, see Section 6 of the QSD
        Paper.

        :param move_dofs: degrees of freedome for which to use a moving basis
            (the first 'move_dofs' freedoms are recentered, and their cutoffs
            adjusted.)
        :type move_dofs: int
        :param delta: probability threshold for the cutoff adjustment
        :type delta: float
        :param width: size of the "pad" for the cutoff
        :type width: int
        :param move_eps: numerical accuracy with which to make the shift. Cf.
            ``shiftAccuracy`` in QSD ``State::recenter`` method
        :type move_eps: float
        """
        if move_dofs <= 0:
            raise ValueError("move_dofs must be an integer >0")
        if move_dofs > len(self._local_spaces):
            raise ValueError("move_dofs must not be larger than the number "
                             "of local Hilbert spaces")
        self._moving_params['move_dofs'] = move_dofs
        self._moving_params['delta'] = delta
        if move_dofs <= 0:
            raise ValueError("width must be an integer >0")
        self._moving_params['width'] = width
        self._moving_params['move_eps'] = move_eps

    def set_trajectories(self, psi_initial, stepper, dt, nt_plot_step,
            n_plot_steps, n_trajectories, add_to_existing_traj=True,
            traj_save=10, rnd_seed=None):
        """Set the parameters that control the trajectories from which a plot
        of expectation values for the registered observables will be generated.

        :param psi_initial: The initial state
        :type psi_initial: qnet.agebra.state_algebra.Ket
        :param stepper: Name of the QSD stepper that should handle propagation
            of a single time step. See ``_known_steppers`` class attribute for
            allowed values
        :type stepper: str
        :param dt: The duration for a single propagation step. Note that the
            plot of expectation values will generally be on a coarser grid, as
            controlled by the ``set_plotting`` routine
        :type dt: float
        :param nt_plot_step: Number of propagation steps per plot step. That
            is, expectation values of the observables will be written out every
            `nt_plot_step` propagation steps
        :type nt_plot_step: int
        :param n_plot_steps: Number of plot steps. The total number of
            propagation steps for each trajectory will be
            ``nt_plot_step * n_plot_steps``, and duration T of the entire
            trajectory will be ``dt * nt_plot_step * n_plot_steps``
        :type n_plot_stpes: int
        :param n_trajectories: The number of trajectories over which to average
            for getting the expectation values of the observables
        :type n_trajectories: int
        :param add_to_existing_traj: If True, and if the output file for every
            observable already exists and contains data for a matching
            trajectory, include the existing data when calculating the averaged
            expectation values. In this case, the average will be determined
            from the existing trajectories and ``n_trajectories`` new
            trajectories. Note that when adding to existing trajectories, it is
            essential that the random number generator starts from a different
            seed.
        :type add_to_existing_traj: boolean
        :param traj_save: Number of trajectories to propagate before writing
            the averaged expectation values of all oberservables to file. This
            ensures that if the program is terminated before the calculation of
            ``n_trajectories`` is complete, the lost data is at most that of
            the last ``traj_save`` trajectories is lost. A value of 0
            indictates that the values are to be written out only after
            completing all trajectories.
        :type traj_save: int
        :rnd_seed: Seed for the QSD random number generator. If None, a new
            random number will be used anytime the QSD program is generated. To
            ensure exact reproducibility, the seed may be set manually. If not
            None, must be in the range of C unsigned integers.
        :type rnd_seed: int, None
        """
        self._psi_initial = psi_initial
        if not stepper in self._known_steppers:
            raise ValueError("stepper '%s' must be one of %s"
                              % (value, self._known_steppers))
        self._traj_params['stepper'] = stepper
        self._traj_params['dt'] = dt
        self._traj_params['nt_plot_step'] = nt_plot_step
        self._traj_params['n_plot_steps'] = n_plot_steps
        self._traj_params['n_trajectories'] = n_trajectories
        self._traj_params['traj_save'] = traj_save
        self._rnd_seed = rnd_seed
        if add_to_existing_traj:
            self._traj_params['read_files'] = 1
        else:
            self._traj_params['read_files'] = 0

    def _ordered_tensor_operands(self, state):
        """Return the operands of the given TensorKet instance ordered by their
        Hilbert space (using self._hilbert_space_index)

        This is necessary because in QNET, state carry a label for their
        Hilbert space, while in QSD they do not. Instead, in a Tensor product
        in QSD the order of the operands associates them with their Hilbert
        space.
        """
        def sort_key(ket):
            return self._hilbert_space_index[ket.space]
        return sorted(state.operands, key=sort_key)

    def _initial_state_lines(self):
        raise NotImplementedError()

    def _trajectory_lines(self):
        logger = logging.getLogger(__name__)
        try:
            read_files = self._traj_params['read_files']
        except KeyError:
            raise QSDCodeGenError("No trajectories set up. Ensure that "
                                  "'set_trajectories' method has been called")
        if read_files == 1:
            for filename in self._outfiles:
                if not os.path.isfile(filename):
                    logger.info("Not all output files exist. "
                                "Disabling 'add_to_existing_traj'")
                    self._traj_params['read_files'] = 0

        lines = [
        'int rndSeed = {rnd_seed};',
        'ACG gen(rndSeed); // random number generator',
        'ComplexNormal rndm(&gen); // Complex Gaussian random numbers',
        '',
        'double dt = {dt};',
        'int dtsperStep = {nt_plot_step};',
        'int nOfSteps = {n_plot_steps};',
        'int nTrajSave = {traj_save};',
        'int nTrajectory = {n_trajectories};',
        'int ReadFile = {read_files};',
        '',
        '{stepper} stepper(psiIni, H, nL, L);',
        'Trajectory traj(psiIni, dt, stepper, &rndm);',
        '',
        ]
        if len(self._moving_params) > 0:
            lines.extend([
                'int move = {move_dofs};',
                'double delta = {delta};',
                'int width = {width};',
                'int moveEps = {move_eps}',
                '',
                'traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,',
                '            nTrajectory, nTrajSave, ReadFile, move,',
                '            delta, width, moveEps);'
            ])
        else:
            lines.extend([
                'traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,',
                '            nTrajectory, nTrajSave, ReadFile);'
            ])
        fmt_mapping = self._traj_params.copy()
        fmt_mapping.update(self._moving_params)
        rnd_seed = self._rnd_seed
        if rnd_seed is None:
            rnd_seed = random.randint(0, UNSIGNED_MAXINT)
        fmt_mapping['rnd_seed'] = rnd_seed
        return "\n".join([line.format(**fmt_mapping) for line in lines])


    def generate_code(self):
        """Return C++ program that corresponds to the circuit as a multiline
        string"""
        return self.template.format(
                OPERATORBASIS=self._operator_basis_lines(),
                PARAMETERS=self._parameters_lines(),
                HAMILTONIAN=self._hamiltonian_lines(),
                LINDBLADS=self._lindblads_lines(),
                OBSERVABLES=self._observables_lines(),
                INITIAL_STATE ='',
                TRAJECTORY =self._trajectory_lines(),
                )

    def write(self, outfile):
        """Write C++ program that corresponds to the circuit"""
        with open(outfile, 'w') as out_fh:
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
        """Return a multiline string of C++ code that defines all numerical
        constants"""
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


    def _observables_lines(self):
        """Return a multiline string of C++ code that defines all
        observables"""
        lines = []
        n_of_out = len(self._observables)
        lines.append('const int nOfOut = %d' % n_of_out)
        outlist_lines = []
        if len(self._observables) < 1:
            raise QSDCodeGenError("Must register at least one observable")
        for observable in self._observables:
            outlist_lines.append(self._operator_str(observable))
        lines.append("Operator outlist[nOfOut] = {\n  "
                     + ",\n  ".join(outlist_lines) + "\n};")
        lines.append('char *flist[nOfOut] = {{{filenames}}};'
                     .format(filenames=", ".join(
                         [('"'+fn+'"') for fn in self._outfiles]
                     )))
        lines.append(r'int pipe[4] = {1,2,3,4};')
        return "\n".join(lines)


    def _operator_str(self, op):
        """For a given instance of ``qnet.algebra.operator_algebra.Operator``,
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
        lines = []
        lines.append('const int nL = {nL};'.format(nL=self.circuit.cdim))
        L_op_lines = []
        for L_op in self.circuit.L.matrix.flatten():
            L_op_lines.append(self._operator_str(L_op))
        lines.append(
            "Operator L[nL]={\n  " + ",\n  ".join(L_op_lines) + "\n};")
        return "\n".join(lines)


