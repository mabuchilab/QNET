import re
import random
import os
import logging
import struct
import shlex
from textwrap import dedent
from collections import OrderedDict
from functools import partial
import subprocess as sp

from qnet.algebra.abstract_algebra import Operation, set_union
from qnet.algebra.hilbert_space_algebra import TrivialSpace, BasisNotSetError
from qnet.algebra.circuit_algebra import Circuit
from qnet.algebra.state_algebra import (
    Ket, LocalKet, BasisKet, CoherentStateKet, TensorKet,
    ScalarTimesKet, KetPlus
)
from qnet.algebra.operator_algebra import (scalar_free_symbols,
        IdentityOperator, Create, Destroy, LocalOperator, Operator, LocalSigma,
        ScalarTimesOperator, OperatorPlus, OperatorTimes)
from qnet.misc.trajectory_data import TrajectoryData
import sympy
from sympy.printing.ccode import CCodePrinter
try:
    issubclass(FileNotFoundError, OSError)
except NameError: # indicates Python 2
    class FileNotFoundError(OSError):
        pass

# max unsigned int in C/C++ when compiled the same way as python
UNSIGNED_MAXINT = 2 ** (struct.Struct('I').size * 8 - 1) - 1


class QSDCCodePrinter(CCodePrinter):
    """A printer for converting SymPy expressions to C++ code, while taking
    into account pre-defined variable names for symbols"""
    def __init__(self, settings={}):
        self._default_settings['user_symbols'] = {}
        super(QSDCCodePrinter, self).__init__(settings=settings)
        self.known_symbols = dict(settings.get('user_symbols'))
    def _print_Symbol(self, expr):
        if expr in self.known_symbols:
            return self.known_symbols[expr]
        else:
            return super(QSDCCodePrinter,self)._print_Symbol(expr)


def local_ops(expr):
    """Given a symbolic expression, extract the set of "atomic" operators
    (instances of :class:`~qnet.algebra.operator_algebra.Operator`) occurring
    in that expression. The set is "atomic" in the sense that the
    operators are not algebraic combinations of other operators.
    """
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
    """Given a :class:`~qnet.algebra.state_algebra.Ket` instance,
    return the set of :class:`~qnet.algebra.state_algebra.LocalKet`
    instances contained in it.
    """
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

    All arguments set the corresponding properties.

    Examples:

        >>> A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
        >>> Ad0 = QSDOperator('Operator', 'Ad0', '= A0.hc()')

    """

    known_types = ['AnnihilationOperator', 'FieldTransitionOperator',
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
        :attr:`known_types` class attribute for allowed type names
        """
        return self._type

    @qsd_type.setter
    def qsd_type(self, value):
        if not value in self.known_types:
            raise ValueError("Type '%s' must be one of %s"
                              % (value, self.known_types))
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
        must either be the constructor arguments of the operator's QSD class,
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
        """Split :obj:`QSDOperator` into a tuple.

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
    """Exception raised for missing data in a :obj:`QSDCodeGen` instance"""
    pass


class QSDCodeGen(object):
    """Class that allows to generate a QSD program for QNET expressions, and
    to run the program to (accumulative) collect expectation values for
    observables

    Parameters:
        circuit (:obj:`~qnet.algebra.circuit_algebra.SLH`): The circuit to be
            simulated via QSD.
        num_vals (dict of :obj:`~sympy.core.symbol.Symbol` to float)): Numeric
            value for any symbol occurring in the `circuit`, or any
            operator/state that may be added later on.
        time_symbol (None or :obj:`~sympy.core.symbol.Symbol`): symbol to
            denote the time dependence in the Hamiltonian (usually `t`). If
            None, the Hamiltonian is time-independent.

    Attributes:
        circuit (:class:`~qnet.algebra.circuit_algebra.SLH`): see `circuit`
            parameter
        time_symbol (None or :obj:`~sympy.core.symbol.Symbol`): see
            `time_symbol` parameter
        syms (set of :obj:`~sympy.core.symbol.Symbol`): The set of symbols used
            either in the circuit, any of the observables, or the initial
            state, excluding `time_symbol`
        num_vals (dict of :obj:`~sympy.core.symbol.Symbol` to float)): Map of
            symbols to numeric value. Must specify a value for any symbol in
            `syms`.
        traj_data (:obj:`~qnet.misc.trajectory_data.TrajectoryData`): The
            accumulated trajectory data. Every time the :meth:`run`,
            respectively the :meth:`run_delayed` method is
            called, the resulting trajectory data is incorporated. Thus, by
            repeatedly calling :meth:`run` (followed by :meth:`run_delayed` if
            ``delay=True``), an arbitrary number of trajectories
            may be accumulated in `traj_data`.
    """

    known_steppers = ['Order4Step', 'AdaptiveStep', 'AdaptiveJump',
                      'AdaptiveOrthoJump']

    _template = dedent(r'''
    #define _USE_MATH_DEFINES
    #include <cmath>
    #include "Complex.h"
    #include "ACG.h"
    #include "CmplxRan.h"
    #include "State.h"
    #include "Operator.h"
    #include "FieldOp.h"
    #include "Traject.h"

    {PARAMETERS}
    {FUNCTIONS}

    int main(int argc, char* argv[])
    {{

      unsigned int rndSeed;
      if (argc != 2){{
        std::cout << "Usage: " << argv[0] << " SEED" << std::endl;
        std::exit(1);
      }} else {{
        if(sscanf(argv[1], "%u", &rndSeed) != 1){{
          std::cout << "ERROR: Could not read SEED" << std::endl;
          std::exit(1);
        }} else {{
          std::cout << "Using rnd seed: " << rndSeed << std::endl;
        }}
      }}

      // Primary Operators
    {OPERATORBASIS}

      // Hamiltonian
    {HAMILTONIAN}

      // Lindblad operators
    {LINDBLADS}

      // Observables
    {OBSERVABLES}

      // Initial state
    {INITIAL_STATE}

      // Trajectory
    {TRAJECTORY}
    }}''').strip()

    _max_op_name_length = 16
    _lib_qsd = 'libqsd.a' # expected name of qsd library
    _link_qsd = '-lqsd' # compiler option to link qsd

    def __init__(self, circuit, num_vals=None, time_symbol=None):
        self.circuit = circuit.toSLH()
        self.time_symbol = time_symbol
        self.num_vals = {}
        self.traj_data = None
        self._psi_initial = None
        self._traj_params = {}
        self._moving_params = {}

        # Set of sympy.core.symbol.Symbol instances
        self.syms = set(circuit.all_symbols())
        self.syms.discard(self.time_symbol)
        # Mapping symbol => variable name (sanitized)
        self._var_names = {}
        if self.time_symbol is not None:
            # it is important to register the time symbol before any other
            # symbols, to detect possible name clashes
            self._var_names[self.time_symbol] = 't'
        self._update_var_names()
        # The add_observable and set_trajectories methods may later extend syms
        # and _var_names

        # Set of qnet.algebra.operator_algebra.Operator, all "atomic"
        # operators required in the code generation
        self._local_ops = local_ops(self.circuit)
        # The add_observable and set_trajectories methods may later extend this
        # set

        self._full_space = self.circuit.space
        self._local_spaces = self._full_space.local_factors()
        self._hilbert_space_index = {space: index
                                    for (index, space)
                                    in enumerate(self._local_spaces)}

        # Dict name => tuple(qnet.algebra.operator_algebra.Operator, outfile)
        self._observables = OrderedDict()
        # Managed via the add_observable method

        # Mapping QNET Operator => QSDOperator for every operator in
        # self._local_ops. This is kept as an ordered dictionary, such that the
        # instantiation of an operator may refer to a previously defined
        # operator
        self._qsd_ops = OrderedDict()
        self._update_qsd_ops(self._local_ops)
        # The add_observable and set_trajectories methods may later extend this
        # mapping

        # Mapping QNET Ket => QSD state name (str) for "atomic" states
        # (instances of LocalKet and TensorKet)
        self._qsd_states = {}
        # This is set in the _define_atomic_kets method

        # The following are set by the compile method to allow for delayed (or
        # remote) compilation. These are stored in "unexpanded" form, i.e.
        # possibly including environment variables. These will only be expanded
        # by the compilation_worker, possibly on a remote system
        self._executable  = None # name of the executable
        self._path        = None # folder for executable (unexpanded)
        self._compile_cmd = None # list of command arguments (unexpanded)
        self._keep_cc     = None # delete C++ file after compilation?
        # _executable will remain None until the compile method has finished
        # without error. Thus, only _executable should be used to check whether
        # the compile method has been called.

        if num_vals is not None:
            self.num_vals.update(num_vals)

        # when the `run` method is called with `delay=True`, the `kwargs`
        # dictionary is appended to the following list. A call to `run_delayed`
        # may then process the whole list in parallel
        self._delayed_runs_kwargs = []
        # We also cache the 'seed' of each kwargs in _delayed_runs_kwargs
        self._delayed_seeds = []

        # for any time-dependent coefficient, we keep
        #   coeff => (time-function-name, time-function-placehold, is_real)
        # in a dictionary
        self._tfuncs = {}

    @property
    def observables(self):
        """Iterator over all defined observables (instances of
        :obj:`~qnet.algebra.operator_algebra.Operator`)
        """
        return iter([op for (op, fn) in self._observables.values()])

    @property
    def observable_names(self):
        """Iterator of all defined observable names (str)"""
        return iter(self._observables.keys())

    @property
    def compile_cmd(self):
        """Command to be used for compilation (after :meth:`compile` method has
        been called). Environment variables and '~' are not expanded"""
        if self._executable is None:
            return ''
        else:
            return _cmd_list_to_str(self._compile_cmd)

    def get_observable(self, name):
        """Return the observable for the given name
        (instance of :obj:`~qnet.algebra.operator_algebra.Operator`), according
        to the mapping defined by :meth:`add_observable`"""
        return self._observables[name][0]

    def _update_qsd_ops(self, operators):
        """Update :attr:`self._qsd_ops` to that every operator in operators is
        mapped to an appropriate :obj:`QSDOperator`. The names of the
        operators are chosen automatically based on the operator type and
        the index of the Hilbert space they act in. For a Hilbert space index
        ``k``, the operators names are chosen as follows::

            IdentityOperator => Id
            Create => A{k}
            Destroy => Ad{k}
            LocalSigma |i><j| => S{k}_{i}_{j}

        Arguments:
            operators (iterable of :obj:`~qnet.operator_algebra.Operator`):
                list or set of operators for which to define :obj:`QSDOperator`
                instances.  These operators must be "atomic", i.e. they must
                not be an algebraic combination of other operators. They must
                be in the Hilbert space of the circuit (otherwise, a ValueError
                is raised), and their Hilbert-space must be (a subspace of) the
                circuit Hilbert space (otherwise, a :exc:`BasisNotSetError` is
                raised)
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
            if not op.space.is_tensor_factor_of(self._full_space):
                raise ValueError(("Operator '%s' is not in the circuit's "
                                  "Hilbert space") % str(op))
            if not op.space is TrivialSpace:
                __ = op.space.basis # raises BasisNotSetError
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
                try:
                    i = op.space.basis.index(op.operands[1])
                    j = op.space.basis.index(op.operands[2])
                except ValueError:
                    raise ValueError(("The states %s in %s are not elements "
                        "of the basis of the Hilbert space %s")
                        % (str(op.operands[1:]), str(op), str(op.space)))
                self._qsd_ops[op] = QSDOperator(
                    qsd_type='FieldTransitionOperator',
                    name="S{k}_{i}_{j}".format(k=k,i=i,j=j),
                    instantiator='({ijk})'.format(
                                 ijk=','.join([str(n) for n in (i, j, k)])))
            else:
                raise TypeError(str(op))

    def _update_var_names(self):
        """Ensure that for every symbol, there is a var name in the cache"""
        used_vars = set(self._var_names.values())
        for sym in self.syms:
            if not sym in self._var_names:
                var = sanitize_varname(str(sym))
                if var in used_vars:
                    raise ValueError("Cannot generate a unique variable name "
                                     "for symbol '%s'" % sym)
                else:
                    self._var_names[sym] = var
                    used_vars.add(var)

    def add_observable(self, op, name=None):
        """Register an operator as an observable, together with a name that
        will be used in the header of the table of expectation values, and on
        which the name of the QSD output files will be based.

        Arguments:
            op (:obj:`~qnet.algebra.operator_algebra.Operator`): Observable
                (does not need to be Hermitian)
            name (str or ``None``): Name of of the operator, to be used in the
                header of the output table. If ``None``, ``str(op)`` is used.

        Raises:
            ValueError: if `name` is invalid or too long, or no unique
                filename can be generated from `name`
        """
        logger = logging.getLogger(__name__)
        if name is None:
            name = str(op).strip()
        if not TrajectoryData._rx['op_name'].match(name):
            raise ValueError(("Operator name '%s' contains invalid "
                              "characters") % name)
        if len(name) > self._max_op_name_length:
            raise ValueError("Operator name '%s' is longer than limit %d"
                             % (name, self._max_op_name_length))
        if name in self._observables:
            logger.warn("Overwriting existing operator '%s'", name)
            # It is necessary to delete the observable, otherwise the check for
            # unique filenames would be tripped
            del self._observables[name]
        filename = sanitize_filename(name)
        if (len(filename) == 0):
            raise ValueError("Cannot generate filename for operator "
                             "%s. You must use a different name" % name)
        filename = filename + '.out'
        if filename in [fn for (__, fn) in self._observables.values()]:
            raise ValueError("Cannot generate unique filename for operator "
                             "%s. You must use a different name" % name)
        op_local_ops = local_ops(op)
        self._local_ops.update(op_local_ops)
        self._update_qsd_ops(op_local_ops)
        self.syms.update(op.all_symbols())
        self.syms.discard(self.time_symbol)
        self._update_var_names()
        self._observables[name] = (op, filename)

    def set_moving_basis(self, move_dofs, delta=1e-4, width=2, move_eps=1e-4):
        """Activate the use of the moving basis, see Section 6 of the QSD
        Paper.

        Arguments:
            move_dofs (int): degrees of freedom for which to use a moving basis
                (the first 'move_dofs' freedoms are re-centered, and their
                cutoffs adjusted.)
            delta (float): probability threshold for the cutoff adjustment
            width (int): size of the "pad" for the cutoff
            move_eps (float): numerical accuracy with which to make the shift.
                Cf. ``shiftAccuracy`` in QSD ``State::recenter`` method

        Raises:
            ValueError: if `move_dofs` is invalid
            QSDCodeGenError: if requesting a moving basis for a degree of
                freedom for which any operator is defined that cannot be
                applied in the moving basis
        """
        if move_dofs <= 0:
            raise ValueError("move_dofs must be an integer >0")
        elif move_dofs > len(self._local_spaces):
            raise ValueError("move_dofs must not be larger than the number "
                             "of local Hilbert spaces")
        else:
            # Ensure that there are no LocalSigma operators for any of the
            # degrees of freedom that are part of the moving basis (LocalSigma
            # is mapped to FieldTransitionOperator in QSD, which is
            # incompatible with a moving basis)
            for op in self._local_ops:
                if isinstance(op, LocalSigma):
                    k = self._hilbert_space_index[op.space]
                    if k < move_dofs:
                        # '<', not '<=', because k counts from 0
                        raise QSDCodeGenError(("A moving basis cannot be used "
                        "for a degree of freedom that has local transition "
                        "operators. Conflicting operator %s acts on Hilbert "
                        "space %d<%d") % (op, k, move_dofs))
        self._moving_params['move_dofs'] = move_dofs
        self._moving_params['delta'] = delta
        if move_dofs <= 0:
            raise ValueError("width must be an integer >0")
        self._moving_params['width'] = width
        self._moving_params['move_eps'] = move_eps

    def set_trajectories(self, psi_initial, stepper, dt, nt_plot_step,
            n_plot_steps, n_trajectories, traj_save=10):
        """Set the parameters that control the trajectories from which a plot
        of expectation values for the registered observables will be generated.

        Arguments:
            psi_initial (:obj:`~qnet.algebra.state_algebra.Ket`): The initial
                state
            stepper (str): Name of the QSD stepper that should handle
                propagation of a single time step. See :attr:`known_steppers`
                for allowed values
            dt (float): The duration for a single propagation step. Note that
                the plot of expectation values will generally be on a coarser
                grid, as controlled by the ``set_plotting`` routine
            nt_plot_step (int): Number of propagation steps per plot step. That
                is, expectation values of the observables will be written out
                every `nt_plot_step` propagation steps
            n_plot_steps (int): Number of plot steps. The total number of
                propagation steps for each trajectory will be ``nt_plot_step *
                n_plot_steps``, and duration T of the entire trajectory will be
                ``dt * nt_plot_step * n_plot_steps``
            n_trajectories (int): The number of trajectories over which to
                average for getting the expectation values of the observables
            traj_save (int): Number of trajectories to propagate before writing
                the averaged expectation values of all observables to file.
                This ensures that if the program is terminated before the
                calculation of ``n_trajectories`` is complete, the lost data is
                at most that of the last ``traj_save`` trajectories is lost. A
                value of 0 indicates that the values are to be written out
                only after completing all trajectories.
        """
        self._psi_initial = psi_initial
        if isinstance(psi_initial, Operation):
            # all non-atomic instances of Ket are also instances of Operation
            psi_local_ops = local_ops(psi_initial)
            self._local_ops.update(psi_local_ops)
            self._update_qsd_ops(psi_local_ops)
            self.syms.update(psi_initial.all_symbols())
            self.syms.discard(self.time_symbol)
            self._update_var_names()
        if not stepper in self.known_steppers:
            raise ValueError("stepper '%s' must be one of %s"
                              % (stepper, self.known_steppers))
        self._traj_params['stepper'] = stepper
        self._traj_params['dt'] = dt
        self._traj_params['nt_plot_step'] = nt_plot_step
        self._traj_params['n_plot_steps'] = n_plot_steps
        self._traj_params['n_trajectories'] = n_trajectories
        self._traj_params['traj_save'] = traj_save

    def _ordered_tensor_operands(self, state):
        """Return the operands of the given TensorKet instance ordered by their
        Hilbert space (using `self._hilbert_space_index`)

        This is necessary because in QNET, state carry a label for their
        Hilbert space, while in QSD they do not. Instead, in a Tensor product
        in QSD the order of the operands associates them with their Hilbert
        space.
        """
        def sort_key(ket):
            return self._hilbert_space_index[ket.space]
        return sorted(state.operands, key=sort_key)

    def _define_atomic_kets(self, state, reset=True):
        """Find all "atomic" kets in the given state and register them in
        self._qsd_states. Return an array of lines of C++ code that defines
        the state in a QSD program. If reset is True, any existing entries in
        self._qsd_states are discarded.

        The "atomic" kets are instances of `LocalKet` or `TensorKet`, both of
        which require to be defined with their own name in QSD code
        """
        if not state.space == self._full_space:
            raise QSDCodeGenError(("State %s is not in the Hilbert "
                                    "space of the Hamiltonian") % state)
        lines = []
        if reset:
            self._qsd_states = {}
        for (prfx, kets) in [
            ('phiL', find_kets(state, cls=LocalKet)),
            ('phiT', find_kets(state, cls=TensorKet))
        ]:
            for k, ket in enumerate(sorted(kets, key=str)):
                # We go through the states in an arbitrary, but well-defined
                # order by sorting them according to str
                name = prfx + str(k)
                self._qsd_states[ket] = name
                try:
                    N = ket.space.dimension
                except BasisNotSetError:
                    raise QSDCodeGenError(("Unknown dimension for Hilbert "
                    "space '%s'. Please set the Hilbert space's dimension "
                    "property. Alternatively, set a basis using "
                    "qnet.algebra.HilbertSpaceAlgebra.BasisRegistry.set_basis")
                    % str(ket.space))
                if isinstance(ket, BasisKet):
                    n = ket.space.basis.index(ket.operands[1])
                    instantiation = '({N:d},{n:d},FIELD)'.format(N=N, n=n)
                    comment = ' // HS %d' \
                              % self._hilbert_space_index[ket.space]
                elif isinstance(ket, CoherentStateKet):
                    alpha = ket.operands[1]
                    if alpha in self.syms:
                        alpha_name = self._var_names[alpha]
                    else:
                        try:
                            alpha = complex(ket.operands[1])
                        except TypeError:
                            raise TypeError(("CoherentStateKet amplitude %s "
                            "is neither a known symbol nor a complex number")
                            % alpha)
                        alpha_name = name + '_alpha'
                        lines.append('Complex {alpha}({re:g},{im:g});'.format(
                                    alpha=alpha_name, re=alpha.real,
                                    im=alpha.imag))
                    instantiation = '({N:d},{alpha},FIELD)'.format(
                                     N=N, alpha=alpha_name)
                    comment = ' // HS %d' \
                              % self._hilbert_space_index[ket.space]
                elif isinstance(ket, TensorKet):
                    operands = [self._ket_str(operand) for operand
                                in self._ordered_tensor_operands(ket)]
                    lines.append("State {name}List[{n}] = {{{ketlist}}};"
                                 .format(name=name, n=len(operands),
                                         ketlist = ", ".join(operands))
                                )
                    instantiation = '({n}, {name}List)'.format(
                                     n=len(operands), name=name)
                    comment = ' // ' + " * ".join(
                                ["HS %d"%self._hilbert_space_index[o.space]
                                for o in self._ordered_tensor_operands(ket)])
                else:
                    raise TypeError("Cannot instantiate QSD state for type %s"
                                    %str(type(ket)))
                lines.append('State '+name+instantiation+';'+comment)
        return lines


    def _initial_state_lines(self, indent=2):
        if not isinstance(self._psi_initial, Ket):
            raise TypeError("Initial state must be a Ket instance")
        lines = self._define_atomic_kets(self._psi_initial)
        lines.append('')
        lines.append('State psiIni = '+self._ket_str(self._psi_initial)+';')
        lines.append('psiIni.normalize();')
        return "\n".join(_indent(lines, indent))

    def _trajectory_lines(self, indent=2):
        try:
            __ = self._traj_params['stepper']
        except KeyError:
            raise QSDCodeGenError("No trajectories set up. Ensure that "
                                  "'set_trajectories' method has been called")
        lines = [
        'ACG gen(rndSeed); // random number generator',
        'ComplexNormal rndm(&gen); // Complex Gaussian random numbers',
        '',
        'double dt = {dt};',
        'int dtsperStep = {nt_plot_step};',
        'int nOfSteps = {n_plot_steps};',
        'int nTrajSave = {traj_save};',
        'int nTrajectory = {n_trajectories};',
        'int ReadFile = 0;',
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
                'double moveEps = {move_eps};',
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
        rendered_lines = [line.format(**fmt_mapping) for line in lines]
        return "\n".join(_indent(rendered_lines, indent))


    def generate_code(self):
        """Return C++ program that corresponds to the circuit as a multiline
        string"""
        return self._template.format(
                OPERATORBASIS=self._operator_basis_lines(),
                PARAMETERS=self._parameters_lines(indent=0),
                FUNCTIONS=self._function_lines(ops=[self.circuit.H, ],
                                               indent=0),
                HAMILTONIAN=self._hamiltonian_lines(),
                LINDBLADS=self._lindblads_lines(),
                OBSERVABLES=self._observables_lines(),
                INITIAL_STATE=self._initial_state_lines(),
                TRAJECTORY=self._trajectory_lines(),
                )

    def write(self, outfile):
        """Write C++ program that corresponds to the circuit"""
        with open(outfile, 'w') as out_fh:
            out_fh.write(self.generate_code())
            out_fh.write("\n")

    def compile(self, qsd_lib, qsd_headers, executable='qsd_run',
            path='.', compiler='g++', compile_options='-O2', delay=False,
            keep_cc=False, remote_apply=None):
        """Compile into an executable

        Arguments:
            qsd_lib (str): full path to the file ``libqsd.a`` containing the
                statically compiled QSD library.  May reference environment
                variables the home directory ('~')
            qsd_headers (str): path to the folder containing the QSD header
                files.  May reference environment variables the home directory
                ('~')
            executable (str): name of executable to which the QSD program
                should be compiled. Must consist only of letters, numbers,
                dashes, and underscores only
            path (str): The path to the folder where executable will be
                generated.  May reference environment variables the home
                directory ('~')
            compiler (str): compiler executable
            compile_options (str): options to pass to the compiler
            delay (bool): Deprecated, must be False
            keep_cc (bool): If True, keep the C++ code from which the
                executable was compiled. It will have the same name as the
                executable, with an added '.cc' file extension.
            remote_apply (callable or None): If not None,
                ``remote_apply(compilation_worker, kwargs)`` must call
                :func:`compilation_worker` on any remote node.
                Typically, this might point to the `apply` method of an
                ``ipyparallel`` View instance. The `remote_apply` argument
                should only be given if :meth:`run_delayed` will be called with
                an argument `map` that will push the calculation of a
                trajectory to a remote node.

        Raises:
            ValueError: if `executable` name or `qsd_lib` are invalid
            subprocess.CalledProcessError: if compilation fails
        """
        if delay:
            raise DeprecationWarning(
                    "`delay` will be removed in future versions")
        logger = logging.getLogger(__name__)
        executable = str(executable)
        self._path = str(path)
        self._keep_cc = keep_cc
        if not re.match(r'^[\w-]{1,128}$', executable):
            if len(executable) > 128:
                raise ValueError("Executable name too long")
            else:
                raise ValueError("Invalid executable name '%s'" % executable)
        self._compile_cmd = self._build_compile_cmd(qsd_lib, qsd_headers,
                executable, self._path, compiler, compile_options)
        kwargs = {'executable': executable, 'path': self._path,
                    'cc_code': self.generate_code(),
                    'keep_cc': self._keep_cc, 'cmd': self._compile_cmd}
        if remote_apply is None:
            if not os.path.isdir(_full_expand(qsd_headers)):
                logger.warn("Header directory "+qsd_headers+" does not exist")
            if not os.path.isfile(_full_expand(qsd_lib)):
                logger.warn("File "+qsd_lib+" does not exist")
            try:
                compilation_worker(kwargs)
            except sp.CalledProcessError as exc_info:
                logger.error("command '{cmd:s}' failed with code {code:d}"
                                .format(cmd=self._compile_cmd,
                                        code=int(exc_info.returncode)))
                raise
        else:
            remote_apply(compilation_worker, kwargs)
        # We set the executable only at the very end so that we can use it as
        # an indicator whether the compile method is complete
        self._executable = executable

    def _build_compile_cmd(self, qsd_lib, qsd_headers, executable, path,
            compiler, compile_options):
        # For debugging purposes, it can be useful to call
        # _cmd_list_to_str(_build_compile_cmd(...))
        # instead of the compile method
        link_dir, libqsd_a = os.path.split(qsd_lib)
        cc_file = executable + '.cc'
        if not libqsd_a == self._lib_qsd:
            raise ValueError("qsd_lib "+qsd_lib+" does not point to a "
                             "file of the name "+self._lib_qsd)
        return ([compiler, ] + shlex.split(compile_options)
                + ['-I%s'%qsd_headers, '-o', executable, cc_file]
                + ['-L%s'%link_dir, self._link_qsd])

    def run(self, seed=None, workdir=None, keep=False, delay=False):
        """Run the QSD program. The :meth:`compile` method must have been
        called before `run`. If :meth:`compile` was called with
        ``delay=True``, compile at this point and run the resulting program.
        Otherwise, just run the existing program from the earlier compilation.
        The resulting directory data is returned, and in addition the
        `traj_data` attribute is updated to include the new trajectories (in
        addition to any previous trajectories)

        The `run` method may be called repeatedly to accumulate trajectories.

        Arguments:
            seed (int): Random number generator seed (unsigned integer), will
                be passed to the executable as the only argument.
            workdir (str or None): The directory in which to (temporarily)
                create the output files. If None, a temporary directory will be
                used. Otherwise, the `workdir` must exist. Environment
                variables and '~' will be expanded.
            keep (bool): If True, keep QSD output files inside `workdir`.
            delay (bool): If True, schedule the run to be performed at a later
                point in time, when the :meth:`run_delayed` routine is called.

        Returns:
            qnet.misc.trajectory_data.TrajectoryData: Averaged data obtained
            from the newly simulated trajectories only. None if `delay=True`.

        Raises:
            QSDCodeGenError: if :meth:`compile` was not called
            OSError: if creating/removing files/folders fails
            subprocess.CalledProcessError: if delayed compilation fails or
                executable returns with non-zero exit code
            ValueError: if seed is not unique

        Note:
            The only way to run multiple trajectories in parallel is by giving
            ``delay=True``. After preparing an arbitrary number of trajectories
            by repeated calls to :meth:`run`. Then :meth:`run_delayed` must be
            called with a `map` argument that supports parallel execution.

        """
        if self._executable is None:
            raise QSDCodeGenError("Call compile method first")
        if self.traj_data is not None:
            if ( (seed in self.traj_data.record_seeds)
            or   (seed in self._delayed_seeds) ):
                raise ValueError("Seed %d already in record or in delayed run"
                                 % seed)
        if seed is None:
            seed = random.randint(0, UNSIGNED_MAXINT)
            # ensure we don't reuse an existing or schedules seed
            while ( (seed in self.traj_data.record_seeds)
            or      (seed in self._delayed_seeds) ):
                seed = random.randint(0, UNSIGNED_MAXINT)
            if delay:
                while seed in delayed_seed:
                    seed = random.randint(0, UNSIGNED_MAXINT)
        kwargs = {
                'executable': self._executable, 'keep': keep,
                'path': self._path, 'seed': seed, 'workdir': workdir,
                'operators': OrderedDict(
                                [(name, fn) for (name, (__, fn))
                                 in self._observables.items()]),
        }
        if delay:
            self._delayed_runs_kwargs.append(kwargs)
            self._delayed_seeds.append(kwargs['seed'])
        else:
            traj = qsd_run_worker(kwargs)
            if self.traj_data is None:
                self.traj_data = traj.copy()
            else:
                self.traj_data += traj
            return traj

    def run_delayed(self, map=map, n_procs_extend=1, _run_worker=None):
        """Execute all scheduled runs (see `delay` option in :meth:`run`
        method), possibly in parallel.

        Arguments:
            map (callable): ``map(qsd_run_worker, list_of_kwargs)``
                must be equivalent to
                ``[qsd_run_worker(kwargs) for kwargs in list_of_kwargs]``.
                Defaults to the builtin `map` routine, which will process the
                scheduled runs serially.
            n_procs_extend (int): Number of local processes to use when
                averaging over trajectories.

        Raises:
            TypeError: If `map` does not return a list of
                :class:`~qnet.misc.trajectory_data.TrajectoryData` instances.

        Note:
            Parallel execution is achieved by passing an appropriate `map`
            routine. For example, ``map=multiprocessing.Pool(5).map`` would use
            a local thread pool of 5 workers. Another alternative would be the
            `map` method of an ``ipyparallel`` View. If (and only if) the View
            connects remote IPython engines, :meth:`compile` must have been
            called with an appropriate `remote_apply` argument that compiled
            the QSD program on all of the remote engines.
        """
        if _run_worker is None:
            _run_worker = qsd_run_worker
        trajs = []
        try:
            trajs = list(map(_run_worker, self._delayed_runs_kwargs))
            self._delayed_runs_kwargs = []
            self._delayed_seeds = []
            if self.traj_data is None:
                self.traj_data = trajs[0].copy()
                if len(trajs) > 1:
                    self.traj_data.extend(*trajs[1:], n_procs=n_procs_extend)
            else:
                self.traj_data.extend(*trajs, n_procs=n_procs_extend)
        except (IndexError, TypeError, AttributeError):
            raise TypeError("`map ` does not return a list of TrajectoryData "
                            "instances.")
        return trajs

    def __str__(self):
        return self.generate_code()

    def _operator_basis_lines(self, indent=2):
        """Return a multiline string of C++ code that defines and initializes
        all operators in the system"""
        # QSD demands that we first define all "special" operators (instances
        # of classes derived from PrimaryOperator) before an instance of the
        # class Operator (algebraic combinations of other operators).
        # Therefore, we collect the lines for these two cases separately.
        special_op_lines = []
        general_op_lines = []
        for k in range(len(self._local_spaces)):
            special_op_lines.append("IdentityOperator Id{k}({k});".format(k=k))
        for op in self._qsd_ops:
            # We assume that self._qsd_ops is an OrderedDict, so that
            # instantiations may refer to earlier operators
            line = self._qsd_ops[op].instantiation
            if line.startswith("Operator "):
                general_op_lines.append(line)
            else:
                special_op_lines.append(line)
        return "\n".join(_indent(special_op_lines + general_op_lines, indent))

    def _parameters_lines(self, indent=2):
        """Return a multiline string of C++ code that defines all numerical
        constants"""
        self._update_var_names() # should be superfluous, but just to be safe
        lines = set() # sorting will happen at the very end
        lines.add("Complex I(0.0,1.0);")
        for s in list(self.syms):
            var = self._var_names[s]
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
                          .format(var, val.real))
            else:
                lines.add("Complex {!s}({:g},{:g});"
                          .format(var, val.real, val.imag))
        return "\n".join(_indent(sorted(lines), indent))


    def _observables_lines(self, indent=2):
        """Return a multiline string of C++ code that defines all
        observables"""
        lines = []
        n_of_out = len(self._observables)
        lines.append('const int nOfOut = %d;' % n_of_out)
        outlist_lines = []
        outfiles = []
        if len(self._observables) < 1:
            raise QSDCodeGenError("Must register at least one observable")
        for (observable, outfile) in self._observables.values():
            outlist_lines.append(self._operator_str(observable))
            outfiles.append(outfile)
        lines.extend(("Operator outlist[nOfOut] = {\n  "
                      + ",\n  ".join(outlist_lines)).split("\n"))
        lines.append("};")
        lines.append('char *flist[nOfOut] = {{{filenames}}};'
                     .format(filenames=", ".join(
                         [('"'+fn+'"') for fn in outfiles]
                     )))
        lines.append(r'int pipe[4] = {1,2,3,4};')
        return "\n".join(_indent(lines, indent))


    def _operator_str(self, op):
        """For a given instance of ``qnet.algebra.operator_algebra.Operator``,
        recursively generate the C++ expression that will instantiate the
        operator.
        """
        if isinstance(op, LocalOperator):
            return str(self._qsd_ops[op])
        elif isinstance(op, ScalarTimesOperator):
            if op.coeff in self._tfuncs:
                func_placeholder = self._tfuncs[op.coeff][1]
                return "%s * (%s)" % (func_placeholder,
                                      self._operator_str(op.term))
            else:
                return "(%s) * (%s)" % (self._scalar_str(op.coeff),
                                        self._operator_str(op.term))
        elif isinstance(op, OperatorPlus):
            str_expr = " + ".join([self._operator_str(o) for o in op.operands])
            return "(" + str_expr + ")"
        elif isinstance(op, OperatorTimes):
            str_expr = " * ".join([self._operator_str(o) for o in op.operands])
            return "(" + str_expr + ")"
        elif op is IdentityOperator:
            return str(self._qsd_ops[op])
        else:
            raise TypeError(str(op))

    def _ket_str(self, ket):
        """For a given instance of ``qnet.algebra.state_algebra.Ket``,
        recursively generate the C++ expression that will instantiate the
        state.
        """
        if isinstance(ket, (LocalKet, TensorKet)):
            return str(self._qsd_states[ket])
        elif isinstance(ket, ScalarTimesKet):
            return "({}) * ({})".format(self._scalar_str(ket.coeff),
                    self._ket_str(ket.term))
        elif isinstance(ket, KetPlus):
            return "({})".format(" + ".join([self._ket_str(o)
                for o in ket.operands]))
        else:
            raise TypeError(str(ket))


    def _scalar_str(self, sc, assign_to=None):
        ccode = QSDCCodePrinter(settings={'user_symbols':self._var_names})
        return ccode.doprint(sc, assign_to=assign_to)

    def _hamiltonian_lines(self, indent=2):
        H = self.circuit.H
        return " "*indent + "Operator H = "+self._operator_str(H)+";"

    def _lindblads_lines(self, indent=2):
        lines = []
        lines.append('const int nL = {nL};'.format(nL=self.circuit.cdim))
        L_op_lines = []
        for L_op in self.circuit.L.matrix.flatten():
            L_op_lines.append(self._operator_str(L_op))
        lines.extend(
            ("Operator L[nL]={\n  " + ",\n  ".join(L_op_lines) + "\n};")
            .split("\n"))
        return "\n".join(_indent(lines, indent))

    def _function_lines(self, ops, indent=2):
        if self.time_symbol is None:
            return ''
        func_lines = []
        tfunc_counter = 0
        for op in ops:
            for coeff in _find_time_dependent_coeffs(op, self.time_symbol):
                if coeff in self._tfuncs:
                    continue
                is_real = coeff.is_real
                if is_real:
                    func_type = 'double'
                else:
                    func_type = 'Complex'
                tfunc_counter += 1
                func_name = "tfunc%d" % tfunc_counter
                # choose a variable name for the time-dependent coefficient
                u = "u%d" % tfunc_counter
                func_placeholder = u
                u_counter = 1
                while func_placeholder in self.syms:
                    func_placeholder = "%s_%d" % (u, u_counter)
                    u_counter += 1
                # remember that the _var_name for the time_symbol was set to
                # 't' in the __init__ routine
                func_lines.append("%s %s(double t)" % (func_type, func_name))
                func_lines.append("{")
                func_lines.append("  "+func_type+" "+func_placeholder+";")
                func_lines.append("  "+self._scalar_str(coeff,
                                       assign_to=func_placeholder))
                func_lines.append("  return "+func_placeholder+";")
                func_lines.append("}")
                func_lines.append("")
                self._tfuncs[coeff] = (func_name, func_placeholder, is_real)
        if len(func_lines) > 0:
            lines = ["", ] + func_lines + ["", ]
            for coeff in self._tfuncs:
                func_name, func_placeholder, is_real = self._tfuncs[coeff]
                if is_real:
                    lines.append("RealFunction %s = %s;"
                                 % (func_placeholder, func_name))
                else:
                    lines.append("ComplexFunction %s = %s;"
                                 % (func_placeholder, func_name))
        else:
            lines = []
        return "\n".join(_indent(lines, indent))



def _find_time_dependent_coeffs(op, time_symbol):
    if isinstance(op, ScalarTimesOperator):
        if time_symbol in scalar_free_symbols(op.coeff):
            yield op.coeff
    else:
        try:
            for operand in op.operands:
                for coeff in _find_time_dependent_coeffs(operand, time_symbol):
                    yield coeff
        except AttributeError:
            # e.g. IdentityOperator has no attribute 'operands'
            pass


def _full_expand(s):
    if s is None:
        return s
    return os.path.expanduser(os.path.expandvars(s))


def expand_cmd(cmd):
    """Return a copy of the array `cmd`, where for each element of the `cmd`
    array, environment variables and '~' are expanded"""
    if isinstance(cmd, str):
        raise TypeError("cmd must be a list")
    cmd_expanded = []
    for part in cmd:
        cmd_expanded.append(_full_expand(part))
    return cmd_expanded


def compilation_worker(kwargs, _runner=None):
    """Worker to perform compilation, suitable e.g. for being run on an
    IPython cluster. All arguments are in the `kwargs` dictionary.

    Keys:
        executable (str): Name of the executable to be created. Nothing will be
            expanded.
        path (str): Path where the executable should be created, as absolute
            path or relative to the current working directory. Environment
            variables and '~' will be expanded.
        cc_code (str): Multiline string that contains the entire C++ program to
            be compiled
        keep_cc (bool): Keep C++ file after compilation? It will have the
            same name as the executable, with an added ``.cc`` file extension.
        cmd (list of str): Command line arguments (see `args` in
            `subprocess.check_output`).  In each argument, environment
            variables are expanded, and '~' is
            expanded to ``$HOME``. It must meet the following requirements:

            * the compiler (first argument) must be in the ``$PATH``
            * Invocation of the command must compile a C++ file with the name
              `executable`.cc in the current working directory to `exectuable`,
              also in the current working directoy. It must *not* take into
              account `path`. This is because the working directory for the
              subprocess handling the command invocation will be set to `path`.
              Thus, that is where the executable will be created.

    Returns:
        Absolute path of the compiled executable

    Raises:
        subprocess.CalledProcessError: if compilation fails
        OSError: if creating/removing files/folder fails
    """
    # we import subprocess locally, to make the routine completely
    # self-contained. This is a requirement e.g. to be a worker on an IPython
    # cluster. To still allow testing, we have the undocumented _runner
    # parameter
    import subprocess as sp
    import os
    if _runner is None:
        _runner = sp.check_output
    executable = str(kwargs['executable'])
    path = _full_expand(str(kwargs['path']))
    cc_code = str(kwargs['cc_code'])
    keep_cc = kwargs['keep_cc']
    cmd = expand_cmd(kwargs['cmd'])
    cc_file = executable + '.cc'
    try:
        os.makedirs(path)
    except OSError:
        # Ignore existing directory
        pass
    with open(os.path.join(path, cc_file), 'w') as out_fh:
        out_fh.write(cc_code)
        out_fh.write("\n")
    executable_abs = os.path.abspath(os.path.join(path, executable))
    try:
        _runner(cmd, stderr=sp.STDOUT, cwd=path)
    finally:
        if not keep_cc:
            os.unlink(cc_file)
    if os.path.isfile(executable_abs) and os.access(executable_abs, os.X_OK):
        return executable_abs
    else:
        raise sp.CalledProcessError("Compilation did not create executable %s"
                                    % executable_abs)


def qsd_run_worker(kwargs, _runner=None):
    """Worker to perform run of a previously compiled program (see
    :func:`compilation_worker`), suitable e.g. for being run on an
    IPython cluster. All arguments are in the `kwargs` dictionary.

    Keys:
        executable (str): Name of the executable to be run. Nothing will be
            expanded.  This should generally be only the name of the
            executable, but it can also be a path relative to
            ``kwargs['path']``, or a (fully expanded) absolute path, in which
            case ``kwargs['path']`` is ignored.
        path (str): Path where the executable can be found, as absolute path or
            relative to the current working directory. Environment variables
            and '~' will be expanded.
        seed (int): Seed (unsigned int) to be passed as argument to the
            executable
        operators(dict or OrderedDict of str to str)): Mapping of operator name
            to filename, see `operators` parameter of
            :meth:`~qnet.misc.trajectory_data.TrajectoryData.from_qsd_data`
        workdir (str or None): The working directory in which to execute the
            executable (relative to the current working directory). The output
            files defined in `operators` will be created in this folder. If
            None, a temporary directory will be used. If `workdir` does not
            exist yet, it will be created.
        keep (bool): If True, keep the QSD output files. If False,
            remove the output files as well as any parent folders that may have
            been created alongside with `workdir`

    Raises:
        FileNotFoundError: if `executable` does not exist in `path`

    Returns:
        Expectation values and variances of the observables, from the newly
        simulated trajectories only (instance of
        :obj:`~qnet.misc.trajectory_data.TrajectoryData`)
    """
    # imports *must* be local so that we can send this to an IPython engine
    import subprocess as sp
    import os
    import shutil
    import tempfile
    from qnet.misc.trajectory_data import TrajectoryData
    if _runner is None:
        _runner = sp.check_output
    executable = str(kwargs['executable'])
    path = os.path.abspath(_full_expand(str(kwargs['path'])))
    seed = int(kwargs['seed'])
    operators = kwargs['operators']
    workdir = _full_expand(kwargs['workdir'])
    if workdir is None:
        workdir = tempfile.mkdtemp()
    keep = kwargs['keep']
    delete_folder = None
    if not os.path.isdir(workdir):
        # If keep is True, we want to remove not only the QSD output files, but
        # also any folder that is newly created as part of workdir. Therefore,
        # before creating workdir, we walk up the path to find the topmost
        # nonexisting folder
        folder = os.path.abspath(workdir)
        while not os.path.isdir(folder):
            delete_folder = folder
            folder = os.path.abspath(os.path.join(folder, '..'))
        try:
            os.makedirs(workdir)
        except OSError:
            # This might happen sometimes when using multi-threading and
            # another thread has created the directory since the "isdir" check
            pass
    local_executable = _full_expand(os.path.join(path, executable))
    is_exe = lambda f: os.path.isfile(f) and os.access(f, os.X_OK)
    if not is_exe(local_executable):
        raise FileNotFoundError("No executable "+local_executable)
    cmd = [local_executable, str(seed)]
    _runner(cmd, stderr=sp.STDOUT, cwd=workdir)
    traj = TrajectoryData.from_qsd_data(operators, seed, workdir=workdir)
    if not keep:
        for filename in operators.values():
            os.unlink(os.path.join(workdir, filename))
        if delete_folder is not None:
            shutil.rmtree(delete_folder, ignore_errors=True)
    return traj


def sanitize_name(name, allowed_letters, replacements):
    """Return a sanitized `name`, where all letters that occur as keys in
    `replacements` are replaced by their corresponding values, and any letters
    that do not match `allowed_letters` are dropped

    Arguments:
        name (str): string to be sanitized
        allowed_letters (regex): compiled regular expression
            that any allowed letter must match
        replacement (dict of str to str): dictionary of mappings

    Returns:
        str: sanitized name

    Example:

    >>> sanitize_filename = partial(sanitize_name,
    ...         allowed_letters=re.compile(r'[.a-zA-Z0-9_-]'),
    ...         replacements={'^':'_', '+':'_', '*':'_', ' ':'_'})
    >>> sanitize_filename.__doc__ = "Sanitize name to be used as a filename"
    >>> sanitize_filename('\chi^{(1)}_1')
    'chi_1_1'
    """
    sanitized = ''
    for letter in name:
        if letter in replacements:
            letter = replacements[letter]
        if allowed_letters.match(letter):
            sanitized += letter
    return sanitized


def _indent(lines, indent=2):
    indented_lines = []
    for line in lines:
        if len(line) > 0:
            indented_lines.append(" "*indent + line)
        else:
            indented_lines.append(line)
    return indented_lines


def _cmd_list_to_str(cmd_list):
    result = ''
    for part in cmd_list:
        part = part.replace('"', '\\"')
        if " " in part:
            result += ' "%s"' % part
        else:
            result += ' %s' % part
    return result.strip()

sanitize_filename = partial(sanitize_name,
        allowed_letters=re.compile(r'[.a-zA-Z0-9_-]'),
        replacements={'^':'_', '+':'_', '*':'_', ' ':'_'})
sanitize_filename.__doc__ = "Sanitize name to be used as a filename"


sanitize_varname = partial(sanitize_name,
        allowed_letters=re.compile(r'[a-zA-Z0-9_]'),
        replacements={'^':'_', '+':'_', '*':'_', ' ':'_', '-':'_', '.':'_'})
sanitize_filename.__doc__ = "Sanitize name to be used as a C++ variable name"
