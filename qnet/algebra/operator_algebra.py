# coding=utf-8
"""
operator_algebra.py

The specification of a quantum mechanical symbolic operator algebra.
Basic elements from which expressions can be built are operator symbols and locally acting operators.
Each operator has an associated `space` property which gives the Hilbert space on which it acts non-trivially.
In order to not have to specify all degrees of freedom in advance, an operator is assumed to act as the identity on
all degrees of freedom that are independent of its space, as is customary in the physics literature.

    >>> x = OperatorSymbol("x", "1")
    >>> x.space
    LocalSpace("1",...)

The available local operator types are

    Create(localspace)
    Destroy(localspace)
    LocalSigma(localspace, j, k)

There exist some useful constants to specify neutral elements of Operator addition and multiplication:
    OperatorZero
    OperatorIdentity

Quantum Operator objects can be added together in code via the infix '+' operator and multiplied with the infix '*' operator.
They can also be added to or multiplied by scalar objects.
In the first case, the scalar object is multiplied by the IdentityOperator constant.

Operations involving at least one quantum Operator argument are

    OperatorPlus(A, B, C, ...)
    OperatorTimes(A, B, C, ...)
    ScalarTimesOperator(coefficient, term)
    Adjoint(op)
    PseudoInverse(op)

"""
from __future__ import division
from abstract_algebra import *
from hilbert_space_algebra import *
from permutation_algebra import *
from itertools import product as cartesian_product
import qutip
from sympy import exp, log, cos, sin, cosh, sinh, tan, cot,\
    acos, asin, acosh, asinh, atan, atan2, atanh, acot, sqrt,\
    factorial, pi, I, sympify, Basic as SympyBasic, symbols, Mul, Add

sympyOne = sympify(1)

from numpy import array as np_array,\
    shape as np_shape,\
    hstack as np_hstack,\
    vstack as np_vstack,\
    diag as np_diag,\
    ones as np_ones,\
    conjugate as np_conjugate,\
    zeros as np_zeros,\
    ndarray,\
    arange,\
    cos as np_cos,\
    sin as np_sin,\
    eye as np_eye


class Operator(object):
    """
    The basic operator class, which fixes the abstract interface of operator objects 
    and where possible also defines the default behavior under operations.
    Any operator contains an associated HilbertSpace object, 
    on which it is taken to act non-trivially.
    """

    # which data types may serve as scalar coefficients
    scalar_types = int, long, float, complex, SympyBasic


    @property
    def space(self):
        """
        The Hilbert space associated with the operator on which it acts non-trivially
        """
        return self._space

    @property
    def _space(self):
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """
        :return: The Hermitian adjoint of the operator.
        :rtype: Operator
        """
        return self._adjoint()

    def _adjoint(self):
        return Adjoint.create(self)

    conjugate = dag = adjoint

    def pseudo_inverse(self):
        """
        :return: The pseudo-Inverse of the Operator, i.e., it inverts the operator on the orthogonal complement of its nullspace
        :rtype: Operator
        """
        return self._pseudo_inverse()

    def _pseudo_inverse(self):
        return PseudoInverse.create(self)

    def to_qutip(self, full_space = None):
        """
        Create a numerical representation of the operator as a QuTiP object.
        Note that all symbolic scalar parameters need to be replaced by numerical values before calling this method.
        :param full_space: The full Hilbert space in which to represent the operator.
        :type full_space: HilbertSpace
        :return: The matrix representation of the operator.
        :rtype: qutip.Qobj
        """
        if full_space is None:
            full_space = self.space
        return self._to_qutip(full_space)

    def _to_qutip(self, full_space):
        raise NotImplementedError(str(self.__class__))

    def expand(self):
        """
        Expand out distributively all products of sums. Note that this does not expand out sums of scalar coefficients.
        :return: A fully expanded sum of operators.
        :rtype: Operator
        """
        return self._expand()

    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def series_expand(self, param, about, order):
        """
        Expand the operator expression as a truncated power series in a scalar parameter.
        :param param: Expansion parameter.
        :type param: sympy.core.symbol.Symbol
        :param about: Point about which to expand.
        :type about:  Any one of Operator.scalar_types
        :param order: Maximum order of expansion.
        :type order: int >= 0
        """
        return self._series_expand(param, about, order)

    def _series_expand(self, param, about, order):
        raise NotImplementedError(self.__class__.__name__)


    def __add__(self, other):
        if isinstance(other, Operator.scalar_types):
            return OperatorPlus.create(self, other * IdentityOperator)
        elif isinstance(other, Operator):
            return OperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        elif isinstance(other, Operator):
            return OperatorTimes.create(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented


@check_signature
class OperatorSymbol(Operator, Operation):
    """
    Operator Symbol class, parametrized by an identifier string and an associated Hilbert space.
    OperatorSymbol(
    :param name: Symbol identifier
    :type name: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """
    signature = str, (HilbertSpace, str, int, tuple)

    def __init__(self, name, hs):
        if isinstance(hs, (str, int)):
            hs = local_space(hs)
        elif isinstance(hs, tuple):
            hs = prod([local_space(h) for h in hs], neutral=TrivialSpace)
        super(OperatorSymbol, self).__init__(name, hs)

    def __str__(self):
        return self.operands[0]

    def _tex(self):
        return "{" + self.operands[0] + "}"

    def _to_qutip(self, full_space=None):
        raise AlgebraError("Cannot convert operator symbol to representation matrix. Substitute first.")

    @property
    def _space(self):
        return self.operands[1]

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return self


@singleton
class IdentityOperator(Operator):
    """
    IdentityOperator constant (singleton) object. Use as IdentityOperator, *NOT* as IdentityOperator().
    """

    @property
    def _space(self):
        return TrivialSpace

    def _adjoint(self):
        return self

    def _to_qutip(self, full_space):
        return qutip.tensor(*[qutip.qeye(s.dimension) for s in full_space.local_factors()])

#    def mathematica(self):
#        return "IdentityOperator"

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return self

    def _pseudo_inverse(self):
        return self

    def _tex(self):
        return "1"

    def __repr__(self):
        return "IdentityOperator"

    def __str__(self):
        return "1"

    def __eq__(self, other):
        return self is other or other == 1

    def __hash__(self):
        return hash(self.__class__)



from scipy.sparse import csr_matrix

@singleton
class OperatorZero(Operator):
    """
    ZeroOperator constant (singleton) object. Use as IdentityOperator, *NOT* as IdentityOperator().
    """

    @property
    def _space(self):
        return TrivialSpace

    def _adjoint(self):
        return self

    def _to_qutip(self, full_space):
        return qutip.tensor(
            *[qutip.Qobj(csr_matrix((), (s.dimension, s.dimension))) for s in full_space.local_factors()])

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return self

    def _pseudo_inverse(self):
        return self

    def _tex(self):
        return "0"

    def __eq__(self, other):
        return self is other or other == 0

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "0"

    def __repr__(self):
        return "OperatorZero"


def implied_local_space_mtd(dcls, clsmtd, cls, space, *ops):
    """
    For Operations whose first operand is a local space, accept int or str arguments and convert these to a LocalSpace via local_space().
    """
    if isinstance(space, (int, str)):
        space = local_space(space)
    return clsmtd(cls, space, *ops)

implied_local_space = preprocess_create_with(implied_local_space_mtd)

class LocalOperator(Operator, Operation):
    """
    Base class for all kinds of operators that act *locally*,
    i.e. only on a single degree of freedom.
    """

    def __init__(self, hs, *args):
        if isinstance(hs, (str, int)):
            hs = local_space(hs)
        super(LocalOperator, self).__init__(hs, *args)

    @property
    def _space(self):
        return self.operands[0]

    def _to_qutip(self, full_space=None):
        if full_space is None or full_space == self.space:
            return self.to_qutip_local_factor()
        else:
            all_spaces = full_space.local_factors()
            own_space_index = all_spaces.index(self.space)
            return qutip.tensor(*([qutip.qeye(s.dimension) for s in all_spaces[:own_space_index]]
                                  + [self.to_qutip_local_factor()]
                                  + [qutip.qeye(s.dimension) for s in all_spaces[own_space_index + 1:]]))

    def to_qutip_local_factor(self):
        """
        :return: Return a qutip representation for the local operator only on its local space.
        :rtype: qutip.Qobj
        """
        return self._to_qutip_local_factor()

    def _to_qutip_local_factor(self):
        raise NotImplementedError(self.__class__.__name__)

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return self


@implied_local_space
@check_signature
class Create(LocalOperator):
    """
    Create(space) yields a bosonic creation operator acting on a particular local space/degree of freedom.
    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """
    signature = (LocalSpace, str, int),

    def _to_qutip_local_factor(self):
        return qutip.create(self.space.dimension)

    def _tex(self):
        return r"{{a_{}^\dagger}}".format(self.space.tex())


@implied_local_space
@check_signature
class Destroy(LocalOperator):
    """
    Destroy(space) yields a bosonic annihilation operator acting on a particular local space/degree of freedom.
    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """
    signature = (LocalSpace, str, int),

    def _to_qutip_local_factor(self):
        return qutip.destroy(self.space.dimension)

    def _tex(self):
        return r"{{a_{}}}".format(self.space.tex())

@implied_local_space
@match_replace
@check_signature
class Phase(LocalOperator):
    """
        Phase(space, phi)
    yields a Phase operator acting on a particular local space/degree of freedom:
        `exp(I * phi * Create(space) * Destroy(space))`

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param phi: Displacement amplitude.
    :type phi: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, str, int), Operator.scalar_types
    rules = []

    def _to_qutip_local_factor(self):
        arg = complex(self.operands[1]) * arange(self.space.dimension)
        d = np_cos(arg) + 1j * np_sin(arg)
        return qutip.Qobj(np_diag(d))

    def _adjoint(self):
        return Phase(self.operands[0], -self.operands[1].conjugate())

    def _pseudo_inverse(self):
        return Phase(self.operands[0], -self.operands[1])

    def _tex(self):
        return r"{{P_{}({})}}".format(self.space.tex(), tex(self.operands[1]))

@implied_local_space
@match_replace
@check_signature
class Displace(LocalOperator):
    """
        Displace(space, alpha)
    yields a coherent Displacement operator acting on a particular local space/degree of freedom:
        `exp(alpha * Create(space) - alpha.conjugate() * Destroy(space))`
    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param alpha: Displacement amplitude.
    :type alpha: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, str, int), Operator.scalar_types
    rules = []
    def _to_qutip_local_factor(self):
        return qutip.displace(self.space.dimension, complex(self.operands[1]))

    def _adjoint(self):
        return Displace(self.operands[0], -self.operands[1])

    _pseudo_inverse = _adjoint

    def _tex(self):
        return r"{{D_{}({})}}".format(self.space.tex(), tex(self.operands[1]))

@implied_local_space
@match_replace
@check_signature
class Squeeze(LocalOperator):
    """
        Squeeze(space, eta)
    yields a Squeeze operator acting on a particular local space/degree of freedom:
        `exp((1/2) * (eta * Create(space) * Create(space) - eta.conjugate() * Destroy(space) * Destroy(space)))`
    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param eta: Squeeze parameter.
    :type eta: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, str, int), Operator.scalar_types
    rules = []
    def _to_qutip_local_factor(self):
        return qutip.displace(self.space.dimension, complex(self.operands[1]))

    def _adjoint(self):
        return Squeeze(self.operands[0], -self.operands[1])

    _pseudo_inverse = _adjoint

    def _tex(self):
        return r"{{S_{}({})}}".format(self.space.tex(), tex(self.operands[1]))




@implied_local_space
@check_signature
class LocalSigma(LocalOperator):
    """
    LocalSigma(space, j, k) yields a sigma_jk = | j >< k | operator acting on a particular local space/degree of freedom.
    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param j: State label j.
    :type j: int or str
    :param k: State label k.
    :type k: int or str
    """

    signature = (LocalSpace, str, int), (int, str), (int, str)

    def _to_qutip_local_factor(self):
        k, j = self.operands[1:]
        ket = qutip.basis(self.space.dimension, self.space.basis.index(k))
        bra = qutip.basis(self.space.dimension, self.space.basis.index(j)).dag()
        return ket * bra

    def _tex(self):
        j, k = self.operands[1:]
        if k == j:
            return r"{{\Pi_{}^{}}}".format(tex(k), self.space.tex())
        return r"{{\sigma_{{{},{}}}^{}}}".format(tex(j), tex(k), self.space.tex())



LocalProjector = lambda spc, state: LocalSigma.create(spc, state, state)


def X(local_space, states=("h", "g")):
    """
    Pauli-type X-operator
    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels (|0>, |1>), where Z|0> = +|0>.
    :type states: tuple with two elements of type int or str
    :return: Local X-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h)


def Y(local_space, states=("h", "g")):
    """
    Pauli-type Y-operator
    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels (|0>, |1>), where Z|0> = +|0>.
    :type states: tuple with two elements of type int or str
    :return: Local Y-operator.
    :rtype: Operator
    """
    h, g = states
    return I * (-LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h))


def Z(local_space, states=("h", "g")):
    """
    Pauli-type Z-operator
    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels (|0>, |1>), where Z|0> = +|0>.
    :type states: tuple with two elements of type int or str
    :return: Local Z-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalProjector(local_space, h) - LocalProjector(local_space, g)


class OperatorOperation(Operator, Operation):
    """
    Base class for Operations acting only on Operator arguments.
    """
    signature = Operator,

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)




@flat
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorPlus(OperatorOperation):
    """
    A sum of Operators
        OperatorPlus(*summands)
    :param summands: Operator summands.
    :type summands: Operator
    """
    neutral_element = OperatorZero
    binary_rules = []

    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesOperator):
            return order_key(a.term), a.coeff
        return order_key(a), 1

    def _to_qutip(self, full_space=None):
        if full_space == None:
            full_space = self.space
        assert self.space <= full_space
        return sum((op.to_qutip(full_space) for op in self.operands), 0)

    def _expand(self):
        return sum((o.expand() for o in self.operands), OperatorZero)

    def _series_expand(self, param, about, order):
        res = sum((o.series_expand(param, about, order) for o in self.operands), OperatorZero)
        return res

    def _tex(self):
        ret = self.operands[0].tex()

        for o in self.operands[1:]:
            if isinstance(o, ScalarTimesOperator) and ScalarTimesOperator.has_minus_prefactor(o.coeff):
                ret += " - " + tex(-o)
            else:
                ret += " + " + tex(o)
        return ret


@flat
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorTimes(OperatorOperation):
    """
    A product of Operators that serves both as a product within a Hilbert space as well as a tensor product.
        OperatorTimes(*factors)
    :param factors: Operator factors.
    :type factors: Operator
    """

    neutral_element = IdentityOperator
    binary_rules = []

    class OperatorOrderKey(object):
        """
        Auxiliary class that generates the correct pseudo-order relation for operator products.
        Only operators acting on different Hilbert spaces are commuted to achieve the order specified in the full HilbertSpace.
        I.e., sorted(factors, key = OperatorOrderKey) achieves this ordering.
        """
        def __init__(self, op, space):
            self.op = op
            self.full = False
            self.trivial = False
            if isinstance(space, LocalSpace):
                self.local_spaces = {space.operands, }
            elif space is TrivialSpace:
                self.local_spaces = set(())
                self.trivial = True
            elif space is FullSpace:
                self.full = True
            else:
                assert isinstance(space, ProductSpace)
                self.local_spaces = {s.operands for s in space.operands}

        def __lt__(self, other):
            if self.trivial and other.trivial:
                return order_key(self.op) < order_key(other.op)

            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) < tuple(other.local_spaces)

        def __gt__(self, other):
            if self.trivial and other.trivial:
                return order_key(self.op) > order_key(other.op)

            if self.full or len(self.local_spaces & other.local_spaces):
                return False

            return tuple(self.local_spaces) > tuple(other.local_spaces)

        def __eq__(self, other):
            if self.trivial and other.trivial:
                return order_key(self.op) == order_key(other.op)

            return self.full or len(self.local_spaces & other.local_spaces) > 0

    @classmethod
    def order_key(cls, a):
        return cls.OperatorOrderKey(a, a.space)

    @classmethod
    def create(cls, *ops):
        if any(o == OperatorZero for o in ops):
            return OperatorZero
        return cls(*ops)

    def _to_qutip(self, full_space=None):

        # if any factor acts non-locally, we need to expand distributively.
        if any(len(op.space) > 1 for op in self.operands):
            return self.expand().to_qutip(full_space)

        if full_space == None:
            full_space = self.space

        all_spaces = full_space.local_factors()
        by_space = []
        ck = 0
        for ls in all_spaces:
            # group factors by associated local space
            ls_ops = [o.to_qutip() for o in self.operands if o.space == ls]
            if len(ls_ops):
                # compute factor associated with local space
                by_space.append(prod(ls_ops))
                ck += len(ls_ops)
            else:
                # if trivial action, take identity matrix
                by_space.append(qutip.qeye(ls.dimension))
        assert ck == len(self.operands)
        # combine local factors in tensor product
        return qutip.tensor(*by_space)

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,) for eo in eops]
        # iterate over a cartesian product of all factor summands, form product of each tuple and sum over result
        return sum((OperatorTimes.create(*combo) for combo in cartesian_product(*eopssummands)), OperatorZero)

    def _tex(self):
        ret = self.operands[0].tex()
        for o in self.operands[1:]:
            if isinstance(o, OperatorPlus):
                ret += r" \left({}\right) ".format(tex(o))
            else:
                ret += " {}".format(tex(o))
        return ret


@match_replace
@check_signature
class ScalarTimesOperator(Operator, Operation):
    """
    Multiply an operator by a scalar coefficient.
        ScalarTimesOperator(coefficient, term)
    :param coefficient: Scalar coefficient.
    :type coefficient: Any of Operator.scalar_types
    :param term: The operator that is multiplied.
    :type term: Operator
    """
    signature = Operator.scalar_types, Operator
    rules = []

    @staticmethod
    def has_minus_prefactor(c):
        """
        For a scalar object c, determine whether it is prepended by a "-" sign.
        """
        cs = str(c).strip()
        return cs[0] == "-"



    @property
    def _space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def __str__(self):
        coeff, term = self.operands

        if coeff == -1:
            return "(-%s)" % (term,)

        if isinstance(coeff,(int, float)) and coeff < 0:
            return "(%g) * %s" % (coeff, term)

        return "%s * %s" % (coeff, term)

    def _tex(self):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            cs = r" \left({}\right)".format(tex(coeff))
        else:
            cs = " {}".format(tex(coeff))

        if term == IdentityOperator:
            ct = ""
        if isinstance(term, OperatorPlus):
            ct = r" \left({}\right)".format(term.tex())
        else:
            ct = r" {}".format(term.tex())

        return cs + ct



    def _to_qutip(self, full_space=None):
        return complex(self.coeff) * self.term.to_qutip(full_space)

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, OperatorPlus):
            return sum((c * eto for eto in et.operands), OperatorZero)
        return c * et

    def _pseudo_inverse(self):
        c, t = self.operands
        return t.pseudo_inverse() / c



@check_signature
@match_replace
class Adjoint(OperatorOperation):
    """
    The symbolic Adjoint of an operator.
        Adjoint(op)
    :param op: The operator to take the adjoint of.
    :type op: Operator
    """
    @property
    def operand(self):
        return self.operands[0]

    rules = []

    def _to_qutip(self, full_space=None):
        return qutip.dag(self.operands[0].to_qutip(full_space))

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, OperatorPlus):
            return sum((eoo.adjoint() for eoo in eo.operands), OperatorZero)
        return eo._adjoint()

    def _pseudo_inverse(self):
        return self.operand.pseudo_inverse().adjoint()

    def _tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^\dagger"

    def __str__(self):
        return "({})^*".format(str(self.operand))

# for hilbert space dimensions less than or equal to this,
# compute numerically PseudoInverse and NullSpaceProjector representations
DENSE_DIMENSION_LIMIT = 1000

@check_signature
@match_replace
class PseudoInverse(OperatorOperation):
    """
    The symbolic pseudo-inverse of an operator.
        PseudoInverse(op)
    :param op: The operator to take the adjoint of.
    :type op: Operator
    """
    delegate_to_method = ScalarTimesOperator, Squeeze, Displace, OperatorZero.__class__, IdentityOperator.__class__

    @classmethod
    def create(cls, op):
        if isinstance(op, cls.delegate_to_method):
            return op._pseudo_inverse()
        return super(PseudoInverse, cls).create(op)

    @property
    def operand(self):
        return self.operands[0]

    rules = []

    def _to_qutip(self, full_space=None):
        mo = self.operand.to_qutip(full_space)
        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
            arr = mo.data.toarray()
            from scipy.linalg import pinv
            piarr = pinv(arr)
            pimo = qutip.Qobj(piarr)
            pimo.dims = mo.dims
            pimo.isherm = mo.isherm
            pimo.type = 'oper'
            return pimo
        raise NotImplementedError("Only implemented for smaller state spaces")
#        return qutip.dag(self.operands[0].to_qutip(full_space))

    def _expand(self):
        return self

    def _pseudo_inverse(self):
        return self.operand

    def _tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^+"

    def __str__(self):
        return "({})^+".format(str(self.operand))

PseudoInverse.delegate_to_method = PseudoInverse.delegate_to_method + (PseudoInverse,)


@check_signature
@match_replace
class NullSpaceProjector(OperatorOperation):
    """
    Returns a projection operator that projects onto the nullspace of its operand
        NullSpaceProjector(op)
    I.e. `op * NullSpaceProjector(op) == 0`
    :param op: Operator argument
    :type op: Operator
    """

    rules = []

    @property
    def operand(self):
        return self.operands[0]

    def to_qutip(self, full_space=None):
        mo = self.operand.to_qutip(full_space)
        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
            arr = mo.data.toarray()
            from scipy.linalg import svd
            # compute Singular Value Decomposition
            U, s, Vh = svd(arr)
            tol = 1e-8 * s[0]
            zero_svs = s < tol
            Vhzero = Vh[zero_svs,:]
            PKarr = Vhzero.conjugate().transpose().dot(Vhzero)
            PKmo = qutip.Qobj(PKarr)
            PKmo.dims = mo.dims
            PKmo.isherm = True
            PKmo.type = 'oper'
            return PKmo
        raise NotImplementedError("Only implemented for smaller state spaces")


    def _tex(self):
        return r"\mathcal{P}_{{\rm Ker}" + tex(self.operand) + "}"

    def __str__(self):
        return "P_ker({})".format(str(self.operand))




@implied_local_space
@match_replace
@check_signature
class OperatorTrace(Operator, Operation):
    signature = HilbertSpace, Operator
    rules = []

    def __init__(self, space, op):
        if isinstance(space, (int, str)):
            space = local_space(space)
        super(OperatorTrace, self).__init__(space, op)

    @property
    def _space(self):
        over_space, op = self.operands
        return op.space / over_space

    def _expand(self):
        s, o = self.operands
        return OperatorTrace.create(s, o.expand())

    def _tex(self):
        s, o = self.operands
        return r"{{\rm Tr}}_{{{}}} \left[ {} \right]".format(tex(s), tex(o))



## Expression rewriting rules
u = wc("u", head=Operator.scalar_types)
v = wc("v", head=Operator.scalar_types)

n = wc("n", head=(int, str))
m = wc("m", head=(int, str))

A = wc("A", head=Operator)
B = wc("B", head=Operator)
A_plus = wc("A", head=OperatorPlus)
A_times = wc("A", head=OperatorTimes)

ls = wc("ls", head=LocalSpace)
h1 = wc("h1", head = HilbertSpace)
h2 = wc("h2", head = HilbertSpace)
H_ProductSpace = wc("H", head = ProductSpace)

ra = wc("ra", head=(int, str))
rb = wc("rb", head=(int, str))
rc = wc("rc", head=(int, str))
rd = wc("rd", head=(int, str))

ScalarTimesOperator.rules += [
    ((1, A), lambda A: A),
    ((0, A), lambda A: OperatorZero),
    ((u, OperatorZero), lambda u: OperatorZero),
    ((u, ScalarTimesOperator(v, A)), lambda u, v, A: (u * v) * A)
]

OperatorPlus.binary_rules += [
    ((ScalarTimesOperator(u, A), ScalarTimesOperator(v, A)), lambda u, v, A: (u + v) * A),
    ((ScalarTimesOperator(u, A), A), lambda u, A: (u + 1) * A),
    ((A, ScalarTimesOperator(v, A)), lambda v, A: (1 + v) * A),
    ((A, A), lambda A: 2 * A),
]

OperatorTimes.binary_rules += [
    ((ScalarTimesOperator(u, A), B), lambda u, A, B: u * (A * B)),

    ((A, ScalarTimesOperator(u, B)), lambda A, u, B: u * (A * B)),

    ((LocalSigma(ls, ra, rb), LocalSigma(ls, rc, rd)),
     lambda ls, ra, rb, rc, rd: LocalSigma(ls, ra, rd)
                                    if rb == rc else OperatorZero),

    ((Create(ls), LocalSigma(ls, rc, rd)),
     lambda ls, rc, rd: sqrt(rc + 1) * LocalSigma(ls, rc + 1, rd)),

    ((Destroy(ls), LocalSigma(ls, rc, rd)),
     lambda ls, rc, rd: sqrt(rc) * LocalSigma(ls, rc - 1, rd)),

    ((LocalSigma(ls, rc, rd), Destroy(ls)),
     lambda ls, rc, rd: sqrt(rd + 1) * LocalSigma(ls, rc, rd + 1)),

    ((LocalSigma(ls, rc, rd), Create(ls)),
     lambda ls, rc, rd: sqrt(rd) * LocalSigma(ls, rc, rd - 1)),

    ((Destroy(ls), Create(ls)),
     lambda ls: IdentityOperator + Create(ls) * Destroy(ls)),

    ((Phase(ls, u), Phase(ls, v)), lambda ls, u, v: Phase.create(ls, u + v)),
    ((Displace(ls, u), Displace(ls, v)), lambda ls, u, v: exp((u*v.conjugate() - u.conjugate() * v)/2) * Displace.create(ls, u + v)),

    ((Destroy(ls), Phase(ls, u)), lambda ls, u: exp(I*u) * Phase(ls, u) * Destroy(ls)),
    ((Destroy(ls), Displace(ls, u)), lambda ls, u: Displace(ls, u) * (Destroy(ls) + u)),

    ((Phase(ls, u), Create(ls)), lambda ls, u: exp(I*u) * Create(ls) * Phase(ls, u)),
    ((Displace(ls, u), Create(ls)), lambda ls, u: (Create(ls) - u.conjugate())* Displace(ls, u)),

    ((Phase(ls, u), LocalSigma(ls, n, m)), lambda ls, u, n, m: exp(I* u * n) * LocalSigma(ls, n, m)),
    ((LocalSigma(ls, n, m), Phase(ls, u)), lambda ls, u, n, m: exp(I* u * m) * LocalSigma(ls, n, m)),
]

Adjoint.rules += [
    ((ScalarTimesOperator(u, A),), lambda u, A: conjugate(u) * A.adjoint()),
    ((A_plus,), lambda A: OperatorPlus.create(*[o.adjoint() for o in A.operands])),
    ((A_times,), lambda A: OperatorTimes.create(*[o.adjoint() for o in A.operands[::-1]])),
    ((Adjoint(A),), lambda A: A),
    ((Create(ls),), lambda ls: Destroy(ls)),
    ((Destroy(ls),), lambda ls: Create(ls)),
    ((LocalSigma(ls, ra, rb),), lambda ls, ra, rb: LocalSigma(ls, rb, ra)),
]

Displace.rules +=[
    ((ls, 0), lambda ls: IdentityOperator)
]
Phase.rules +=[
    ((ls, 0), lambda ls: IdentityOperator)
]
Squeeze.rules +=[
    ((ls, 0), lambda ls: IdentityOperator)
]


def factor_for_trace(ls, op):
    """
    Given a local space ls to take the partial trace over and an operator, factor the trace such that operators acting on
    disjoint degrees of freedom are pulled out of the trace. If the operator acts trivially on ls the trace yields only
    a pre-factor equal to the dimension of ls. If there are LocalSigma operators among a product, the trace's cyclical property
    is used to move to sandwich the full product by LocalSigma operators:
        Tr A sigma_jk B = Tr sigma_jk B A sigma_jj
    :param ls: Degree of Freedom to trace over
    :type ls: HilbertSpace
    :param op: Operator to take the trace of
    :type op: Operator
    :return: The (partial) trace over the operator's spc-degrees of freedom
    :rtype: Operator
    """
    if op.space == ls:
        if isinstance(op, OperatorTimes):
            pull_out = [o for o in op.operands if o.space is TrivialSpace]
            rest = [o for o in op.operands if o.space is not TrivialSpace]
            if pull_out:
                return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
        raise CannotSimplify()
    if ls & op.space == TrivialSpace:
        return ls.dimension * op
    if ls < op.space and isinstance(op, OperatorTimes):

        pull_out = [o for o in op.operands if (o.space & ls) == TrivialSpace]

        rest = [o for o in op.operands if (o.space & ls) != TrivialSpace]
        if not isinstance(rest[0], LocalSigma) or not isinstance(rest[-1], LocalSigma):
            found_ls = False
            for j, r in enumerate(rest):
                if isinstance(r, LocalSigma):
                    found_ls = True
                    break
            if found_ls:
                m, n = r.operands[1:]
                rest = rest[j:] + rest[:j] + [LocalSigma(ls, m, m)]
        if not rest:
            rest = [IdentityOperator]
        if len(pull_out):
            return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
    raise CannotSimplify()


def decompose_space(H, A):
    return OperatorTrace.create(ProductSpace.create(*H.operands[:-1]),
        OperatorTrace.create(H.operands[-1], A))

OperatorTrace.rules += [
    ((TrivialSpace, A), lambda A: A),
    ((h1, OperatorZero), lambda h1: OperatorZero),
    ((h1, IdentityOperator), lambda h1: h1.dimension * IdentityOperator),
    ((h1, A_plus), lambda h1, A: sum(OperatorTrace.create(h1, o) for o in A.operands)),
    ((h1, Adjoint(A)), lambda h1, A: Adjoint.create(OperatorTrace.create(h1, A))),
    ((h1, ScalarTimesOperator(u, A)), lambda h1, u, A: u * OperatorTrace.create(h1, A)),
    ((H_ProductSpace, A), lambda H, A : decompose_space(H, A)),
    ((ls, Create(ls)), lambda ls: OperatorZero),
    ((ls, Destroy(ls)), lambda ls: OperatorZero),
    ((ls, LocalSigma(ls, n, m)), lambda ls, n, m: IdentityOperator if n == m else OperatorZero),
    ((ls, A), lambda ls, A: factor_for_trace(ls, A)),
]




class NonSquareMatrix(Exception):
    pass


class OperatorMatrix(Expression):
    """
    Matrix with Operator (or scalar-) valued elements.

    """
    matrix = None
    _hash = None

    def __init__(self, m):
        if isinstance(m, ndarray):
            self.matrix = m
        elif isinstance(m, OperatorMatrix):
            self.matrix = np_array(m.matrix)
        else:
            self.matrix = np_array(m)
        if len(self.matrix.shape) < 2:
            self.matrix = self.matrix.reshape((1, self.matrix.shape[0]))
        if len(self.matrix.shape) > 2:
            raise ValueError()


    @property
    def shape(self):
        return self.matrix.shape

    def __hash__(self):
        if not self._hash:
            self._hash = hash((tuple(self.matrix.flatten()), self.matrix.shape, OperatorMatrix))
        return self._hash

    def __eq__(self, other):
        return isinstance(other, OperatorMatrix) and (self.matrix == other.matrix).all()

    def __add__(self, other):
        if isinstance(other, OperatorMatrix):
            return OperatorMatrix(self.matrix + other.matrix)
        else: return OperatorMatrix(self.matrix + other)

    def __radd__(self, other):
        return OperatorMatrix(other + self.matrix)

    def __mul__(self, other):
        if isinstance(other, OperatorMatrix):
            return OperatorMatrix(self.matrix.dot(other.matrix))
        else: return OperatorMatrix(self.matrix * other)

    def __rmul__(self, other):
        return OperatorMatrix(other * self.matrix)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self


    #    @trace
    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented

    __truediv__ = __div__

    #    def __pow__(self, power):
    #        return OperatorMatrix(self.matrix.__pow__(power))

    def transpose(self):
        """
        :return: The transpose matrix
        :rtype: OperatorMatrix
        """
        return OperatorMatrix(self.matrix.T)

    def conjugate(self):
        """
        The element-wise conjugate matrix, i.e., if an element is an operator this means the adjoint operator,
        but no transposition of matrix elements takes place.
        :return: Element-wise hermitian conjugate matrix.
        :rtype: OperatorMatrix
        """
        return OperatorMatrix(np_conjugate(self.matrix))

    @property
    def T(self):
        """
        :return: Transpose matrix
        :rtype: OperatorMatrix
        """
        return self.transpose()

    def adjoint(self):
        """
        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
        """
        return self.T.conjugate()

    dag = adjoint

    def __repr__(self):
        return "OperatorMatrix({})".format(repr(self.matrix.tolist()))

    def trace(self):
        if self.shape[0] == self.shape[1]:
            return sum(self.matrix[k, k] for k in range(self.shape[0]))
        raise NonSquareMatrix(repr(self))


    @property
    def H(self):
        """
        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
        """
        return self.adjoint()


    def __getitem__(self, item_id):
        item = self.matrix.__getitem__(item_id)
        if isinstance(item, ndarray):
            return OperatorMatrix(item)
        return item

    def element_wise(self, method):
        """
        Apply a method to each matrix element and return the result in a new operator matrix of the same shape.
        :param method: A method taking a single argument.
        :type method: FunctionType
        :return: Operator matrix with results of method applied element-wise.
        :rtype: OperatorMatrix
        """
        s = self.shape
        emat = [method(o) for o in self.matrix.flatten()]
        return OperatorMatrix(np_array(emat).reshape(s))


    def expand(self):
        """
        Expand each matrix element distributively.
        :return: Expanded matrix.
        :rtype: OperatorMatrix
        """
        m = lambda o: o.expand() if isinstance(o, Operator) else o
        return self.element_wise(m)

    def _substitute(self, var_map):
        m = lambda o: substitute(o, var_map) if isinstance(o, Operation) else o
        return self.element_wise(m)

    def _tex(self):
        ret = r"\begin{pmatrix} "
#        for row in self.matrix:
        ret += r""" \\
""".join([" & ".join([tex(o) for o in row]) for row in self.matrix])
        ret += r"\end{pmatrix}"



    @property
    def space(self):
        """
        :return: Return the combined Hilbert space of all matrix elements.
        :rtype: HilbertSpace
        """
        return prod((o.space for o in self.matrix.flatten()), TrivialSpace)


def hstack(matrices):
    """
    Generalizes `numpy.hstack` to OperatorMatrix objects.
    """
    return OperatorMatrix(np_hstack(matrices))


def vstack(matrices):
    """
    Generalizes `numpy.vstack` to OperatorMatrix objects.
    """
    return OperatorMatrix(np_vstack(matrices))


def diag(v, k=0):
    """
    Generalizes the diagonal matrix creation capabilities of `numpy.diag` to OperatorMatrix objects.
    """
    return OperatorMatrix(np_diag(v, k))


def block_matrix(A, B, C, D):
    """
    Generate the operator matrix with quadrants
       [[A B]
        [C D]]
    :return: The combined block matrix [[A, B], [C, D]].
    :type: OperatorMatrix
    """
    return vstack((hstack((A, B)), hstack((C, D))))


def identity_matrix(N):
    """
    Generate the N-dimensional identity matrix.
    :param N: Dimension
    :type N: int
    :return: Identity matrix in N dimensions
    :rtype: OperatorMatrix
    """
    return diag(np_ones(N, dtype=int))


def zeros(shape):
    """
    Generalizes `numpy.zeros` to OperatorMatrix objects.
    """
    return OperatorMatrix(np_zeros(shape))


def permutation_matrix(permutation):
    """
    Return an orthogonal permutation matrix M_sigma
    for a permutation sigma given by a tuple
    (sigma(1), sigma(2),... sigma(n)), such that
    such that M_sigma e_i = e_sigma(i), where e_k
    is the k-th standard basis vector.
    This definition ensures a composition law:
    M_{sigma . pi} = M_sigma M_pi.
    In column form M_sigma is thus given by
    M = [e_sigma(1), e_sigma(2), ... e_sigma(n)].
    """
    assert(check_permutation(permutation))
    n = len(permutation)
    op_matrix = zeros((n, n))
    for i, j in enumerate(permutation):
        op_matrix[j, i] = 1
    return op_matrix

# :deprecated:
# for backwards compatibility
OperatorMatrixInstance = OperatorMatrix
IdentityMatrix = identity_matrix

def Im(op):
    """
    The imaginary part of a number or operator. Acting on OperatorMatrices, it produces the element-wise imaginary parts.
    :param op: Anything that has a conjugate method.
    :type op: Operator or OperatorMatrix or any of Operator.scalar_types
    :return: The imaginary part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() - op) * I / 2

def Re(op):
    """
    The real part of a number or operator. Acting on OperatorMatrices, it produces the element-wise real parts.
    :param op: Anything that has a conjugate method.
    :type op: Operator or OperatorMatrix or any of Operator.scalar_types
    :return: The real part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate()+ op) / 2


def ImAdjoint(opmatrix):
    """
    The imaginary part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: OperatorMatrix
    :return: The matrix imaginary part of the operand.
    :rtype: OperatorMatrix
    """
    return (opmatrix.H - opmatrix) * I / 2


def ReAdjoint(opmatrix):
    """
    The real part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: OperatorMatrix
    :return: The matrix real part of the operand.
    :rtype: OperatorMatrix
    """
    return (opmatrix.H + opmatrix) / 2






