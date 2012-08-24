from __future__ import division
from abstract_algebra import *
from hilbert_space_algebra import *
from permutation_algebra import *
from itertools import product as cartesian_product
import qutip
from sympy import exp, log, cos, sin, cosh, sinh, tan, cot,\
    acos, asin, acosh, asinh, atan, atan2, atanh, acot, sqrt,\
    factorial, pi, I, sympify, Basic as SympyBasic, symbols

sympyOne = sympify(1)

from numpy import array as np_array,\
    shape as np_shape,\
    hstack as np_hstack,\
    vstack as np_vstack,\
    diag as np_diag,\
    ones as np_ones,\
    conjugate as np_conjugate,\
    zeros as np_zeros,\
    ndarray


class Operator(object):
    """
    The basic operator class, which fixes the abstract interface of operator objects 
    and where possible also defines the default behavior under operations.
    Any operator contains an associated HilbertSpace object, 
    on which it is taken to act non-trivially.
    """

    scalar_types = int, long, float, complex, SympyBasic


    @property
    def space(self):
        raise NotImplementedError(self.__class__.__name__)

    def n(self):
        return self

    def __neg__(self):
        return (-1) * self

    def adjoint(self):
        return Adjoint.create(self)

    conjugate = dag = adjoint

    def to_qutip(self, full_space=None):
        raise NotImplementedError(str(self.__class__))

    def expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def series_expand(self, param, about, order):
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

    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented


@check_signature
class OperatorSymbol(Operator, Operation):
    signature = str, (HilbertSpace, str, tuple)

    def __init__(self, name, hs):
        if isinstance(hs, str):
            hs = local_space(hs)
        elif isinstance(hs, tuple):
            hs = prod([local_space(h) for h in hs], neutral=TrivialSpace)
        super(OperatorSymbol, self).__init__(name, hs)

    def __str__(self):
        return self.operands[0]

    def tex(self):
        return "{" + self.operands[0] + "}"

    def __repr__(self):
        return "OperatorSymbol({},{})".format(*map(repr, self.operands))


    def to_qutip(self, full_space=None):
        raise AlgebraError("Cannot convert operator symbol to representation matrix. Substitute first.")

    @property
    def space(self):
        return self.operands[1]

    def expand(self):
        return self

    def series_expand(self, param, about, order):
        return self


@singleton
class IdentityOperator(Operator):
    def tex(self):
        return "1"

    def __repr__(self):
        return "IdentityOperator"

    @property
    def space(self):
        return TrivialSpace

    def adjoint(self):
        return self

    def to_qutip(self, full_space):
        return qutip.tensor(*[qutip.qeye(s.dimension) for s in full_space.local_factors()])

    def __eq__(self, other):
        return self is other or other == 1

    def __hash__(self):
        return hash(self.__class__)

    def mathematica(self):
        return "IdentityOperator"

    def expand(self):
        return self

    def series_expand(self, param, about, order):
        return self


from scipy.sparse import csr_matrix

@singleton
class OperatorZero(Operator):
    @property
    def space(self):
        return TrivialSpace

    def adjoint(self):
        return self

    def to_qutip(self, full_space):
        return qutip.tensor(
            *[qutip.Qobj(csr_matrix((), (s.dimension, s.dimension))) for s in full_space.local_factors()])

    def __eq__(self, other):
        return self is other or other == 0

    def __hash__(self):
        return hash(self.__class__)

    def expand(self):
        return self

    def series_expand(self, param, about, order):
        return self


class LocalOperator(Operator, Operation):
    @property
    def space(self):
        return self.operands[0]

    def __init__(self, hs, *args):
        if isinstance(hs, str):
            hs = local_space(hs)
        super(LocalOperator, self).__init__(hs, *args)

    def to_qutip(self, full_space=None):
        if full_space is None or full_space == self.space:
            return self.to_qutip_local_factor()
        else:
            all_spaces = full_space.local_factors()
            own_space_index = all_spaces.index(self.space)
            return qutip.tensor(*([qutip.qeye(s.dimension) for s in all_spaces[:own_space_index]]
                                  + [self.to_qutip_local_factor()]
                                  + [qutip.qeye(s.dimension) for s in all_spaces[own_space_index + 1:]]))


    def to_qutip_local_factor(self):
        raise NotImplementedError(self.__class__.__name__)

    def expand(self):
        return self

    def series_expand(self, param, about, order):
        return self


@check_signature
class Create(LocalOperator):
    signature = (LocalSpace, str),

    def to_qutip_local_factor(self):
        return qutip.create(self.space.dimension)


@check_signature
class Destroy(LocalOperator):
    signature = (LocalSpace, str),

    def to_qutip_local_factor(self):
        return qutip.destroy(self.space.dimension)


@check_signature
class LocalSigma(LocalOperator):
    signature = (LocalSpace, str), (int, str), (int, str)

    def to_qutip_local_factor(self):
        k, j = self.operands[1:]
        ket = qutip.basis(self.space.dimension, self.space.states.index(k))
        bra = qutip.basis(self.space.dimension, self.space.states.index(j)).dag()
        return ket * bra


LocalProjector = lambda spc, state: LocalSigma.create(spc, state, state)


def Z(local_space, states=("h", "g")):
    h, g = states
    return LocalProjector(local_space, h) - LocalProjector(local_space, g)


def X(local_space, states=("h", "g")):
    h, g = states
    return LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h)


def Y(local_space, states=("h", "g")):
    h, g = states
    return I * (-LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h))


class OperatorOperation(Operator, Operation):
    signature = Operator,

    @property
    def space(self):
        return prod((o.space for o in self.operands), TrivialSpace)

    def n(self):
        return self.__class__.apply_with_rules(*map(n, self.operands))




@flat
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorPlus(OperatorOperation):
    neutral_element = OperatorZero

    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesOperator):
            return Operation.order_key(a.operands[1]), a.operands[0]
        return Operation.order_key(a), 1


    binary_rules = [
    ]

    def to_qutip(self, full_space=None):
        if full_space == None:
            full_space = self.space
        assert self.space <= full_space
        return sum((op.to_qutip(full_space) for op in self.operands), 0)

    def expand(self):
        return sum((o.expand() for o in self.operands), OperatorZero)

    def series_expand(self, param, about, order):
        res = sum((o.series_expand(param, about, order) for o in self.operands), OperatorZero)
        return res


@flat
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorTimes(OperatorOperation):
    neutral_element = IdentityOperator

    binary_rules = []

    class OperatorOrderKey(object):
        def __init__(self, space):
            self.full = False
            if isinstance(space, LocalSpace):
                self.local_spaces = {space.operands, }
            elif space is TrivialSpace:
                self.local_spaces = set(())
            elif space is FullSpace:
                self.full = True
            else:
                assert isinstance(space, ProductSpace)
                self.local_spaces = {s.operands for s in space.operands}

        def __lt__(self, other):
            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) < tuple(other.local_spaces)

        def __gt__(self, other):
            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) > tuple(other.local_spaces)

        def __eq__(self, other):
            return self.full or len(self.local_spaces & other.local_spaces) > 0

    @classmethod
    def order_key(cls, a):
        return cls.OperatorOrderKey(a.space)

    @classmethod
    def create(cls, *ops):
        if any(o == OperatorZero for o in ops):
            return OperatorZero
        return cls(*ops)

    def to_qutip(self, full_space=None):
        if any(len(op.space) > 1 for op in self.operands):
            return self.expand().to_qutip(full_space)
        if full_space == None:
            full_space = self.space
        all_spaces = full_space.local_factors()
        by_space = []
        ck = 0
        for ls in all_spaces:
            ls_ops = [o.to_qutip() for o in self.operands if o.space == ls]
            if len(ls_ops):
                by_space.append(prod(ls_ops))
                ck += len(ls_ops)
            else:
                by_space.append(qutip.qeye(ls.dimension))
        assert ck == len(self.operands)
        return qutip.tensor(*by_space)

    def expand(self):
        eops = [o.expand() for o in self.operands]
        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,) for eo in eops]
        return sum((OperatorTimes.create(*combo) for combo in cartesian_product(*eopssummands)), OperatorZero)


@match_replace
@check_signature
class ScalarTimesOperator(Operator, Operation):
    signature = Operator.scalar_types, Operator
    rules = []

    @property
    def space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def n(self):
        # print complex(self.coeff)
        return complex(self.operands[0]) * self.operands[1]

    def __str__(self):
        coeff, term = self.operands

        if coeff == -1:
            return "(-%s)" % (term,)

        if isinstance(coeff,(int, float)) and coeff < 0:
            return "(%g) * %s" % (coeff, term)

        return "%s * %s" % (coeff, term)

    def tex(self):
        coeff, term = self.operands
        if term == IdentityOperator:
            return tex(coeff)
        if coeff == -1:
            return "(-%s)" % tex(term)
        return "(%s)  %s" % (tex(coeff), tex(term))


    def to_qutip(self, full_space=None):
        return complex(self.coeff) * self.term.to_qutip(full_space)

    def expand(self):
        c, t = self.operands
        et = expand(t)
        if isinstance(et, OperatorPlus):
            return sum((c * eto for eto in et.operands), OperatorZero)
        return c * et



@check_signature
@match_replace
class Adjoint(OperatorOperation):
    @property
    def operand(self):
        return self.operands[0]

    rules = []

    def to_qutip(self, full_space=None):
        return qutip.dag(self.operands[0].to_qutip(full_space))

    def tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^\dagger"

    def expand(self):
        eo = self.operand.expand()
        if isinstance(eo, OperatorPlus):
            return sum((eoo.adjoint() for eoo in eo.operands), OperatorZero)
        return eo.adjoint()


## Expression rewriting rules
u = wc("u", head=Operator.scalar_types)
v = wc("v", head=Operator.scalar_types)

A = wc("A", head=Operator)
B = wc("B", head=Operator)
A_plus = wc("A", head=OperatorPlus)
A_times = wc("A", head=OperatorTimes)

ls = wc("ls", head=LocalSpace)
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



class NonSquareMatrix(Exception):
    pass


class OperatorMatrix(object):
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
        return OperatorMatrix(self.matrix.T)

    def conjugate(self):
        return OperatorMatrix(np_conjugate(self.matrix))

    @property
    def T(self):
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
        s = self.shape
        emat = [method(o) for o in self.matrix.flatten()]
        return OperatorMatrix(np_array(emat).reshape(s))


    def expand(self):
        m = lambda o: o.expand() if isinstance(o, Operator) else o
        return self.element_wise(m)


def hstack(matrices):
    return OperatorMatrix(np_hstack(matrices))


def vstack(matrices):
    return OperatorMatrix(np_vstack(matrices))


def diag(v, k=0):
    return OperatorMatrix(np_diag(v, k))


def block_matrix(A, B, C, D):
    return vstack((hstack((A, B)), hstack((C, D))))


def identity_matrix(N):
    return diag(np_ones(N, dtype=int))


def zeros(shape):
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


OperatorMatrixInstance = OperatorMatrix
IdentityMatrix = identity_matrix

def Im(op):
    return (conjugate(op) - op) * I / 2


def Re(op):
    return (conjugate(op) + op) / 2


def ImAdjoint(opmatrix):
    return (opmatrix.H - opmatrix) * I / 2


def ReAdjoint(opmatrix):
    return (opmatrix.H + opmatrix) / 2
