from sympy import Basic as SympyBasic, I, exp, sqrt

from .core.circuit_algebra import (
    CIdentity, CPermutation, Circuit, Concatenation, Feedback, SLH,
    SeriesInverse, SeriesProduct, circuit_identity as cid)
from .core.exceptions import CannotSimplify, IncompatibleBlockStructures
from .core.hilbert_space_algebra import (
    HilbertSpace, LocalSpace, ProductSpace, TrivialSpace)
from .core.operator_algebra import (
    Adjoint, Commutator, IdentityOperator, LocalOperator,
    LocalSigma, Operator, OperatorIndexedSum, OperatorPlus, OperatorTimes,
    OperatorTrace, PseudoInverse, ScalarTimesOperator, ZeroOperator,
    decompose_space, factor_for_trace)
from .library.spin_algebra import (
    Jz, Jplus, Jminus, Jpjmcoeff, Jzjmcoeff, Jmjmcoeff)
from .library.fock_operators import (
    Destroy, Create, Phase,
    Displace, Squeeze)
from .core.scalar_algebra import (
    Scalar, ScalarExpression, ScalarValue, ScalarPlus, ScalarTimes,
    ScalarPower, ScalarIndexedSum, Zero, One, KroneckerDelta)
from .core.state_algebra import (
    BasisKet, Bra, BraKet, CoherentStateKet, State, KetBra, KetIndexedSum,
    KetPlus, LocalKet, OperatorTimesKet, ScalarTimesKet, TensorKet, TrivialKet,
    ZeroKet, KetSymbol)
from .core.super_operator_algebra import (
    IdentitySuperOperator, SPost, SPre, ScalarTimesSuperOperator, SuperAdjoint,
    SuperOperator, SuperOperatorPlus, SuperOperatorTimes,
    SuperOperatorTimesOperator, ZeroSuperOperator)
from .pattern_matching import pattern, pattern_head, wc
from ..utils.check_rules import check_rules_dict
from ..utils.indices import IndexRangeBase, SymbolicLabelBase
from ..utils.permutations import concatenate_permutations


SCALAR_TYPES = (Scalar, ) + Scalar._val_types
SCALAR_VAL_TYPES = (ScalarValue, ) + Scalar._val_types


# Scalar Rules

def _algebraic_rules_scalar():
    """Set the default algebraic rules for scalars"""
    a = wc("a", head=SCALAR_VAL_TYPES)
    b = wc("b", head=SCALAR_VAL_TYPES)
    x = wc("x", head=SCALAR_TYPES)
    y = wc("y", head=SCALAR_TYPES)
    z = wc("z", head=SCALAR_TYPES)

    indranges__ = wc("indranges__", head=IndexRangeBase)

    ScalarTimes._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(a, b),
            lambda a, b: a * b)),
        ('R002', (
            pattern_head(x, x),
            lambda x: x**2)),
        ('R003', (
            pattern_head(Zero, x),
            lambda x: Zero)),
        ('R004', (
            pattern_head(x, Zero),
            lambda x: Zero)),
        ('R005', (
            pattern_head(
                pattern(ScalarPower, x, y),
                pattern(ScalarPower, x, z)),
            lambda x, y, z: x**(y+z))),
        ('R006', (
            pattern_head(x, pattern(ScalarPower, x, -1)),
            lambda x: One)),
    ]))

    ScalarPower._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(a, b),
            lambda a, b: a**b)),
        ('R002', (
            pattern_head(x, 0),
            lambda x: One)),
        ('R003', (
            pattern_head(x, 1),
            lambda x: x)),
        ('R004', (
            pattern_head(pattern(ScalarPower, x, y), z),
            lambda x, y, z: x**(y*z))),
    ]))

    def pull_constfactor_from_sum(x, y, indranges):
        bound_symbols = set([r.index_symbol for r in indranges])
        if len(x.free_symbols.intersection(bound_symbols)) == 0:
            return x * ScalarIndexedSum.create(y, *indranges)
        else:
            raise CannotSimplify()

    ScalarIndexedSum._rules.update(check_rules_dict([
        ('R001', (  # sum over zero -> zero
            pattern_head(Zero, indranges__),
            lambda indranges: Zero)),
        ('R002', (  # pull constant prefactor out of sum
            pattern_head(pattern(ScalarTimes, x, y), indranges__),
            lambda x, y, indranges:
                pull_constfactor_from_sum(x, y, indranges))),
    ]))


# Operator rules

def _algebraic_rules_operator():
    """Set the default algebraic rules for the operations defined in this
    module"""
    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    n = wc("n", head=(int, str, SymbolicLabelBase))
    m = wc("m", head=(int, str, SymbolicLabelBase))

    A = wc("A", head=Operator)
    B = wc("B", head=Operator)

    A_plus = wc("A", head=OperatorPlus)
    A_times = wc("A", head=OperatorTimes)

    ls = wc("ls", head=LocalSpace)
    h1 = wc("h1", head=HilbertSpace)
    H_ProductSpace = wc("H", head=ProductSpace)

    localsigma = wc(
        'localsigma', head=LocalSigma, kwargs={'hs': ls})

    ra = wc("ra", head=(int, str, SymbolicLabelBase))
    rb = wc("rb", head=(int, str, SymbolicLabelBase))
    rc = wc("rc", head=(int, str, SymbolicLabelBase))
    rd = wc("rd", head=(int, str, SymbolicLabelBase))

    indranges__ = wc("indranges__", head=IndexRangeBase)

    ScalarTimesOperator._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(1, A),
            lambda A: A)),
        ('R002', (
            pattern_head(0, A),
            lambda A: ZeroOperator)),
        ('R003', (
            pattern_head(u, ZeroOperator),
            lambda u: ZeroOperator)),
        ('R004', (
            pattern_head(u, pattern(ScalarTimesOperator, v, A)),
            lambda u, v, A: (u * v) * A)),
        ('R005', (
            pattern_head(-1, A_plus),
            lambda A: OperatorPlus.create(*[-1 * op for op in A.args]))),
    ]))

    OperatorTimes._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(pattern(ScalarTimesOperator, u, A), B),
            lambda u, A, B: u * (A * B))),

        ('R002', (
            pattern_head(ZeroOperator, B),
            lambda B: ZeroOperator)),
        ('R003', (
            pattern_head(A, ZeroOperator),
            lambda A: ZeroOperator)),

        ('R004', (
            pattern_head(A, pattern(ScalarTimesOperator, u, B)),
            lambda A, u, B: u * (A * B))),

        ('R005', (
            pattern_head(
                pattern(LocalSigma, ra, rb, hs=ls),
                pattern(LocalSigma, rc, rd, hs=ls)),
            lambda ls, ra, rb, rc, rd: (
                KroneckerDelta(
                    BasisKet(rb, hs=ls).index,
                    BasisKet(rc, hs=ls).index) *
                LocalSigma.create(ra, rd, hs=ls)))),

        # Harmonic oscillator rules
        ('R009', (
            pattern_head(pattern(Create, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j + 1) * localsigma.raise_jk(j_incr=1))),

        ('R010', (
            pattern_head(pattern(Destroy, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j) * localsigma.raise_jk(j_incr=-1))),

        ('R011', (
            pattern_head(localsigma, pattern(Destroy, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k + 1) * localsigma.raise_jk(k_incr=1))),

        ('R012', (
            pattern_head(localsigma, pattern(Create, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k) * localsigma.raise_jk(k_incr=-1))),

        # Normal ordering for harmonic oscillator <=> all a^* to the left, a to
        # the right.
        ('R013', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Create, hs=ls)),
            lambda ls: IdentityOperator + Create(hs=ls) * Destroy(hs=ls))),

        # Oscillator unitary group rules
        ('R014', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Phase, v, hs=ls)),
            lambda ls, u, v: Phase.create(u + v, hs=ls))),
        ('R015', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(Displace, v, hs=ls)),
            lambda ls, u, v: (
                exp((u * v.conjugate() - u.conjugate() * v) / 2) *
                Displace.create(u + v, hs=ls)))),

        ('R016', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Phase, u, hs=ls)),
            lambda ls, u:
                exp(I * u) * Phase.create(u, hs=ls) * Destroy(hs=ls))),
        ('R017', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Displace, u, hs=ls)),
            lambda ls, u: Displace.create(u, hs=ls) * (Destroy(hs=ls) + u))),

        ('R018', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u:
                exp(I * u) * Create(hs=ls) * Phase.create(u, hs=ls))),
        ('R019', (
            pattern_head(pattern(Displace, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u: (((Create(hs=ls) - u.conjugate()) *
                            Displace.create(u, hs=ls))))),

        ('R020', (
            pattern_head(pattern(Phase, u, hs=ls), localsigma),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_j) * localsigma)),
        ('R021', (
            pattern_head(localsigma, pattern(Phase, u, hs=ls)),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_k) * localsigma)),

        # Spin rules
        ('R022', (
            pattern_head(pattern(Jplus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=1))),

        ('R023', (
            pattern_head(pattern(Jminus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=-1))),

        ('R024', (
            pattern_head(pattern(Jz, hs=ls), localsigma),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_j, shift=True) * localsigma)),

        ('R025', (
            pattern_head(localsigma, pattern(Jplus, hs=ls)),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=-1))),

        ('R026', (
            pattern_head(localsigma, pattern(Jminus, hs=ls)),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=+1))),

        ('R027', (
            pattern_head(localsigma, pattern(Jz, hs=ls)),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_k, shift=True) * localsigma)),

        # Normal ordering for angular momentum <=> all J_+ to the left, J_z to
        # center and J_- to the right
        ('R028', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: -2 * Jz(hs=ls) + Jplus(hs=ls) * Jminus(hs=ls))),

        ('R029', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jz, hs=ls)),
            lambda ls: Jz(hs=ls) * Jminus(hs=ls) + Jminus(hs=ls))),

        ('R030', (
            pattern_head(pattern(Jz, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: Jplus(hs=ls) * Jz(hs=ls) + Jplus(hs=ls))),
    ]))

    Displace._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Phase._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Squeeze._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))

    OperatorTrace._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(A, over_space=TrivialSpace),
            lambda A: A)),
        ('R002', (
            pattern_head(ZeroOperator, over_space=h1),
            lambda h1: ZeroOperator)),
        ('R003', (
            pattern_head(IdentityOperator, over_space=h1),
            lambda h1: h1.dimension * IdentityOperator)),
        ('R004', (
            pattern_head(A_plus, over_space=h1),
            lambda h1, A: OperatorPlus.create(
                *[OperatorTrace.create(o, over_space=h1)
                  for o in A.operands]))),
        ('R005', (
            pattern_head(pattern(Adjoint, A), over_space=h1),
            lambda h1, A: Adjoint.create(
                OperatorTrace.create(A, over_space=h1)))),
        ('R006', (
            pattern_head(pattern(ScalarTimesOperator, u, A), over_space=h1),
            lambda h1, u, A: u * OperatorTrace.create(A, over_space=h1))),
        ('R007', (
            pattern_head(A, over_space=H_ProductSpace),
            lambda H, A: decompose_space(H, A))),
        ('R008', (
            pattern_head(pattern(Create, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('R009', (
            pattern_head(pattern(Destroy, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('R010', (
            pattern_head(pattern(LocalSigma, n, m, hs=ls), over_space=ls),
            lambda ls, n, m:
                KroneckerDelta(
                    BasisKet(n, hs=ls).index,
                    BasisKet(m, hs=ls).index) *
                IdentityOperator)),
        ('R011', (
            pattern_head(A, over_space=ls),
            lambda ls, A: factor_for_trace(ls, A))),
    ]))

    Commutator._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(A, A), lambda A: ZeroOperator)),
        ('R002', (
            pattern_head(
                pattern(ScalarTimesOperator, u, A),
                pattern(ScalarTimesOperator, v, B)),
            lambda u, v, A, B: u * v * Commutator.create(A, B))),
        ('R003', (
            pattern_head(pattern(ScalarTimesOperator, v, A), B),
            lambda v, A, B: v * Commutator.create(A, B))),
        ('R004', (
            pattern_head(A, pattern(ScalarTimesOperator, v, B)),
            lambda v, A, B: v * Commutator.create(A, B))),

        # special known commutators
        ('R005', (
            pattern_head(pattern(Create, hs=ls), pattern(Destroy, hs=ls)),
            lambda ls: ScalarTimesOperator(-1, IdentityOperator))),
        # the remaining  rules basically defer to OperatorTimes; just writing
        # out the commutator will generate something simple
        ('R006', (
            pattern_head(
                wc('A', head=(
                    Create, Destroy, LocalSigma, Phase, Displace)),
                wc('B', head=(
                    Create, Destroy, LocalSigma, Phase, Displace))),
            lambda A, B: A * B - B * A)),
        ('R007', (
            pattern_head(
                wc('A', head=(LocalSigma, Jplus, Jminus, Jz)),
                wc('B', head=(LocalSigma, Jplus, Jminus, Jz))),
            lambda A, B: A * B - B * A)),
    ]))

    def pull_constfactor_from_sum(u, A, indranges):
        bound_symbols = set([r.index_symbol for r in indranges])
        if len(u.free_symbols.intersection(bound_symbols)) == 0:
            return u * OperatorIndexedSum.create(A, *indranges)
        else:
            raise CannotSimplify()

    OperatorIndexedSum._rules.update(check_rules_dict([
        ('R001', (  # sum over zero -> zero
            pattern_head(ZeroOperator, indranges__),
            lambda indranges: ZeroOperator)),
        ('R002', (  # pull constant prefactor out of sum
            pattern_head(pattern(ScalarTimesOperator, u, A), indranges__),
            lambda u, A, indranges:
                pull_constfactor_from_sum(u, A, indranges))),
    ]))


# Super-Operator rules

def _algebraic_rules_superop():
    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    A = wc("A", head=Operator)
    B = wc("B", head=Operator)
    C = wc("C", head=Operator)

    sA = wc("sA", head=SuperOperator)
    sA__ = wc("sA__", head=SuperOperator)
    sB = wc("sB", head=SuperOperator)

    sA_plus = wc("sA", head=SuperOperatorPlus)
    sA_times = wc("sA", head=SuperOperatorTimes)

    ScalarTimesSuperOperator._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(1, sA),
            lambda sA: sA)),
        ('R002', (
            pattern_head(0, sA),
            lambda sA: ZeroSuperOperator)),
        ('R003', (
            pattern_head(u, ZeroSuperOperator),
            lambda u: ZeroSuperOperator)),
        ('R004', (
            pattern_head(u, pattern(ScalarTimesSuperOperator, v, sA)),
            lambda u, v, sA: (u * v) * sA)),
    ]))

    SuperOperatorTimes._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sB),
            lambda u, sA, sB: u * (sA * sB))),
        ('R002', (
            pattern_head(sA, pattern(ScalarTimesSuperOperator, u, sB)),
            lambda sA, u, sB: u * (sA * sB))),
        ('R003', (
            pattern_head(pattern(SPre, A), pattern(SPre, B)),
            lambda A, B: SPre.create(A*B))),
        ('R004', (
            pattern_head(pattern(SPost, A), pattern(SPost, B)),
            lambda A, B: SPost.create(B*A))),
    ]))

    SPre._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPre.create(A))),
        ('R002', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('R003', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SPost._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPost.create(A))),
        ('R002', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('R003', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SuperOperatorTimesOperator._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(sA_plus, B),
            lambda sA, B:
                OperatorPlus.create(*[o*B for o in sA.operands]))),
        ('R002', (
            pattern_head(IdentitySuperOperator, B),
            lambda B: B)),
        ('R003', (
            pattern_head(ZeroSuperOperator, B),
            lambda B: ZeroOperator)),
        ('R004', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), B),
            lambda u, sA, B: u * (sA * B))),
        ('R005', (
            pattern_head(sA, pattern(ScalarTimesOperator, u, B)),
            lambda u, sA, B: u * (sA * B))),
        ('R006', (
            pattern_head(sA, pattern(SuperOperatorTimesOperator, sB, C)),
            lambda sA, sB, C: (sA * sB) * C)),
        ('R007', (
            pattern_head(pattern(SPre, A), B),
            lambda A, B: A*B)),
        ('R008', (
            pattern_head(
                pattern(
                    SuperOperatorTimes, sA__, wc('sB', head=(SPost, SPre))),
                C),
            lambda sA, sB, C: (
                SuperOperatorTimes.create(*sA) * (sB * C)))),
        ('R009', (
            pattern_head(pattern(SPost, A), B),
            lambda A, B: B*A)),
    ]))


# State rules

def act_locally(op, ket):
    ket_on, ket_off = ket.factor_for_space(op.space)
    if ket_off != TrivialKet:
        return (op * ket_on) * ket_off
    raise CannotSimplify()


def act_locally_times_tensor(op, ket):
    local_spaces = op.space.local_factors
    for spc in local_spaces:
        while spc < ket.space:
            op_on, op_off = op.factor_for_space(spc)
            ket_on, ket_off = ket.factor_for_space(spc)

            if (op_on.space <= ket_on.space and
                    op_off.space <= ket_off.space and ket_off != TrivialKet):
                return (op_on * ket_on) * (op_off * ket_off)
            else:
                spc = op_on.space * ket_on.space
    raise CannotSimplify()


def tensor_decompose_kets(a, b, operation):
    full_space = a.space * b.space
    local_spaces = full_space.local_factors
    for spc in local_spaces:
        while spc < full_space:
            a_on, a_off = a.factor_for_space(spc)
            b_on, b_off = b.factor_for_space(spc)
            if (a_on.space == b_on.space and a_off.space == b_off.space and
                    a_off != TrivialKet):
                return operation(a_on, b_on) * operation(a_off, b_off)
            else:
                spc = a_on.space * b_on.space
    raise CannotSimplify()


def _algebraic_rules_state():
    """Set the default algebraic rules for the operations defined in this
    module"""
    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    n = wc("n", head=(int, str, SymbolicLabelBase))
    m = wc("m", head=(int, str, SymbolicLabelBase))
    k = wc("k", head=(int, str, SymbolicLabelBase))

    A = wc("A", head=Operator)
    A__ = wc("A__", head=Operator)
    B = wc("B", head=Operator)

    A_times = wc("A", head=OperatorTimes)
    A_local = wc("A", head=LocalOperator)
    B_local = wc("B", head=LocalOperator)

    Psi_sym = wc("Psi", head=KetSymbol)
    Psi = wc("Psi", head=State)
    Phi = wc("Phi", head=State)
    Psi_local = wc("Psi", head=LocalKet)
    Psi_tensor = wc("Psi", head=TensorKet)
    Phi_tensor = wc("Phi", head=TensorKet)

    ls = wc("ls", head=LocalSpace)

    basisket = wc('basisket', BasisKet, kwargs={'hs': ls})
    ket_a = wc('a', BasisKet)
    ket_b = wc('b', BasisKet)

    indranges__ = wc("indranges__", head=IndexRangeBase)
    sum = wc('sum', head=KetIndexedSum)
    sum2 = wc('sum2', head=KetIndexedSum)

    ScalarTimesKet._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(1, Psi),
            lambda Psi: Psi)),
        ('R002', (
            pattern_head(0, Psi),
            lambda Psi: ZeroKet)),
        ('R003', (
            pattern_head(u, ZeroKet),
            lambda u: ZeroKet)),
        ('R004', (
            pattern_head(u, pattern(ScalarTimesKet, v, Psi)),
            lambda u, v, Psi: (u * v) * Psi))
    ]))

    def local_rule(A, B, Psi):
        return OperatorTimes.create(*A) * (B * Psi)

    OperatorTimesKet._rules.update(check_rules_dict([
        ('R001', (  # Id * Psi = Psi
            pattern_head(IdentityOperator, Psi),
            lambda Psi: Psi)),
        ('R002', (  # 0 * Psi = 0
            pattern_head(ZeroOperator, Psi),
            lambda Psi: ZeroKet)),
        ('R003', (  # A * 0 = 0
            pattern_head(A, ZeroKet),
            lambda A: ZeroKet)),
        ('R004', (  # A * v * Psi = v * A * Psi (pull out scalar)
            pattern_head(A, pattern(ScalarTimesKet, v, Psi)),
            lambda A, v, Psi:  v * (A * Psi))),

        ('R005', (  # |n><m| * |k> = delta_mk * |n>
            pattern_head(
                pattern(LocalSigma, n, m, hs=ls),
                pattern(BasisKet, k, hs=ls)),
            lambda ls, n, m, k:
                KroneckerDelta(
                    BasisKet(m, hs=ls).index, BasisKet(k, hs=ls).index) *
                BasisKet(n, hs=ls))),

        # harmonic oscillator
        ('R006', (  # a^+ |n> = sqrt(n+1) * |n+1>
            pattern_head(pattern(Create, hs=ls), basisket),
            lambda basisket, ls:
                sqrt(basisket.index + 1) * basisket.next())),
        ('R007', (  # a |n> = sqrt(n) * |n-1>
            pattern_head(pattern(Destroy, hs=ls), basisket),
            lambda basisket, ls:
                sqrt(basisket.index) * basisket.prev())),
        ('R008', (  # a |alpha> = alpha * |alpha> (eigenstate of annihilator)
            pattern_head(
                pattern(Destroy, hs=ls),
                pattern(CoherentStateKet, u, hs=ls)),
            lambda ls, u: u * CoherentStateKet(u, hs=ls))),

        # spin
        ('R009', (
            pattern_head(pattern(Jplus, hs=ls), basisket),
            lambda basisket, ls:
                Jpjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket.next())),
        ('R010', (
            pattern_head(pattern(Jminus, hs=ls), basisket),
            lambda basisket, ls:
                Jmjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket.prev())),
        ('R011', (
            pattern_head(pattern(Jz, hs=ls), basisket),
            lambda basisket, ls:
                Jzjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket)),

        ('R012', (
            pattern_head(A_local, Psi_tensor),
            lambda A, Psi: act_locally(A, Psi))),
        ('R013', (
            pattern_head(A_times, Psi_tensor),
            lambda A, Psi: act_locally_times_tensor(A, Psi))),
        ('R014', (
            pattern_head(A, pattern(OperatorTimesKet, B, Psi)),
            lambda A, B, Psi: (
                (A * B) * Psi
                if (B * Psi) == OperatorTimesKet(B, Psi)
                else A * (B * Psi)))),
        ('R015', (
            pattern_head(pattern(OperatorTimes, A__, B_local), Psi_local),
            local_rule)),
        ('R016', (
            pattern_head(pattern(ScalarTimesOperator, u, A), Psi),
            lambda u, A, Psi: u * (A * Psi))),
        ('R017', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(BasisKet, 0, hs=ls)),
            lambda ls, u: CoherentStateKet(u, hs=ls))),
        ('R018', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(CoherentStateKet, v, hs=ls)),
            lambda ls, u, v:
                ((Displace(u, hs=ls) * Displace(v, hs=ls)) *
                 BasisKet(0, hs=ls)))),
        ('R019', (
            pattern_head(
                pattern(Phase, u, hs=ls), pattern(BasisKet, m, hs=ls)),
            lambda ls, u, m: exp(I * u * m) * BasisKet(m, hs=ls))),
        ('R020', (
            pattern_head(
                pattern(Phase, u, hs=ls),
                pattern(CoherentStateKet, v, hs=ls)),
            lambda ls, u, v: CoherentStateKet(v * exp(I * u), hs=ls))),

        ('R021', (
            pattern_head(A, sum),
            lambda A, sum: KetIndexedSum.create(A * sum.term, *sum.ranges))),
    ]))

    TensorKet._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u * (Psi * Phi))),
        ('R002', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u * (Psi * Phi))),
        ('R003', (  # delegate to __mul__
            pattern_head(sum, sum2),
            lambda sum, sum2: sum * sum2)),
        ('R004', (  # delegate to __mul__
            pattern_head(Psi, sum),
            lambda Psi, sum: Psi * sum)),
        ('R005', (  # delegate to __mul__
            pattern_head(sum, Psi),
            lambda sum, Psi: sum * Psi)),
    ]))

    BraKet._rules.update(check_rules_dict([
        # All rules must result in scalars or objects in the TrivialSpace
        ('R001', (
            pattern_head(Phi, ZeroKet),
            lambda Phi: Zero)),
        ('R002', (
            pattern_head(ZeroKet, Phi),
            lambda Phi: Zero)),
        ('R003', (
            pattern_head(ket_a, ket_b),
            lambda a, b: KroneckerDelta(a.index, b.index))),
        ('R004', (
            pattern_head(Psi_sym, Psi_sym),
            lambda Psi: One)),
            # we're assuming every KetSymbol is normalized. If we ever want
            # to allow non-normalized states, the best thing to dou would be to
            # add a `norm` attribute
        ('R005', (
            pattern_head(Psi_tensor, Phi_tensor),
            lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, BraKet.create))),
        ('R006', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u.conjugate() * (Psi.adjoint() * Phi))),
        ('R007', (
            pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
            lambda A, Psi, Phi: (Psi.adjoint() * (A.dag() * Phi)))),
        ('R008', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u * (Psi.adjoint() * Phi))),
        ('R009', (  # delegate to __mul__
            pattern_head(sum, sum2),
            lambda sum, sum2: Bra.create(sum) * sum2)),
        ('R010', (  # delegate to __mul__
            pattern_head(Psi, sum),
            lambda Psi, sum: Bra.create(Psi) * sum)),
        ('R011', (  # delegate to __mul__
            pattern_head(sum, Psi),
            lambda sum, Psi: Bra.create(sum) * Psi)),
    ]))

    KetBra._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(
                pattern(BasisKet, m, hs=ls),
                pattern(BasisKet, n, hs=ls)),
            lambda ls, m, n: LocalSigma(m, n, hs=ls))),
        ('R002', (
            pattern_head(pattern(CoherentStateKet, u, hs=ls), Phi),
            lambda ls, u, Phi: (
                Displace(u, hs=ls) * (BasisKet(0, hs=ls) * Phi.adjoint())))),
        ('R003', (
            pattern_head(Phi, pattern(CoherentStateKet, u, hs=ls)),
            lambda ls, u, Phi: (
                (Phi * BasisKet(0, hs=ls).adjoint()) * Displace(-u, hs=ls)))),
        ('R004', (
            pattern_head(Psi_tensor, Phi_tensor),
            lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, KetBra.create))),
        ('R005', (
            pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
            lambda A, Psi, Phi: A * (Psi * Phi.adjoint()))),
        ('R006', (
            pattern_head(Psi, pattern(OperatorTimesKet, A, Phi)),
            lambda Psi, A, Phi: (Psi * Phi.adjoint()) * A.adjoint())),
        ('R007', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u * (Psi * Phi.adjoint()))),
        ('R008', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u.conjugate() * (Psi * Phi.adjoint()))),
        ('R009', (  # delegate to __mul__
            pattern_head(sum, sum2),
            lambda sum, sum2: sum * Bra.create(sum2))),
        ('R010', (  # delegate to __mul__
            pattern_head(Psi, sum),
            lambda Psi, sum: Psi * Bra.create(sum))),
        ('R011', (  # delegate to __mul__
            pattern_head(sum, Psi),
            lambda sum, Psi: sum * Bra.create(Psi))),
    ]))

    def pull_constfactor_from_sum(u, Psi, indranges):
        bound_symbols = set([r.index_symbol for r in indranges])
        if len(u.free_symbols.intersection(bound_symbols)) == 0:
            return u * KetIndexedSum.create(Psi, *indranges)
        else:
            raise CannotSimplify()

    KetIndexedSum._rules.update(check_rules_dict([
        ('R001', (  # sum over zero -> zero
            pattern_head(ZeroKet, indranges__),
            lambda indranges: ZeroKet)),
        ('R002', (  # pull constant prefactor out of sum
            pattern_head(pattern(ScalarTimesKet, u, Psi), indranges__),
            lambda u, Psi, indranges:
                pull_constfactor_from_sum(u, Psi, indranges))),
    ]))


# Circuit rules
def _get_common_block_structure(lhs_bs, rhs_bs):
    """For two block structures ``aa = (a1, a2, ..., an)``, ``bb = (b1, b2,
    ..., bm)`` generate the maximal common block structure so that every block
    from aa and bb is contained in exactly one block of the resulting
    structure.  This is useful for determining how to apply the distributive
    law when feeding two concatenated Circuit objects into each other.

    Examples:
        ``(1, 1, 1), (2, 1) -> (2, 1)``
        ``(1, 1, 2, 1), (2, 1, 2) -> (2, 3)``

    Args:
        lhs_bs (tuple): first block structure
        rhs_bs (tuple): second block structure
    """

    # for convenience the arguments may also be Circuit objects
    if isinstance(lhs_bs, Circuit):
        lhs_bs = lhs_bs.block_structure
    if isinstance(rhs_bs, Circuit):
        rhs_bs = rhs_bs.block_structure

    if sum(lhs_bs) != sum(rhs_bs):
        raise IncompatibleBlockStructures(
            'Blockstructures have different total channel numbers.')

    if len(lhs_bs) == len(rhs_bs) == 0:
        return ()

    i = j = 1
    lsum = 0
    while True:
        lsum = sum(lhs_bs[:i])
        rsum = sum(rhs_bs[:j])
        if lsum < rsum:
            i += 1
        elif rsum < lsum:
            j += 1
        else:
            break

    return (lsum, ) + _get_common_block_structure(lhs_bs[i:], rhs_bs[j:])


def _tensor_decompose_series(lhs, rhs):
    """Simplification method for lhs << rhs

    Decompose a series product of two reducible circuits with compatible block
    structures into a concatenation of individual series products between
    subblocks.  This method raises CannotSimplify when rhs is a CPermutation in
    order not to conflict with other _rules.
    """
    if isinstance(rhs, CPermutation):
        raise CannotSimplify()
    lhs_structure = lhs.block_structure
    rhs_structure = rhs.block_structure
    res_struct = _get_common_block_structure(lhs_structure, rhs_structure)
    if len(res_struct) > 1:
        blocks, oblocks = (
            lhs.get_blocks(res_struct),
            rhs.get_blocks(res_struct))
        parallel_series = [SeriesProduct.create(lb, rb)
                           for (lb, rb) in zip(blocks, oblocks)]
        return Concatenation.create(*parallel_series)
    raise CannotSimplify()


def _factor_permutation_for_blocks(cperm, rhs):
    """Simplification method for cperm << rhs.
    Decompose a series product of a channel permutation and a reducible circuit
    with appropriate block structure by decomposing the permutation into a
    permutation within each block of rhs and a block permutation and a residual
    part.  This allows for achieving something close to a normal form for
    circuit expression.
    """
    rbs = rhs.block_structure
    if rhs == cid(rhs.cdim):
        return cperm
    if len(rbs) > 1:
        residual_lhs, transformed_rhs, carried_through_lhs \
                = cperm._factorize_for_rhs(rhs)
        if residual_lhs == cperm:
            raise CannotSimplify()
        return SeriesProduct.create(residual_lhs, transformed_rhs,
                                    carried_through_lhs)
    raise CannotSimplify()


def _pull_out_perm_lhs(lhs, rest, out_port, in_port):
    """Pull out a permutation from the Feedback of a SeriesProduct with itself.

    Args:
        lhs (CPermutation): The permutation circuit
        rest (tuple): The other SeriesProduct operands
        out_port (int): The feedback output port index
        in_port (int): The feedback input port index

    Returns:
        Circuit: The simplified circuit
    """
    out_inv, lhs_red = lhs._factor_lhs(out_port)
    return lhs_red << Feedback.create(SeriesProduct.create(*rest),
                                      out_port=out_inv, in_port=in_port)


def _pull_out_unaffected_blocks_lhs(lhs, rest, out_port, in_port):
    """In a self-Feedback of a series product, where the left-most operand is
    reducible, pull all non-trivial blocks outside of the feedback.

   Args:
       lhs (Circuit): The reducible circuit
       rest (tuple): The other SeriesProduct operands
       out_port (int): The feedback output port index
       in_port (int): The feedback input port index

   Returns:
       Circuit: The simplified circuit
   """

    _, block_index = lhs.index_in_block(out_port)

    bs = lhs.block_structure

    nbefore, nblock, nafter = (sum(bs[:block_index]),
                               bs[block_index],
                               sum(bs[block_index + 1:]))
    before, block, after = lhs.get_blocks((nbefore, nblock, nafter))

    if before != cid(nbefore) or after != cid(nafter):
        outer_lhs = before + cid(nblock - 1) + after
        inner_lhs = cid(nbefore) + block + cid(nafter)
        return outer_lhs << Feedback.create(
                SeriesProduct.create(inner_lhs, *rest),
                out_port=out_port, in_port=in_port)
    elif block == cid(nblock):
        outer_lhs = before + cid(nblock - 1) + after
        return outer_lhs << Feedback.create(
                SeriesProduct.create(*rest),
                out_port=out_port, in_port=in_port)
    raise CannotSimplify()


def _pull_out_perm_rhs(rest, rhs, out_port, in_port):
    """Similar to :func:`_pull_out_perm_lhs` but on the RHS of a series
    product self-feedback."""
    in_im, rhs_red = rhs._factor_rhs(in_port)
    return (Feedback.create(
                SeriesProduct.create(*rest),
                out_port=out_port, in_port=in_im) << rhs_red)


def _pull_out_unaffected_blocks_rhs(rest, rhs, out_port, in_port):
    """Similar to :func:`_pull_out_unaffected_blocks_lhs` but on the RHS of a
    series product self-feedback.
    """
    _, block_index = rhs.index_in_block(in_port)
    rest = tuple(rest)
    bs = rhs.block_structure
    (nbefore, nblock, nafter) = (sum(bs[:block_index]),
                                 bs[block_index],
                                 sum(bs[block_index + 1:]))
    before, block, after = rhs.get_blocks((nbefore, nblock, nafter))
    if before != cid(nbefore) or after != cid(nafter):
        outer_rhs = before + cid(nblock - 1) + after
        inner_rhs = cid(nbefore) + block + cid(nafter)
        return Feedback.create(SeriesProduct.create(*(rest + (inner_rhs,))),
                               out_port=out_port, in_port=in_port) << outer_rhs
    elif block == cid(nblock):
        outer_rhs = before + cid(nblock - 1) + after
        return Feedback.create(SeriesProduct.create(*rest),
                               out_port=out_port, in_port=in_port) << outer_rhs
    raise CannotSimplify()


def _series_feedback(series, out_port, in_port):
    """Invert a series self-feedback twice to get rid of unnecessary
    permutations."""
    series_s = series.series_inverse().series_inverse()
    if series_s == series:
        raise CannotSimplify()
    return series_s.feedback(out_port=out_port, in_port=in_port)


def _algebraic_rules_circuit():
    """Set the default algebraic rules for the operations defined in this
    module"""
    A_CPermutation = wc("A", head=CPermutation)
    B_CPermutation = wc("B", head=CPermutation)
    C_CPermutation = wc("C", head=CPermutation)
    D_CPermutation = wc("D", head=CPermutation)

    A_Concatenation = wc("A", head=Concatenation)
    B_Concatenation = wc("B", head=Concatenation)

    A_SeriesProduct = wc("A", head=SeriesProduct)

    A_Circuit = wc("A", head=Circuit)
    B_Circuit = wc("B", head=Circuit)
    C_Circuit = wc("C", head=Circuit)

    A__Circuit = wc("A__", head=Circuit)
    B__Circuit = wc("B__", head=Circuit)
    C__Circuit = wc("C__", head=Circuit)

    A_SLH = wc("A", head=SLH)
    B_SLH = wc("B", head=SLH)

    j_int = wc("j", head=int)
    k_int = wc("k", head=int)

    SeriesProduct._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(A_CPermutation, B_CPermutation),
            lambda A, B: A.series_with_permutation(B))),
        ('R002', (
            pattern_head(A_SLH, B_SLH),
            lambda A, B: A.series_with_slh(B))),
        ('R003', (
            pattern_head(A_Circuit, B_Circuit),
            lambda A, B: _tensor_decompose_series(A, B))),
        ('R004', (
            pattern_head(A_CPermutation, B_Circuit),
            lambda A, B: _factor_permutation_for_blocks(A, B))),
        ('R005', (
            pattern_head(A_Circuit, pattern(SeriesInverse, A_Circuit)),
            lambda A: cid(A.cdim))),
        ('R006', (
            pattern_head(pattern(SeriesInverse, A_Circuit), A_Circuit),
            lambda A: cid(A.cdim))),
    ]))

    Concatenation._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(A_SLH, B_SLH),
            lambda A, B: A.concatenate_slh(B))),
        ('R002', (
            pattern_head(A_CPermutation, B_CPermutation),
            lambda A, B: CPermutation.create(
                concatenate_permutations(A.permutation, B.permutation)))),
        ('R003', (
            pattern_head(A_CPermutation, CIdentity),
            lambda A: CPermutation.create(
                concatenate_permutations(A.permutation, (0,))))),
        ('R004', (
            pattern_head(CIdentity, B_CPermutation),
            lambda B: CPermutation.create(
                concatenate_permutations((0,), B.permutation)))),
        ('R005', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation),
                pattern(SeriesProduct, C__Circuit, D_CPermutation)),
            lambda A, B, C, D: (
                (SeriesProduct.create(*A) + SeriesProduct.create(*C)) <<
                (B + D)))),
        ('R006', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation), C_Circuit),
            lambda A, B, C: (
                (SeriesProduct.create(*A) + C) << (B + cid(C.cdim))))),
        ('R007', (
            pattern_head(
                A_Circuit, pattern(SeriesProduct, B__Circuit, C_CPermutation)),
            lambda A, B, C: ((A + SeriesProduct.create(*B)) <<
                             (cid(A.cdim) + C)))),
    ]))

    Feedback._rules.update(check_rules_dict([
        ('R001', (
            pattern_head(A_SeriesProduct, out_port=j_int, in_port=k_int),
            lambda A, j, k: _series_feedback(A, out_port=j, in_port=k))),
        ('R002', (
            pattern_head(
                pattern(SeriesProduct, A_CPermutation, B__Circuit),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_perm_lhs(A, B, j, k))),
        ('R003', (
            pattern_head(
                pattern(SeriesProduct, A_Concatenation, B__Circuit),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_unaffected_blocks_lhs(A, B, j, k))),
        ('R004', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_perm_rhs(A, B, j, k))),
        ('R005', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_Concatenation),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_unaffected_blocks_rhs(A, B, j, k))),
    ]))


def _algebraic_rules():
    _algebraic_rules_scalar()
    _algebraic_rules_operator()
    _algebraic_rules_superop()
    _algebraic_rules_state()
    _algebraic_rules_circuit()


_algebraic_rules()
