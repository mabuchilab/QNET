from sympy import Basic as SympyBasic, I, exp, sqrt

from .core.abstract_algebra import all_symbols
from .core.circuit_algebra import (
    ABCD, CIdentity, CPermutation, Circuit, Concatenation, Feedback, SLH,
    SeriesInverse, SeriesProduct, cid, get_common_block_structure, )
from .core.exceptions import CannotSimplify
from .core.hilbert_space_algebra import (
    HilbertSpace, LocalSpace, ProductSpace, TrivialSpace, )
from .core.operator_algebra import (
    Adjoint, Commutator, Create, Destroy, Displace, IdentityOperator, Jminus,
    Jmjmcoeff, Jpjmcoeff, Jplus, Jz, Jzjmcoeff, LocalOperator, LocalProjector,
    LocalSigma, Operator, OperatorIndexedSum, OperatorPlus, OperatorTimes,
    OperatorTrace, Phase, PseudoInverse, ScalarTimesOperator, Squeeze,
    ZeroOperator, decompose_space, factor_for_trace, )
from .core.scalar_types import SCALAR_TYPES
from .core.state_algebra import (
    BasisKet, Bra, BraKet, CoherentStateKet, State, KetBra, KetIndexedSum,
    KetPlus, LocalKet, OperatorTimesKet, ScalarTimesKet, TensorKet, TrivialKet,
    ZeroKet, )
from .core.super_operator_algebra import (
    IdentitySuperOperator, SPost, SPre, ScalarTimesSuperOperator, SuperAdjoint,
    SuperOperator, SuperOperatorPlus, SuperOperatorTimes,
    SuperOperatorTimesOperator, ZeroSuperOperator, )
from .pattern_matching import pattern, pattern_head, wc
from ..utils.check_rules import check_rules_dict
from ..utils.indices import IndexRangeBase, KroneckerDelta, SymbolicLabelBase
from ..utils.permutations import concatenate_permutations


# Operator rules

def _algebraic_rules_operator():
    """Set the default algebraic rules for the operations defined in this
    module"""
    PseudoInverse._delegate_to_method += (PseudoInverse,)

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
        'localsigma', head=(LocalSigma, LocalProjector), kwargs={'hs': ls})

    ra = wc("ra", head=(int, str, SymbolicLabelBase))
    rb = wc("rb", head=(int, str, SymbolicLabelBase))
    rc = wc("rc", head=(int, str, SymbolicLabelBase))
    rd = wc("rd", head=(int, str, SymbolicLabelBase))

    indranges__ = wc("indranges__", head=IndexRangeBase)

    ScalarTimesOperator._rules.update(check_rules_dict([
        ('1A', (
            pattern_head(1, A),
            lambda A: A)),
        ('0A', (
            pattern_head(0, A),
            lambda A: ZeroOperator)),
        ('u0', (
            pattern_head(u, ZeroOperator),
            lambda u: ZeroOperator)),
        ('assoc_coeff', (
            pattern_head(u, pattern(ScalarTimesOperator, v, A)),
            lambda u, v, A: (u * v) * A)),
        ('negsum', (
            pattern_head(-1, A_plus),
            lambda A: OperatorPlus.create(*[-1 * op for op in A.args]))),
    ]))

    OperatorPlus._binary_rules.update(check_rules_dict([
        ('upv', (
            pattern_head(
                pattern(ScalarTimesOperator, u, A),
                pattern(ScalarTimesOperator, v, A)),
            lambda u, v, A: (u + v) * A)),
        ('up1', (
            pattern_head(pattern(ScalarTimesOperator, u, A), A),
            lambda u, A: (u + 1) * A)),
        ('1pv', (
            pattern_head(A, pattern(ScalarTimesOperator, v, A)),
            lambda v, A: (1 + v) * A)),
        ('2A', (
            pattern_head(A, A),
            lambda A: 2 * A)),
    ]))

    OperatorTimes._binary_rules.update(check_rules_dict([
        ('uAB', (
            pattern_head(pattern(ScalarTimesOperator, u, A), B),
            lambda u, A, B: u * (A * B))),

        ('zero1', (
            pattern_head(ZeroOperator, B),
            lambda B: ZeroOperator)),
        ('zero2', (
            pattern_head(A, ZeroOperator),
            lambda A: ZeroOperator)),

        ('coeff', (
            pattern_head(A, pattern(ScalarTimesOperator, u, B)),
            lambda A, u, B: u * (A * B))),

        ('sig', (
            pattern_head(
                pattern(LocalSigma, ra, rb, hs=ls),
                pattern(LocalSigma, rc, rd, hs=ls)),
            lambda ls, ra, rb, rc, rd: (
                KroneckerDelta(rb, rc) * LocalSigma.create(ra, rd, hs=ls)))),
        ('sigproj', (
            pattern_head(
                pattern(LocalSigma, ra, rb, hs=ls),
                pattern(LocalProjector, rc, hs=ls)),
            lambda ls, ra, rb, rc: (
                KroneckerDelta(rb, rc) * LocalSigma.create(ra, rc, hs=ls)))),
        ('projsig', (
            pattern_head(
                pattern(LocalProjector, ra, hs=ls),
                pattern(LocalSigma, rc, rd, hs=ls)),
            lambda ls, ra, rc, rd: (
                KroneckerDelta(ra, rc) * LocalSigma.create(ra, rd, hs=ls)))),
        ('projproj', (
            pattern_head(
                pattern(LocalProjector, ra, hs=ls),
                pattern(LocalProjector, rc, hs=ls)),
            lambda ls, ra, rc: (
                KroneckerDelta(ra, rc) * LocalProjector(ra, hs=ls)))),

        # Harmonic oscillator rules
        ('hamos1', (
            pattern_head(pattern(Create, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j + 1) * localsigma.raise_jk(j_incr=1))),

        ('hamos2', (
            pattern_head(pattern(Destroy, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j) * localsigma.raise_jk(j_incr=-1))),

        ('hamos3', (
            pattern_head(localsigma, pattern(Destroy, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k + 1) * localsigma.raise_jk(k_incr=1))),

        ('hamos4', (
            pattern_head(localsigma, pattern(Create, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k) * localsigma.raise_jk(k_incr=-1))),

        # Normal ordering for harmonic oscillator <=> all a^* to the left, a to
        # the right.
        ('hamosord', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Create, hs=ls)),
            lambda ls: IdentityOperator + Create(hs=ls) * Destroy(hs=ls))),

        # Oscillator unitary group rules
        ('phase2', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Phase, v, hs=ls)),
            lambda ls, u, v: Phase.create(u + v, hs=ls))),
        ('displace2', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(Displace, v, hs=ls)),
            lambda ls, u, v: (
                exp((u * v.conjugate() - u.conjugate() * v) / 2) *
                Displace.create(u + v, hs=ls)))),

        ('dphase', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Phase, u, hs=ls)),
            lambda ls, u:
                exp(I * u) * Phase.create(u, hs=ls) * Destroy(hs=ls))),
        ('ddisplace', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Displace, u, hs=ls)),
            lambda ls, u: Displace.create(u, hs=ls) * (Destroy(hs=ls) + u))),

        ('phasec', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u:
                exp(I * u) * Create(hs=ls) * Phase.create(u, hs=ls))),
        ('displacec', (
            pattern_head(pattern(Displace, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u: (((Create(hs=ls) - u.conjugate()) *
                            Displace.create(u, hs=ls))))),

        ('phasesig', (
            pattern_head(pattern(Phase, u, hs=ls), localsigma),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_j) * localsigma)),
        ('sigphase', (
            pattern_head(localsigma, pattern(Phase, u, hs=ls)),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_k) * localsigma)),

        # Spin rules
        ('spin1', (
            pattern_head(pattern(Jplus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=1))),

        ('spin2', (
            pattern_head(pattern(Jminus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=-1))),

        ('spin3', (
            pattern_head(pattern(Jz, hs=ls), localsigma),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_j, shift=True) * localsigma)),

        ('spin4', (
            pattern_head(localsigma, pattern(Jplus, hs=ls)),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=-1))),

        ('spin5', (
            pattern_head(localsigma, pattern(Jminus, hs=ls)),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=+1))),

        ('spin6', (
            pattern_head(localsigma, pattern(Jz, hs=ls)),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_k, shift=True) * localsigma)),

        # Normal ordering for angular momentum <=> all J_+ to the left, J_z to
        # center and J_- to the right
        ('spinord1', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: -2 * Jz(hs=ls) + Jplus(hs=ls) * Jminus(hs=ls))),

        ('spinord2', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jz, hs=ls)),
            lambda ls: Jz(hs=ls) * Jminus(hs=ls) + Jminus(hs=ls))),

        ('spinord3', (
            pattern_head(pattern(Jz, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: Jplus(hs=ls) * Jz(hs=ls) + Jplus(hs=ls))),
    ]))

    Displace._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Phase._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Squeeze._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    LocalSigma._rules.update(check_rules_dict([
        ('projector', (
            pattern_head(n, n, hs=ls), lambda n, ls: LocalProjector(n, hs=ls)))
    ]))

    OperatorTrace._rules.update(check_rules_dict([
        ('triv', (
            pattern_head(A, over_space=TrivialSpace),
            lambda A: A)),
        ('zero', (
            pattern_head(ZeroOperator, over_space=h1),
            lambda h1: ZeroOperator)),
        ('id', (
            pattern_head(IdentityOperator, over_space=h1),
            lambda h1: h1.dimension * IdentityOperator)),
        ('plus', (
            pattern_head(A_plus, over_space=h1),
            lambda h1, A: OperatorPlus.create(
                *[OperatorTrace.create(o, over_space=h1)
                  for o in A.operands]))),
        ('adjoint', (
            pattern_head(pattern(Adjoint, A), over_space=h1),
            lambda h1, A: Adjoint.create(
                OperatorTrace.create(A, over_space=h1)))),
        ('uA', (
            pattern_head(pattern(ScalarTimesOperator, u, A), over_space=h1),
            lambda h1, u, A: u * OperatorTrace.create(A, over_space=h1))),
        ('prodspace', (
            pattern_head(A, over_space=H_ProductSpace),
            lambda H, A: decompose_space(H, A))),
        ('create', (
            pattern_head(pattern(Create, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('destroy', (
            pattern_head(pattern(Destroy, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('sigma', (
            pattern_head(pattern(LocalSigma, n, m, hs=ls), over_space=ls),
            lambda ls, n, m: KroneckerDelta(n, m) * IdentityOperator)),
        ('proj', (
            pattern_head(pattern(LocalProjector, n, hs=ls), over_space=ls),
            lambda ls, n: IdentityOperator)),
        ('factor', (
            pattern_head(A, over_space=ls),
            lambda ls, A: factor_for_trace(ls, A))),
    ]))

    PseudoInverse._rules.update(check_rules_dict([
        ('sig', (
            pattern_head(pattern(LocalSigma, m, n, hs=ls)),
            lambda ls, m, n: LocalSigma(n, m, hs=ls))),
        ('proj', (
            pattern_head(pattern(LocalProjector, m, hs=ls)),
            lambda ls, m: LocalProjector(m, hs=ls))),
    ]))

    Commutator._rules.update(check_rules_dict([
        ('AA', (
            pattern_head(A, A), lambda A: ZeroOperator)),
        ('uAvB', (
            pattern_head(
                pattern(ScalarTimesOperator, u, A),
                pattern(ScalarTimesOperator, v, B)),
            lambda u, v, A, B: u * v * Commutator.create(A, B))),
        ('vAB', (
            pattern_head(pattern(ScalarTimesOperator, v, A), B),
            lambda v, A, B: v * Commutator.create(A, B))),
        ('AvB', (
            pattern_head(A, pattern(ScalarTimesOperator, v, B)),
            lambda v, A, B: v * Commutator.create(A, B))),

        # special known commutators
        ('crdest', (
            pattern_head(pattern(Create, hs=ls), pattern(Destroy, hs=ls)),
            lambda ls: ScalarTimesOperator(-1, IdentityOperator))),
        # the remaining  rules basically defer to OperatorTimes; just writing
        # out the commutator will generate something simple
        ('expand1', (
            pattern_head(
                wc('A', head=(
                    Create, Destroy, LocalSigma, LocalProjector, Phase,
                    Displace)),
                wc('B', head=(
                    Create, Destroy, LocalSigma, LocalProjector, Phase,
                    Displace))),
            lambda A, B: A * B - B * A)),
        ('expand2', (
            pattern_head(
                wc('A', head=(LocalSigma, LocalProjector, Jplus, Jminus, Jz)),
                wc('B', head=(LocalSigma, LocalProjector, Jplus, Jminus, Jz))),
            lambda A, B: A * B - B * A)),
    ]))

    def pull_constfactor_from_sum(u, A, indranges):
        bound_symbols = set([r.index_symbol for r in indranges])
        if len(all_symbols(u).intersection(bound_symbols)) == 0:
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
        ('one', (
            pattern_head(1, sA),
            lambda sA: sA)),
        ('zero', (
            pattern_head(0, sA),
            lambda sA: ZeroSuperOperator)),
        ('zeroSop', (
            pattern_head(u, ZeroSuperOperator),
            lambda u: ZeroSuperOperator)),
        ('uvSop', (
            pattern_head(u, pattern(ScalarTimesSuperOperator, v, sA)),
            lambda u, v, sA: (u * v) * sA)),
    ]))

    SuperOperatorPlus._binary_rules.update(check_rules_dict([
        ('upv', (
            pattern_head(
                pattern(ScalarTimesSuperOperator, u, sA),
                pattern(ScalarTimesSuperOperator, v, sA)),
            lambda u, v, sA: (u + v) * sA)),
        ('up1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sA),
            lambda u, sA: (u + 1) * sA)),
        ('1pv', (
            pattern_head(sA, pattern(ScalarTimesSuperOperator, v, sA)),
            lambda v, sA: (1 + v) * sA)),
        ('two', (
            pattern_head(sA, sA),
            lambda sA: 2 * sA)),
    ]))

    SuperOperatorTimes._binary_rules.update(check_rules_dict([
        ('uSaSb1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sB),
            lambda u, sA, sB: u * (sA * sB))),
        ('uSaSb2', (
            pattern_head(sA, pattern(ScalarTimesSuperOperator, u, sB)),
            lambda sA, u, sB: u * (sA * sB))),
        ('spre', (
            pattern_head(pattern(SPre, A), pattern(SPre, B)),
            lambda A, B: SPre.create(A*B))),
        ('spost', (
            pattern_head(pattern(SPost, A), pattern(SPost, B)),
            lambda A, B: SPost.create(B*A))),
    ]))

    SPre._rules.update(check_rules_dict([
        ('scal', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPre.create(A))),
        ('id', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('zero', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SPost._rules.update(check_rules_dict([
        ('scal', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPost.create(A))),
        ('id', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('zero', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SuperOperatorTimesOperator._rules.update(check_rules_dict([
        ('plus', (
            pattern_head(sA_plus, B),
            lambda sA, B:
                SuperOperatorPlus.create(*[o*B for o in sA.operands]))),
        ('id', (
            pattern_head(IdentitySuperOperator, B),
            lambda B: B)),
        ('zero', (
            pattern_head(ZeroSuperOperator, B),
            lambda B: ZeroOperator)),
        ('uSaSb1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), B),
            lambda u, sA, B: u * (sA * B))),
        ('uSaSb2', (
            pattern_head(sA, pattern(ScalarTimesOperator, u, B)),
            lambda u, sA, B: u * (sA * B))),
        ('sAsBC', (
            pattern_head(sA, pattern(SuperOperatorTimesOperator, sB, C)),
            lambda sA, sB, C: (sA * sB) * C)),
        ('AB', (
            pattern_head(pattern(SPre, A), B),
            lambda A, B: A*B)),
        ('xxx1', (  # XXX
            pattern_head(
                pattern(SuperOperatorTimes, sA__, pattern(SPre, B)), C),
            lambda sA, B, C: (
                SuperOperatorTimes.create(*sA) * (pattern(SPre, B) * C)))),
        ('spost', (
            pattern_head(pattern(SPost, A), B),
            lambda A, B: B*A)),
        ('xxx2', (  # XXX
            pattern_head(
                pattern(SuperOperatorTimes, sA__, pattern(SPost, B)), C),
            lambda sA, B, C: (
                SuperOperatorTimes.create(*sA) * (pattern(SPost, B) * C)))),
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

    nsym = wc("nsym", head=(int, str, SympyBasic))

    Psi = wc("Psi", head=State)
    Phi = wc("Phi", head=State)
    Psi_local = wc("Psi", head=LocalKet)
    Psi_tensor = wc("Psi", head=TensorKet)
    Phi_tensor = wc("Phi", head=TensorKet)

    ls = wc("ls", head=LocalSpace)

    basisket = wc('basisket', BasisKet, kwargs={'hs': ls})

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
            lambda ls, n, m, k: KroneckerDelta(m, k) * BasisKet(n, hs=ls))),

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

    KetPlus._binary_rules.update(check_rules_dict([
        ('R001', (
            pattern_head(
                pattern(ScalarTimesKet, u, Psi),
                pattern(ScalarTimesKet, v, Psi)),
            lambda u, v, Psi: (u + v) * Psi)),
        ('R002', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Psi),
            lambda u, Psi: (u + 1) * Psi)),
        ('R003', (
            pattern_head(Psi, pattern(ScalarTimesKet, v, Psi)),
            lambda v, Psi: (1 + v) * Psi)),
        ('R004', (
            pattern_head(Psi, Psi),
            lambda Psi: 2 * Psi)),
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
            lambda Phi: 0)),
        ('R002', (
            pattern_head(ZeroKet, Phi),
            lambda Phi: 0)),
        ('R003', (
            pattern_head(
                pattern(BasisKet, m, hs=ls), pattern(BasisKet, n, hs=ls)),
            lambda ls, m, n: KroneckerDelta(m, n))),
        ('R004', (
            pattern_head(
                pattern(BasisKet, nsym, hs=ls),
                pattern(BasisKet, nsym, hs=ls)),
            lambda ls, nsym: 1)),
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
        if len(all_symbols(u).intersection(bound_symbols)) == 0:
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

def _tensor_decompose_series(lhs, rhs):
    """Simplification method for lhs << rhs

    Decompose a series product of two reducible circuits with compatible block
    structures into a concatenation of individual series products between
    subblocks.  This method raises CannotSimplify when rhs is a CPermutation in
    order not to conflict with other _rules.

    :type lhs: Circuit
    :type rhs: Circuit
    :return: The combined reducible circuit
    :rtype: Circuit
    :raise: CannotSimplify
    """
    if isinstance(rhs, CPermutation):
        raise CannotSimplify()
    lhs_structure = lhs.block_structure
    rhs_structure = rhs.block_structure
    res_struct = get_common_block_structure(lhs_structure, rhs_structure)
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

    :type cperm: CPermutation
    :type rhs: Circuit
    :rtype: Circuit
    :raise: CannotSimplify
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

    :param lhs: The permutation circuit
    :type lhs: CPermutation
    :param rest: The other SeriesProduct operands
    :type rest: OperandsTuple
    :param out_port: The feedback output port index
    :type out_port: int
    :param in_port: The feedback input port index
    :type in_port: int
    :return: The simplified circuit
    :rtype: Circuit
    """
    out_inv, lhs_red = lhs._factor_lhs(out_port)
    return lhs_red << Feedback.create(SeriesProduct.create(*rest),
                                      out_port=out_inv, in_port=in_port)


def _pull_out_unaffected_blocks_lhs(lhs, rest, out_port, in_port):
    """In a self-Feedback of a series product, where the left-most operand is
    reducible, pull all non-trivial blocks outside of the feedback.

   :param lhs: The reducible circuit
   :type lhs: Circuit
   :param rest: The other SeriesProduct operands
   :type rest: OperandsTuple
   :param out_port: The feedback output port index
   :type out_port: int
   :param in_port: The feedback input port index
   :type in_port: int
   :return: The simplified circuit
   :rtype: Circuit
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
    """Similar to :py:func:_pull_out_perm_lhs: but on the RHS of a series
    product self-feedback."""
    in_im, rhs_red = rhs._factor_rhs(in_port)
    return (Feedback.create(
                SeriesProduct.create(*rest),
                out_port=out_port, in_port=in_im) << rhs_red)


def _pull_out_unaffected_blocks_rhs(rest, rhs, out_port, in_port):
    """Similar to :py:func:_pull_out_unaffected_blocks_lhs: but on the RHS of a
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

    A_ABCD = wc("A", head=ABCD)
    B_ABCD = wc("B", head=ABCD)

    j_int = wc("j", head=int)
    k_int = wc("k", head=int)

    SeriesProduct._binary_rules.update(check_rules_dict([
        ('perm', (
            pattern_head(A_CPermutation, B_CPermutation),
            lambda A, B: A.series_with_permutation(B))),
        ('slh', (
            pattern_head(A_SLH, B_SLH),
            lambda A, B: A.series_with_slh(B))),
        ('abcd', (
            pattern_head(A_ABCD, B_ABCD),
            lambda A, B: A.series_with_abcd(B))),
        ('circuit', (
            pattern_head(A_Circuit, B_Circuit),
            lambda A, B: _tensor_decompose_series(A, B))),
        ('permcirc', (
            pattern_head(A_CPermutation, B_Circuit),
            lambda A, B: _factor_permutation_for_blocks(A, B))),
        ('inv2', (
            pattern_head(A_Circuit, pattern(SeriesInverse, A_Circuit)),
            lambda A: cid(A.cdim))),
        ('inv1', (
            pattern_head(pattern(SeriesInverse, A_Circuit), A_Circuit),
            lambda A: cid(A.cdim))),
    ]))

    Concatenation._binary_rules.update(check_rules_dict([
        ('slh', (
            pattern_head(A_SLH, B_SLH),
            lambda A, B: A.concatenate_slh(B))),
        ('abcd', (
            pattern_head(A_ABCD, B_ABCD),
            lambda A, B: A.concatenate_abcd(B))),
        ('perm', (
            pattern_head(A_CPermutation, B_CPermutation),
            lambda A, B: CPermutation.create(
                concatenate_permutations(A.permutation, B.permutation)))),
        ('permId', (
            pattern_head(A_CPermutation, CIdentity),
            lambda A: CPermutation.create(
                concatenate_permutations(A.permutation, (0,))))),
        ('Idperm', (
            pattern_head(CIdentity, B_CPermutation),
            lambda B: CPermutation.create(
                concatenate_permutations((0,), B.permutation)))),
        ('sp1', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation),
                pattern(SeriesProduct, C__Circuit, D_CPermutation)),
            lambda A, B, C, D: (
                (SeriesProduct.create(*A) + SeriesProduct.create(*C)) <<
                (B + D)))),
        ('sp2', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation), C_Circuit),
            lambda A, B, C: (
                (SeriesProduct.create(*A) + C) << (B + cid(C.cdim))))),
        ('sp3', (
            pattern_head(
                A_Circuit, pattern(SeriesProduct, B__Circuit, C_CPermutation)),
            lambda A, B, C: ((A + SeriesProduct.create(*B)) <<
                             (cid(A.cdim) + C)))),
    ]))

    Feedback._rules.update(check_rules_dict([
        ('series', (
            pattern_head(A_SeriesProduct, out_port=j_int, in_port=k_int),
            lambda A, j, k: _series_feedback(A, out_port=j, in_port=k))),
        ('pull1', (
            pattern_head(
                pattern(SeriesProduct, A_CPermutation, B__Circuit),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_perm_lhs(A, B, j, k))),
        ('pull2', (
            pattern_head(
                pattern(SeriesProduct, A_Concatenation, B__Circuit),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_unaffected_blocks_lhs(A, B, j, k))),
        ('pull3', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_CPermutation),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_perm_rhs(A, B, j, k))),
        ('pull4', (
            pattern_head(
                pattern(SeriesProduct, A__Circuit, B_Concatenation),
                out_port=j_int, in_port=k_int),
            lambda A, B, j, k: _pull_out_unaffected_blocks_rhs(A, B, j, k))),
    ]))


def _algebraic_rules():
    _algebraic_rules_operator()
    _algebraic_rules_superop()
    _algebraic_rules_state()
    _algebraic_rules_circuit()


_algebraic_rules()
