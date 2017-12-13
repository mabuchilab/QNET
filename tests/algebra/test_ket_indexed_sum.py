from qnet.printing import ascii, unicode, latex, srepr, configure_printing
from qnet.algebra.abstract_algebra import all_symbols, InfiniteSumError
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.state_algebra import (
    KetPlus, ScalarTimesKet, KetIndexedSum, BasisKet, CoherentStateKet,
    KetSymbol)
from qnet.algebra.indices import (
    IdxSym, StrLabel, FockIndex, IndexOverFockSpace, IndexOverList,
    IndexOverRange)
from qnet.algebra.toolbox import expand_indexed_sum
import sympy
from sympy import symbols, IndexedBase

import pytest


def test_qubit_state():
    """Test  sum_i alpha_i |i> for TLS"""
    i = IdxSym('i')
    alpha = IndexedBase('alpha')
    alpha_i = alpha[i]
    hs_tls = LocalSpace('tls', basis=('g', 'e'))

    term = alpha_i * BasisKet(FockIndex(i), hs=hs_tls)

    expr1 = KetIndexedSum.create(
        term, IndexOverFockSpace(i, hs=hs_tls))

    expr2 = KetIndexedSum.create(
        term, IndexOverList(i, [0, 1]))

    expr3 = KetIndexedSum.create(
        term, IndexOverRange(i, start_from=0, to=1))

    assert IndexOverFockSpace(i, hs=hs_tls) in expr1.args

    assert ascii(expr1) == "Sum_{i in H_tls} alpha_i * |i>^(tls)"
    assert unicode(expr1) == "∑_{i ∈ ℌ_tls} α_i |i⟩⁽ᵗˡˢ⁾"
    assert (
        srepr(expr1) ==
        "KetIndexedSum(ScalarTimesKet(Indexed(IndexedBase(Symbol('alpha')), "
        "IdxSym('i', integer=True)), BasisKet(FockIndex(IdxSym('i', "
        "integer=True)), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverFockSpace(IdxSym('i', integer=True), LocalSpace('tls', "
        "basis=('g', 'e'))))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr1) ==
            r'\sum_{i \in \mathcal{H}_{tls}} \alpha_{i} \Ket{i}^{(tls)}')

    assert ascii(expr2) == 'Sum_{i in {0,1}} alpha_i * |i>^(tls)'
    assert unicode(expr2) == '∑_{i ∈ {0,1}} α_i |i⟩⁽ᵗˡˢ⁾'
    assert (
        srepr(expr2) ==
        "KetIndexedSum(ScalarTimesKet(Indexed(IndexedBase(Symbol('alpha')), "
        "IdxSym('i', integer=True)), BasisKet(FockIndex(IdxSym('i', "
        "integer=True)), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverList(IdxSym('i', integer=True), (0, 1)))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr2) == r'\sum_{i \in \{0,1\}} \alpha_{i} \Ket{i}^{(tls)}')

    assert ascii(expr3) == 'Sum_{i=0}^{1} alpha_i * |i>^(tls)'
    assert unicode(expr3) == '∑_{i=0}^{1} α_i |i⟩⁽ᵗˡˢ⁾'
    assert (
        srepr(expr3) ==
        "KetIndexedSum(ScalarTimesKet(Indexed(IndexedBase(Symbol('alpha')), "
        "IdxSym('i', integer=True)), BasisKet(FockIndex(IdxSym('i', "
        "integer=True)), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverRange(IdxSym('i', integer=True), 0, 1))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr3) == r'\sum_{i=0}^{1} \alpha_{i} \Ket{i}^{(tls)}')

    for expr in (expr1, expr2, expr3):
        assert all_symbols(expr.term) == set([i, symbols('alpha')])
        assert all_symbols(expr) == set([symbols('alpha'), ])
        assert expr.variables == [i]
        assert len(expr) == len(expr.ranges[0]) == 2
        assert 0 in expr.ranges[0]
        assert 1 in expr.ranges[0]
        assert expr.space == hs_tls
        assert len(expr.args) == 2
        assert len(expr.operands) == 1
        assert expr.args[0] == term
        assert expr.term == term
        assert len(expr.kwargs) == 0
        expr_expand = expr.expand_sum().substitute(
            {alpha[0]: alpha['g'], alpha[1]: alpha['e']})
        assert expr_expand == (
            alpha['g'] * BasisKet('g', hs=hs_tls) +
            alpha['e'] * BasisKet('e', hs=hs_tls))
        assert (
            ascii(expr_expand) == 'alpha_e * |e>^(tls) + alpha_g * |g>^(tls)')

    with pytest.raises(IndexError) as exc_info:
        KetIndexedSum.create(term, IndexOverRange(i, 0, 2)).expand_sum()
    assert "tuple index out of range" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        KetIndexedSum.create(
            alpha_i * BasisKet(i, hs=hs_tls),
            IndexOverFockSpace(i, hs=hs_tls))
    assert "label_or_index must be an instance of" in str(exc_info.value)


def test_qubit_state_bra():
    """Test  sum_i alpha_i <i| for TLS"""
    i = IdxSym('i')
    alpha = IndexedBase('alpha')
    alpha_i = alpha[i]
    hs_tls = LocalSpace('tls', basis=('g', 'e'))

    term = alpha_i * BasisKet(FockIndex(i), hs=hs_tls).dag

    expr = KetIndexedSum.create(
        term, IndexOverFockSpace(i, hs=hs_tls))

    assert IndexOverFockSpace(i, hs=hs_tls) in expr.args

    assert ascii(expr) == "Sum_{i in H_tls} alpha_i * <i|^(tls)"

    assert all_symbols(expr.term) == set([i, symbols('alpha')])
    assert all_symbols(expr) == set([symbols('alpha'), ])
    assert expr.variables == [i]
    assert expr.space == hs_tls
    assert len(expr.args) == 2
    assert len(expr.operands) == 1
    assert expr.args[0] == term
    assert expr.term == term
    assert len(expr.kwargs) == 0
    expr_expand = expr.expand_sum().substitute(
        {alpha[0]: alpha['g'], alpha[1]: alpha['e']})
    assert expr_expand == (
        alpha['g'] * BasisKet('g', hs=hs_tls).dag +
        alpha['e'] * BasisKet('e', hs=hs_tls).dag)
    assert (
        ascii(expr_expand) == 'alpha_e * <e|^(tls) + alpha_g * <g|^(tls)')


def test_coherent_state():
    """Test fock representation of coherent state"""
    alpha = symbols('alpha')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1, dimension=3)
    i = IdxSym('i')
    n = IdxSym('n')

    psi = CoherentStateKet(alpha, hs=hs0)
    psi_focksum_3 = psi.to_fock_representation(max_terms=3)
    assert len(psi_focksum_3.term) == 3
    for n_val in (0, 1, 2):
        assert n_val in psi_focksum_3.term.ranges[0]
    psi_focksum_inf = psi.to_fock_representation()
    with pytest.raises(InfiniteSumError):
        len(psi_focksum_inf.term)
    for n_val in (0, 1, 2, 3):
        assert n_val in psi_focksum_inf.term.ranges[0]

    assert (
        all_symbols(psi_focksum_inf.term.term) == set([n, symbols('alpha')]))
    assert all_symbols(psi_focksum_inf) == set([symbols('alpha'), ])
    assert psi_focksum_inf.term.variables == [n]

    assert psi_focksum_inf.substitute({n: i}).term.variables == [i]
    assert (
        psi_focksum_inf.substitute({hs0: hs1}) ==
        CoherentStateKet(alpha, hs=hs1).to_fock_representation())

    assert (
        psi.to_fock_representation(index_symbol='i').substitute({i: n}) ==
        psi_focksum_inf)
    assert (
        psi.to_fock_representation(index_symbol=i).substitute({i: n}) ==
        psi_focksum_inf)

    assert(
        expand_indexed_sum(psi_focksum_3) ==
        expand_indexed_sum(psi_focksum_inf, max_terms=3))
    psi_expanded_3 = expand_indexed_sum(psi_focksum_3)
    assert(
        psi_expanded_3 ==
        (sympy.exp(-alpha*alpha.conjugate()/2) *
            KetPlus(
                BasisKet(0, hs=LocalSpace(0)),
                ScalarTimesKet(alpha, BasisKet(1, hs=LocalSpace(0))),
                ScalarTimesKet(
                    alpha**2 / sympy.sqrt(2), BasisKet(2, hs=LocalSpace(0))))))

    psi = CoherentStateKet(alpha, hs=hs1)
    assert(
        expand_indexed_sum(psi.to_fock_representation()) ==
        psi_expanded_3.substitute({hs0: hs1}))


def test_two_hs_symbol_sum():
    """Test sum_{ij} a_{ij} Psi_{ij}"""
    i = IdxSym('i')
    j = IdxSym('j')
    a = IndexedBase('a')
    hs1 = LocalSpace('1', dimension=3)
    hs2 = LocalSpace('2', dimension=3)
    hs = hs1 * hs2
    Psi = IndexedBase('Psi')
    a_ij = a[i, j]
    Psi_ij = Psi[i, j]
    KetPsi_ij = KetSymbol(StrLabel(Psi_ij), hs=hs)
    term = a_ij * KetPsi_ij

    expr1 = KetIndexedSum(
        term, IndexOverFockSpace(i, hs=hs1), IndexOverFockSpace(j, hs=hs2))

    expr2 = KetIndexedSum(
        term, IndexOverRange(i, 0, 2), IndexOverRange(j, 0, 2))

    assert all_symbols(expr1.term) == set([i, j, symbols('a'), symbols('Psi')])
    assert all_symbols(expr1) == set([symbols('a'), symbols('Psi')])
    assert expr1.variables == [i, j]

    assert (
        ascii(expr1) == 'Sum_{i in H_1} Sum_{j in H_2} a_ij * |Psi_ij>^(1*2)')
    assert (
        unicode(expr1) == '∑_{i ∈ ℌ₁} ∑_{j ∈ ℌ₂} a_ij |Ψ_ij⟩^(1⊗2)')
    assert (
        latex(expr1) ==
        r'\sum_{i \in \mathcal{H}_{1}} \sum_{j \in \mathcal{H}_{2}} '
        r'a_{i j} \left\lvert \Psi_{i j} \right\rangle^{(1 \otimes 2)}')

    assert ascii(expr2) == 'Sum_{i,j=0}^{2} a_ij * |Psi_ij>^(1*2)'
    assert unicode(expr2) == '∑_{i,j=0}^{2} a_ij |Ψ_ij⟩^(1⊗2)'
    assert (
        latex(expr2) ==
        r'\sum_{i,j=0}^{2} a_{i j} '
        r'\left\lvert \Psi_{i j} \right\rangle^{(1 \otimes 2)}')

    assert expr1.expand_sum() == expr2.expand_sum()
    assert expr1.expand_sum() == KetPlus(
        a[0, 0] * KetSymbol('Psi_00', hs=hs),
        a[0, 1] * KetSymbol('Psi_01', hs=hs),
        a[0, 2] * KetSymbol('Psi_02', hs=hs),
        a[1, 0] * KetSymbol('Psi_10', hs=hs),
        a[1, 1] * KetSymbol('Psi_11', hs=hs),
        a[1, 2] * KetSymbol('Psi_12', hs=hs),
        a[2, 0] * KetSymbol('Psi_20', hs=hs),
        a[2, 1] * KetSymbol('Psi_21', hs=hs),
        a[2, 2] * KetSymbol('Psi_22', hs=hs))


def test_make_disjunct_indices():
    i = IdxSym('i')
    j = IdxSym('j')
    k = IdxSym('k')
    Psi = IndexedBase('Psi')

    def sum(term, *index_symbols):
        return KetIndexedSum(
            term, *[IndexOverRange(i, 0, 2) for i in index_symbols])

    with pytest.raises(ValueError):
        sum(Psi[i, j, j], i, j, j)

    expr = sum(Psi[i, j, k], i, j, k)
    others = [expr, ]
    assert expr.make_disjunct_indices(*others) == expr.substitute({
            i: i.prime, j: j.prime, k: k.prime})

    others = [sum(Psi[i], i), sum(Psi[j], j)]
    assert expr.make_disjunct_indices(*others) == expr.substitute({
            i: i.prime, j: j.prime})

    others = [sum(Psi[i], i), sum(Psi[i.prime], i.prime)]
    assert expr.make_disjunct_indices(*others) == expr.substitute({
            i: i.prime.prime})
