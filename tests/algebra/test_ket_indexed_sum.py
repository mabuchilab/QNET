from qnet.printing import ascii, unicode, latex, srepr, configure_printing
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.state_algebra import KetIndexedSum, BasisKet
from qnet.algebra.indices import (
        FockIndex, IndexOverFockSpace, IndexOverList, IndexOverRange)
from sympy import symbols, Idx, IndexedBase

import pytest


def test_qubit_state():
    """Test  sum_i alpha_i |i> for TLS"""
    i = Idx('i')
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
        "Idx(Symbol('i', integer=True))), BasisKet(FockIndex(Idx(Symbol('i', "
        "integer=True))), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverFockSpace(Idx(Symbol('i', integer=True)), "
        "LocalSpace('tls', basis=('g', 'e'))))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr1) ==
            r'\sum_{i \in \mathcal{H}_{tls}} \alpha_{i} \Ket{i}^{(tls)}')

    assert ascii(expr2) == 'Sum_{i in {0,1}} alpha_i * |i>^(tls)'
    assert unicode(expr2) == '∑_{i ∈ {0,1}} α_i |i⟩⁽ᵗˡˢ⁾'
    assert (
        srepr(expr2) ==
        "KetIndexedSum(ScalarTimesKet(Indexed(IndexedBase(Symbol('alpha')), "
        "Idx(Symbol('i', integer=True))), BasisKet(FockIndex(Idx(Symbol('i', "
        "integer=True))), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverList(Idx(Symbol('i', integer=True)), (0, 1)))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr2) == r'\sum_{i \in \{0,1\}} \alpha_{i} \Ket{i}^{(tls)}')

    assert ascii(expr3) == 'Sum_{i=0}^{1} alpha_i * |i>^(tls)'
    assert unicode(expr3) == '∑_{i=0}^{1} α_i |i⟩⁽ᵗˡˢ⁾'
    assert (
        srepr(expr3) ==
        "KetIndexedSum(ScalarTimesKet(Indexed(IndexedBase(Symbol('alpha')), "
        "Idx(Symbol('i', integer=True))), BasisKet(FockIndex(Idx(Symbol('i', "
        "integer=True))), hs=LocalSpace('tls', basis=('g', 'e')))), "
        "IndexOverRange(Idx(Symbol('i', integer=True)), 0, 1))")
    with configure_printing(tex_use_braket=True):
        assert (
            latex(expr3) == r'\sum_{i=0}^{1} \alpha_{i} \Ket{i}^{(tls)}')

    for expr in (expr1, expr2, expr3):
        syms = list(expr.term.all_symbols())
        assert symbols('alpha') in syms
        assert i in syms
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
    i = Idx('i')
    alpha = IndexedBase('alpha')
    alpha_i = alpha[i]
    hs_tls = LocalSpace('tls', basis=('g', 'e'))

    term = alpha_i * BasisKet(FockIndex(i), hs=hs_tls).dag

    expr = KetIndexedSum.create(
        term, IndexOverFockSpace(i, hs=hs_tls))

    assert IndexOverFockSpace(i, hs=hs_tls) in expr.args

    assert ascii(expr) == "Sum_{i in H_tls} alpha_i * <i|^(tls)"

    syms = list(expr.term.all_symbols())
    assert symbols('alpha') in syms
    assert i in syms
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
