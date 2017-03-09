# This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################

from textwrap import dedent

import pytest

from sympy import (
        symbols, sqrt, exp, I, Pow, Mul, Integer, Symbol, Rational)
from numpy import array, float64, complex128, int64

from qnet.algebra.circuit_algebra import(
        CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
        Concatenation, Feedback, SeriesInverse)
from qnet.algebra.operator_algebra import(
        OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz,
        Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma, tr, Adjoint,
        PseudoInverse, NullSpaceProjector, OperatorPlus, OperatorTimes,
        ScalarTimesOperator, OperatorTrace)
from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, FullSpace, ProductSpace)
from qnet.algebra.matrix_algebra import Matrix
from qnet.algebra.state_algebra import (
        KetSymbol, LocalKet, ZeroKet, TrivialKet, BasisKet, CoherentStateKet,
        UnequalSpaces, OperatorTimesKet, Bra, KetPlus, ScalarTimesKet,
        OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra, TensorKet)
from qnet.algebra.super_operator_algebra import (
        SuperOperatorSymbol, IdentitySuperOperator, ZeroSuperOperator,
        SuperAdjoint, SPre, SPost, SuperOperatorTimesOperator,
        SuperOperatorPlus, SuperOperatorTimes, ScalarTimesSuperOperator)
from qnet.printing import srepr


def test_srepr_circuit_elements():
    """Test the tex representation of "atomic" circuit algebra elements"""
    assert (srepr(CircuitSymbol("C_1", cdim=2)) ==
            "CircuitSymbol('C_1', 2)")
    assert (srepr(CIdentity) ==
            r'CIdentity')
    assert (srepr(CircuitZero) ==
            r'CircuitZero')


def test_foreign_srepr():
    """Test that srepr also works on sympy/numpy components"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    gamma = symbols('gamma')
    phase = exp(-I * gamma / 2)
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    expr = Matrix([[phase*A, B], [C, phase.conjugate()*D]])
    res = srepr(expr)
    expected = "Matrix(array([[ScalarTimesOperator(exp(Mul(Integer(-1), Rational(1, 2), I, Symbol('gamma'))), OperatorSymbol('A', hs=LocalSpace('1'))), OperatorSymbol('B', hs=LocalSpace('1'))], [OperatorSymbol('C', hs=LocalSpace('1')), ScalarTimesOperator(exp(Mul(Rational(1, 2), I, conjugate(Symbol('gamma')))), OperatorSymbol('D', hs=LocalSpace('1')))]], dtype=object))"
    assert res == expected
    expected = "ScalarTimesKet(Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2))), KetPlus(TensorKet(BasisKet('e', hs=LocalSpace('q_1', basis=('g', 'e'))), BasisKet('g', hs=LocalSpace('q_2', basis=('g', 'e')))), ScalarTimesKet(Mul(Integer(-1), I), TensorKet(BasisKet('g', hs=LocalSpace('q_1', basis=('g', 'e'))), BasisKet('e', hs=LocalSpace('q_2', basis=('g', 'e')))))))"
    assert srepr(bell1) == expected
    assert srepr(bell1, indented=True) == dedent(r'''
    ScalarTimesKet(
        Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2))),
        KetPlus(
            TensorKet(
                BasisKet(
                    'e',
                    hs=LocalSpace(
                        'q_1',
                        basis=('g', 'e'),
                    ),
                ),
                BasisKet(
                    'g',
                    hs=LocalSpace(
                        'q_2',
                        basis=('g', 'e'),
                    ),
                ),
            ),
            ScalarTimesKet(
                Mul(Integer(-1), I),
                TensorKet(
                    BasisKet(
                        'g',
                        hs=LocalSpace(
                            'q_1',
                            basis=('g', 'e'),
                        ),
                    ),
                    BasisKet(
                        'e',
                        hs=LocalSpace(
                            'q_2',
                            basis=('g', 'e'),
                        ),
                    ),
                ),
            ),
        ),
    )''').strip()


def circuit_exprs():
    """Prepare a list of circuit algebra expressions"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    return [
        CircuitSymbol("C_1", cdim=2),
        CIdentity,
        CircuitZero,
        A << B << C,
        A + B + C,
        A << (beta + gamma),
        A + (B << C),
        perm,
        SeriesProduct(perm, (A+B)),
        Feedback((A+B), out_port=3, in_port=0),
        SeriesInverse(A+B),
    ]


def hilbert_exprs():
    """Prepare a list of Hilbert space algebra expressions"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    return [
        LocalSpace(1),
        LocalSpace(1, dimension=2),
        LocalSpace(1, basis=(r'g', 'e')),
        LocalSpace('kappa'),
        TrivialSpace,
        FullSpace,
        H1 * H2,
    ]


def matrix_exprs():
    """Prepare a list of Matrix expressions"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    return [
        Matrix([[A, B], [C, D]]),
        Matrix([A, B, C, D]),
        Matrix([[A, B, C, D]]),
        Matrix([[0, 1], [-1, 0]]),
        #Matrix([[], []]),  # see issue #8316 in numpy
        #Matrix([]),
    ]


def operator_exprs():
    """Prepare a list of operator algebra expressions"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma')
    return [
        #OperatorSymbol("A", hs=hs1),
        #OperatorSymbol("A_1", hs=hs1*hs2),
        #OperatorSymbol("Xi_2", hs=(r'q1', 'q2')),
        #OperatorSymbol("Xi_full", hs=1),
        #IdentityOperator,
        #ZeroOperator,
        #Create(hs=1),
        #Create(hs=1, identifier=r'b'),
        #Destroy(hs=1),
        #Destroy(hs=1, identifier=r'b'),
        #Jz(hs=1),
        #Jz(hs=1, identifier='Z'),
        #Jplus(hs=1, identifier='Jp'),
        #Jminus(hs=1, identifier='Jm'),
        #Phase(0.5, hs=1),
        #Phase(0.5, hs=1, identifier=r'Ph'),
        #Displace(0.5, hs=1),
        #Squeeze(0.5, hs=1),
        #LocalSigma('e', 'g', hs=1),
        #LocalSigma('e', 'e', hs=1),
        #A + B,
        #A * B,
        #A * C,
        #2 * A,
        #2j * A,
        #(1+2j) * A,
        gamma**2 * A,
        #-gamma**2/2 * A,
        #tr(A * C, over_space=hs2),
        #Adjoint(A),
        #Adjoint(A + B),
        #PseudoInverse(A),
        #NullSpaceProjector(A),
        #A - B,
        #2 * A - sqrt(gamma) * (B + C),
    ]


def state_exprs():
    """Prepare a list of state algebra expressions"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi1_l = LocalKet("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    phi_l = LocalKet("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma')
    phase = exp(-I * gamma)
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    bra_psi1 = KetSymbol("Psi_1", hs=hs1).dag
    bra_psi1_l = LocalKet("Psi_1", hs=hs1).dag
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag
    bra_phi_l = LocalKet("Phi", hs=hs2).dag
    return [
        KetSymbol('Psi', hs=hs1),
        KetSymbol('Psi', hs=1),
        KetSymbol('Psi', hs=(1, 2)),
        LocalKet('Psi', hs=1),
        ZeroKet,
        TrivialKet,
        BasisKet('e', hs=hs1),
        BasisKet('excited', hs=1),
        BasisKet(1, hs=1),
        CoherentStateKet(2.0, hs=1),
        Bra(KetSymbol('Psi', hs=hs1)),
        Bra(KetSymbol('Psi', hs=1)),
        Bra(KetSymbol('Psi', hs=(1, 2))),
        Bra(KetSymbol('Psi', hs=hs1*hs2)),
        LocalKet('Psi', hs=1).dag,
        Bra(ZeroKet),
        Bra(TrivialKet),
        BasisKet('e', hs=hs1).adjoint(),
        BasisKet(1, hs=1).adjoint(),
        CoherentStateKet(2.0, hs=1).dag,
        psi1 + psi2,
        psi1 - psi2 + psi3,
        psi1 * phi,
        psi1_l * phi_l,
        phase * psi1,
        A * psi1,
        BraKet(psi1, psi2),
        ket_e1.dag * ket_e1,
        ket_g1.dag * ket_e1,
        KetBra(psi1, psi2),
        bell1,
        BraKet.create(bell1, bell2),
        KetBra.create(bell1, bell2),
        (psi1 + psi2).dag,
        bra_psi1 + bra_psi2,
        bra_psi1_l * bra_phi_l,
        Bra(phase * psi1),
        (A * psi1).dag,
    ]


def sop_exprs():
    """Prepare a list of super operator algebra expressions"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    L = SuperOperatorSymbol("L", hs=1)
    M = SuperOperatorSymbol("M", hs=1)
    A_op = OperatorSymbol("A", hs=1)
    gamma = symbols('gamma')
    return [
        SuperOperatorSymbol("A", hs=hs1),
        SuperOperatorSymbol("A_1", hs=hs1*hs2),
        IdentitySuperOperator,
        ZeroSuperOperator,
        A + B,
        A * B,
        A * C,
        2 * A,
        (1+2j) * A,
        -gamma**2/2 * A,
        SuperAdjoint(A + B),
        2 * A - sqrt(gamma) * (B + C),
        SPre(A_op),
        SPost(A_op),
        SuperOperatorTimesOperator(L, sqrt(gamma) * A_op),
        SuperOperatorTimesOperator((L + 2*M), A_op),
    ]


@pytest.mark.parametrize(
    'expr',
    operator_exprs())
    #(circuit_exprs() + hilbert_exprs() + matrix_exprs() + operator_exprs() +
     #state_exprs() + sop_exprs()))
def test_self_eval(expr):
    s = srepr(expr)
    assert eval(s) == expr
    s = srepr(expr, indented=True)
    assert eval(s) == expr
