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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################

import pytest
import sympy

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import (
    OperatorSymbol, DisjunctCommutativeHSOrder, FullCommutativeHSOrder, tr,
    Phase, Displace)
from qnet.algebra.state_algebra import BraKet, KetBra, BasisKet


def disjunct_commutative_test_data():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    C1 = OperatorSymbol("C", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    B2 = OperatorSymbol("B", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    tr_A1 = tr(A1, over_space=1)
    tr_A2 = tr(A2, over_space=2)
    A1_m = OperatorSymbol("A", hs=LocalSpace(1, order_index=2))
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))
    ket_0 = BasisKet(0, hs=1)
    ket_1 = BasisKet(1, hs=1)
    ketbra = KetBra(ket_0, ket_1)
    braket = BraKet(ket_1, ket_1)
    return [
      ([B2, B1, A1],        [B1, A1, B2]),
      ([B2_m, B1_m, A1_m],  [B2_m, B1_m, A1_m]),
      ([B1_m, A1_m, B2_m],  [B2_m, B1_m, A1_m]),
      ([B1, A2, C1, tr_A2], [tr_A2, B1, C1, A2]),
      ([A1, B1+B2],         [A1, B1+B2]),
      ([B1+B2, A1],         [B1+B2, A1]),
      ([A3+B4, A1+A2],      [A1+A2, A3+B4]),
      ([A1+A2, A3+B4],      [A1+A2, A3+B4]),
      ([B4+A3, A2+A1],      [A1+A2, A3+B4]),
      ([tr_A2, tr_A1],      [tr_A1, tr_A2]),
      ([A2, ketbra, A1],    [ketbra, A1, A2]),
      ([A2, braket, A1],    [braket, A1, A2]),
    ]


def full_commutative_test_data():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    C1 = OperatorSymbol("C", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    B2 = OperatorSymbol("B", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    tr_A1 = tr(A1, over_space=1)
    tr_A2 = tr(A2, over_space=2)
    A1_m = OperatorSymbol("A", hs=LocalSpace(1, order_index=2))
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))
    ket_0 = BasisKet(0, hs=1)
    ket_1 = BasisKet(1, hs=1)
    ketbra = KetBra(ket_0, ket_1)
    braket = BraKet(ket_1, ket_1)
    a = sympy.symbols('a')
    Ph = lambda phi: Phase(phi, hs=1)
    Ph2 = lambda phi: Phase(phi, hs=2)
    D = lambda alpha: Displace(alpha, hs=1)
    return [
      ([B2, B1, A1],             [A1, B1, B2]),
      ([B2_m, B1_m, A1_m],       [B2_m, A1_m, B1_m]),
      ([B1_m, A1_m, B2_m],       [B2_m, A1_m, B1_m]),
      ([B1, A2, C1, tr_A2],      [tr_A2, B1, C1, A2]),
      ([A1, B1+B2],              [A1, B1+B2]),
      ([B1+B2, A1],              [A1, B1+B2]),
      ([A3+B4, A1+A2],           [A1+A2, A3+B4]),
      ([A1+A2, A3+B4],           [A1+A2, A3+B4]),
      ([B4+A3, A2+A1],           [A1+A2, A3+B4]),
      ([tr_A2, tr_A1],           [tr_A1, tr_A2]),
      ([A2, ketbra, A1],         [ketbra, A1, A2]),
      ([A2, braket, A1],         [braket, A1, A2]),
      ([A2, 0.5*A1, 2*A1, A1, a*A1, -3*A1],
                                 [0.5*A1, A1, 2*A1, -3*A1, a*A1, A2]),
      ([Ph(1), Ph(0.5), D(2), D(0.1)],
                                 [D(0.1), D(2), Ph(0.5), Ph(1)]),
      ([Ph(1), Ph2(1), Ph(0.5)], [Ph(0.5), Ph(1), Ph2(1)]),
      ([Ph(a), Ph(1)],           [Ph(1), Ph(a)]),
    ]


@pytest.mark.parametrize('unsorted_args, sorted_args',
                         disjunct_commutative_test_data())
def test_disjunct_commutative_hs_order(unsorted_args, sorted_args):
    res = sorted(unsorted_args,  key=DisjunctCommutativeHSOrder)
    assert res == sorted_args


@pytest.mark.parametrize('unsorted_args, sorted_args',
                         full_commutative_test_data())
def test_full_commutative_hs_order(unsorted_args, sorted_args):
    res = sorted(unsorted_args,  key=FullCommutativeHSOrder)
    assert res == sorted_args
