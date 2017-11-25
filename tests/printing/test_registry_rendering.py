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

import os

import pytest

from qnet.misc.testing_tools import datadir
from qnet.algebra.circuit_algebra import (
    cid, P_sigma, FB, SeriesProduct, Feedback, CPermutation, Concatenation,
    CIdentity)
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.three_port_kerr_cavity_cc import (
    ThreePortKerrCavity)
from qnet.circuit_components.phase_cc import Phase
from qnet.circuit_components.displace_cc import Displace
from qnet.printing import srepr, unicode


datadir = pytest.fixture(datadir)


def test_latch_srepr(datadir):
    """Test rendering of the "Latch" circuit component creduce expression"""
    # There is a problem with Components being cached incorrectly, so we have
    # to clear the instance cache until this is fixed
    B11 = Beamsplitter('Latch.B11')
    B12 = Beamsplitter('Latch.B12')
    B21 = Beamsplitter('Latch.B21')
    B22 = Beamsplitter('Latch.B22')
    B3 = Beamsplitter('Latch.B3')
    C1 = ThreePortKerrCavity('Latch.C1')
    C2 = ThreePortKerrCavity('Latch.C2')
    Phase1 = Phase('Latch.Phase1')
    Phase2 = Phase('Latch.Phase2')
    Phase3 = Phase('Latch.Phase3')
    W1 = Displace('Latch.W1')
    W2 = Displace('Latch.W2')
    cache = {
        B11: 'B11', B12: 'B12', B21: 'B21', B22: 'B22', B3: 'B3', C1: 'C1',
        C2: 'C2', Phase1: 'Phase1', Phase2: 'Phase2', Phase3: 'Phase3',
        W1: 'W1', W2: 'W2'}
    expr = (
        P_sigma(1, 2, 3, 4, 5, 6, 7, 0) <<
        FB(((cid(4) + (P_sigma(0, 4, 1, 2, 3) << (B11 + cid(3)))) <<
            P_sigma(0, 1, 2, 3, 4, 6, 7, 8, 5) <<
            ((P_sigma(0, 1, 2, 3, 4, 7, 5, 6) <<
             ((P_sigma(0, 1, 5, 3, 4, 2) <<
              FB((cid(2) +
                  ((B3 + cid(1)) <<
                      P_sigma(0, 2, 1) <<
                      (B12 + cid(1))) +
                  cid(2)) <<
                  P_sigma(0, 1, 4, 5, 6, 2, 3) <<
                  (((cid(1) +
                      ((cid(1) + ((Phase3 + Phase2) << B22) + cid(1)) <<
                       P_sigma(0, 1, 3, 2) << (C2 + W2))) <<
                      ((B21 << (Phase1 + cid(1))) + cid(3))) +
                   cid(2)),
                  out_port=4, in_port=0)) +
              cid(2)) <<
             (cid(4) +
             (P_sigma(0, 2, 3, 1) << ((P_sigma(1, 0, 2) << C1) + W1)))) +
             cid(1))), out_port=8, in_port=4) <<
        P_sigma(7, 0, 6, 3, 1, 2, 4, 5))
    rendered = srepr(expr, indented=True, cache=cache)
    # the rendered expression is directly the Python code for a more efficient
    # evaluation of the same expression
    with open(os.path.join(datadir, 'latch_srepr.dat')) as in_fh:
        expected = in_fh.read().strip()
    assert rendered == expected
    assert eval(rendered) == expr

    expected = (
        r'Perm(1, 2, 3, 4, 5, 6, 7, 0) ◁ '
        r'[(cid(4) ⊞ (Perm(0, 4, 1, 2, 3) ◁ (Latch.B11(theta=π/4) ⊞ cid(3)))) '
        r'◁ Perm(0, 1, 2, 3, 4, 6, 7, 8, 5) ◁ ((Perm(0, 1, 2, 3, 4, 7, 5, 6) '
        r'◁ ((Perm(0, 1, 5, 3, 4, 2) ◁ [(cid(2) ⊞ '
        r'((Latch.B3(theta=π/4) ⊞ cid(1)) ◁ Perm(0, 2, 1) '
        r'◁ (Latch.B12(theta=π/4) ⊞ cid(1))) ⊞ cid(2)) '
        r'◁ Perm(0, 1, 4, 5, 6, 2, 3) ◁ (((cid(1) ⊞ ((cid(1) '
        r'⊞ ((Latch.Phase3(phi=φ) ⊞ Latch.Phase2(phi=φ)) '
        r'◁ Latch.B22(theta=π/4)) ⊞ cid(1)) ◁ Perm(0, 1, 3, 2) '
        r'◁ (Latch.C2(Delta=Δ, chi=χ, kappa_1=κ₁, kappa_2=κ₂, kappa_3=κ₃, '
        r'FOCK_DIM=75) ⊞ Latch.W2(alpha=α)))) ◁ ((Latch.B21(theta=π/4) '
        r'◁ (Latch.Phase1(phi=φ) ⊞ cid(1))) ⊞ cid(3))) ⊞ cid(2))]₄₋₀) '
        r'⊞ cid(2)) ◁ (cid(4) ⊞ (Perm(0, 2, 3, 1) ◁ ((Perm(1, 0, 2) ◁ '
        r'Latch.C1(Delta=Δ, chi=χ, kappa_1=κ₁, kappa_2=κ₂, kappa_3=κ₃, '
        r'FOCK_DIM=75)) ⊞ Latch.W1(alpha=α))))) ⊞ cid(1))]₈₋₄ '
        r'◁ Perm(7, 0, 6, 3, 1, 2, 4, 5)')
    assert unicode(expr) == expected
    cache = {
        B11: 'B11', B12: 'B12', B21: 'B21', B22: 'B22', B3: 'B3', C1: 'C1',
        C2: 'C2', Phase1: 'Phase1', Phase2: 'Phase2', Phase3: 'Phase3',
        W1: 'W1', W2: 'W2'}
    expected = (
        r'Perm(1, 2, 3, 4, 5, 6, 7, 0) ◁ [(cid(4) ⊞ (Perm(0, 4, 1, 2, 3) '
        r'◁ (B11 ⊞ cid(3)))) ◁ Perm(0, 1, 2, 3, 4, 6, 7, 8, 5) '
        r'◁ ((Perm(0, 1, 2, 3, 4, 7, 5, 6) ◁ ((Perm(0, 1, 5, 3, 4, 2) '
        r'◁ [(cid(2) ⊞ ((B3 ⊞ cid(1)) ◁ Perm(0, 2, 1) ◁ (B12 ⊞ cid(1))) '
        r'⊞ cid(2)) ◁ Perm(0, 1, 4, 5, 6, 2, 3) ◁ (((cid(1) ⊞ ((cid(1) '
        r'⊞ ((Phase3 ⊞ Phase2) ◁ B22) ⊞ cid(1)) ◁ Perm(0, 1, 3, 2) '
        r'◁ (C2 ⊞ W2))) ◁ ((B21 ◁ (Phase1 ⊞ cid(1))) ⊞ cid(3))) '
        r'⊞ cid(2))]₄₋₀) ⊞ cid(2)) ◁ (cid(4) ⊞ (Perm(0, 2, 3, 1) '
        r'◁ ((Perm(1, 0, 2) ◁ C1) ⊞ W1)))) ⊞ cid(1))]₈₋₄ '
        r'◁ Perm(7, 0, 6, 3, 1, 2, 4, 5)')
    assert unicode(expr, cache=cache) == expected
    cache = {expr.operands[1]: 'main_term'}
    expected = (
        'Perm(1, 2, 3, 4, 5, 6, 7, 0) ◁ main_term '
        '◁ Perm(7, 0, 6, 3, 1, 2, 4, 5)')
    assert unicode(expr, cache=cache) == expected
