from sympy import symbols
import unittest
from collections import OrderedDict

from qnet.algebra.core.abstract_algebra import (
    Operation)
from qnet.algebra.core.abstract_quantum_algebra import ScalarTimesQuantumExpression
from qnet.algebra.core.algebraic_properties import (
    assoc, assoc_indexed, idem,
    orderby, filter_neutral, match_replace_binary, indexed_sum_over_const)
from qnet.algebra.core.indexed_operations import (
    IndexedSum)
from qnet.algebra.core.exceptions import CannotSimplify
from qnet.utils.indices import IndexOverRange, IdxSym
from qnet.utils.ordering import expr_order_key
from qnet.algebra.pattern_matching import pattern_head, wc
from qnet.algebra.core.operator_algebra import (
    LocalProjector, OperatorTimes)
from qnet.algebra.library.fock_operators import Displace
from qnet.algebra.core.hilbert_space_algebra import LocalSpace


def test_match_replace_binary_complete():
    """Test that replace_binary works correctly for a non-trivial case"""
    x, y, z, alpha = symbols('x y z alpha')
    hs = LocalSpace('f')
    ops = [LocalProjector(0, hs=hs),
           Displace(-alpha, hs=hs),
           Displace(alpha, hs=hs),
           LocalProjector(0, hs=hs)]
    res = OperatorTimes.create(*ops)
    assert res == LocalProjector(0, hs=hs)
