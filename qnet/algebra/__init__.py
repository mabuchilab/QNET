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

import qnet.algebra.operator_algebra
import qnet.algebra.circuit_algebra
import qnet.algebra.hilbert_space_algebra
import qnet.algebra.matrix_algebra
import qnet.algebra.operator_algebra
import qnet.algebra.ordering
import qnet.algebra.pattern_matching
import qnet.algebra.permutations
import qnet.algebra.singleton
import qnet.algebra.state_algebra
import qnet.algebra.super_operator_algebra
import qnet.algebra.scalar_types

from .abstract_algebra import *
from .circuit_algebra import *
from .hilbert_space_algebra import *
from .matrix_algebra import *
from .operator_algebra import *
from .state_algebra import *
from .super_operator_algebra import *
from .pattern_matching import *
from .scalar_types import *
from .toolbox import *

from qnet._flat_api_tools import _combine_all

__all__ = _combine_all(
    'qnet.algebra.abstract_algebra',
    'qnet.algebra.circuit_algebra',
    'qnet.algebra.hilbert_space_algebra',
    'qnet.algebra.matrix_algebra',
    'qnet.algebra.operator_algebra',
    'qnet.algebra.state_algebra',
    'qnet.algebra.super_operator_algebra',
    'qnet.algebra.scalar_types',
    'qnet.algebra.pattern_matching',
    'qnet.algebra.toolbox')

__imported_data__ = {
    'SCALAR_TYPES': ':data:`~qnet.algebra.scalar_types.SCALAR_TYPES`'}
