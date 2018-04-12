r'''This module defines :data:`SCALAR_TYPES`, the list of types that scalars in
any algebraic operation may have. The lists consist of the following:

* The builtin numerical types :obj:`int`, :obj:`float`, and :obj:`complex`
* `Numpy numerical types`_ :obj:`numpy.int64`, :obj:`numpy.complex128`, and
  :obj:`numpy.float64`
* Sympy_ scalars (:class:`sympy.Basic <sympy.core.basic.Basic>`)

.. _Sympy: http://www.sympy.org
.. _Numpy numerical types: https://docs.scipy.org/doc/numpy/user/basics.types.html
'''
import sympy
from numpy import int64, complex128, float64

__all__ = ['SCALAR_TYPES']


#: list of types that are considered scalars
SCALAR_TYPES = (int, float, complex, sympy.Basic, int64, complex128, float64)
