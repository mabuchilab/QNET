"""The :mod:`qnet` package exposes all of QNET's functionality for easy
interactive or programmative use.

Specifically, the subpackages for the following parts of QNET are directly
available:

    * Symbolic quantum and circuit algebra as :mod:`qnet.algebra`
    * Printers for symbolic symbolic expressions as :mod:`qnet.printing`
    * Conversion utilities to Sympy and Numpy as :mod:`qnet.convert`

For interactive usage, the package should be initialized as follows::

    >>> import qnet
    >>> qnet.init_printing()

Note that most subpackages in turn expose their functionality through a "flat"
API. That is, instead of

.. code-block:: python

    from qnet.algebra.operator_algebra import LocalOperator
    from qnet.circuit_components.displace_cc import Displace

the two objects may be more succintly imported from a higher level namespace as

.. code-block:: python

    from qnet.algebra import LocalOperator, Displace

In an interactive context (and only there!), a star import such as

.. code-block:: python

    from qnet.algebra import *

may be useful.

The flat API is defined via the `__all__ <https://docs.python.org/3.5/tutorial/modules.html#importing-from-a-package>`_
attribute of each subpackage (see each package's documentation).

Internally, the flat API (or star imports) must never be used.
"""

from qnet._flat_api_tools import _import_submodules

__all__ = []  # will be set by _import_submodules

__imported_data__ = {
    'SCALAR_TYPES': ':data:`~qnet.algebra.scalar_types.SCALAR_TYPES`'}

__version__ = "2.0.0-dev"

_import_submodules(
    __all__, __path__, __name__,
    exclude=['qnet.printing', 'qnet.circuit_components'])
_import_submodules(
    __all__, __path__, __name__,
    include=['qnet.printing'], recursive=False)
