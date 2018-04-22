"""Main QNET package

The :mod:`qnet` package exposes all of QNET's functionality for easy
interactive or programmative use.

For interactive usage, the package should be initialized as follows::

    >>> import qnet
    >>> qnet.init_printing()

"""

import qnet._flat_api_tools
import qnet.algebra._rules

__doc__ += qnet._flat_api_tools.__doc__

__all__ = []  # will be extended by _import_submodules

__imported_data__ = {
    'SCALAR_TYPES': ':data:`~qnet.algebra.scalar_types.SCALAR_TYPES`'}

__version__ = "2.0.0-dev"


# dynamic initialization

qnet._flat_api_tools._import_submodules(
    __all__, __path__, __name__)

qnet.algebra._rules._algebraic_rules()
