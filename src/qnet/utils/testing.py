"""Collection of routines needed for testing. This includes proto-fixtures,
i.e. routines that should be imported and then turned into a fixture with the
pytest.fixture decorator.

See <https://pytest.org/latest/fixture.html>
"""
import os
from distutils import dir_util

from qnet.printing.asciiprinter import QnetAsciiPrinter

__all__ = []
__private__ = ['QnetAsciiTestPrinter', 'datadir', 'check_idempotent_create']


class QnetAsciiTestPrinter(QnetAsciiPrinter):
    """A Printer subclass for testing"""
    _default_settings = {
        'show_hs_label': True,  # alternatively: False, 'subscript'
        'sig_as_ketbra': True,
    }


def datadir(tmpdir, request):
    '''Proto-fixture responsible for searching a folder with the same name of
    test module and, if available, moving all contents to a temporary directory
    so tests can use them freely.

    In any test, import the datadir routine and turn it into a fixture::

        >>> import pytest
        >>> import qnet.utils.testing
        >>> datadir = pytest.fixture(qnet.utils.testing.datadir)
    '''
    # http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return str(tmpdir)


def check_idempotent_create(expr):
    """Check that an expression is 'idempotent'"""
    from qnet.algebra.core.abstract_algebra import Expression
    print("*** CHECKING IDEMPOTENCY of %s" % expr)
    if isinstance(expr, Expression):
        new_expr = expr.create(*expr.args, **expr.kwargs)
        if new_expr != expr:
            # noinspection PyPackageRequirements
            from IPython.core.debugger import Tracer
            Tracer()()
            print(expr)
            print(new_expr)
    print("*** IDEMPOTENCY OK")
