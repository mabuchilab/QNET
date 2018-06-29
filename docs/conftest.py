"""Set up the environment for doctests

This file is automatically evaluated by py.test. It ensures that we can write
doctests without importing anything. The entire content for qnet, as well as
numpy and sympy will be available in all doctests.
"""
import numpy
import sympy
import qnet
from collections import OrderedDict

# noinspection PyPackageRequirements
import pytest


qnet.init_printing(repr_format='unicode')


@pytest.fixture(autouse=True)
def set_doctest_env(doctest_namespace):
    doctest_namespace['numpy'] = numpy
    doctest_namespace['sympy'] = sympy
    doctest_namespace['OrderedDict'] = OrderedDict
    for name in qnet.__all__:
        doctest_namespace[name] = getattr(qnet, name)
