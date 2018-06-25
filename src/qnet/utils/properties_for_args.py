"""Class decorator for adding properties for arguments"""
from functools import partial

__all__ = []
__private__ = ['properties_for_args']


def properties_for_args(cls):
    """For a class that defines an `_arg_names` property list, add a property
    for every `arg_name` in the list.

    It is assumed that there is an instance attribute  ``self._<arg_name>``,
    which is returned by the `arg_name` property.
    """
    from qnet.algebra.core.scalar_algebra import Scalar
    scalar_args = False
    if hasattr(cls, '_scalar_args'):
        scalar_args = cls._scalar_args
    for arg_name in cls._arg_names:
        def get_arg(self, name):
            val = getattr(self, "_%s" % name)
            if scalar_args:
                assert isinstance(val, Scalar)
            return val
        prop = property(partial(get_arg, name=arg_name))
        doc = "The `%s` argument" % arg_name
        if scalar_args:
            doc += ", as a :class:`.Scalar` instance."
        else:
            doc += "."
        prop.__doc__ = doc
        setattr(cls, arg_name, prop)
    return cls
