"""Class decorator for adding properties for arguments"""
from functools import partial

__all__ = []
__private__ = ['properties_for_args']


def properties_for_args(cls, arg_names='_arg_names'):
    """For a class with an attribute `arg_names` containing a list of names,
    add a property for every name in that list.

    It is assumed that there is an instance attribute  ``self._<arg_name>``,
    which is returned by the `arg_name` property. The decorator also adds a
    class attribute :attr:`_has_properties_for_args` that may be used to ensure
    that a class is decorated.
    """
    from qnet.algebra.core.scalar_algebra import Scalar
    scalar_args = False
    if hasattr(cls, '_scalar_args'):
        scalar_args = cls._scalar_args
    for arg_name in getattr(cls, arg_names):
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
    cls._has_properties_for_args = True
    return cls
