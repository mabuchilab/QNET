r"""
Constant algebraic objects are best implemented as singletons (i.e., they only
exist as a single object). This module provides the means to declare
singletons:

* The :class:`Singleton` metaclass ensures that every class based on it
  produces the same object every time it is instantiated
* The :func:`singleton_object` class decorator converts a singleton class
  definition into the actual singleton object

Singletons in QNET should use both of these.

.. note::

    In order for the Sphinx autodoc extension to correctly recognize
    singletons, a custom documenter will have to be registered. The Sphinx
    ``conf.py`` file must contain the following::

        from sphinx.ext.autodoc import DataDocumenter

        class SingletonDocumenter(DataDocumenter):
            directivetype = 'data'
            objtype = 'singleton'
            priority = 20
            @classmethod
            def can_document_member(cls, member, membername, isattr, parent):
                return isinstance(member, qnet.algebra.singleton.SingletonType)

        def setup(app):
            # ... (other hook settings)
            app.add_autodocumenter(SingletonDocumenter)
"""
from abc import ABCMeta


__all__ = ['singleton_object', 'Singleton', 'SingletonType']

__private__ = []


def singleton_object(cls):
    """Class decorator that transforms (and replaces) a class definition (which
    must have a Singleton metaclass) with the actual singleton object. Ensures
    that the resulting object can still be "instantiated" (i.e., called),
    returning the same object. Also ensures the object can be pickled, is
    hashable, and has the correct string representation (the name of the
    singleton)
    """
    assert isinstance(cls, Singleton), \
        cls.__name__ + " must use Singleton metaclass"

    def self_instantiate(self):
        return self

    cls.__call__ = self_instantiate
    cls.__hash__ = lambda self: hash(cls)
    cls.__repr__ = lambda self: cls.__name__
    cls.__reduce__ = lambda self: cls.__name__
    obj = cls()
    obj.__name__ = cls.__name__
    return obj


class Singleton(ABCMeta):
    """Metaclass for singletons. Any instantiation of a Singleton class yields
    the exact same object, e.g.:

    >>> class MyClass(metaclass=Singleton):
    ...     pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b
    True

    You can check that an object is a singleton using

    >>> isinstance(a, SingletonType)
    True
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]

    @classmethod
    def __instancecheck__(mcs, instance):
        if instance.__class__ is mcs:
            return True
        else:
            return isinstance(instance.__class__, mcs)


class SingletonType(metaclass=Singleton):
    """A dummy type that may be used to check whether an object is a
    Singleton::

        isinstance(obj, SingletonType)
    """
    pass
