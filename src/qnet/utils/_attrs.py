"""Extensions to atts package (http://www.attrs.org)"""

from collections import OrderedDict

import attr

__all__ = ['immutable_attribs']

__private__ = []


def immutable_attribs(cls):
    """Class decorator like ``attr.s(frozen=True)`` with improved __repr__"""
    cls = attr.s(cls, frozen=True)
    defaults = OrderedDict([(a.name, a.default) for a in cls.__attrs_attrs__])

    def repr_(self):
        from qnet.printing import srepr
        real_cls = self.__class__
        class_name = real_cls.__name__
        args = []
        for name in defaults.keys():
            val = getattr(self, name)
            positional = defaults[name] == attr.NOTHING
            if val != defaults[name]:
                args.append(
                    srepr(val) if positional else "%s=%s" % (name, srepr(val)))
        return "{0}({1})".format(class_name, ", ".join(args))

    cls.__repr__ = repr_
    return cls
