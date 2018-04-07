# coding=utf-8
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
