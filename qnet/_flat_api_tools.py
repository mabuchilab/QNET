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

import importlib


def _combine_all(*modules):
    """Modify `all_list` in place, extending it with the entries from all
    modules.
    """
    all_list = []
    for module in modules:
        mod = importlib.import_module(module)
        if hasattr(mod, '__all__'):
            all_list.extend(mod.__all__)
    if len(set(all_list)) != len(all_list):
        raise ValueError("modules have overlapping __all__ lists")
    return sorted(all_list)
