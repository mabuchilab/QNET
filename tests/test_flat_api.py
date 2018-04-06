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

"""Test consistency and completeness of the flat API for QDYN (for interactive
use)"""

import os
import importlib


def get_leaf_modules(package_path):
    """Return all leaf modules in the given package_path"""
    assert os.path.isfile(os.path.join(package_path, '__init__.py'))
    res = []
    root = os.path.join(package_path, '..')
    for path, _, files in os.walk(package_path):
        for f in files:
            if f.endswith(".py") and not f == "__init__.py":
                full_file = os.path.relpath(os.path.join(path, f), start=root)
                module = full_file.replace("/", ".")[:-3]
                res.append(module)
    return res


def test_get_leaf_modules(request):
    """Test that get_leaf_modules produces expected results"""
    filename = request.module.__file__
    qnet_dir = os.path.join(os.path.split(filename)[0], '../qnet')
    modules = get_leaf_modules(qnet_dir)
    assert "qnet.algebra.abstract_algebra" in modules


def test_module_access(request):
    """Test that we can reach all leaf modules by importing just qnet"""
    filename = request.module.__file__
    qnet_dir = os.path.join(os.path.split(filename)[0], '../qnet')
    modules = get_leaf_modules(qnet_dir)

    import qnet

    exclude = ['qnet.misc.circuit_visualization',
               'qnet.misc.parser__CircuitExpressionParser_parsetab',
               'qnet.qhdl']

    def check_excl(module, exclude):
        for excl in exclude:
            if module.startswith(excl):
                return True
        return False

    for module in modules:
        if check_excl(module, exclude):
            continue
        obj = qnet
        for part in module.split(".")[1:]:
            obj = getattr(obj, part)


def test_flat_algebra(request):
    """Check that all items defined in an __all__ list of any of the
    qnet.algebra.* modules are directly accessible through qnet.algebra
    """
    filename = request.module.__file__
    qnet_dir = os.path.join(os.path.split(filename)[0], '../qnet')
    modules = [m for m in get_leaf_modules(qnet_dir)
               if m.startswith('qnet.algebra.')]

    import qnet

    exclude = [
        'Singleton', 'singleton_object', 'SingletonType', 'immutable_attribs']

    for modname in modules:
        mod = importlib.import_module(modname)
        assert hasattr(mod, '__all__'), "All algebra mods must define __all__"
        for name in mod.__all__:
            if name in exclude:
                continue
            assert hasattr(qnet.algebra, name)


def _get_members(obj):
    """Get public members of the module `obj`."""
    public = []
    for name in dir(obj):
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if getattr(value, '__module__', None) == obj.__name__:
            if not name.startswith('_'):
                public.append(name)
    return public


def test_algebra_all(request):
    """Check that all public members of all qnet.algebra.* modules appear
    either in __all__ or in __private__. This ensures that as new members
    are added to these modules, they are not accidentally missing from the flat
    API (i.e., __all__). By requiring that all members not in __all__ must be
    in private, we force the developer to make a conscious choice.
    """
    filename = request.module.__file__
    qnet_dir = os.path.join(os.path.split(filename)[0], '../qnet')
    modules = [m for m in get_leaf_modules(qnet_dir)
               if m.startswith('qnet.algebra.')]
    for modname in modules:
        mod = importlib.import_module(modname)
        public_members = _get_members(mod)
        if len(public_members) > 0:
            assert hasattr(mod, '__private__'), \
                "All algebra mods must define __private__"
            assert set(mod.__all__).isdisjoint(mod.__private__)
        for member in public_members:
            assert member in mod.__all__ or member in mod.__private__
