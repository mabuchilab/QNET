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
                module = full_file.replace(os.sep, ".")[:-3]
                res.append(module)
    return res


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


def test_get_leaf_modules(request):
    """Test that get_leaf_modules produces expected results"""
    filename = request.module.__file__
    qnet_dir = os.path.join(os.path.split(filename)[0], '..', 'src', 'qnet')
    modules = get_leaf_modules(qnet_dir)
    assert "qnet.algebra.core.abstract_algebra" in modules


def test_flat_api(request):
    """Check the promises made by the "flat" API"""
    filename = request.module.__file__
    pkg_dir = os.path.join(os.path.split(filename)[0], '..', 'src', 'qnet')
    modules = get_leaf_modules(pkg_dir)
    # TODO: what about packages that define members?

    for modname in modules:

        if modname.split(".")[-1].startswith('_'):
            continue

        # check that we can reach all leaf modules by importing just qnet
        obj = importlib.import_module('qnet')
        for part in modname.split(".")[1:]:
            assert hasattr(obj, part)
            obj = getattr(obj, part)

        mod = importlib.import_module(modname)
        public_members = _get_members(mod)
        private_symbols = getattr(mod, '__private__', [])
        assert len(set(private_symbols)) == len(private_symbols)

        # check that every public member is either in __all__ or in __private__
        if len(public_members) > 0:
            assert hasattr(mod, '__all__')
            assert set(mod.__all__).isdisjoint(private_symbols)
            assert len(set(mod.__all__)) == len(mod.__all__)
        for symbol in public_members:
            assert symbol in mod.__all__ or symbol in private_symbols

        # check that every member in __all__ is exported in every super-package
        modname_parts = modname.split('.')
        super_packages = [
            ".".join(l) for l in reversed(
                [modname_parts[:i] for i in range(1, len(modname_parts)+1)])]
        for symbol in mod.__all__:
            assert hasattr(mod, symbol)
            for super_package in super_packages:
                pkg = importlib.import_module(super_package)
                assert hasattr(pkg, symbol)
                assert hasattr(pkg, '__all__')
                assert symbol in pkg.__all__
        for symbol in private_symbols:
            assert hasattr(mod, symbol)
