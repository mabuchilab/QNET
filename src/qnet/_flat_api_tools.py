import importlib
import pkgutil
import sys


def _import_submodules(
        __all__, __path__, __name__, include=None, exclude=None,
        include_private_modules=False, require__all__=True, recursive=True):
    """
    Import all available submodules, all objects defined in the `__all__` lists
    of those submodules, and extend `__all__` with the imported objects.

    Args:
        __all__ (list): The list of public objects in the "root" module
        __path__ (str): The path where the ``__init__.py`` file for the "root"
            module is located in the file system (every module has a global
            `__path__` variable which should be passed here)
        __name__ (str): The full name of the "root" module. Again, every module
            has a global `__name__` variable.
        include (list or None): If not None, list of full module names to be
            included. That is, every module not in the `include` list is
            ignored
        exclude (list or None): List of full module names to be
            excluded from the (recursive) input
        include_private_modules (bool): Whether to include modules whose name
            starts with an underscore
        recursive (bool): Whether to recursively act on submodules of the
            "root" module. This will make sub-submodules available both in the
            submodule, and in the "root" module
    """
    mod = sys.modules[__name__]
    if exclude is None:
        exclude = []
    for (_, submodname, ispkg) in pkgutil.iter_modules(path=__path__):
        if submodname.startswith('_') and not include_private_modules:
            continue
        submod = importlib.import_module('.' + submodname, __name__)
        if submod.__name__ in exclude:
            continue
        if include is not None:
            if submod.__name__ not in include:
                continue
        if not hasattr(submod, '__all__'):
            setattr(submod, '__all__', [])
        if recursive and ispkg:
            _import_submodules(
                submod.__all__, submod.__path__, submod.__name__)
        setattr(mod, submodname, submod)
        for obj_name in submod.__all__:
            obj = getattr(submod, obj_name)
            if hasattr(mod, obj_name):
                existing_obj = getattr(mod, obj_name)
                if existing_obj is obj:
                    continue
                else:
                    raise ImportError(
                        "{mod}.{attr} points to {submod1}.{attr}. "
                        "Cannot set to {submod2}.{attr}".format(
                            mod=mod.__name__, attr=obj_name,
                            submod1=existing_obj.__module__,
                            submod2=obj.__module__))
            setattr(mod, obj_name, obj)
            __all__.append(obj_name)
    __all__.sort()
