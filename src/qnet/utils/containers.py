"""Tools for working with data structures built from native containers.

"""

from collections import OrderedDict
from collections.abc import Container, Iterable, Sized, Sequence, Mapping

__all__ = []

__private__ = [
    "sorted_if_possible",
    "nested_tuple",
]  # anything not in __all__ must be in __private__


def sorted_if_possible(iterable, **kwargs):
    """Create a sorted list of elements of an iterable if they are orderable.

    See `sorted` for details on optional arguments to customize the sorting.

    Parameters
    ----------
    iterable : Iterable
        Iterable returning a finite number of elements to sort.
    kwargs
        Keyword arguments are passed on to `sorted`.

    Returns
    -------
    list
        List of elements, sorted if orderable, otherwise kept in the order of iteration.

    """
    try:
        return sorted(iterable, **kwargs)
    except TypeError:
        return list(iterable)


def nested_tuple(container):
    """Recursively transform a container structure to a nested tuple.

    The function understands container types inheriting from the selected abstract base
    classes in `collections.abc`, and performs the following replacements:
    `Mapping`
        `tuple` of key-value pair `tuple`s. The order is preserved in the case of an
        `OrderedDict`, otherwise the key-value pairs are sorted if orderable and
        otherwise kept in the order of iteration.
    `Sequence`
        `tuple` containing the same elements in unchanged order.
    `Container and Iterable and Sized` (equivalent to `Collection` in python >= 3.6)
        `tuple` containing the same elements in sorted order if orderable and otherwise
        kept in the order of iteration.
    The function recurses into these container types to perform the same replacement,
    and leaves objects of other types untouched.

    The returned container is hashable if and only if all the values contained in the
    original data structure are hashable.

    Parameters
    ----------
    container
        Data structure to transform into a nested tuple.

    Returns
    -------
    tuple
        Nested tuple containing the same data as `container`.

    """
    if isinstance(container, OrderedDict):
        return tuple(map(nested_tuple, container.items()))
    if isinstance(container, Mapping):
        return tuple(sorted_if_possible(map(nested_tuple, container.items())))
    if not isinstance(container, (str, bytes)):
        if isinstance(container, Sequence):
            return tuple(map(nested_tuple, container))
        if (
            isinstance(container, Container)
            and isinstance(container, Iterable)
            and isinstance(container, Sized)
        ):
            return tuple(sorted_if_possible(map(nested_tuple, container)))
    return container
