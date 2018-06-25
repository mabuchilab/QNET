from ..core.circuit_algebra import (
    Concatenation, SLH, _cumsum, map_signals_circuit, )

__all__ = ["connect", ]


def connect(components, connections, force_SLH=False, expand_simplify=True):
    """Connect a list of components according to a list of connections.

    Args:
        components (list): List of Circuit instances
        connections (list): List of pairs ``((c1, port1), (c2, port2))`` where
            ``c1`` and ``c2`` are elements of `components` (or the index of the
            element in `components`), and ``port1`` and ``port2`` are the
            indices (or port names) of the ports of the two components that
            should be connected
        force_SLH (bool): If True, convert the result to an SLH object
        expand_simplify (bool): If the result is an SLH object, expand and
            simplify the circuit after each feedback connection is added

    Example:
        >>> A = CircuitSymbol('A', cdim=2)
        >>> B = CircuitSymbol('B', cdim=2)
        >>> BS = Beamsplitter('BS')
        >>> circuit = connect(
        ...     components=[A, B, BS],
        ...     connections=[
        ...         ((A, 0),  (BS, 'in')),
        ...         ((BS, 'tr'), (B, 0)),
        ...         ((A, 1), (B, 1))])
        >>> print(unicode(circuit).replace('cid(1)', '1'))
        (B ⊞ 1) ◁ Perm(0, 2, 1) ◁ (BS(π/4) ⊞ 1) ◁ Perm(0, 2, 1) ◁ (A ⊞ 1)

        The above example corresponds to the circuit diagram::

             ┌─┐    ┌───────┐    ┌─┐
            >┤ ├───>┤       ├───>┤ ├
             │A│    │BS(π/4)│    │B│
            >┤ ├┐┌─>┤       ├┐┌─>┤ ├
             └─┘└│  └───────┘└│  └─┘
                 │┐           │┐
            ─────┘└───────────┘└────

    Raises:
        ValueError: if `connections` includes any invalid entries
    """
    combined = Concatenation.create(*components)
    cdims = [c.cdim for c in components]
    offsets = _cumsum([0] + cdims[:-1])
    imap = []
    omap = []
    for (ic, ((c1, op), (c2, ip))) in enumerate(connections):
        try:
            if not isinstance(c1, int):
                c1 = components.index(c1)
            else:
                if c1 < 0 or c1 >= len(components):
                    raise ValueError("Invalid index %d" % c1)
        except ValueError:
            raise ValueError(
                "The connection %r refers to the component %s that is not in "
                "the list of components %r"
                % (connections[ic], c1, components))
        try:
            if not isinstance(c2, int):
                c2 = components.index(c2)
            else:
                if c2 < 0 or c2 >= len(components):
                    raise ValueError("Invalid index %d" % c2)
        except ValueError:
            raise ValueError(
                "The connection %r refers to the component %s that is not in "
                "the list of components %r"
                % (connections[ic], c2, components))
        try:
            if not (isinstance(op, int)):
                try:
                    op = components[c1].PORTSOUT.index(op)
                except AttributeError:
                    raise ValueError(
                        "The component %s does not define PORTSOUT labels. "
                        "You cannot use the string %r to refer to a port"
                        % (components[c1], op))
            else:
                if op < 0 or op >= components[c1].cdim:
                    raise ValueError("Invalid index %d" % op)
        except ValueError:
            raise ValueError(
                "The connection %r refers to an invalid output channel %s for "
                "component %r" % (connections[ic], op, components[c1]))
        try:
            if not (isinstance(ip, int)):
                try:
                    ip = components[c2].PORTSIN.index(ip)
                except AttributeError:
                    raise ValueError(
                        "The component %s does not define PORTSIN labels. "
                        "You cannot use the string %r to refer to a port"
                        % (components[c2], ip))
            else:
                if ip < 0 or ip >= components[c2].cdim:
                    raise ValueError("Invalid index %d" % op)
        except ValueError:
            raise ValueError(
                "The connection %r refers to an invalid input channel %s for "
                "component %r" % (connections[ic], ip, components[c2]))
        op_idx = offsets[c1] + op
        ip_idx = offsets[c2] + ip
        imap.append(ip_idx)
        omap.append(op_idx)
    n = combined.cdim
    nfb = len(connections)

    imapping = map_signals_circuit(
        {k: im for (k, im) in zip(range(n-nfb, n), imap)},
        n)

    omapping = map_signals_circuit(
        {om: k for (k, om) in zip(range(n-nfb, n), omap)},
        n)

    combined = omapping << combined << imapping

    if force_SLH:
        combined = combined.toSLH()

    for k in range(nfb):
        combined = combined.feedback()
        if isinstance(combined, SLH) and expand_simplify:
            combined = combined.expand().simplify_scalar()

    return combined
