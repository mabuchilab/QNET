from ..core.circuit_algebra import (
    Concatenation, SLH, _cumsum, map_signals_circuit, )

__all__ = ["connect", ]


def connect(components, connections, force_SLH=False, expand_simplify=True):
    """Connect a list of components according to a list of connections.

    Args:
        components (list): List of Circuit instances
        connections (list): List of pairs ``((c1, port1), (c2, port2))`` where
            ``c1`` and ``c2`` are elements of `components` (or the index of the
            element in `components`, and ``port1`` and ``port2`` are the
            indices of the ports of the two components that should be connected
        force_SLH (bool): If True, convert the result to an SLH object
        expand_simplify (bool): If the result is an SLH object, expand and
            simplify the circuit after each feedback connection is added
    """
    combined = Concatenation.create(*components)
    cdims = [c.cdim for c in components]
    offsets = _cumsum([0] + cdims[:-1])
    imap = []
    omap = []
    for (c1, op), (c2, ip) in connections:
        if not isinstance(c1, int):
            c1 = components.index(c1)
        if not isinstance(c2, int):
            c2 = components.index(c2)
        op_idx = offsets[c1] + op
        ip_idx = offsets[c2] + ip
        imap.append(ip_idx)
        omap.append(op_idx)
    n = combined.cdim
    nfb = len(connections)

    imapping = map_signals_circuit({k: im for k, im
                                    in zip(range(n-nfb, n), imap)}, n)

    omapping = map_signals_circuit({om: k for k, om
                                    in zip(range(n-nfb, n), omap)}, n)

    combined = omapping << combined << imapping

    if force_SLH:
        combined = combined.toSLH()

    for k in range(nfb):
        combined = combined.feedback()
        if isinstance(combined, SLH) and expand_simplify:
            combined = combined.expand().simplify_scalar()

    return combined
