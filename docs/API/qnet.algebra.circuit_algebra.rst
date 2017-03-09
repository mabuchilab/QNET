qnet.algebra.circuit_algebra module
===================================

.. currentmodule:: qnet.algebra.circuit_algebra

.. automodule:: qnet.algebra.circuit_algebra
    :members: ABCD, CIdentity, CPermutation, CannotConvertToABCD, CannotConvertToSLH, CannotEliminateAutomatically, CannotVisualize, Circuit, CircuitSymbol, CircuitZero, Concatenation, FB, Feedback, IncompatibleBlockStructures, P_sigma, SLH, SeriesInverse, SeriesProduct, WrongCDimError, check_cdims, cid, cid_1, circuit_identity, connect, eval_adiabatic_limit, extract_signal, extract_signal_circuit, getABCD, get_common_block_structure, map_signals, map_signals_circuit, move_drive_to_H, pad_with_identity, prepare_adiabatic_limit, try_adiabatic_elimination
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

    Summary
    -------

    Exceptions:

    .. autosummary::
        :nosignatures:

        CannotConvertToABCD
        CannotConvertToSLH
        CannotEliminateAutomatically
        CannotVisualize
        IncompatibleBlockStructures
        WrongCDimError

    Classes:

    .. autosummary::
        :nosignatures:

        ABCD
        CPermutation
        Circuit
        CircuitSymbol
        Concatenation
        Feedback
        SLH
        SeriesInverse
        SeriesProduct

    Functions:

    .. autosummary::
        :nosignatures:

        FB
        P_sigma
        check_cdims
        cid
        circuit_identity
        connect
        eval_adiabatic_limit
        extract_signal
        extract_signal_circuit
        getABCD
        get_common_block_structure
        map_signals
        map_signals_circuit
        move_drive_to_H
        pad_with_identity
        prepare_adiabatic_limit
        try_adiabatic_elimination

    ``__all__``: :class:`ABCD <qnet.algebra.circuit_algebra.ABCD>`, :data:`CIdentity <CIdentity>`, :class:`CPermutation <qnet.algebra.circuit_algebra.CPermutation>`, :exc:`CannotConvertToABCD <qnet.algebra.circuit_algebra.CannotConvertToABCD>`, :exc:`CannotConvertToSLH <qnet.algebra.circuit_algebra.CannotConvertToSLH>`, :exc:`CannotEliminateAutomatically <qnet.algebra.circuit_algebra.CannotEliminateAutomatically>`, :exc:`CannotVisualize <qnet.algebra.circuit_algebra.CannotVisualize>`, :class:`Circuit <qnet.algebra.circuit_algebra.Circuit>`, :class:`CircuitSymbol <qnet.algebra.circuit_algebra.CircuitSymbol>`, :data:`CircuitZero <CircuitZero>`, :class:`Concatenation <qnet.algebra.circuit_algebra.Concatenation>`, :func:`FB <qnet.algebra.circuit_algebra.FB>`, :class:`Feedback <qnet.algebra.circuit_algebra.Feedback>`, :exc:`IncompatibleBlockStructures <qnet.algebra.circuit_algebra.IncompatibleBlockStructures>`, :func:`P_sigma <qnet.algebra.circuit_algebra.P_sigma>`, :class:`SLH <qnet.algebra.circuit_algebra.SLH>`, :class:`SeriesInverse <qnet.algebra.circuit_algebra.SeriesInverse>`, :class:`SeriesProduct <qnet.algebra.circuit_algebra.SeriesProduct>`, :exc:`WrongCDimError <qnet.algebra.circuit_algebra.WrongCDimError>`, :func:`cid <qnet.algebra.circuit_algebra.circuit_identity>`, :data:`cid_1 <cid_1>`, :func:`circuit_identity <qnet.algebra.circuit_algebra.circuit_identity>`, :func:`connect <qnet.algebra.circuit_algebra.connect>`, :func:`eval_adiabatic_limit <qnet.algebra.circuit_algebra.eval_adiabatic_limit>`, :func:`extract_signal <qnet.algebra.circuit_algebra.extract_signal>`, :func:`extract_signal_circuit <qnet.algebra.circuit_algebra.extract_signal_circuit>`, :func:`getABCD <qnet.algebra.circuit_algebra.getABCD>`, :func:`get_common_block_structure <qnet.algebra.circuit_algebra.get_common_block_structure>`, :func:`map_signals <qnet.algebra.circuit_algebra.map_signals>`, :func:`map_signals_circuit <qnet.algebra.circuit_algebra.map_signals_circuit>`, :func:`move_drive_to_H <qnet.algebra.circuit_algebra.move_drive_to_H>`, :func:`pad_with_identity <qnet.algebra.circuit_algebra.pad_with_identity>`, :func:`prepare_adiabatic_limit <qnet.algebra.circuit_algebra.prepare_adiabatic_limit>`, :func:`try_adiabatic_elimination <qnet.algebra.circuit_algebra.try_adiabatic_elimination>`

    Module data:


    .. data:: CIdentity
    .. data:: CircuitZero
    .. data:: cid_1


    Reference
    ---------