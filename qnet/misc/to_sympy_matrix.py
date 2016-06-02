"""Conversion of QNET expressions to sympy matrices. For small Hilbert spaces,
this facilitates some analytic treatments, such as decomposition into a basis.
"""
import qnet
import sympy
from sympy.physics.quantum import TensorProduct as tensor
from qnet.algebra.operator_algebra import (IdentityOperator, ZeroOperator,
        LocalOperator, Create, Destroy, Jz, Jplus, Jminus, Phase, Displace,
        Squeeze, LocalSigma, OperatorOperation, OperatorPlus, OperatorTimes,
        ScalarTimesOperator, Adjoint, PseudoInverse, OperatorTrace)


def basis_state(i, n):
    """``n x 1`` `sympy.Matrix` representing the `i`'th eigenstate of an
    `n`-dimensional Hilbert space (`i` >= 0)"""
    v = sympy.zeros(n, 1)
    v[i] = 1
    return v

def SympyCreate(n):
    """Creation operator for a Hilbert space of dimension `n`, as an instance
    of `sympy.Matrix`"""
    a = sympy.zeros(n)
    for i in range(1, n):
        a += sympy.sqrt(i) * basis_state(i, n) * basis_state(i-1, n).H
    return a


def convert_to_sympy_matrix(expr, full_space=None):
    """Convert a QNET expression to an explicit ``n x n`` instance of
    `sympy.Matrix`, where ``n`` is the dimension of `full_space`. The entries
    of the matrix may contain symbols.

    Parameters:
        expr: a QNET expression
        full_space (qnet.algebra.hilbert_space_algebra.HilbertSpace): The
            Hilbert space in which `expr` is defined. If not given,
            ``expr.space`` is used. The Hilbert space must have a well-defined
            basis.

    Raises:
        qnet.algebra.hilbert_space_algebra.BasisNotSetError: if `full_space`
            does not have a defined basis
        ValueError: if `expr` is not in `full_space`, or if `expr` cannot be
            converted.
    """
    if full_space is None:
        full_space = expr.space
    if not expr.space.is_tensor_factor_of(full_space):
        raise ValueError("expr must be in full_space")
    if expr is IdentityOperator:
        return sympy.eye(full_space.dimension)
    elif expr is ZeroOperator:
        return 0
    elif isinstance(expr, LocalOperator):
        n = full_space.dimension
        if full_space != expr.space:
            all_spaces = full_space.local_factors()
            own_space_index = all_spaces.index(expr.space)
            return tensor(*( [sympy.eye(s.dimension)
                              for s in all_spaces[:own_space_index]]
                           + convert_to_sympy_matrix(expr, expr.space)
                           + [sympy.eye(s.dimension)
                              for s in all_spaces[own_space_index + 1:]]
                         ))
        if isinstance(expr, (Create, Jz, Jplus)):
            return SympyCreate(n)
        elif isinstance(expr, (Destroy, Jminus)):
            return SympyCreate(n).H
        elif isinstance(expr, Phase):
            phi = expr.operands[1]
            result = sympy.zeros(n)
            for i in range(n):
                result[i,i] = sympy.exp(sympy.I * i * phi)
            return result
        elif isinstance(expr, Displace):
            alpha = expr.operands[1]
            a = SympyCreate(n)
            return  (alpha * a - alpha.conjugate() * a.H).exp()
        elif isinstance(expr, Squeeze):
            eta = expr.operands[1]
            a = SympyCreate(n)
            return ((eta/2) * a**2 - (eta.conjugate()/2) * (a.H)**2).exp()
        elif isinstance(expr, LocalSigma):
            k, j = expr.operands[1:]
            ket = basis_state(k, n)
            bra = basis_state(j, n).H
            return ket * bra
        else:
            raise ValueError("Cannot convert '%s' of type %s"
                             % (str(expr), type(expr)))
    elif isinstance(expr, OperatorOperation):
        if isinstance(expr, OperatorPlus):
            s = convert_to_sympy_matrix(expr.operands[0], full_space)
            for op in expr.operands[1:]:
                s += convert_to_sympy_matrix(op, full_space)
            return s
        elif isinstance(expr, OperatorTimes):
            # if any factor acts non-locally, we need to expand distributively.
            if any(len(op.space) > 1 for op in expr.operands):
                se = expr.expand()
                if se == expr:
                    raise ValueError("Cannot represent as sympy matrix: %s"
                                     % expr)
                return convert_to_sympy_matrix(se, full_space)
            all_spaces = full_space.local_factors()
            by_space = []
            ck = 0
            for ls in all_spaces:
                # group factors by associated local space
                ls_ops = [convert_to_sympy_matrix(o, o.space)
                          for o in expr.operands if o.space == ls]
                if len(ls_ops):
                    # compute factor associated with local space
                    by_space.append(ls_ops[0])
                    for ls_op in ls_ops[1:]:
                        by_space[-1] *= ls_op
                    ck += len(ls_ops)
                else:
                    # if trivial action, take identity matrix
                    by_space.append(sympy.eye(ls.dimension))
            assert ck == len(expr.operands)
            # combine local factors in tensor product
            if len(by_space) == 1:
                return by_space[0]
            else:
                return tensor(*by_space)
        elif isinstance(expr, Adjoint):
            return convert_to_sympy_matrix(expr.operand, full_space).H
        elif isinstance(expr, PseudoInverse):
            raise NotImplementedError('Cannot convert PseudoInverse to sympy '
                    'matrix')
        elif isinstance(expr, NullSpaceProjector):
            raise NotImplementedError('Cannot convert NullSpaceProjector to '
                    'sympy')
        else:
            raise ValueError("Cannot convert '%s' of type %s"
                             % (str(expr), type(expr)))
    elif isinstance(expr, ScalarTimesOperator):
        return expr.coeff * convert_to_sympy_matrix(expr.term, full_space)
    elif isinstance(expr, ScalarTimesOperator):
        raise NotImplementedError('Cannot convert OperatorTrace to '
                'sympy')
        # actually, this is perfectly doable in principle, but requires a bit
        # of work
    else:
        raise ValueError("Cannot convert '%s' of type %s"
                         % (str(expr), type(expr)))

