import sympy
from sympy import I
from numpy import array as np_array
import pytest

from qnet.algebra.core.circuit_algebra import (
    SLH, CircuitSymbol, CPermutation, circuit_identity, map_signals,
    SeriesProduct, invert_permutation, Concatenation, P_sigma, cid,
    map_signals_circuit, FB, getABCD, CIdentity,
    pad_with_identity, move_drive_to_H, try_adiabatic_elimination)
from qnet.algebra.toolbox.circuit_manipulation import connect
from qnet.utils.permutations import (
    permute, full_block_perm, block_perm_and_perms_within_blocks)
from qnet.algebra.core.operator_algebra import (
    Operator, OperatorSymbol, sympyOne, Destroy, ZeroOperator, LocalSigma,
    LocalProjector, IdentityOperator)
from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.matrix_algebra import Matrix, identity_matrix
from qnet.algebra.library.circuit_components import (
    CoherentDriveCC, PhaseCC, Beamsplitter)


symbol_counter = 0


def get_symbol(cdim):
    global symbol_counter
    sym =  CircuitSymbol('test_%d' % symbol_counter, cdim)
    symbol_counter +=1
    return sym


def get_symbols(*cdim):
    return [get_symbol(n) for n in cdim]


#qnet.algebra.abstract_algebra.CHECK_OPERANDS = True
#qnet.algebra.abstract_algebra.PRINT_PRETTY = True


def test_circuit_symbol_hashing():
    """Check that CircuitSymbol have the same hash value if they are the
    same"""
    A1 = CircuitSymbol('A', 1)
    A2 = CircuitSymbol('A', 2)
    B1 = CircuitSymbol('B', 1)
    a1 = CircuitSymbol('A', '1')
    assert A1 == a1
    assert hash(A1) == hash(a1)
    assert A1 is not B1
    assert A1 != B1
    assert A1 is not A2
    assert A1 != A2


def test_permutation():
    n = 5
    assert CPermutation.create(()) == circuit_identity(0)
    invalid_permutation = (1,1)
    with pytest.raises(Exception):
        CPermutation.create((invalid_permutation,))
    p_id = tuple(range(n))
    assert CPermutation.create(p_id) == circuit_identity(n)
    assert map_signals({0:1,1:0}, 2) == (1,0)
    assert map_signals({0:5,1:0}, 6) == (5,0,1,2,3,4)
    assert map_signals({0:5,1:0, 3:2}, 6) == invert_permutation(map_signals({5:0,0:1, 2:3}, 6))


def test_series():
    A, B = get_symbol(1), get_symbol(1)
    assert A << B == SeriesProduct(A,B)
    assert A << B == SeriesProduct.create(A,B)
    assert SeriesProduct.create(CIdentity, CIdentity) == CIdentity
    # need at least two operands
    # self.assertRaises(Exception, SeriesProduct, ())
    # self.assertRaises(Exception, SeriesProduct.create, ())
    # self.assertRaises(Exception, SeriesProduct, (A,))
    assert SeriesProduct.create(A) == A


def test_series_filter_identities():
    for n in (1, 2, 3, 10):
        A, B = get_symbol(n), get_symbol(n)
        idn = circuit_identity(n)
        assert A << idn == A
        assert idn << A == A
        assert SeriesProduct.create(idn, idn, A, idn, idn, B, idn, idn) == A << B


def test_concatenation():
    n = 4
    A, B = get_symbol(n), get_symbol(n)
    id0 = circuit_identity(0)
    assert A+B == Concatenation(A,B)
    assert A+B == Concatenation.create(A,B)
    assert id0 + id0 + A + id0 + id0 + B + id0 + id0 == A + B
    #self.assertRaises(Exception, Concatenation, ())
    #self.assertRaises(Exception, Concatenation, (A,))

    assert (A+B).block_structure == (n,n)
    assert (A+B).get_blocks((n,n)) == (A,B)
    #test index_in_block()
    assert (A+B).index_in_block(0) == (0,0)
    assert (A+B).index_in_block(1) == (1,0)
    assert (A+B).index_in_block(2) == (2,0)
    assert (A+B).index_in_block(3) == (3,0)
    assert (A+B).index_in_block(4) == (0,1)
    assert (A+B).index_in_block(5) == (1,1)
    assert (A+B).index_in_block(7) == (3,1)

    res = Concatenation.create(CIdentity, CIdentity, CPermutation((1,0)))
    assert res == CPermutation((0, 1, 3, 2))


def test_distributive_law():
    A = CircuitSymbol('A', 2)
    B = CircuitSymbol('B', 1)
    C = CircuitSymbol('C', 1)
    D = CircuitSymbol('D', 1)
    E = CircuitSymbol('E', 1)
    assert (A+B) << (C+D+E) == Concatenation(A<<(C+D), B << E)
    assert (C+D+E) << (A+B) == Concatenation((C+D)<< A,  E<< B)
    assert (A+B) << (C+D+E) << (A+B) == Concatenation(A << (C+D)<< A,  B << E<< B)
    assert SeriesProduct.create((A+B), (C+D+E), (A+B)) == Concatenation(A << (C+D)<< A,  B << E<< B)
    test_perm = (0,1,3,2)
    qtp = CPermutation(test_perm)
    assert CPermutation((1,0)) << ( B + C) == SeriesProduct(Concatenation(C, B), CPermutation((1,0)))
    assert qtp << (A + B + C) == (A + C+ B) <<  qtp
    assert qtp << ( B + C + A) == B + C + (CPermutation((1,0)) << A)
    test_perm2 = (1,0,3,2)
    qtp2 = CPermutation(test_perm2)
    assert qtp2 << (A + B + C) == (CPermutation((1,0)) << A) + ((C+B) << CPermutation((1,0)))
    assert qtp << qtp2 == CPermutation(permute(test_perm, test_perm2))


def test_permutation():
    test_perm = (0,1,2,5,6,3,4)
    qtp = CPermutation.create(test_perm)
    assert qtp.series_inverse() == CPermutation.create(invert_permutation(test_perm))
    assert qtp.block_structure == (1,1,1,4)
    id1 = circuit_identity(1)
    assert qtp.get_blocks() == (id1, id1, id1, CPermutation.create((2,3,0,1)))

    assert CPermutation((1,0,3,2)).get_blocks() == (CPermutation((1,0)), CPermutation((1,0)))
    nt = len(test_perm)
    assert qtp << qtp.series_inverse() == circuit_identity(nt)
    assert permute(list(invert_permutation(test_perm)), test_perm) == list(range(nt))


def test_factorize_permutation():
    assert full_block_perm((0,1,2), (1,1,1)) == (0,1,2)
    assert full_block_perm((0,2,1), (1,1,1)) == (0,2,1)
    assert full_block_perm((0,2,1), (1,1,2)) == (0,3,1,2)
    assert full_block_perm((0,2,1), (1,2,3)) == (0,4,5,1,2,3)
    assert full_block_perm((1,2,0), (1,2,3)) == (3,4,5,0,1,2)
    assert full_block_perm((3,1,2,0), (1,2,3,4)) == (9, 4, 5, 6, 7, 8, 0, 1, 2, 3 )
    assert block_perm_and_perms_within_blocks((9, 4, 5, 6, 7, 8, 0, 1, 2, 3 ), (1,2,3,4)) == \
                                                                    ((3,1,2,0), [(0,),(0,1),(0,1,2),(0,1,2,3)])

    A1,A2,A3,A4 = get_symbols(1,2,3,4)

    new_lhs, permuted_rhs, new_rhs = P_sigma(9, 4, 5, 6, 7, 8, 0, 1, 2, 3 )._factorize_for_rhs(A1+A2+A3+A4)
    assert new_lhs == cid(10)
    assert permuted_rhs == (A4+A2+A3+A1)
    assert new_rhs == P_sigma(9, 4, 5, 6, 7, 8, 0, 1, 2, 3 )

    p = P_sigma(0,1,4,2,3,5)
    expr = A2 + A3 + A1
    new_lhs, permuted_rhs, new_rhs = p._factorize_for_rhs(expr)
    assert new_lhs == cid(6)
    assert permuted_rhs == A2 + (P_sigma(2,0,1) << A3) + A1
    assert new_rhs == cid(6)

    p = P_sigma(0, 3, 1, 2)

    p_r = P_sigma(2, 0, 1)
    assert p == cid(1) + p_r
    A = get_symbol(2)

    new_lhs, permuted_rhs, new_rhs = p._factorize_for_rhs(cid(1) + A+ cid(1))

    assert new_lhs == P_sigma(0,1,3,2)
    assert permuted_rhs == (cid(1) + (P_sigma(1,0) << A)  + cid(1))
    assert new_rhs == cid(4)

    new_lhs, permuted_rhs, new_rhs = p._factorize_for_rhs(cid(2) + A)

    assert new_lhs == cid(4)
    assert permuted_rhs == (cid(1) + A  + cid(1))
    assert new_rhs == p

    assert p.series_inverse() << (cid(2) + A) == cid(1) + SeriesProduct(P_sigma(0,2,1), Concatenation(SeriesProduct(P_sigma(1,0), A), cid(1)),P_sigma(2,0,1))

    assert p.series_inverse() << (cid(2) + A) << p == cid(1) + (p_r.series_inverse() << (cid(1) + A) << p_r)

    new_lhs, permuted_rhs, new_rhs = P_sigma(4,2,1,3,0)._factorize_for_rhs((A4 + cid(1)))
    assert new_lhs == cid(5)
    assert permuted_rhs == (cid(1) + (P_sigma(3,1,0,2) << A4))
    assert new_rhs == map_signals_circuit({4:0}, 5)

    ## special test case that helped find the major permutation block structure factorization bug
    p = P_sigma(3, 4, 5, 0, 1, 6, 2)
    q = cid(3) + CircuitSymbol('NAND1', 4)

    new_lhs, permuted_rhs, new_rhs = p._factorize_for_rhs(q)
    assert new_lhs == P_sigma(0,1,2,6,3,4,5)
    assert permuted_rhs == (P_sigma(0,1,3,2) << CircuitSymbol('NAND1', 4)) + cid(3)
    assert new_rhs == P_sigma(4,5,6, 0,1,2,3)


def _symbmatrix(a):
    try:
        return sympy.Matrix(a)
    except (TypeError, ValueError):
        return np_array(a)

typelist = [lambda x: x, np_array, _symbmatrix, Matrix]


def test_SLH_equality():
    # Verify that equality between SLH objects is symmetric (ref. #55)
    a = Destroy(hs=1)
    b, c, d = sympy.symbols('b c d')

    S1 = [[b, c], [2.0, 4]]
    L1 = [d, a]
    H1 = a.dag() * a

    S2 = [[c, d], [-I, 2]]
    L2 = [a, b]
    H2 = 0

    slh1 = [SLH(typ(S1), typ(L1), H1) for typ in typelist]
    slh2 = [SLH(typ(S2), typ(L2), H2) for typ in typelist]

    m = len(typelist)
    for i in range(m):
        for j in range(m):
            assert slh1[i] == slh1[j]
            assert slh1[j] == slh1[i]

            assert slh2[i] == slh2[j]
            assert slh2[j] == slh2[i]

            assert slh1[i] != slh2[j]
            assert slh2[j] != slh1[i]


def test_SLH_elements():
    # Check that all elements in the SLH triple are operator valued, even if
    # only scalars are provided (ref #55). Consider removing this test if
    # a different way to satisfy test_SLH_equality is found.

    def check(S, L, H):
        for typ in typelist:
            slh = SLH(typ(S), typ(L), H)
            assert all(isinstance(s, Operator) for s in slh.S.matrix.ravel())
            assert all(isinstance(l, Operator) for l in slh.L.matrix.ravel())
            assert isinstance(slh.H, Operator)

    # Numbers
    S = [[1.0, 2.0], [3.0, 4.0]]
    L = [5.0, 6.0]
    H = 7.0

    check(S, L, H)

    # Symbols
    a, b, c, d, e, f, g = sympy.symbols('a b c d e f g')
    S = [[a, b], [c, d]]
    L = [e, f]
    H = g

    check(S, L, H)


def test_feedback():
    A, B, C, D, A1, A2 = get_symbols(3, 2, 1, 1, 1, 1)
    circuit_identity(1)

    assert FB(A+B) == A + FB(B)
    smq = map_signals_circuit({2:1}, 3) # == 'cid(1) + X'
    assert smq == smq.series_inverse()

    assert ( smq << (B + C)).feedback(out_port = 2, in_port = 1) == B.feedback() + C

    assert ( smq << (B + C) << smq).feedback() == B.feedback() + C

    assert (B + C).feedback(out_port=1, in_port=1) == B.feedback() + C

    #check that feedback is resolved into series when possible
    b_feedback = B.feedback(out_port=1, in_port=0)
    series_D_C = b_feedback.substitute({B:(C+D)})
    assert series_D_C == C << D
    assert (A << (B + cid(1))).feedback() == A.feedback() << B
    assert (A << (B + cid(1)) << (cid(1) + P_sigma(1,0))).feedback(out_port=2, in_port=1) == A.feedback() << B
    assert (A << (cid(1) + P_sigma(1,0)) << (B + cid(1)) << (cid(1) + P_sigma(1,0))).feedback(out_port=1, in_port=1) == A.feedback(out_port=1, in_port=1) << B
    assert (B << (cid(1)  + C)).feedback(out_port=0, in_port=1).substitute({B: (A1 + A2)}) == A2 << C << A1
    assert ((cid(1)  + C)<< P_sigma(1,0) << B).feedback(out_port=1, in_port=1).substitute({B: (A1 + A2)}) == A2 << C << A1
    assert ((cid(1)  + C)<< P_sigma(1,0) << B << (cid(1) + D)).feedback(out_port=1, in_port=1).substitute({B: (A1 + A2)}) == A2 << D<< C << A1

    #check for correctness of the SLH triple in non-trivial cases, i.e.,
    #S[k, l] != 0 _and_ L[k] != 0 when feeding back from port k to port l.
    a = Destroy(hs=1)
    theta, phi = sympy.symbols('theta phi', real=True)
    gamma, epsilon = sympy.symbols('gamma epsilon', positive=True)

    eip = sympy.exp(I * phi)
    eimp = sympy.conjugate(eip)
    ct, st = sympy.cos(theta), sympy.sin(theta)
    sqe, sqg, sq2 = sympy.sqrt(epsilon), sympy.sqrt(gamma), sympy.sqrt(2)
    div = 1 + eip * st ** 2

    cav = SLH([[1]], [sq2 * sqg * a], 0)
    bs = Beamsplitter('theta', theta=theta)
    ph = PhaseCC('phi', phi=phi)
    flip = map_signals_circuit({1: 0}, 2)

    sys = (
        (cid(1) + ph + cid(1)) <<
        (cid(1) + bs) <<
        (flip + cav) <<
        (cid(1) + bs)
    ).coherent_input(0, 0, sqe)
    sys_fb = sys.feedback(out_port=1, in_port=1).toSLH()

    fb = SLH(
        Matrix([[ eip * ct ** 2, -(1 + eip) * st],
                [(1 + eip) * st,         ct ** 2]]) / div,
        Matrix([-(1 + eip) * sqe * st - sq2 * sqg * eip * st * ct * a,
                                   sqe * ct ** 2 + sq2 * sqg * ct * a]) / div,
        (I * (sqe * sqg / sq2) * ct * (
            (1 +  eip * st ** 2) * a -
            (1 + eimp * st ** 2) * a.dag()
        ) + I * gamma * (eip - eimp) * st ** 2 * a.dag() * a) /
        (div * sympy.conjugate(div))
    )

    assert sys_fb.expand().simplify_scalar() == fb.expand().simplify_scalar()


def test_ABCD():
    a = Destroy(hs=1)
    H = 2 * a.dag() * a
    slh1 = SLH(identity_matrix(1), [a], H)
    slh = slh1.coherent_input(3)
    A, B, C, D, a, c = getABCD(slh, doubled_up=True)
    assert A[0, 0] == -sympyOne / 2 - 2 * I
    assert A[1, 1] == -sympyOne / 2 + 2 * I
    assert B[0, 0] == -1
    assert C[0, 0] == 1
    assert D[0, 0] == 1


def test_inverse():
    """Test that the series product of a circuit and its inverse gives the
    identity"""
    X = CircuitSymbol('X', cdim=3)
    right = X << X.series_inverse()
    left = X.series_inverse() << X
    expected = cid(X.cdim)
    assert left == right == expected


def test_pad_with_identity():
    """Test that pad_with_identity inserts identity channels correctly"""
    A = CircuitSymbol('A', 1)
    B = CircuitSymbol('B', 1)
    res = pad_with_identity(A+B, 1, 2)
    expected = A + cid(2) + B
    assert res == expected


def connect_data():
    A = CircuitSymbol('A', 1)
    B = CircuitSymbol('B', 1)
    C = CircuitSymbol('C', 2)
    return [
        ([A, B],                 # components
         [((0, 0), (1, 0))],     # connection
         B << A                  # expected
        ),
        ([A, B, C],              # components
         [((0, 0), (2, 0)),      # connection 1
          ((1, 0), (2, 1))],     # connection 2
         (C << A + B)            # expected
        ),
        ([A, B, C],              # components
         [((A, 0), (C, 0)),      # connection 1
          ((B, 0), (C, 1))],     # connection 2
         (C << A + B)            # expected
        ),
        ([A, C],                 # components
         [((A, 0), (C, 0))],     # connections
         C << (A + cid(1))       # expected
        ),
    ]


@pytest.mark.parametrize('components, connections, expected', connect_data())
def test_connect(components, connections, expected):
    res = connect(components, connections, force_SLH=False)
    assert res == expected


def test_adiabatic_elimination():
    fock = LocalSpace('fock')
    tls = LocalSpace('tls', basis=('e', 'g'))

    Delta, Theta, g = sympy.symbols('Delta, Theta, g', real=True)
    kappa, gamma = sympy.symbols('kappa, gamma', positive=True)

    a = Destroy(hs=fock)
    sigma = LocalSigma('g', 'e', hs=tls)

    S = identity_matrix(2)

    L1 = sympy.sqrt(kappa) * a
    L2 = sympy.sqrt(gamma) * sigma
    L = Matrix([[L1],
                [L2]])

    H = (Delta * sigma.dag() * sigma +
         Theta * a.dag() * a +
         I * g * (sigma * a.dag() - sigma.dag() * a))

    slh = SLH(S, L, H)

    k = sympy.symbols("k", positive=True)
    subst = {
        kappa: k**2 * kappa,
        g: g * k,
    }

    slh_limit = try_adiabatic_elimination(
        slh.substitute(subst), k)

    expected = SLH(
        Matrix([[-1 * IdentityOperator, ZeroOperator],
                [ZeroOperator, IdentityOperator]]),
        Matrix([[(2*g/sympy.sqrt(kappa)) * LocalSigma('g', 'e', hs=tls)],
                [sympy.sqrt(gamma) * LocalSigma('g', 'e', hs=tls)]]),
        Delta * LocalProjector('e', hs=tls))

    assert slh_limit == expected



def test_move_drive_to_H():
    """Test moving inhomogeneities in the Lindblad operators to the
    Hamiltonian. This occurs when adding a coherent drive input to a circuit
    """

    # Single channel
    S = identity_matrix(1)
    L = Matrix([[OperatorSymbol('L', hs=1)], ])
    H = OperatorSymbol('H', hs='1')
    SLH1 = SLH(S, L, H)
    assert move_drive_to_H(SLH1) == SLH1

    # Two Channels
    S2 = identity_matrix(2)
    L2 = Matrix([[OperatorSymbol('L_1', hs=2)], [OperatorSymbol('L_2', hs=2)]])
    H2 = OperatorSymbol('H', hs='2')
    SLH2 = SLH(S2, L2, H2)
    assert move_drive_to_H(SLH2) == SLH2

    # Single Drive
    α = sympy.symbols('alpha')
    W = CoherentDriveCC('W', alpha=α)
    SLH_driven = (SLH1 << W).toSLH()
    SLH_driven_out = move_drive_to_H(SLH_driven)
    assert SLH_driven_out.S == SLH1.S
    assert SLH_driven_out.L == SLH1.L
    assert SLH_driven_out.H == (SLH1.H - I * α * L[0,0].dag() +
                                I * α.conjugate() * L[0,0])

    # Concatenated drives (single channel)
    β = sympy.symbols('beta')
    Wb = CoherentDriveCC('W', alpha=β)
    SLH_concat_driven = (SLH1 << Wb << W).toSLH()
    SLH_concat_driven_out = move_drive_to_H(SLH_concat_driven)
    assert SLH_concat_driven_out.S == SLH1.S
    assert SLH_concat_driven_out.L == SLH1.L
    term = SLH_concat_driven.H.expand().operands
    assert (
        SLH_concat_driven_out.H - (term[0] + term[1] + 2*term[2] + 2*term[3])
    ).expand().simplify_scalar() == 0

    # Two Drives (two channels)
    α1 = sympy.symbols('alpha_1')
    α2 = sympy.symbols('alpha_2')
    W1 = CoherentDriveCC('W', alpha=α1)
    W2 = CoherentDriveCC('W', alpha=α2)
    SLH2_driven = (SLH2 << (W1 + W2)).toSLH()
    term2 = SLH2_driven.H.expand().operands
    # ###  remove both inhomogeneities (implicitly)
    SLH2_driven_out = move_drive_to_H(SLH2_driven)
    assert SLH2_driven_out.S == SLH2.S
    assert SLH2_driven_out.L == SLH2.L
    assert SLH2_driven_out.H == (
        SLH2.H + (2 * (SLH2_driven.H.expand() - SLH2.H)).expand())
    # ###  remove first inhomogeneity only
    SLH2_driven_out1 = move_drive_to_H(SLH2_driven, [0, ])
    assert SLH2_driven_out1.S == SLH2.S
    assert SLH2_driven_out1.L[0, 0] == SLH2.L[0,0]
    assert SLH2_driven_out1.L[1, 0] == SLH2_driven.L[1,0]
    assert (
        SLH2_driven_out1.H - term2[0] - 2*term2[1] - 2*term2[2] -
        term2[3] - term2[4] == 0)
    # ###  remove second inhomogeneity only
    SLH2_driven_out2 = move_drive_to_H(SLH2_driven, [1, ])
    assert SLH2_driven_out2.S == SLH2.S
    assert SLH2_driven_out2.L[0,0] == SLH2_driven.L[0,0]
    assert SLH2_driven_out2.L[1,0] == SLH2.L[1,0]
    assert (
        SLH2_driven_out2.H - term2[0] - term2[1] - term2[2] -
        2*term2[3] - 2*term2[4] == 0)
    # ###  remove both inhomogeneities (explicitly)
    SLH2_driven_out12 = move_drive_to_H(SLH2_driven, [0, 1])
    assert SLH2_driven_out12 == SLH2_driven_out
