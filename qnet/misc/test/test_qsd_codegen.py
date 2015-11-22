from qnet.misc.qsd_codegen import (local_ops, find_kets, QSDCodeGen,
    QSDOperator, QSDCodeGenError, UNSIGNED_MAXINT)
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, SLH, set_union, TrivialSpace, symbols, sqrt,
    LocalSigma, identity_matrix, I
)
from qnet.algebra.state_algebra import BasisKet, LocalKet, TensorKet
import re
from textwrap import dedent
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND
import pytest

def test_local_ops():
    psa = PseudoNAND()
    assert isinstance(psa, Circuit)
    l_ops = local_ops(psa)
    a = Destroy(psa.space)
    assert type(local_ops(a)) is set
    assert set([IdentityOperator, a, a.dag()]) == l_ops
    assert local_ops(a) == set([a])
    assert local_ops(a*a) == set([a])
    assert local_ops(a + a.dag()) == set([a,a.dag()])
    assert local_ops(10*a) == set([a])
    with pytest.raises(TypeError):
        local_ops({})


def test_find_kets():
    #                  hs n
    psi_A_0 = BasisKet(0, 0)
    psi_A_1 = BasisKet(0, 1)
    psi_B_0 = BasisKet(1, 0)
    psi_B_1 = BasisKet(1, 1)
    local_psi = [psi_A_0, psi_A_1, psi_B_0, psi_B_1]

    psi_00 = psi_A_0 * psi_B_0
    psi_01 = psi_A_0 * psi_B_1
    psi_10 = psi_A_1 * psi_B_0
    psi_11 = psi_A_1 * psi_B_1
    tensor_psi = [psi_00, psi_01, psi_10, psi_11]
    a1 = Destroy(psi_A_1.space)

    assert set((psi_A_0, )) == find_kets(psi_A_0, LocalKet)

    psi = 0.5*(psi_00 + psi_01 + psi_10 + psi_11)
    assert set(local_psi) == find_kets(psi, LocalKet)
    assert set(tensor_psi) == find_kets(psi, TensorKet)

    psi = 0.5 * a1 * psi_10 # = 0.5 * psi_00
    assert set([psi_A_0, psi_B_0]) == find_kets(psi, LocalKet)
    assert set([psi_00, ]) == find_kets(psi, TensorKet)

    with pytest.raises(TypeError):
        find_kets({}, cls=LocalKet)


def test_qsd_codegen_operator_basis():
    a = Destroy(1)
    ad = a.dag()
    s = LocalSigma(2, 1, 0)
    sd = s.dag()
    circuit = SLH(identity_matrix(0), [], a*ad + s + sd)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    IdentityOperator Id0(0);
    IdentityOperator Id1(1);
    Operator Id = Id0*Id1;
    AnnihilationOperator A0(0);
    Operator Ad0 = A0.hc();
    TransitionOperator S1_0_1(1,0,1);
    TransitionOperator S1_1_0(1,1,0);
    """).strip()
    circuit = SLH(identity_matrix(0), [], ad)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    IdentityOperator Id0(0);
    Operator Id = Id0;
    AnnihilationOperator A0(0);
    Operator Ad0 = A0.hc();
    """).strip()


def test_qsd_codegen_parameters():
    k = symbols("kappa", positive=True)
    x = symbols("chi", real=True)
    c = symbols("c")

    a = Destroy(1)
    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    slh = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit=slh, num_vals={x: 2., c: 1+2j, k: 2})

    scode = codegen._parameters_lines()
    assert dedent(scode).strip() == dedent("""
    Complex I(0.0,1.0);
    Complex c(1,2);
    double chi = 2;
    double kappa = 2;""").strip()

    codegen.num_vals.update({c: 1})
    scode = codegen._parameters_lines()
    assert dedent(scode).strip() == dedent("""
    Complex I(0.0,1.0);
    Complex c(1,0);
    double chi = 2;
    double kappa = 2;""").strip()

    del codegen.num_vals[c]
    with pytest.raises(KeyError) as excinfo:
        scode = codegen._parameters_lines()
    assert "There is no value for symbol c" in str(excinfo.value)


def test_qsd_codegen_hamiltonian():
    k = symbols("kappa", positive=True)
    x = symbols("chi", real=True)
    c = symbols("c",real=True)

    a = Destroy(1)
    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    slh = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit=slh, num_vals={x: 2., c: 1, k: 2})

    codegen._operator_basis_lines()
    scode = codegen._hamiltonian_lines()
    assert scode.strip() == (r'Operator H = ((c) * (Ad0) + (c) * (A0) + (chi) '
                             r'* (((Ad0 * Ad0) + (A0 * A0))));')

def test_QSDOperator():
    A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
    assert A0.__repr__() == "QSDOperator('AnnihilationOperator', 'A0', '(0)')"
    assert A0.qsd_type == 'AnnihilationOperator'
    assert A0.name == 'A0'
    assert A0.instantiator == '(0)'
    assert A0.instantiation == 'AnnihilationOperator A0(0);'
    assert str(A0) == str(A0.name)
    assert len(A0) == 3
    qsd_type, name, instantiator = A0
    assert qsd_type == A0.qsd_type
    assert name == A0.name
    assert instantiator == A0.instantiator

    Id0 = QSDOperator('IdentityOperator', 'Id0', '(0)')
    assert Id0.instantiation == 'IdentityOperator Id0(0);'

    S = QSDOperator('TransitionOperator', 'S1_0_1', '(1,0,1)')
    assert S.instantiation == 'TransitionOperator S1_0_1(1,0,1);'

    Ad0 = QSDOperator('Operator', 'Ad0', '= A0.hc()')
    assert Ad0.instantiator == '= A0.hc()'
    assert Ad0.instantiation == 'Operator Ad0 = A0.hc();'

    # spaces around the '=' should be stripped out
    Ad0_2 = QSDOperator('Operator', 'Ad0', '  =   A0.hc()')
    assert Ad0_2 == Ad0
    Ad0_2 = QSDOperator('Operator', 'Ad0', '=A0.hc()')
    assert Ad0_2 == Ad0

    # having __hash__ implemente allows usage in dicts and sets
    d = {Ad0: 1}
    assert d[Ad0] == 1
    assert d[Ad0_2] == 1
    s = set([Ad0, ])
    assert Ad0_2 in s

    # Changing an attribute is a change in __key()
    Ad0.name = 'Adagger0'
    assert Ad0 != Ad0_2
    with pytest.raises(KeyError):
        v = d[Ad0]
    assert Ad0 not in s
    d[Ad0] = 2
    assert len(d) == 2

    with pytest.raises(ValueError) as excinfo:
        Ad0 = QSDOperator('CreationOperator', 'Ad0', '(0)')
    assert "Type 'CreationOperator' must be one of" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        A0 = QSDOperator('AnnihilationOperator', 'A 0', '(0)')
    assert "Name 'A 0' is not a valid C++ variable name" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Ad0 = QSDOperator('Operator', 'Ad0', 'A0.hc()')
    assert "Instantiator 'A0.hc()' does not start with" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Id0 = QSDOperator('IdentityOperator', 'Id0', '(0')
    assert "Instantiator '(0' does not end with ')'" in str(excinfo.value)


@pytest.fixture
def slh_Sec6():
    """SHL for the model in Section 6 of the QSD paper"""
    E      = symbols("E", positive=True)
    chi    = symbols("chi", real=True)
    omega  = symbols("omega",real=True)
    eta    = symbols("eta",real=True)
    gamma1 = symbols("gamma1", positive=True)
    gamma2 = symbols("gamma2", positive=True)
    kappa  = symbols("kamma", positive=True)
    A1 = Destroy(0)
    Ac1 = A1.dag()
    N1 = Ac1*A1
    Id1 = identity_matrix(0)
    A2 = Destroy(1)
    Ac2 = A2.dag()
    N2 = Ac2*A2
    Id2 = identity_matrix(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    Id3 = identity_matrix(3)

    H  = E*I*(Ac1-A1) + 0.5*chi*I*(Ac1*Ac1*A2 - A1*A1*Ac2) \
         + omega*Sp*Sm + eta*I*(A2*Sp-Ac2*Sm)
    Lindblads = [sqrt(2*gamma1)*A1, sqrt(2*gamma2)*A2, sqrt(2*kappa)*Sm]

    return SLH(identity_matrix(3), Lindblads, H)


@pytest.fixture
def slh_Sec6_vals():
    return {
        symbols("E", positive=True):     20.0,
        symbols("chi", real=True):        0.4,
        symbols("omega",real=True):      -0.7,
        symbols("eta",real=True):         0.001,
        symbols("gamma1", positive=True): 1.0,
        symbols("gamma2", positive=True): 1.0,
        symbols("kamma", positive=True):  0.1
    }


def test_qsd_codegen_lindblads(slh_Sec6):
    codegen = QSDCodeGen(circuit=slh_Sec6)
    scode = codegen._lindblads_lines()
    assert dedent(scode).strip() == dedent(r'''
    const int nL = 3;
    Operator L[nL]={
      (sqrt(2)*sqrt(gamma1)) * (A0),
      (sqrt(2)*sqrt(gamma2)) * (A1),
      (sqrt(2)*sqrt(kamma)) * (S2_0_1)
    };
    ''').strip()


def test_qsd_codegen_observables(slh_Sec6, slh_Sec6_vals):
    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    codegen = QSDCodeGen(circuit=slh_Sec6, num_vals=slh_Sec6_vals)
    with pytest.raises(QSDCodeGenError) as excinfo:
        scode = codegen._observables_lines()
    assert "Must register at least one observable" in str(excinfo.value)
    codegen.add_observable(Sp*A2*Sm*Sp, "X1.out")
    codegen.add_observable(Sm*Sp*A2*Sm, "X2.out")
    codegen.add_observable(A2, "A2.out")
    scode = codegen._observables_lines()
    assert dedent(scode).strip() == dedent(r'''
    const int nOfOut = 3
    Operator outlist[nOfOut] = {
      (A1 * S2_1_0),
      (A1 * S2_0_1),
      A1
    };
    char *flist[nOfOut] = {"X1.out", "X2.out", "A2.out"};
    int pipe[4] = {1,2,3,4};
    ''').strip()
    # Note how the observables have been simplified
    assert Sp*A2*Sm*Sp == Sp*A2
    assert codegen._operator_str(Sp*A2) == '(A1 * S2_1_0)'
    assert Sm*Sp*A2*Sm == Sm*A2
    assert codegen._operator_str(Sm*A2) == '(A1 * S2_0_1)'
    # If the oberservables introduce new operators or symbols, these should
    # extend the existing ones
    P1 = LocalSigma(2, 1, 1)
    zeta = symbols("zeta", real=True)
    codegen.add_observable(zeta*P1, "P1.out")
    assert P1 in codegen._local_ops
    assert str(codegen._qsd_ops[P1]) == 'S2_1_1'
    assert zeta in codegen.syms
    codegen.num_vals.update({zeta: 1.0})
    assert 'zeta' in codegen._parameters_lines()
    assert str(codegen._qsd_ops[P1]) in codegen._operator_basis_lines()


def test_qsd_codegen_traj(slh_Sec6):
    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    codegen = QSDCodeGen(circuit=slh_Sec6)
    codegen.add_observable(Sp*A2*Sm*Sp, "X1.out")
    codegen.add_observable(Sm*Sp*A2*Sm, "X2.out")
    codegen.add_observable(A2, "A2.out")

    with pytest.raises(QSDCodeGenError) as excinfo:
        scode = codegen._trajectory_lines()
    assert "No trajectories set up"  in str(excinfo.value)

    codegen.set_trajectories(psi_initial=None, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            add_to_existing_traj=True, traj_save=10, rnd_seed=None)
    scode = codegen._trajectory_lines()
    m = re.search(r'int rndSeed = (\d+);', scode)
    assert int(m.group(1)) <= UNSIGNED_MAXINT

    codegen.set_trajectories(psi_initial=None, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            add_to_existing_traj=True, traj_save=10, rnd_seed=0)
    scode = codegen._trajectory_lines()
    assert dedent(scode).strip() == dedent(r'''
    int rndSeed = 0;
    ACG gen(rndSeed); // random number generator
    ComplexNormal rndm(&gen); // Complex Gaussian random numbers

    double dt = 0.01;
    int dtsperStep = 100;
    int nOfSteps = 5;
    int nTrajSave = 10;
    int nTrajectory = 1;
    int ReadFile = 0;

    AdaptiveStep stepper(psiIni, H, nL, L);
    Trajectory traj(psiIni, dt, stepper, &rndm);

    traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,
                nTrajectory, nTrajSave, ReadFile);
    ''').strip()

    with pytest.raises(ValueError) as excinfo:
        codegen.set_moving_basis(move_dofs=0, delta=0.01, width=2,
                                 move_eps=0.01)
    assert "move_dofs must be an integer >0" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        codegen.set_moving_basis(move_dofs=4, delta=0.01, width=2,
                                 move_eps=0.01)
    assert "move_dofs must not be larger" in str(excinfo.value)
    codegen.set_moving_basis(move_dofs=2, delta=0.01, width=2, move_eps=0.01)
    scode = codegen._trajectory_lines()
    assert dedent(scode).strip() == dedent(r'''
    int rndSeed = 0;
    ACG gen(rndSeed); // random number generator
    ComplexNormal rndm(&gen); // Complex Gaussian random numbers

    double dt = 0.01;
    int dtsperStep = 100;
    int nOfSteps = 5;
    int nTrajSave = 10;
    int nTrajectory = 1;
    int ReadFile = 0;

    AdaptiveStep stepper(psiIni, H, nL, L);
    Trajectory traj(psiIni, dt, stepper, &rndm);

    int move = 2;
    double delta = 0.01;
    int width = 2;
    int moveEps = 0.01

    traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,
                nTrajectory, nTrajSave, ReadFile, move,
                delta, width, moveEps);
    ''').strip()
