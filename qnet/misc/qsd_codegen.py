from qnet.algebra.abstract_algebra import prod
from qnet.algebra.hilbert_space_algebra import TrivialSpace
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, set_union, TrivialSpace, LocalSigma,
    ScalarTimesOperator, OperatorPlus, OperatorTimes
)
import sympy
from textwrap import dedent


def local_ops(expr):
    """Given a QNET symbolic expression, return a set of operators
    occuring in that expression (instances of
    qnet.algebra.operator_algebra.Operator)"""
    if isinstance(expr, Operator.scalar_types + (str,)):
        return set()
    elif isinstance(expr, (LocalOperator, IdentityOperator.__class__)):
        return set([expr])
    elif isinstance(expr, Circuit):
        slh = expr.toSLH()
        Ls = slh.L.matrix.flatten().tolist()
        H = slh.H
        return set_union(*tuple(map(local_ops, Ls))) | local_ops(H)
    elif isinstance(expr, Operation):
        return set_union(*tuple(map(local_ops, expr.operands)))
    else:
        raise TypeError(str(expr))


class QSDCodeGen(object):
    """Object that allows to generate QSD programs for QNET expressions"""
    template = dedent(r'''
    #include "Complex.h"
    #include "ACG.h"
    #include "CmplxRan.h"
    #include "State.h"
    #include "Operator.h"
    #include "FieldOp.h"
    #include "SpinOp.h"
    #include "Traject.h"

    int main()
    {{
    // Primary Operators
    {OPERATORBASIS}
      /*AnnihilationOperator A1(0);  // 1st freedom*/
      /*NumberOperator N1(0);*/
      /*IdentityOperator Id1(0);*/
      /*AnnihilationOperator A2(1);  // 2nd freedom*/
      /*NumberOperator N2(1);*/
      /*IdentityOperator Id2(1);*/
      /*SigmaPlus Sp(2);             // 3rd freedom*/
      /*IdentityOperator Id3(2);*/
      /*Operator Sm = Sp.hc();       // Hermitian conjugate*/
      /*Operator Ac1 = A1.hc();*/
      /*Operator Ac2 = A2.hc();*/


    // Hamiltonian
    {PARAMETERS}
      /*double E = 20.0;           */
      /*double chi = 0.4;      */
      /*double omega = -0.7;       */
      /*double eta = 0.001;*/
      /*Complex I(0.0,1.0);*/
      /*double gamma1 = 1.0;       */
      /*double gamma2 = 1.0;       */
      /*double kappa = 0.1;        */



    {HAMILTONIAN}
      /*Operator H = (E*I)*(Ac1-A1)*/
                 /*+ (0.5*chi*I)*(Ac1*Ac1*A2 - A1*A1*Ac2)*/
                 /*+ omega*Sp*Sm + (eta*I)*(A2*Sp-Ac2*Sm);*/
    // Lindblad operators
    {NLINDBLADS}
      /*const int nL = 3;*/
    {LINDBLADS}
      /*Operator L[nL]={{
      sqrt(2*gamma1)*A1,
      sqrt(2*gamma2)*A2,
      sqrt(2*kappa)*Sm
      }};*/

    // Initial state
    {INITIAL_STATE}
      /*State phi1(50,FIELD);       // see paper Section 4.2*/
      /*State phi2(50,FIELD);*/
      /*State phi3(2,SPIN);*/
      /*State stateList[3] = {{phi1,phi2,phi3}};*/
      /*State psiIni(3,stateList);*/

    // Trajectory
    {TRAJECTORYPARAMS}
      /*double dt = 0.01;    // basic time step                            */
      /*int numdts = 100;    // time interval between outputs = numdts*dt  */
      /*int numsteps = 5;    // total integration time = numsteps*numdts*dt*/
      /*int nOfMovingFreedoms = 2;*/
      /*double epsilon = 0.01;     // cutoff probability*/
      /*int nPad = 2;              // pad size*/
      /*ACG gen(38388389);         // random number generator with seed*/
      /*ComplexNormal rndm(&gen);  // Complex Gaussian random numbers*/
      /*AdaptiveStep stepper(psiIni, H, nL, L);       // see paper Section 5*/
    // Output
    {OBSERVABLES}
      /*const int nOfOut = 3;*/
      /*Operator outlist[nOfOut]={{ Sp*A2*Sm*Sp, Sm*Sp*A2*Sm, A2 }};*/
      /*char *flist[nOfOut]={{"X1.out","X2.out","A2.out"}};*/
      /*int pipe[] = {{1,5,9,11}}; // controls standard output (see `onespin.cc')*/
      int pipe[4] = {{1,2,3,4}};

    // Simulate one trajectory (for several trajectories see `onespin.cc')
      Trajectory traj(psiIni, dt, stepper, &rndm);  // see paper Section 5
      traj.plotExp( nOfOut, outlist, flist, pipe, numdts, numsteps,
                    nOfMovingFreedoms, epsilon, nPad );
    }}''').strip()

    def __init__(self, circuit, num_vals=None):
        self.circuit = circuit.toSLH()
        self.num_vals = {}
        self.syms = circuit.all_symbols()
        self._local_ops = local_ops(self.circuit)
        self._full_space = self.circuit.space
        self.lop_str = {}

        self._local_factors = {space: index for (index, space)
                in enumerate(self._full_space.local_factors())}

        if num_vals is not None:
            self.num_vals.update(num_vals)

    @property
    def local_ops(self):
        """Return set of operators occuring in circuit"""
        return self._local_ops


    def generate_code(self):
        """Return C++ program that corresponds to the circuit as a multiline
        string"""
        return self.template.format(
                OPERATORBASIS=self._operator_basis_lines(),
                PARAMETERS=self._parameters_lines(),
                HAMILTONIAN=self._hamiltonian_lines(),
                NLINDBLADS='const int nL = {nL:d};'\
                           .format(nL=self.circuit.cdim),
                LINDBLADS='',
                INITIAL_STATE ='',
                TRAJECTORYPARAMS ='',
                OBSERVABLES ='',
                )

    def write(self, filename):
        """Write C++ program that corresponds to the circuit"""
        with open(filename, 'w') as out_fh:
            out_fh.write(self.generate_code())

    def __str__(self):
        return self.generate_code()


    def _operator_basis_lines(self):
        """Given a set of qnet.algebra.operator_algebra.Operator instances,
        return a multiline string of QSD C++ code that defines an appropriate
        set of QSD operators. These are limited to AnnihilationOperator,
        IdentityOperator, and TansitionOperator, labeled by a unique index
        for each (local) Hilbert space"""
        lines = set()
        visited = set()
        for k, s in enumerate(self._local_factors):
            lines.add("IdentityOperator Id{k}({k});".format(k=k))
        self.lop_str[IdentityOperator] = "({})".format(
                "*".join(["Id{k}".format(k=k)
                    for k, __ in enumerate(self._local_factors)]))
        for op in self.local_ops:
            if isinstance(op, IdentityOperator.__class__):
                continue
            elif isinstance(op, (Create, Destroy)):
                if op.space in visited:
                    continue
                else:
                    visited.add(op.space)
                a = Destroy(op.space)
                ad = a.dag()
                k = self._local_factors[op.space]
                a_str = "A{}".format(k)
                ad_str = "A{}c".format(k)

                self.lop_str[a] = a_str
                self.lop_str[ad] = ad_str

                # QSD only has annihilation operators, so for every creation
                # operator, we must add the corresponding annihilation operator
                lines.add("AnnihilationOperator {a_str}({k});".format(a_str=a_str, k=k))
                lines.add("Operator {ad_str} = {a_str}.hc();".format(ad_str=ad_str, a_str=a_str))
            elif isinstance(op, LocalSigma):
                k = self._local_factors[op.space]
                i,j = op.operands[1:]
                op_str = "S{}_{}_{}".format(k,i,j)
                self.lop_str[op] = op_str
                lines.add("TransitionOperator {op_str}({k},{i},{j});".format(
                        op_str=op_str, k=k, i=i, j=j))
            else:
                raise TypeError(str(op))
        return "\n".join(sorted(lines))

    def _parameters_lines(self):
        lines = set()
        lines.add("Complex I(0.0,1.0);")
        for s in self.syms:
            if s.is_real is True:
                if self.num_vals[s].imag != 0:
                    raise ValueError(self.num_vals[s])
                lines.add("double {!s} = {:g};".format(s,
                          self.num_vals[s].real))
            else:
                lines.add("Complex {!s}({:g},{:g});".format(
                        s, self.num_vals[s].real, self.num_vals[s].imag))
        return "\n".join(sorted(lines))

    def _operator_str(self, op):
        if isinstance(op, LocalOperator):
            return self.lop_str[op]
        elif isinstance(op, ScalarTimesOperator):
            return "({}) * ({})".format(self._scalar_str(op.coeff),
                    self._operator_str(op.term))
        elif isinstance(op, OperatorPlus):
            return "({})".format(" + ".join([self._operator_str(o)
                for o in op.operands]))
        elif isinstance(op, OperatorTimes):
            return "({})".format(" * ".join([self._operator_str(o)
                for o in op.operands]))
        elif op is IdentityOperator:
            return self.lop_str[op]
        else:
            raise TypeError(str(op))

    def _scalar_str(self, sc):
        if isinstance(sc, sympy.Basic):
            return str(sc)
        return "{:g}".format(sc)

    def _hamiltonian_lines(self):
        H = self.circuit.H
        return "Operator H = {};".format(self._operator_str(H))


