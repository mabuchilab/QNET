#include "Complex.h"
#include "ACG.h"
#include "CmplxRan.h"
#include "State.h"
#include "Operator.h"
#include "FieldOp.h"
#include "Traject.h"

int main()
{
// Primary Operators
IdentityOperator Id0(0);
IdentityOperator Id1(1);
IdentityOperator Id2(2);
AnnihilationOperator A0(0);
AnnihilationOperator A1(1);
FieldTransitionOperator S2_1_1(1,1,2);
FieldTransitionOperator S2_0_1(0,1,2);
FieldTransitionOperator S2_1_0(1,0,2);
Operator Id = Id0*Id1*Id2;
Operator Ad0 = A0.hc();
Operator Ad1 = A1.hc();

// Hamiltonian
Complex I(0.0,1.0);
double E = 20;
double chi = 0.4;
double eta = 0.001;
double gamma1 = 1;
double gamma2 = 1;
double kamma = 0.1;
double omega = -0.7;

Operator H = ((omega) * (S2_1_1) + (I*E) * ((Ad0 + (-1) * (A0))) + (0.5*I*chi) * (((Ad0 * Ad0 * A1) + (-1) * ((A0 * A0 * Ad1)))) + (I*eta) * (((-1) * ((Ad1 * S2_0_1)) + (A1 * S2_1_0))));

// Lindblad operators
const int nL = 3;
Operator L[nL]={
  (sqrt(2)*sqrt(gamma1)) * (A0),
  (sqrt(2)*sqrt(gamma2)) * (A1),
  (sqrt(2)*sqrt(kamma)) * (S2_0_1)
};

// Observables
const int nOfOut = 3;
Operator outlist[nOfOut] = {
  (A1 * S2_1_0),
  (A1 * S2_0_1),
  A1
};
char *flist[nOfOut] = {"X1.out", "X2.out", "A2.out"};
int pipe[4] = {1,2,3,4};

// Initial state
State phiL0(50,0,FIELD); // HS 0
State phiL1(50,0,FIELD); // HS 1
State phiL2(2,0,FIELD); // HS 2
State phiT0List[3] = {phiL0, phiL1, phiL2};
State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2

State psiIni = phiT0;
psiIni.normalize();

// Trajectory
int rndSeed = 38388389;
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
}