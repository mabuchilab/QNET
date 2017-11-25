#define _USE_MATH_DEFINES
#include <cmath>
#include "Complex.h"
#include "ACG.h"
#include "CmplxRan.h"
#include "State.h"
#include "Operator.h"
#include "FieldOp.h"
#include "Traject.h"

Complex I(0.0,1.0);
double E = 20;
double chi = 0.4;
double eta = 0.001;
double gamma_1 = 1;
double gamma_2 = 1;
double kappa = 0.1;
double omega = -0.7;


int main(int argc, char* argv[])
{

  unsigned int rndSeed;
  if (argc != 2){
    std::cout << "Usage: " << argv[0] << " SEED" << std::endl;
    std::exit(1);
  } else {
    if(sscanf(argv[1], "%u", &rndSeed) != 1){
      std::cout << "ERROR: Could not read SEED" << std::endl;
      std::exit(1);
    } else {
      std::cout << "Using rnd seed: " << rndSeed << std::endl;
    }
  }

  // Primary Operators
  IdentityOperator Id0(0);
  IdentityOperator Id1(1);
  IdentityOperator Id2(2);
  AnnihilationOperator A0(0);
  AnnihilationOperator A1(1);
  FieldTransitionOperator S2_0_1(0,1,2);
  FieldTransitionOperator S2_1_0(1,0,2);
  FieldTransitionOperator S2_1_1(1,1,2);
  Operator Id = Id0*Id1*Id2;
  Operator Ad0 = A0.hc();
  Operator Ad1 = A1.hc();

  // Hamiltonian
  Operator H = ((I*E) * ((Ad0 + (-1) * (A0))) + (0.5*I*chi) * (((Ad0 * Ad0 * A1) + (-1) * ((A0 * A0 * Ad1)))) + (I*eta) * (((-1) * ((Ad1 * S2_0_1)) + (A1 * S2_1_0))) + (omega) * (S2_1_1));

  // Lindblad operators
  const int nL = 3;
  Operator L[nL]={
    (sqrt(2)*sqrt(gamma_1)) * (A0),
    (sqrt(2)*sqrt(gamma_2)) * (A1),
    (sqrt(2)*sqrt(kappa)) * (S2_0_1)
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
