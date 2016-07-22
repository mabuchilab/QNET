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
double E_0 = 6.28319;
double T = 10;
double omega = 1;

double tfunc1(double t)
{
  double u1;
  u1 = 0.5*E_0*(-0.5*cos(2*M_PI*t/T) + 0.08*cos(4*M_PI*t/T) + 0.42);
  return u1;
}


RealFunction u1 = tfunc1;

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
  AnnihilationOperator A0(0);
  FieldTransitionOperator S0_1_0(1,0,0);
  FieldTransitionOperator S0_0_1(0,1,0);
  FieldTransitionOperator S0_1_1(1,1,0);
  Operator Id = Id0;
  Operator Ad0 = A0.hc();

  // Hamiltonian
  Operator H = (u1 * ((S0_1_0 + S0_0_1)) + (omega) * ((Ad0 * A0)));

  // Lindblad operators
  const int nL = 0;
  Operator L[nL]={
    
  };

  // Observables
  const int nOfOut = 1;
  Operator outlist[nOfOut] = {
    S0_1_1
  };
  char *flist[nOfOut] = {"P_e.out"};
  int pipe[4] = {1,2,3,4};

  // Initial state
  State phiL0(2,1,FIELD); // HS 0

  State psiIni = phiL0;
  psiIni.normalize();

  // Trajectory
  ACG gen(rndSeed); // random number generator
  ComplexNormal rndm(&gen); // Complex Gaussian random numbers

  double dt = 0.01;
  int dtsperStep = 5;
  int nOfSteps = 200;
  int nTrajSave = 10;
  int nTrajectory = 1;
  int ReadFile = 0;

  AdaptiveStep stepper(psiIni, H, nL, L);
  Trajectory traj(psiIni, dt, stepper, &rndm);

  traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,
              nTrajectory, nTrajSave, ReadFile);
}
