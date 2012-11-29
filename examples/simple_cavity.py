#!/usr/bin/env python
# encoding=utf-8

from qnet.circuit_components.single_sided_jaynes_cummings_cc import SingleSidedJaynesCummings as CavityModel
from qnet.algebra import circuit_algebra as ca
from sympy import symbols
import qutip
import numpy as np
import matplotlib.pyplot as plt

alpha = symbols('alpha')
print alpha

# Fock space dimension
N = 10

cm = CavityModel('', FOCK_DIM = N)

# circuit triplet with coherent input field with amplitude alpha in first channel
S, L, H = cm.coherent_input(alpha, 0).toSLH()

L1 = L[0,0]
L2 = L[1,0]

# Full Hilbert space is a tensor product space of the two-level system and the cavity mode 
full_space = H.space | L1.space | L2.space
fock_space = L1.space
tls_space = L2.space

print "Product space ordering:", full_space

fock_dimension = fock_space.dimension  # == CavityModel.FOCK_DIM at time of instantiation of cm
tls_dimension = tls_space.dimension    # == 2, obviously

print 'full_space =', full_space, "dimension =", full_space.dimension
print 'Hamiltonian in symbolic form:', H
print 'Lindblad collapse operators:'
print 'L1', L1
print 'L2', L2

n_fock = ca.Create(fock_space) * ca.Destroy(fock_space)
n_e = ca.LocalProjector(tls_space,'h')
observables = [n_fock, n_e]
print "excitation number observables", observables


# to prepare objects for qutip, we need to replace all symbolic parameters by actual numbers
params = {alpha : 1., cm.g_c : 1., cm.kappa : .1, cm.Delta : .5, cm.gamma_0 : .05}

# substitute model parameters and convert to QUTIP numerical objects
Hq = H.substitute(params).to_qutip(full_space)
L1q = L1.substitute(params).to_qutip(full_space)
L2q = L2.substitute(params).to_qutip(full_space)


times = np.linspace(0, 50., 501)

# Obviously the order of the tensor product matters in general! 
# By printing out the full_space object, we find that 
# the first d.o.f is the atom and the second is the fock space
# initial state = |h>|0>
initial_state = qutip.tensor(qutip.basis(2,1), qutip.basis(N, 0))

expectation_vals_mc = qutip.mcsolve(Hq, initial_state, times, 10, [L1q, L2q], [obs.to_qutip(full_space) for obs in observables])
expectation_vals_master = qutip.odesolve(Hq, initial_state, times, [L1q, L2q], [obs.to_qutip(full_space) for obs in observables])
#plot results
fig = plt.figure(figsize = [6,4])
plt.plot(times, expectation_vals_master[0], 'r', times, expectation_vals_master[1], 'b', lw = 1.5)
plt.plot(times, expectation_vals_mc[0], 'r--', times, expectation_vals_mc[1], 'b--', lw = 1.5)
plt.xlabel('Time')
plt.ylabel('Excitations')
plt.legend(('Photons', 'Atom', 'Photons (MC)', 'Atom (MC)') , loc = 'lower right')
plt.savefig('jaynes-cummings.png')
plt.close(fig)

