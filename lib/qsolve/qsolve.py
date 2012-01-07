#!/usr/bin/env python
# encoding: utf-8
"""
qsolve.py

Created by Nikolas Tezak on 2011-05-23.
Copyright (c) 2011 . All rights reserved.
"""
import algebra.circuit_algebra as ca
from scipy.sparse import kron as sparse_kron_scipy, linalg as sparse_la, identity as sparse_id
from math import factorial, exp, pi, sqrt
from itertools import izip

from numpy import kron, zeros, zeros_like, \
                    dot, ones, array, inf, \
                    vdot, linalg, identity, \
                    sqrt as np_sqrt, abs as np_abs, \
                    ndarray, shape, meshgrid, arange, arctan2, \
                    trace, float64, complex128, \
                    loadtxt, fromfile, int64, shape

import numpy.random as random

from scipy.linalg import norm


class MyCOO(object):
    def __init__(self, shape, row, col, data):
        self.shape = shape
        self.row = row
        self.col = col
        self.data = data
        assert data.shape == row.shape == col.shape
    
    @property
    def nnz(self):
        return self.getnnz()
    
    def getnnz(self):
        return len(self.data)
        
    def __iter__(self):
        return izip(self.row, self.col, self.data)

    def tocoo(self):
        return self
    
    def __add__(self, other):
        if self.shape != other.shape:
            raise Exception((self.shape, other.shape))
        
        
        if other == 0:
            return self
        
        elif not isinstance(other, MyCOO):
            raise Exception(other)
        
        max_nnz = self.getnnz() + other.getnnz()
        res_row = zeros(max_nnz, dtype = int64)
        res_col = zeros(max_nnz, dtype = int64)
        res_data = zeros(max_nnz, dtype = complex)
        ii = jj = -1
        ll = mm = -2 
        val1 = val2 = 0
        kk = 0
        self_iter = iter(self)
        other_iter = iter(other)
        self_finished = other_finished = False
        while(True):
            if ii == ll and jj == mm:
                val = val1 + val2
                if val:
                    res_row[kk] = ii
                    res_col[kk] = jj
                    res_data[kk] = val
                    kk += 1
                try:
                    ii, jj, val1 = self_iter.next()
                except StopIteration:
                    self_finished = True
                    
                try:
                    ll, mm, val2 = other_iter.next()
                except StopIteration:
                    other_finished = True
            elif ii < ll or (ii == ll and  jj < mm):
                if val1:
                    res_row[kk] = ii
                    res_col[kk] = jj
                    res_data[kk] = val1
                    kk += 1
                try:
                    ii, jj, val1 = self_iter.next()
                except StopIteration:
                    self_finished = True
            else:
                if val2:
                    res_row[kk] = ll
                    res_col[kk] = mm
                    res_data[kk] = val2
                    kk += 1
                try:
                    ll, mm, val2 = other_iter.next()
                except StopIteration:
                    other_finished = True
            
            if self_finished and other_finished:
                break
        return MyCOO(self.shape, res_row[:kk], res_col[:kk], res_data[:kk])
    
    class RowEmpty(Exception):
        def __init__(self, offset):
            self.offset = offset
    
    
    def row_offset(self, row_index, start = 0, end = -1):
        nnz = self.getnnz()
        end = end % nnz
        while(True):
            center = int((start + end)/2)
            
            rc = self.row[center]
            
            if rc >= row_index:
                end = center
            if rc < row_index:
                start = center
            elif start == center:
                if rc == row_index:
                    return start
                raise MyCOO.RowEmpty(start)
            
    
    def iterate_over_row(self, row_index, start = 0, end = -1):
        try:
            row_offset = self.row_offset(row_index, start, end)
        except MyCOO.RowEmpty:
            return 
        for r, c, val in izip(self.row[row_offset:], self.col[row_offset:], self.data[row_offset:]):
            if r > row_index:
                break
            yield c, val
            
    def __mul__(self, other):
        try:
            self.data *= other
        except:
            return NotImplemented

    def __rmul__(self, other):
        try:
            self.data.__rmul__(other)
        except:
            return NotImplemented

# def sparse_kron(A, B, *args, **kwargs):
#     #Assume sorted inputs
#     n,m = A.shape
#     q,p = B.shape
#     if n*q < 2**32 and m*p < 2**32:
#         return sparse_kron_scipy(A, B, *args, **kwargs)
#         
#     if not isinstance(A, MyCOO):
#         A = A.tocoo()
#         A = MyCOO(A.shape, A.row, A.col, A.data)
#         
#     if not isinstance(B, MyCOO):
#         B = B.tocoo()
#         B = MyCOO(B.shape, B.row, B.col, B.data)    
#     
#     max_nnz = A.getnnz() * B.getnnz()
#     res_row = zeros(max_nnz, dtype = int64)
#     res_col = zeros(max_nnz, dtype = int64)
#     res_data = zeros(max_nnz, dtype = complex)
#         
#     A_cnt = 0
#     B_cnt = 0
#     
#     kk = 0
#     for ii in xrange(n):
#         A_row = list(A.iterate_over_row(ii, start = A_cnt, end = A_cnt + 1))
#         A_cnt += len(A_row)
#         
#         for jj in xrange(q):
#             B_row = list(B.iterate_over_row(jj, start = B_cnt, end = B_cnt + 1))
#             B_cnt += len(B_row)
#             for ll, A_il in A_row:
#                 for mm, B_jm in B_row:
#                     res_row[kk] = ii * n + jj
#                     res_col[kk] = ll * m + mm
#                     res_data[kk] = A_il * B_jm
#                     kk += 1
#     return MyCOO((n*q, m*p), res_row, res_col, res_data)
    
    
def sparse_kron(A,B, *args, **kwargs):
    
    #Assume sorted inputs
    (n,m) = shape(A)
    
    q,p = shape(B)
    if n*q < 2**32 and m*p < 2**32:
        return sparse_kron_scipy(A, B, *args, **kwargs)
        
    if not isinstance(A, MyCOO):
        A = A.tocoo()
        A = MyCOO(A.shape, A.row, A.col, A.data)
        
    if not isinstance(B, MyCOO):
        B = B.tocoo()
        B = MyCOO(B.shape, B.row, B.col, B.data)    

    output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # kronecker product is the zero matrix
        return coo_matrix( output_shape )

    # expand entries of a into blocks
    row  = A.row.repeat(B.nnz)
    col  = A.col.repeat(B.nnz)
    data = A.data.repeat(B.nnz)

    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row,col = row.reshape(-1,B.nnz), col.reshape(-1, B.nnz)
    row += B.row
    col += B.col
    row,col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape(-1,B.nnz) * B.data
    data = data.reshape(-1)

    return MyCOO(output_shape, row, col, data)
                




def prod(sequence):
    return reduce(lambda a, b: a*b, sequence, 1)

def factorial_sequence(N):
    """
    Generator that returns the factorial sequence 0! = 1, 1! = 1, 2! = 2*1 = 2, 3!, ... (N-1)!
    """
    yield 1
    f = 1
    for k in xrange(1, N):
        f *= k
        yield f


def factorial_vector(N):
    """
    Return an array with N elements and values [0!, 1!, 2!, ... , (N-1)!].
    """
    return array([f_n for f_n in factorial_sequence(N)])
    

def my_vector_sqrt(vector):
    return array(map(sqrt, vector))

my_cache = {}
def one_over_sqrt_factorial(N):
    if N in my_cache:
        return my_cache[N]
    result = 1. / my_vector_sqrt(factorial_vector(N))
    my_cache[N] = result
    return result

def coherent_state(alpha, N):
    amp = np_abs(alpha)
    result = zeros_like(alpha)
    state_vector = (((alpha * ones((N,)))**arange(N)) * one_over_sqrt_factorial(N)) * exp(-amp**2 / 2.)
    return state_vector


# def make_array_func(func):
#     
#     def array_func(array_arg, *args, **kwargs):
#         if isinstance(array_arg, (int, long, float, complex)):
#             return func(array_arg, *args, **kwargs)
#         s = shape(array_arg)
#         
#         ret = 
        

def symbolic_lindbladian_schroedinger(circuit, rho = None):
    if isinstance(circuit, ca.SLH):
        L, H = circuit.L, circuit.H
        
    elif isinstance(circuit, tuple):
        if len(circuit) == 2:
            L, H = circuit
        elif len(circuit) == 3:
            S, L, H = circuit
        else:
            raise ValueError(str(circuit))
    if rho is None:
        rho = ca.OperatorSymbol('rho', L.space | H.space)
    return -1j*(H*rho - rho*H) + sum( Lk * rho * ca.adjoint(Lk)
                         -  (ca.adjoint(Lk)*Lk * rho + rho * ca.adjoint(Lk)*Lk) / 2
                                            for Lk in L.array.flatten())


def symbolic_lindbladian_heisenberg(circuit, M = None):
    if isinstance(circuit, ca.SLH):
        L, H = circuit.L, circuit.H
    elif isinstance(circuit, tuple):
        if len(circuit) == 2:
            L, H = circuit
        elif len(circuit) == 3:
            S, L, H = circuit
        else:
            raise ValueError(str(circuit))
    if M is None:
        M = ca.OperatorSymbol('M', L.space | H.space)            
    return 1j*(H*M - M*H) + sum(ca.adjoint(Lk)* M * Lk \
                -  (ca.adjoint(Lk)*Lk * M + M * ca.adjoint(Lk)*Lk) / 2 \
                                                        for Lk in L.array.flatten())


def get_lindbladian(circuit_triplet, full_space = None, order = 'F'):
    """
    Return the lindblad superoperator for the circuit represented in the space described by the HilbertSpace object full_space.
    
    The object circuit_triplet may either be a SLH instance or a tuple(S, L, H) or just (L,H) containing appropriate operator (matrix) objects.
    If full_space is None, the representation space is assumed to be the full product space of the associated operators contained in the triplet.
    """
    
    if isinstance(circuit_triplet, tuple):
        if len(circuit_triplet) == 3:
            S, L, H = circuit_triplet
        elif len(circuit_triplet) == 2:
            L, H = circuit_triplet
        else:
            raise ValueError('The argument circuit_triplet may either be a SLH instance or a tuple(S, L, H) or just (L,H): %r' % (circuit_triplet,))
    elif isinstance(circuit_triplet, ca.SLH):
        S, L, H = circuit_triplet.S, circuit_triplet.L, circuit_triplet.H
    else:
        raise ValueError('The argument circuit_triplet may either be a SLH instance or a tuple(S, L, H) or just (L,H): %r' % (circuit_triplet,))
    
    # determine full space
    full_space = full_space or (S.space | L.space | H.space)
    dimension = full_space.dimension
    id_dim = sparse_id(dimension)
    
    # print "dimension:", full_space.dimension
    
    assert isinstance(full_space, ca.HilbertSpace)
    
    print 'calculating the L_matrices'
    # calculate individual representation matrices for the L operators
    L_matrices = [ca.representation_matrix(ca.n(ca.expand(Lk)), full_space) for Lk in L.array[:,0]]
    
    
    # calculate the L contribution to the K operator
    L_dagger_L = sum((ca.adjoint(Lk)*Lk for Lk in L.array[:,0]))
    
    L_dagger_L_matrix  = ca.representation_matrix(ca.n(ca.expand(L_dagger_L)), full_space)
    
    # calculate the Hamilton and K-operator matrices
    print "calculating the Hamiltonian matrix"
    H_matrix = ca.representation_matrix(ca.n(ca.expand(H)), full_space)
    
    print "calculating the effective Hamiltonian (K) matrix"
    K_matrix = -(1j * H_matrix + 0.5 * L_dagger_L_matrix)
    
    # empty the matrix cache so reclaim memory
    ca.matrix_cache.clear()
    
    """
    if order = 'C'
    we map rho to a single column S[rho] = rho_v = (rho_11, rho_12, ... rho_1n, rho_21,    ...    rho_nn)^T,
    
    if order = 'F' (default)
    we map rho to a single column S[rho] = rho_v = (rho_11, rho_21, ... rho_n1, rho_12,    ...    rho_nn)^T,
    
    we can for all operators A,B introduce the super operators A_L = S_L[A], B_R = S_R[B], such that
        S[A * rho] == S_L[A] * rho_v
                and
        S[rho * B] == S_R[B] * rho_v
        
    A calculation reveals that for order = 'C':
        S_L[A] == kron(A, id_n)
                and 
        S_R[B] == kron(id_n, B^T)
    and for order = 'F'    
    
        S_L[A] == kron(id_n, A)
                and 
        S_R[B] == kron(B^T, id_n)
        
    where kron denotes the tensor-product:                                       
    kron(A,B) =    [[A_11 B_11, A_11 B_12, .. A_11 B_1n, A_12 B_11, ...             A_1n B_1n],
                    [A_11 B_21,         ...                                         A_1n B_2n],
                        ...                                                             ...
                    [A_11 B_n1,         ...                                         A_1n B_nn],
                    [A_21 B_11,         ...                                         A_2n B_1n],
                        .                                                               .
                        .                                                               .
                        .                                                               .
                    [A_n1 B_n1,         ...                                         A_nn B_nn]]
                    
    (cf http://en.wikipedia.org/wiki/Kronecker_product)
    
    It also follows that:
        S[A * rho * B] == S_L[A] * S_R[B] * rho_v == S_R[B] * S_L[A]  * rho_v 
            = kron(A, B^T) * rho_v # for order = 'C'
            = kron(B^T, A) * rho_v # for order = 'F'
    """
    
    #Lindblad[rho] = K \rho - \rho K^\dagger + \sum_j L_j rho L_j^\dagger
    
    if order == 'C':
        print "calculating the K*rho contribution to L_sup"
        # Now calculate the actual super operator contributions to the lindbladian!
        # for the term: K * rho
        Lindblad_super = sparse_kron(K_matrix, id_dim, format = 'lil')
        
        print "calculating the rho * K^dagger contribution"
        # for the term: rho * K^dagger
        # (K^dagger)^T == K^conjugate
        Lindblad_super += sparse_kron(id_dim, K_matrix.conjugate(), format = 'lil')
        
        lll = len(L_matrices)
        # for the terms: L_k * rho * L_k^dagger
        for k, Lk in enumerate(L_matrices):
            print "calculating the %d-th (out of %d) L_k rho L_k^\dagger contribution" % (k, lll)
            Lindblad_super += sparse_kron(Lk, Lk.conjugate(), format = 'lil')
        
    elif order == 'F':
        print "calculating the K*rho contribution to L_sup"
        # Now calculate the actual super operator contributions to the lindbladian!
        # for the term: K * rho
        Lindblad_super = sparse_kron(id_dim, K_matrix, format = 'lil')

        print "calculating the rho * K^dagger contribution"        
        # for the term: rho * K^dagger
        # (K^dagger)^T == K^conjugate
        Lindblad_super += sparse_kron(K_matrix.conjugate(), id_dim, format = 'lil')

        lll = len(L_matrices)
        # for the terms: L_k * rho * L_k^dagger
        for k, Lk in enumerate(L_matrices):        
            print "calculating the %d-th (out of %d) L_k rho L_k^\dagger contribution" % (k, lll)
            Lindblad_super += sparse_kron(Lk.conjugate(), Lk, format = 'lil')
        
    if isinstance(Lindblad_super, MyCOO):
        return Lindblad_super
    else:
        return Lindblad_super.tocsr()


def state_vector_representation(full_space, *state_rep):
    """
    Return a numerical vector representation in the HilbertSpace full_space for a product state.
    
    If full_space is None, it is automatically assumed the full product space of all registered local spaces.
    The state information is passed in the order of the HilbertSpace space ids 
            (state information for ALL "factor"-HilbertSpaces of full_space must be provided)
    Example: 
    >>> state_vector_representation(HilbertSpace(('Q1__atom', 'Q1__fock',)), 'e', 1)
    
    For Fock-Spaces, a coherent state can be initialized by passing a complex number as its representation.
    So be careful!
        rep = 1 
            -> fock-state |1>
    while 
        rep = 1+0j 
            -> coherent state |alpha = 1+0j>
    """
    if not len(state_rep) == len(full_space):
        raise ValueError('You need to specify state information for every local space')
    
    full_vector = array([1+0j])
    
    for space, rep in zip(full_space, state_rep):
        basis_states = ca.HilbertSpace.retrieve_by_descriptor(space)[1]
        local_dim = len(basis_states)
        if list(basis_states) == range(local_dim) and isinstance(rep, complex):
            
            # coherent state!
            amplitude = abs(rep)
            # |alpha> = exp(-|alpha|**2 / 2) \sum_j (alpha^j/sqrt(j!)) |j>
            amps = [rep**j / sqrt(factorial(j)) for j in range(local_dim)]
            local_vector = array(amps) * exp(-0.5 * amplitude**2)
            if amplitude*(1 + amplitude) >= local_dim:
                print "WARNING: your state space truncation will probably affect this coherent state", local_dim, rep
        else:            
            rep_index = basis_states.index(rep)
            local_vector = zeros((local_dim,), dtype = complex)
            local_vector[rep_index] = 1+0j
        
        # kron product is associative, so this iterative order is fine
        full_vector = kron(full_vector, local_vector)
    
    return full_vector

def _is_vector(vector):
    return len(vector.shape) == 1 or (len(vector.shape) > 1 and all(s == 1 for s in vector.shape[1:]))


def vector_to_density_matrix(vector):
    N = vector.shape[0]
    if not _is_vector(vector):
        raise ValueError(repr(vector))
    # \rho = v \otimes v^\dagger
    return dot(vector.reshape((N, 1)), vector.reshape((1, N)).conjugate())


def mixed_state(*amp_state_pairs):
    """
    Return a mixed state density matrix as specified by the amp_state_pairs (amplitude, state), 
    where each state object must either be a state vector or a density matrix.
    If the state is given as a vector, it is only multiplied by the amplitudes AFTER conversion to a 
    density matrix. Hence, all amplitudes must be positive reals.
    The resulting density matrix is normalized before being returned.
    """
    # take first state to determine dimensions
    state0 = amp_state_pairs[0][1]
    N = state0.shape[0]
    full_state = zeros((N,N), dtype = complex)
    for amp, state in amp_state_pairs:
        if amp.imag != 0 or amp.real < 0:
            raise ValueError('The amplitudes need to be positive real numbers: %s' % amp)
        if _is_vector(state):
            full_state += amp * vector_to_density_matrix(state)
        else:
            full_state += amp * state.reshape(N, N)
    
    if normalize:
        full_state /= full_state.trace()
    
    return full_state

def normalize_density_matrix_vector(s_vector):
    N = s_vector.shape[0]
    id_N = identity(int(sqrt(N))).reshape((N,))
    assert int(sqrt(N))**2 == N
    return s_vector / dot(s_vector, id_N)

def steady_state(L_super, max_iter = 20, tol = 1e-6, initial_rho = None):
    N = L_super.shape[0]
    assert L_super.shape[1] == N
    
    from sys import float_info
    L_effective = L_super + float_info.epsilon * sparse_id(L_super.shape[0])
    
    if initial_rho == None:
        from numpy import random
        initial_rho = random.randn(N)
    else:
        assert initial_rho.shape == (N,)
        
    rho_p = initial_rho / linalg.norm(initial_rho, inf)
    
    for k in xrange(max_iter):
        rho = sparse_la.spsolve(L_effective, rho_p)
        rho_p = rho/linalg.norm(rho, inf)
        
        err = linalg.norm(L_super * rho_p, inf)
        if err < tol:
            
            return normalize_density_matrix_vector(rho_p)
    print "Warning, not converged: err = %g" % err
    return normalize_density_matrix_vector(rho_p)
    
# 
# ### BELOW COPIED FROM QuTIP
# #This file is part of QuTIP.
# #
# #    QuTIP is free software: you can redistribute it and/or modify
# #    it under the terms of the GNU General Public License as published by
# #    the Free Software Foundation, either version 3 of the License, or
# #   (at your option) any later version.
# #
# #    QuTIP is distributed in the hope that it will be useful,
# #    but WITHOUT ANY WARRANTY; without even the implied warranty of
# #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #    GNU General Public License for more details.
# #
# #    You should have received a copy of the GNU General Public License
# #    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
# #
# # Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
# #
# ###########################################################################
# #-------------------------------------------------------------------------------
# # Q FUNCTION
# #
def qfunc(state, xvec, yvec):
    g=sqrt(2)
    
    X,Y = meshgrid(xvec, yvec)
    amat = 0.5*g*(X + Y * 1j);
    

    if _is_vector(state):
        qmat = qfunc_vector(state, amat)
        
    elif len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        from scipy.linalg import eigh
        d, v = eigh(state)
        # d[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        qmat = zeros(amat.shape, dtype = float)
        for k in arange(0, len(d)):
            qmat1 = qfunc_vector(v[:,k], amat)
            qmat += d[k] * qmat1
    else:
        raise ValueError()
    
    # apply normalization
    qmat = 0.25 * qmat * g**2;
    return qmat
# 
# #
# Q-function for a pure state: Q = |<alpha|psi>|^2 / pi
#
# |psi>   = the state in fock basis
# |alpha> = the coherent state with amplitude alpha
#
def qfunc_vector(psi, alpha_mat):
    
    a_shape = shape(alpha_mat)
    
    q_vals = map(lambda alpha: abs(vdot(psi, coherent_state(alpha, len(psi))))**2, alpha_mat.flatten())

    return array(q_vals).reshape(a_shape) / pi


def phases(values, fix_jumps = True):
    phases = arctan2(values.imag, values.real)

    if fix_jumps:    # make phases continuous across n*2*pi crossings        
        phases_diffs = phases[1:]-phases[:-1]
        for i, dp in enumerate(phases_diffs):
            if dp > pi: # threshold
                phases[i+1:] = phases[i+1:] - 2 * pi
            elif dp < -pi: 
                phases[i + 1:] = phases[i + 1:] + 2 * pi
    
    return phases
    
def evaluate_observable(rho_series, observable_operator, full_space = None):
    if not isinstance(observable_operator, ca.Operator): # assume it is already a number
        return ones(rho_series.shape[0]) * observable_operator
        
    sparse_obs_matrix = ca.representation_matrix(observable_operator, full_space)
    obs_transpose_vector = sparse_obs_matrix.transpose().toarray().flatten()
    return dot(rho_series, obs_transpose_vector)
    
def sq_dim_vector_to_matrix(vector):
    vector = vector.flatten()
    from math import floor
    dim = int(sqrt(vector.shape[0]))
    if dim**2 == vector.shape[0]:
        return vector.reshape((dim, dim))
    else:
        raise Exception("Vector does not have square dimension! %s, %s" % (vector, vector.shape))
    
def partial_trace(rho_matrix, full_space, trace_over_index):
    local_dims = tuple(ca.HilbertSpace((ls,)).dimension for ls in full_space)
    return sq_dim_vector_to_matrix(
                trace(rho_matrix.reshape(local_dims*2), 
                    axis1 = trace_over_index, 
                    axis2 = len(local_dims) + trace_over_index
                    ).flatten()
                )

def integrate_master_eq(network_triplet, times, initial_state):
    L_super = get_lindbladian(network_triplet)
    if isinstance(initial_state, ca.Operator):
        initial_state = initial_state.representation_matrix(network_triplet.full_space).toarray()
    initial_state = initial_state.flatten()
    if initial_state.shape[0] != L_super.shape[0]:
        raise Exception("Arrays not aligned")
    if initial_state.dtype != complex128:
        initial_state = array(initial_state, dtype = complex128)

    # do reinterpretation trick to use real-only odeint
    # the data is unaffected but the number of array entries doubles.
    initial_state.dtype = float64
    def drho_dt(rho, t):
        
        # print "called for %s" % t
        rho.dtype = complex128
        
        # perform matrix vector product with correct complex datatype
        rho_prime = L_super * rho
        
        # switch back to float interpretation
        rho.dtype = float64
        rho_prime.dtype = float64
        return rho_prime
        
    from scipy.integrate import odeint
    data = odeint(drho_dt, initial_state, times)
    
    # switch back to correct type
    data.dtype = complex128
    initial_state.dtype = complex128
    
    return times, data
    
def integrate_master_eq_o2(network_triplet, duration, initial_state, stepsize = 1e-5, iterations_per_sample = 1000):

    L_super = get_lindbladian(network_triplet)
    if isinstance(initial_state, ca.Operator):
        initial_state = initial_state.representation_matrix(network_triplet.full_space).toarray()
    initial_state = initial_state.flatten()

    if initial_state.shape[0] != L_super.shape[0]:
        raise Exception("Arrays not aligned")

    # make sure array is complex
    if initial_state.dtype != complex128:
        initial_state = array(initial_state, dtype = complex128)
    print "creating stepper: dt*L_sup*(1+ .5*dt*L_sup)"
    stepperO2 = stepsize * L_super  + (0.5*stepsize**2) * L_super * L_super
    
    times = arange(0., duration+.1*stepsize*iterations_per_sample, stepsize*iterations_per_sample)
    rhos = zeros((times.shape[0], initial_state.shape[0]), dtype = complex)
    current_state = initial_state
    for k, t in enumerate(times[:-1]):
        rhos[k,:] = current_state
        for k in xrange(iterations_per_sample):
            current_state += stepperO2 * current_state
    rhos[-1,:] = current_state
    return times, rhos

def integrate_master_eq_o4(network_triplet, duration, initial_state, stepsize = 1e-5, iterations_per_sample = 1000):

    L_super = get_lindbladian(network_triplet)
    if isinstance(initial_state, ca.Operator):
        initial_state = initial_state.representation_matrix(network_triplet.full_space).toarray()
    initial_state = initial_state.flatten()

    if initial_state.shape[0] != L_super.shape[0]:
        raise Exception("Arrays not aligned")

    # make sure array is complex
    if initial_state.dtype != complex128:
        initial_state = array(initial_state, dtype = complex128)
    print "creating stepper: dt*L_sup*(1+ .5*dt*L_sup)"
    M1 = stepsize * L_super
    M2 = M1 * M1 / 2.0
    M3 = M1 * M2 / 3.0
    M4 = M1 * M3 / 4.0
    diff_stepper_o4 = (M1 + M2 + M3 + M4).tocsr()
    del M1, M2, M3, M4, L_super
    times = arange(0., duration+.1*stepsize*iterations_per_sample, stepsize*iterations_per_sample)
    rhos = zeros((times.shape[0], initial_state.shape[0]), dtype = complex)
    current_state = initial_state
    for k, t in enumerate(times[:-1]):
        rhos[k,:] = current_state
        for k in xrange(iterations_per_sample):
            current_state += diff_stepper_o4 * current_state
    rhos[-1,:] = current_state
    return times, rhos
    
    
def jsim_nt(network_triplet, initial_state, observables, ndatp = 50000, rns = 0, dt = 1e-5, pperd = 1000):
    full_space = network_triplet.space
    
    cdim = network_triplet.cdim
    
    # full dimension
    dtot = full_space.dimension
    
    # random number generator seed
    if rns == 0:
        import time, random as pyrandom
        pyrandom.seed(time.clock())
        rns = pyrandom.randint(0, 2**29 - 1)
    
    
    # create necessary operators
    # effective hamiltonian
    H_eff = network_triplet.H -.5j*sum( ca.adjoint(Lk)*Lk for Lk in network_triplet.L.array[:,0])
    
    
    # convert to matrix
    H_eff_m = ca.representation_matrix(H_eff.expand().n(), full_space)
    
    # calculate stepper by expanding exp(i dt H_eff) to 4th order
    M1 = -1j * dt * H_eff_m
    M2 = M1 * M1 / 2.0
    M3 = M1 * M2 / 3.0
    M4 = M1 * M3 / 4.0
    Propagator =  (sparse_id(dtot) + M1 + M2 + M3 + M4).tocsr()
    # Propagator =  (sparse_id(dtot) + M1).tocsr()
    

       
    # calculate the rep matrices for the jump operators
    Ljs = [ca.representation_matrix(ca.n(ca.expand(Lk)), full_space).tocsr() for Lk in network_triplet.L.array[:,0]]
    
    observable_matrices = [ca.representation_matrix(ca.n(ca.expand(obs)), full_space).tocsr() for obs in observables]
    
    if initial_state.dtype != complex:
        initial_state = (1+0j) * initial_state
    
    psi_t = initial_state
    
    data = zeros((ndatp, 1 + len(observable_matrices)))
    t = 0
    
    data[0,:] = [t] + [vdot(psi_t, obs * psi_t).real for obs in observable_matrices]
    random.seed(rns)
    for data_row in data[1:]:
        
        jump_rnd_sample = random.random_sample(pperd)
        for jmpp in jump_rnd_sample: #iterate over random numbers to compare with jump probability
            psi_plus_dpsi = Propagator * psi_t
            
            
            psi_norm_sq = vdot(psi_plus_dpsi, psi_plus_dpsi).real
            
            # decide whether to jump
            if jmpp > psi_norm_sq: # if j_rnd \in [psi_norm, 1.) jump
            
                Ljpsis = [Lj * psi_t for Lj in Ljs]
                # do only a single jump by selecting one of the jump operators
                # according to their conditional probabilities
                L_probs_over_dt = [vdot(Ljpsi, Ljpsi).real for  Ljpsi in Ljpsis]
                psum_over_dt = sum(L_probs_over_dt)
                print (1. - psi_norm_sq), psum_over_dt * dt
                choice_rnd = random.random() * psum_over_dt
                Lp_sum = 0
                
                # partition the interval [0, p1 + ... + pN) into N subintervals
                # [0,p1). [p1, p1+p2), ...[p1 + ... pN-1, p1 + ... + pN)
                # generate random number between r \in [0, p1 + ... + pN)
                # select that operator L_k which interval the random number fell:
                # r \in [p1 + ... + pk-1, p1 + ... pk)
                for k, Lp in enumerate(L_probs_over_dt):
                    Lp_sum += Lp
                    if Lp_sum > choice_rnd:
                        break
                        
                # k is now the chosen jump operator index
                # dpsi = (-iH_eff dt + (L_k - 1)) psi
                psi_t = Ljpsis[k] / sqrt(L_probs_over_dt[k])
                print "jump at time %g in channel %d" % (t, k)
                
            else:    
                psi_t = psi_plus_dpsi / sqrt(psi_norm_sq)
            t += dt
        # add data entry, time, observable expectation values
        # psi_t_norm_sq = vdot(psi_t, psi_t)
        data_row[:] = [t] + [(vdot(psi_t, obs * psi_t)).real for obs in observable_matrices]
    return rns, data, psi_t


def prepare_jsim3_simulation(network_triplet, working_directory, initial_state, observables, ndatp = 500000, rns = 0,  dt = 1e-5, pperd = 1000):
    # dt : stepsize for timestep 
    # pperd : timesteps (without detected photons) per cycle 
    # ndatp : total number of cycles
    
    # number of observables
    nE = len(observables)
    
    full_space = network_triplet.space
    
    # full dimension
    dtot = full_space.dimension
    
    # random number generator seed
    if rns == 0:
        import time, random
        random.seed(time.clock())
        rns = random.randint(0, 2**29 - 1)
    
    
    # create necessary operators
    # effective hamiltonian
    H_eff = (network_triplet.H + (-.5j*sum(ca.adjoint(Lk)*Lk for Lk in network_triplet.L.array[:,0])))

    # prepare for numerical representation
    H_eff = H_eff.expand().n()
    
    print "created effective symbolic hamiltonian, it has {} summands".format(len(H_eff.operands))
    # print H_eff
   
    # # calculate stepper by expanding exp(i dt H_eff) to 4th order
    # M1 = -1j * dt * H_eff
    # print "created M1"

    # M2 = ((M1 * M1) * (1/ 2.0)).expand().n()
    # print "created M2"

    # M3 = ((M1 * M2) * (1/ 3.0)).expand().n()
    # print "created M3"

    # M4 = ((M1 * M3) * (1/ 4.0)).expand().n()
    # print "created M4, it has {} summands}".format(len(M3.summands))
    # print "calculating the representation matrix of T0 = exp(-i dt H_eff) expanded to 4th order"

    # # convert to matrix
    # T0 =  ca.representation_matrix((1 + M1 + M2 + M3 + M4).n(), full_space).tocoo()
    # print "generated representation matrix of T0 with {} non-zero entries".format(T0.nnz)
    
    T0 = ca.representation_matrix(H_eff, full_space).tocoo()
    print "generated the effective Hamiltonian in matrix representation, {} non-zero elements".format(T0.nnz)
    
    # calculate the rep matrices for the jump operators
    Tjs = [ca.representation_matrix(ca.n(ca.expand(Lk)), full_space).tocoo() for Lk in network_triplet.L.array[:,0] if Lk != 0]
    print "generated all representation matrices for the jump operators"

    Ts = [T0] + Tjs

    # number of non-zero noises 
    mm = len(Ts) - 1


    import os
    
    
    # stepping operator plus lindbladian jump operators
    tmat_file_name = os.path.join(working_directory, 'Tmatsp.dat')
    tmat_file = open(tmat_file_name, 'wb')
    
    
    for k,M in enumerate(Ts):
        tmat_file.write(M.row.tostring())
        tmat_file.write(M.col.tostring())
        tmat_file.write(M.data.real.tostring())
        tmat_file.write(M.data.imag.tostring())
        tmat_file.flush()
        print "wrote T{} to disk".format(k)
    
    tmat_file.close()
    
    # observables
    emat_file_name = os.path.join(working_directory, 'Ematsp.dat')
    emat_file = open(emat_file_name, 'wb')
    
    observable_matrices = [ca.representation_matrix(ca.n(ca.expand(obs)), full_space).tocoo() for obs in observables]
    for k, M in enumerate(observable_matrices):
        emat_file.write(M.row.tostring())
        emat_file.write(M.col.tostring())
        emat_file.write(M.data.real.tostring())
        emat_file.write(M.data.imag.tostring())
        emat_file.flush()
        print "wrote M{} to disk".format(k)
        
    emat_file.close()
    
    # initial state
    if initial_state.dtype != complex:
        initial_state = (1+0j) * initial_state
    psi_file_name = os.path.join(working_directory, 'psivecsp.dat')
    psi_file = open(psi_file_name, 'wb')
    psi_file.write(arange(initial_state.shape[0]).tostring())
    psi_file.write(initial_state.real.tostring())
    psi_file.write(initial_state.imag.tostring())
    psi_file.flush()
    psi_file.close()
    print "wrote inital state"
    
    # params
    params_file_name = os.path.join(working_directory, 'params.txt')
    params_file = open(params_file_name, 'wt')
    params_file.write(' %d %d %d %d %d %d %e' % (rns, dtot, mm, nE, ndatp, pperd, dt))
    params_file.write(''.join([' %d' % M.nnz for M in Ts]))
    params_file.write(''.join([' %d' % M.nnz for M in observable_matrices]))
    params_file.write(' %d' % initial_state.shape[0])
    params_file.close()
    
    return Ts, observable_matrices
    

def retrieve_jsim3_simulation_data(working_directory):
    import os
    
    time_series = loadtxt(os.path.join(working_directory, 'jsim.out'), skiprows = 1)
    final_state_data = fromfile(os.path.join(working_directory, 'psi.jsim'), dtype = float)
    assert final_state_data.shape[0] % 2 == 0
    final_state = final_state_data[:final_state_data.shape[0]/2] + 1j * final_state_data[final_state_data.shape[0]/2:]
    return time_series, final_state

def save_big_sparse_matrix(filename, matrix):
    mat_file = open(mat_file_name, 'wb')
    mat_file.write(m.row.tostring())
    mat_file.write(m.col.tostring())
    mat_file.write(m.data.real.tostring())
    mat_file.write(m.data.imag.tostring())
    mat_file.flush()
    mat_file.close()
    
