# coding=utf-8

from __future__ import division

import numpy as np
from qnet.algebra.circuit_algebra import *




def model_matrices(slh, dynamic_input_ports, apply_kerr_diagonal_correction=True, epsilon = 0., return_eoms=False):
    """
    Return the matrices necessary to carry out a semi-classical simulation 
    of the SLH system driven by some dynamic inputs.

    Params
    ------
    slh: SLH object

    dynamic_input_ports: python dict {port_index: input_name_str,...}

    apply_kerr_diagonal_correction: bool that specifies whether there should be an 
                                    effective detuning of 2 \chi for every kerr-cavity.

    epsilon: for non-zero epsilon (and a numerical coefficient slh) remove 
             expressions with coefficents smaller than epsilon.

    return_eoms: Whether to also return the symbolic e.o.m.'s as well as the output processes.
    
    Returns
    -------
    A tuple (A, B, C, D, A_kerr, B_input, D_input, u_c, U_c[, eoms, dA'])

    A: coupling of modes to each other
    B: coupling of external input fields to modes
    C: coupling of internal modes to output
    D: coupling of external input fields to output fields

    A_kerr: kerr-type coupling between modes
    B_input: coupling of dynamic inputs to modes
    D_input: coupling of dynamic inputs to external output fields
    u_c: constant coherent input driving to modes
    U_c: constant coherent input contribution to output field

    Optional
    --------

    eoms: symbolic QSDEs for the internal modes
    dA': symbolic expression for the output fields

    The overall SDE is then:
    da_t/dt = (A * a_t + (A_kerr * (a_t (*) a_t^*)) (*) a_t + u_c + B_input * u_t) + B * dA_t/dt
    dA'_t/dt = (C * a_t + U_c + D_input * u_t) + D * dA_t/dt

    Here A * b is a matrix product, whereas a (*) b is an element-wise
    product of two vectors.  It is assumed that all degrees of freedom
    are cavities with their only non-linearity being of the Kerr-type,
    i.e. either self coupling H_kerr = a^*a^* a a or cross-coupling
    H_kerr = a^*a b^*b.
    """

    # the different degrees of freedom
    modes = sorted(slh.space.local_factors())
    
    # various dimensions
    ncav = len(modes)
    cdim = slh.cdim
    ninputs = len(dynamic_input_ports)
    
    # initialize the matrices
    A = np.zeros((ncav, ncav), dtype=object)
    B = np.zeros((ncav, cdim), dtype=object)
    C = np.zeros((cdim, ncav), dtype=object)
    def ascomplex(o):
        try:
            return complex(o.coeff) if (isinstance(o, ScalarTimesOperator) and o.term is IdentityOperator) else o
        except:
            return o

    D = np.array([[ascomplex(o) for o in Sjj] for Sjj in slh.S.matrix])
    A_kerr = np.zeros((ncav, ncav), dtype=object)
    B_input = np.zeros((ncav, ninputs), dtype=object)
    D_input = np.zeros((cdim, ninputs), dtype=object)
    u_c = np.zeros(ncav, dtype=object)
    U_c = np.zeros(cdim, dtype=object)
    
    # make symbols for the external field modes
    noises = [OperatorSymbol('b_{{{}}}'.format(n), "ext({})".format(n)) for n in range(cdim)]
    
    # make symbols for the dynamic inputs
    inputs = [OperatorSymbol('u_{{{}}}'.format(u_name), TrivialSpace) for  
               n, u_name in sorted(dynamic_input_ports.items())]
    
    inputs_extended = [0] * cdim
    for ii, n in zip(inputs, sorted(dynamic_input_ports.keys())):
        inputs_extended[n] = ii

    # feed in the dynamic inputs
    slh_input = slh.coherent_input(*inputs_extended).expand().simplify_scalar()
    
    print("computing QSDEs")
    # compute the QSDEs for the internal operators
    eoms = [slh_input.symbolic_heisenberg_eom(Destroy(s), noises=noises).expand().simplify_scalar() for s in modes]
    
    
    print("Extracting matrices")
    # use the coefficients to generate A, B matrices
    for jj, sjj in enumerate(modes):
        coeffsjj = get_coeffs(eoms[jj], epsilon=epsilon)
        for kk, skk in enumerate(modes):
            A[jj, kk] = coeffsjj[Destroy(skk)]
            chi_jjkk = coeffsjj[Create(skk) * Destroy(skk) * Destroy(sjj)]
            if apply_kerr_diagonal_correction:
                A[jj, kk] += -(1 + int(jj==kk)) * chi_jjkk / 2
            A_kerr[jj, kk] = chi_jjkk
        for kk, dAkk in enumerate(noises):
            B[jj, kk] = coeffsjj[dAkk]
        for kk, u_kk in enumerate(inputs):
            if inputs == 0:
                continue
            B_input[jj,kk] = coeffsjj[u_kk]
        u_c[jj] = coeffsjj[IdentityOperator]
    
    
    # use the coefficients in the L vector to generate the C, D
    # matrices
    for jj, Ljj in enumerate(slh_input.L.matrix[:,0]):
        coeffsjj = get_coeffs(Ljj)
        for kk, skk in enumerate(modes):
            C[jj,kk] = coeffsjj[Destroy(skk)]
        U_c[jj] = coeffsjj[IdentityOperator]
        
        for kk, u_kk in enumerate(inputs):
            D_input[jj, kk] = coeffsjj[u_kk]
    

    if return_eoms:
        # compute output processes
        dAps =  (slh_input.S * Matrix([noises]).T + slh_input.L).expand().simplify_scalar()
        return A, B, C, D, A_kerr, B_input, D_input, u_c, U_c, eoms, dAps
    
    return A, B, C, D, A_kerr, B_input, D_input, u_c, U_c


def model_matrices_complex(*args, **kwargs):
    "Same as model_matrices() but tries to convert all output to purely numerical matrices"
    matrices = model_matrices(*args, **kwargs)
    if len(matrices) <= 9:
        return [arr.astype(complex) for arr in matrices]
    else:
        return [arr.astype(complex) for arr in matrices[:9]] + list(matrices[9:])
model_matrices_complex.__doc__ += "\n--\ndoc of model_matrices():\n" + model_matrices.__doc__


def model_matrices_symbolic(*args, **kwargs):
    "Same as model_matrices() but converts all output to Matrix() objects."
    matrices = model_matrices(*args, **kwargs)
    if len(matrices) <= 9:
        return [Matrix(arr) for arr in matrices]
    else:
        return [Matrix(arr) for arr in matrices[:9]] + list(matrices[9:])
model_matrices_symbolic.__doc__ += "\n--\ndoc of model_matrices():\n" + model_matrices.__doc__
        


def substitute_into_symbolic_model_matrices(model_matrices, params):
    
    return [m.substitute(params).matrix.astype(complex) for m in model_matrices[:9]] + list(model_matrices[9:])


def prepare_sde(numeric_model_matrices, input_fn, return_jac=False):
    """
    Compute the SDE functions f, g and (optionally) the Jacobian of f (see euler_mayurama docs) for the model matrices.
    
    Returns f, g[, Jf]

    The overall SDE is:
    da_t/dt = (A * a_t + (A_kerr * (a_t (*) a_t^*)) (*) a_t + u_c + B_input * u_t) + B * dA_t/dt
    dA'_t/dt = (C * a_t + U_c + D_input * u_t) + D * dA_t/dt
    """
    A, B, C, D, A_kerr, B_input, D_input, u_c, U_c = numeric_model_matrices[:9]
    B_over_2 = B/2.
    u_c = u_c.ravel()

    def f(a, t):
        "Return A.dot(a) +  (A_kerr.dot(a.conjugate() * a)) * a + u_c + B_input.dot(input_fn(t))."
        return A.dot(a) +  (A_kerr.dot(a.conjugate() * a)) * a + u_c + B_input.dot(input_fn(t))
    
    def g(a, t):
        return B_over_2

    if not return_jac:    
        return f, g
    
    def Jf(a, t):
        AA, BB = A + np.diag(A_kerr.dot(a * a.conjugate())) + A_kerr * np.outer(a, a.conjugate()), A_kerr*np.outer(a, a)
        return np.vstack([
            np.hstack([AA, BB]),
            np.hstack([BB.conjugate(), AA.conjugate()]),
        ])
    return f, g, Jf

def wrap_fqp(f):
    "Wrap a complex ode function f(a,t) as f(qp, t) where qp = [a1r, a1i, a2r, a2i,...]"
    def fqp(qp, t):
        qp.dtype = np.complex128
        fa = f(qp, t)
        qp.dtype = np.float64
        fa.dtype = np.float64
        return fa
    return fqp

def T_qp_a(n):
    "Basis transfer matrix, qp = T_qp_a.dot([[a],[a.conjugate()]])"
    ret = np.zeros((2*n, 2*n), dtype=complex)
    ret[::2,:n] = ret[::2,n:] = np.eye(n)
    ret[1::2,:n] = -1j* np.eye(n)
    ret[1::2,n:] = 1j * np.eye(n)
    return ret/2.
[]
def T_a_qp(n):
    """
    Basis transfer matrix, [[a],[a.conjugate()]] = T_a_qp.dot(qp),
    where qp = [a1r, a1i, a2r, a2i,...]
    """
    ret = np.zeros((2*n, 2*n), dtype=complex)
    ret[::2,:n] = ret[::2,n:] = np.eye(n)
    ret[1::2,:n] = -1j* np.eye(n)
    ret[1::2,n:] = 1j * np.eye(n)
    return ret.T.conjugate()

def wrap_Jqp(J):
    """
    Wrap the jacobian of a complex ode function f(a,t) as f(qp, t),
    where qp = [a1r, a1i, a2r, a2i,...]
    """
    def Jqp(qp, t):
        qp.dtype = np.complex128
        Ja = J(qp, t)
        qp.dtype = np.float64
        n = qp.shape[0]/2
        ret = T_qp_a(n).dot(Ja).dot(T_a_qp(n))
        assert np.allclose(ret.imag, np.zeros_like(ret))
        return ret.real
    return Jqp
