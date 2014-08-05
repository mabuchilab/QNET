#cython: embedsignature=True


import numpy as np
from libc.math cimport sin, cos

ctypedef int c_input_fn(double t, complex * inputs, void * input_params) nogil

# cdef struct combinator_params:
#     Py_ssize_t k
#     Py_ssize_t n_terms
#     c_input_fn * input_fns
#     void ** input_params


# cdef int add_combinator(double t, complex * inputs, void * input_params) nogil:
#     cdef combinator_params cp = (<combinator_params*> input_params)[0]
#     cdef complex * tmp = malloc(cp.k * sizeof(complex))
#     cdef int jj, kk
#     for jj in range(cp.n_terms):
#         cp.input_fns[kk](t, tmp, cp.input_params[jj])
#         for kk in range(cp.k):
#             inputs[kk] += tmp[kk]


cdef struct osc_params:
    Py_ssize_t k
    Py_ssize_t * n_omegas
    complex ** amplitudes
    double ** omegas
    


cdef int osc_inputs(double t, complex * inputs, void * input_params) nogil:
    cdef Py_ssize_t jj, kk

    cdef osc_params op = (<osc_params*>input_params)[0]
    for jj in range(op.k):
        inputs[jj] = 0.
        for kk in range(op.n_omegas[jj]):
            inputs[jj] += op.amplitudes[jj][kk] * (cos(op.omegas[jj][kk] * t) + 1j * sin(op.omegas[jj][kk] * t))

    return 0

# cdef struct ramp_params:
#     Py_ssize_t k
#     double * periods
#     double * amplitudes

# cdef int ramp_inputs(double t, complex * inputs, void * input_params) nogil:
#     cdef Py_ssize_t jj, kk
#     cdef osc_params op = (<osc_params*>input_params)[0]
#     for jj in range(op.k):
#         inputs[jj] = 0.
#         for kk in range(op.n_omegas[jj]):
#             inputs[jj] += op.amplitudes[jj][kk] * (cos(op.omegas[jj][kk] * t) + 1j * sin(op.omegas[jj][kk] * t))

#     return 0


cdef struct KerrSDEParams:
    Py_ssize_t n                 # number of complex internal modes
    Py_ssize_t k                 # number of complex dynamic inputs
    Py_ssize_t m                 # number of complex noise inputs m >= k
    complex * A
    complex * B
    complex * A_kerr
    complex * B_input
    complex * u_c
    void * input_params
    complex * inputs
    c_input_fn input_fn
    

cdef int KerrODE(double t, double * y, double * ydot, void *params) nogil:
    cdef Py_ssize_t jj, kk
    cdef KerrSDEParams kp = (<KerrSDEParams*>params)[0]
    cdef complex * ycmp = <complex*>y
    cdef complex * ydotcmp = <complex*>ydot

    kp.input_fn(t, kp.inputs, kp.input_params)

    for jj in range(kp.n):
        ydotcmp[jj] = kp.u_c[jj]
        for kk in range(kp.n):
            ydotcmp[jj] += kp.A[jj * kp.n + kk] * ycmp[kk]
            ydotcmp[jj] += kp.A_kerr[jj * kp.n + kk] * (ycmp[kk].real**2 + ycmp[kk].imag**2) * ycmp[jj]
        for kk in range(kp.k):
            ydotcmp[jj] += kp.B_input[jj * kp.k + kk] * kp.inputs[kk]

    return 0

cdef int KerrNoise(double t, double * y, double * w, double * b, void * params) nogil:
    cdef KerrSDEParams kp = (<KerrSDEParams*>params)[0]
    cdef Py_ssize_t jj, kk

    cdef complex * wcmp = <complex*>w
    cdef complex * bcmp = <complex*>b
    
    

    for jj in range(kp.n):
        bcmp[jj] = 0
        for kk in range(kp.m):
            bcmp[jj] += kp.B[jj * kp.m + kk] * wcmp[kk]/2

    return 0


        
        

cdef class KerrSDE(cysolve.ode.SDEs):

    cdef KerrSDEParams kparams
    cdef osc_params oparams

    cdef np.ndarray _amplitude_ptrs
    cdef np.ndarray _omega_ptrs
    cdef np.ndarray _n_omegas
    cdef np.ndarray _inputs

    cdef public np.ndarray A
    cdef public np.ndarray B
    cdef public np.ndarray C
    cdef public np.ndarray D
    cdef public np.ndarray A_kerr
    cdef public np.ndarray B_input
    cdef public np.ndarray D_input
    cdef public np.ndarray u_c
    cdef public np.ndarray U_c

    cdef object omegas
    cdef object amplitudes

    
    def __init__(self, model_matrices, omegas, amplitudes):
        A, B, C, D, A_kerr, B_input, D_input, u_c, U_c = model_matrices
        cdef Py_ssize_t n, m, k, jj
        cdef complex * Ap, * Bp, * A_kerrp, * B_inputp, * u_cp
        
        self.amplitudes = [np.array(amps, dtype=complex) for amps in amplitudes]
        self.omegas = [np.array(omegs, dtype=float) for omegs in omegas]
        
        n = A.shape[0]
        m = B.shape[1]
        k = B_input.shape[1]
        
        
        self.A = A = A.astype(complex)
        self.B = B = B.astype(complex)
        self.C = C = C.astype(complex)
        self.D = D = D.astype(complex)
        self.A_kerr = A_kerr = A_kerr.astype(complex)
        self.B_input = B_input = B_input.astype(complex)
        self.D_input = D_input = D_input.astype(complex)
        self.u_c = u_c = u_c.astype(complex)
        self.U_c = U_c = U_c.astype(complex)
        
        Ap = <complex*>np.PyArray_DATA(A)
        Bp = <complex*>np.PyArray_DATA(B)
        A_kerrp = <complex*>np.PyArray_DATA(A_kerr)
        B_inputp = <complex*>np.PyArray_DATA(B_input)
        u_cp = <complex*>np.PyArray_DATA(u_c)
        self.c_params = & self.kparams
        self.kparams.n = n
        self.kparams.m = m
        self.kparams.k = k
        self.kparams.A = Ap
        self.kparams.B = Bp
        self.kparams.A_kerr = A_kerrp
        self.kparams.B_input = B_inputp
        self.kparams.u_c = u_cp
        
        self.oparams.k = k
        self._amplitude_ptrs = np.zeros(k, dtype=np.int64)
        self._omega_ptrs = np.zeros(k, dtype=np.int64)
        self._n_omegas = np.zeros(k, dtype=np.int64)
        self._inputs = np.zeros(k, dtype=np.complex128)
        
        self.oparams.amplitudes = <complex **>np.PyArray_DATA(self._amplitude_ptrs)
        self.oparams.omegas = <double **>np.PyArray_DATA(self._omega_ptrs)
        self.oparams.n_omegas = <Py_ssize_t*>np.PyArray_DATA(self._n_omegas)
        self.kparams.inputs = <complex *>np.PyArray_DATA(self._inputs)
        self.kparams.input_fn = osc_inputs
        self.kparams.input_params = <void*>&self.oparams
    
        for jj in range(k):            
            self.oparams.amplitudes[jj] = <complex *>np.PyArray_DATA(self.amplitudes[jj])
            self.oparams.omegas[jj] = <double*>np.PyArray_DATA(self.omegas[jj])
            self.oparams.n_omegas[jj] = len(self.amplitudes[jj])

        
        print "setting up self.y"
        self.y = np.zeros((1, 2* n), dtype=float)
        self.t = 0.
        self.dim = 2 * n
        self.N_noises = 2 * m
        self.c_noise_coeff = KerrNoise
        self.c_ode = KerrODE
        self.N_params = 1
        self.param_size = sizeof(KerrSDEParams)

    def _get_y_complex(self):
        return self.get_y()[0].view(dtype=complex)

    def _set_y_complex(self, y):
        y2 = np.zeros((1, self.dim))
        y2[0,::2] = y.real
        y2[0,1::2] = y.imag
        self.set_y(y2)

    y_c = property(_get_y_complex, _set_y_complex, doc="Complex internal mode amplitudes")

    def eval_inputs(self, double delta_t, Py_ssize_t N, int include_initial=0):
        cdef Py_ssize_t k = self.kparams.k
        cdef Py_ssize_t jj
        cdef np.ndarray ret = np.zeros((N+1, k), dtype=complex)
        cdef complex * retp = <complex*>np.PyArray_DATA(ret)
        cdef double t = self.t
        for jj in range(N+1):
            t = jj * delta_t / N + self.t
            self.kparams.input_fn(t, &retp[jj*k], self.kparams.input_params)
        if not include_initial:
            ret = ret[1:]
        return ret
        
    
    def integrate_kerr_sde(self, double delta_t, double h, Py_ssize_t N, int include_initial=0, 
                           int update_state=1, int sep_output_noise=1, int rngseed=0):
        
        inputs = self.eval_inputs(delta_t, N, include_initial)
        Y, W, B, success = self.integrate_sde(delta_t, h, N, include_initial=include_initial, 
                                              update_state=update_state, return_noises=1, rngseeds=np.array([rngseed], dtype=np.int64))

        Yc = Y[0].view(dtype=complex)
        Ac = W[0].view(dtype=complex)/2

        outputs = Yc.dot(self.C.T) + self.U_c.reshape((1, -1)) + inputs.dot(self.D_input.T)

        if sep_output_noise:
            return Yc, inputs, outputs, Ac.dot(self.D.T)

        return Yc, inputs, outputs +  Ac.dot(self.D.T)

    def integrate_kerr_ode(self, double delta_t, Py_ssize_t N, int include_initial=0, int update_state=1):
        inputs = self.eval_inputs(delta_t, N, include_initial)
        Y, success = self.integrate(delta_t, N, include_initial=include_initial, update_state=update_state)
        Yc = Y[0].view(dtype=complex)
        outputs = Yc.dot(self.C.T) + self.U_c.reshape((1, -1)) + inputs.dot(self.D_input.T) 

        return Yc, inputs, outputs
    
    def evaluate_c(self):
        return self.evaluate().view(dtype=complex)

    def steady_state_c(self, double T_max, double tol):
        y, success = self.steady_state(T_max, tol)
        return y.view(dtype=complex), success
            
