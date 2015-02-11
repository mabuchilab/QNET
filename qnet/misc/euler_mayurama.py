# coding=utf-8

import numpy as np


def euler_mayurama_complex(f, g, x0, m, h, steps_per_sample, n_samples, include_initial=False):
    """
    Evaluate the Ito-SDE dx = f(x,t)dt + g(x,t)dA_t using an Euler-Mayurama scheme with fixed stepsize h.
    
    Params
    ------
    f: function for drift term
    g: function for diffusion term
    x0: initial value
    m: number of complex noises
    steps_per_sample: The number of h-sized steps between samples
    n_samples: the total number of integration steps
    include_initial: whether or not to include the initial value (and time) in the output.

    Returns
    -------
    A tuple(times, xs, dAs)
    
    times: array of times at which x is evaluated
    x: x at different times
    dAs: The complex Ito noise increments for the time interval [t, t+h)
    """
    times = np.zeros(n_samples + 1)
    xs = np.zeros((n_samples + 1, x0.shape[0]), dtype=complex)
    xs[0] = x0
    dAs = np.zeros((n_samples + 1, m), dtype=complex)
    x = np.array(x0, dtype=complex)
    t = 0.
    for nn in range(n_samples):
        dA = np.random.randn(steps_per_sample, m) * np.sqrt(h) + (np.sqrt(h) * 1j) * np.random.randn(steps_per_sample, m)
        for mm in range(steps_per_sample):
            x += f(x, t) * h + g(x, t).dot(dA[mm])
            t += h
        # print(".",)
        times[nn+1] = t
        xs[nn+1] = x
        dAs[nn] = dA[0]
    dAs[-1] = np.random.randn(m) * np.sqrt(h) + (np.sqrt(h) * 1j) * np.random.randn(m)

    if not include_initial:
        return times[1:], xs[1:], dAs[1:]

    return times, xs, dAs

# from itertools import reduce

def euler_mayurama_complex(f, g, x0, m, h, steps_per_sample, n_samples, include_initial=False):
    """
    Evaluate the Ito-SDE dx = f(x,t)dt + g(x,t)dA_t using an Euler-Mayurama scheme with fixed stepsize h.
    
    Params
    ------
    f: function for drift term
    g: function for diffusion term
    x0: initial value
    m: number of complex noises
    steps_per_sample: The number of h-sized steps between samples
    n_samples: the total number of integration steps
    include_initial: whether or not to include the initial value (and time) in the output.

    Returns
    -------
    A tuple(times, xs, dAs)
    
    times: array of times at which x is evaluated
    x: x at different times
    dAs: The complex Ito noise increments for the time interval [t, t+h)
    """
    times = np.zeros(n_samples + 1)
    xs = np.zeros((n_samples + 1, x0.shape[0]), dtype=complex)
    xs[0] = x0
    dAs = np.zeros((n_samples + 1, m), dtype=complex)
    x = np.array(x0, dtype=complex)
    t = 0.
    for nn in range(n_samples):
        dA = np.random.randn(steps_per_sample, m) * np.sqrt(h) + (np.sqrt(h) * 1j) * np.random.randn(steps_per_sample, m)
        update_step = lambda x_t, dAt: (x_t[0] + f(*x_t) * h + g(*x_t).dot(dAt), x_t[1]+h)
        # for mm in range(steps_per_sample):
        #     x += f(x, t) * h + g(x, t).dot(dA[mm])
        #     t += h
        # print(".",)
        x, t = reduce(update_step, dA, (x, t))
        times[nn+1] = t
        xs[nn+1] = x
        dAs[nn+1] = dA.sum(axis=0)


    if not include_initial:
        return times[1:], xs[1:], dAs[1:]

    return times, xs, dAs
