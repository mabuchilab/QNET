# coding=utf-8

from functools import reduce

import numpy as np


def euler_mayurama_complex(
        f, g, x0, m, h, steps_per_sample, n_samples, include_initial=False):
    """Evaluate the Ito-SDE

    .. math::

        dx = f(x,t)dt + g(x,t)dA_t

    using an Euler-Mayurama scheme with fixed stepsize $h$.

    Args:
        f: function for drift term
        g: function for diffusion term
        x0: initial value
        m: number of complex noises
        steps_per_sample: The number of $h$-sized steps between samples
        n_samples: the total number of integration steps
        include_initial: whether or not to include the initial value (and time)
            in the output.

    Returns:
        a tuple ``(times, xs, dAs)``, where ``times`` is an array of times at
        which x is evaluated, ``xs`` is $x$ at different times, and ``dAs`` are
        the noise increments for the time interval $[t, t+h)$
    """
    times = np.zeros(n_samples + 1)
    xs = np.zeros((n_samples + 1, x0.shape[0]), dtype=complex)
    xs[0] = x0
    dAs = np.zeros((n_samples + 1, m), dtype=complex)
    x = np.array(x0, dtype=complex)
    t = 0.
    for nn in range(n_samples):
        dA = (np.random.randn(steps_per_sample, m) * np.sqrt(h) +
              (np.sqrt(h) * 1j) * np.random.randn(steps_per_sample, m))

        def update_step(x_t, dAt):
            return (x_t[0] + f(*x_t) * h + g(*x_t).dot(dAt), x_t[1]+h)

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
