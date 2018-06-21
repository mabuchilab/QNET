"""Main QNET package

The :mod:`qnet` package exposes all of QNET's functionality for easy
interactive or programmative use.

For interactive usage, the package should be initialized as follows::

    >>> import qnet
    >>> qnet.init_printing()

"""

import qnet._flat_api_tools
import qnet.algebra._rules

__doc__ += qnet._flat_api_tools.__doc__

__all__ = []  # will be extended by _import_submodules

__version__ = "2.0.0-dev"


def _git_version():
    """If installed with 'pip installe -e .' from inside a git repo, the
    current git revision as a string"""

    import subprocess
    import os

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        FNULL = open(os.devnull, 'w')
        cwd = os.path.dirname(os.path.realpath(__file__))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=FNULL, env=env, cwd=cwd)
        out = proc.communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        return out.strip().decode('ascii')
    except OSError:
        return "unknown"


__git_version__ = _git_version()


# dynamic initialization

qnet._flat_api_tools._import_submodules(
    __all__, __path__, __name__)

qnet.algebra._rules._algebraic_rules()
qnet.algebra.init_algebra()
