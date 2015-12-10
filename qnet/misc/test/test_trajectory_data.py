import os
from os.path import join
from qnet.misc.trajectory_data import TrajectoryData
from distutils import dir_util
from collections import OrderedDict
import uuid
import numpy as np
import pytest

@pytest.fixture
def datadir(tmpdir, request):
    '''Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.'''
    # http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return str(tmpdir)


def test_new_id():
    id1 = TrajectoryData.new_id()
    assert id1 == str(uuid.UUID(id1))
    id2 = TrajectoryData.new_id()
    id3 = TrajectoryData.new_id('foo')
    assert(id1 != id2 != id3)
    id4 = TrajectoryData.new_id('foo')
    assert(id3 == id4)
    id5 = TrajectoryData.new_id('bar')
    assert(id5 != id4)


def test_from_qsd_data(datadir):
    operators = OrderedDict([
        ('X1', join(datadir, 'traj1', 'X1.out')),
        ('X2', join(datadir, 'traj1', 'X2.out'))
    ])
    init_seed = 103212
    traj = TrajectoryData.from_qsd_data(operators, seed=init_seed)
    assert traj.dt == 0.1
    assert traj.nt == 51
    assert len(traj.record) == 1
    assert traj.operators == ['X1', 'X2']
    for (id, (seed, n_traj, op_list)) in traj.record.items():
        assert seed == init_seed
        assert n_traj == 1
        assert op_list == ['X1', 'X2']
    assert traj.record_seeds == set([init_seed, ])
    assert traj.record_IDs == set([traj.ID, ])
    assert list(traj.table.keys()) == ['Re[<X1>]', 'Im[<X1>]', 'Re[var(X1)]',
            'Im[var(X1)]', 'Re[<X2>]', 'Im[<X2>]', 'Re[var(X2)]',
            'Im[var(X2)]']
    fname = join(datadir, 'traj1', 'X1.out')
    (tgrid, re_exp, im_exp, re_var, im_var) \
    = np.genfromtxt(fname, dtype=np.float64, skip_header=1, unpack=True)
    assert np.max(np.abs(re_exp - traj.table['Re[<X1>]'])) < 1.0e-15
    assert np.max(np.abs(im_exp - traj.table['Im[<X1>]'])) < 1.0e-15
    assert np.max(np.abs(re_var - traj.table['Re[var(X1)]'])) < 1.0e-15
    assert np.max(np.abs(im_var - traj.table['Im[var(X1)]'])) < 1.0e-15
