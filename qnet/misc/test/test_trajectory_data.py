import os
from os.path import join
from glob import glob
from textwrap import dedent
from qnet.misc.trajectory_data import TrajectoryData
from distutils import dir_util
from collections import OrderedDict
import uuid
import hashlib
import numpy as np
import pytest

TRAJ1_SEED = 103212

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


def qsd_traj(folder, seed):
    """Return a fixture that returns a TrajectoryData instance based on all the
    *.out file in the given folder (relative to the test datadir), and with the
    given seed"""
    def fixture(datadir):
        operators = OrderedDict()
        datafiles = sorted(glob(join(datadir, folder, '*.out')))
        assert len(datafiles) >0, "No files *.out in %s"%folder
        for file in datafiles:
            op_name = os.path.splitext(os.path.split(file)[1])[0]
            operators[op_name] = file
        return TrajectoryData.from_qsd_data(operators, seed=seed)
    return pytest.fixture(fixture)


traj1        = qsd_traj('traj1', TRAJ1_SEED)
traj1_coarse = qsd_traj('traj1_coarse', TRAJ1_SEED)


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


def test_from_qsd_data(traj1, datadir):
    traj = traj1
    md5sum = lambda f: hashlib.md5(open(f,'rb').read()).hexdigest()
    X1_file = join(datadir, 'traj1', 'X1.out')
    X2_file = join(datadir, 'traj1', 'X2.out')
    md5 = md5sum(X1_file) + md5sum(X2_file)
    assert traj.ID == str(uuid.uuid3(TrajectoryData._uuid_namespace, md5))
    assert traj.dt == 0.1
    assert traj.nt == 51
    assert traj.shape == (51, 9)
    assert len(traj.record) == 1
    assert traj.operators == ['X1', 'X2']
    for (id, (seed, n_traj, op_list)) in traj.record.items():
        assert seed == TRAJ1_SEED
        assert n_traj == 1
        assert op_list == ['X1', 'X2']
    assert traj.record_seeds == set([TRAJ1_SEED, ])
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


def test_data_line(traj1):
    assert traj1._data_line(0, fmt='%15.6e')  == "   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00"
    assert traj1._data_line(1, fmt='%15.6e')  == "   1.000000e-01   1.484680e-08  -2.097820e-10  -2.203820e-16   6.229170e-18   1.479510e-08   2.055640e-10  -2.188530e-16  -6.082700e-18"
    assert traj1._data_line(1, fmt='%5.1f')  == "  0.1  0.0 -0.0 -0.0  0.0  0.0  0.0 -0.0 -0.0"


def test_to_str(traj1, traj1_coarse):
    assert traj1_coarse.to_str().strip() == dedent('''
    # QNET Trajectory Data ID 8d102e4b-4a30-3460-9746-dc642f740cfb
    # Record 8d102e4b-4a30-3460-9746-dc642f740cfb (seed 103212): 1
    #                       t                 Re[<X1>]                 Im[<X1>]              Re[var(X1)]              Im[var(X1)]                 Re[<X2>]                 Im[<X2>]              Re[var(X2)]              Im[var(X2)]
       0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00
       1.0000000000000000e+00   1.2617600000000000e-02  -2.5277300000000002e-03  -1.5281299999999999e-04   6.3787600000000006e-05   1.2773600000000000e-02   1.0170900000000000e-03  -1.6212900000000001e-04  -2.5983600000000000e-05
       2.0000000000000000e+00   4.2612299999999999e-02  -2.0798400000000002e-02  -1.3832300000000000e-03   1.7725400000000000e-03   4.7045200000000002e-02   4.4810300000000004e-03  -2.1931699999999999e-03  -4.2162200000000002e-04
       3.0000000000000000e+00   6.1951199999999998e-02  -4.2463000000000001e-02  -2.0348499999999999e-03   5.2612700000000002e-03   5.7477100000000003e-02   4.8158100000000002e-02  -9.8441599999999989e-04  -5.5359700000000003e-03
       4.0000000000000000e+00   6.7414600000000005e-02  -6.9206400000000001e-02   2.4479599999999998e-04   9.3310400000000005e-03   2.6293199999999999e-02   9.2851000000000003e-02   7.9299799999999997e-03  -4.8827000000000002e-03
       5.0000000000000000e+00  -4.0601099999999996e-03  -9.8784499999999997e-02   9.7418899999999996e-03  -8.0215199999999999e-04   3.1303499999999998e-02   9.3702499999999994e-02   7.8002499999999999e-03  -5.8664399999999997e-03
    ''').strip()
    assert traj1.to_str(show_rows=4).strip() == dedent('''
    # QNET Trajectory Data ID d9831647-f2e7-3793-8b24-7c49c5c101a7
    # Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1
    #                       t                 Re[<X1>]                 Im[<X1>]              Re[var(X1)]              Im[var(X1)]                 Re[<X2>]                 Im[<X2>]              Re[var(X2)]              Im[var(X2)]
       0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00
       1.0000000000000001e-01   1.4846800000000000e-08  -2.0978199999999999e-10  -2.2038200000000000e-16   6.2291700000000000e-18   1.4795100000000000e-08   2.0556399999999999e-10  -2.1885300000000000e-16  -6.0827000000000000e-18
               ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...
       4.9000000000000004e+00   1.5875000000000000e-02  -1.0133800000000000e-01   1.0017500000000000e-02   3.2174999999999999e-03   2.0664200000000001e-02   1.0037800000000000e-01   9.6487799999999992e-03  -4.1484699999999996e-03
       5.0000000000000000e+00  -4.0601099999999996e-03  -9.8784499999999997e-02   9.7418899999999996e-03  -8.0215199999999999e-04   3.1303499999999998e-02   9.3702499999999994e-02   7.8002499999999999e-03  -5.8664399999999997e-03
    ''').strip()
    assert traj1.to_str(show_rows=5).strip() == dedent('''
    # QNET Trajectory Data ID d9831647-f2e7-3793-8b24-7c49c5c101a7
    # Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1
    #                       t                 Re[<X1>]                 Im[<X1>]              Re[var(X1)]              Im[var(X1)]                 Re[<X2>]                 Im[<X2>]              Re[var(X2)]              Im[var(X2)]
       0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00
       1.0000000000000001e-01   1.4846800000000000e-08  -2.0978199999999999e-10  -2.2038200000000000e-16   6.2291700000000000e-18   1.4795100000000000e-08   2.0556399999999999e-10  -2.1885300000000000e-16  -6.0827000000000000e-18
               ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...
       4.8000000000000007e+00   3.0328200000000000e-02  -1.0106200000000000e-01   9.2936500000000005e-03   6.1300399999999998e-03   1.4357300000000000e-02   1.0441300000000001e-01   1.0696000000000001e-02  -2.9981900000000000e-03
       4.9000000000000004e+00   1.5875000000000000e-02  -1.0133800000000000e-01   1.0017500000000000e-02   3.2174999999999999e-03   2.0664200000000001e-02   1.0037800000000000e-01   9.6487799999999992e-03  -4.1484699999999996e-03
       5.0000000000000000e+00  -4.0601099999999996e-03  -9.8784499999999997e-02   9.7418899999999996e-03  -8.0215199999999999e-04   3.1303499999999998e-02   9.3702499999999994e-02   7.8002499999999999e-03  -5.8664399999999997e-03
    ''').strip()
    assert (   traj1_coarse.to_str(show_rows=6)
            == traj1_coarse.to_str(show_rows=10)
            == traj1_coarse.to_str(show_rows=20)
            == traj1_coarse.to_str()
           )
    assert traj1.to_str(show_rows=0).strip() == dedent('''
    # QNET Trajectory Data ID d9831647-f2e7-3793-8b24-7c49c5c101a7
    # Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1
    #                       t                 Re[<X1>]                 Im[<X1>]              Re[var(X1)]              Im[var(X1)]                 Re[<X2>]                 Im[<X2>]              Re[var(X2)]              Im[var(X2)]
    ''').strip()
    assert traj1.to_str(show_rows=1).strip() == dedent('''
    # QNET Trajectory Data ID d9831647-f2e7-3793-8b24-7c49c5c101a7
    # Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1
    #                       t                 Re[<X1>]                 Im[<X1>]              Re[var(X1)]              Im[var(X1)]                 Re[<X2>]                 Im[<X2>]              Re[var(X2)]              Im[var(X2)]
               ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...                      ...
       5.0000000000000000e+00  -4.0601099999999996e-03  -9.8784499999999997e-02   9.7418899999999996e-03  -8.0215199999999999e-04   3.1303499999999998e-02   9.3702499999999994e-02   7.8002499999999999e-03  -5.8664399999999997e-03
    ''').strip()
    traj1_coarse.col_width = 12
    assert traj1_coarse.to_str().strip() == dedent('''
    # QNET Trajectory Data ID 8d102e4b-4a30-3460-9746-dc642f740cfb
    # Record 8d102e4b-4a30-3460-9746-dc642f740cfb (seed 103212): 1
    #          t    Re[<X1>]    Im[<X1>] Re[var(X1)] Im[var(X1)]    Re[<X2>]    Im[<X2>] Re[var(X2)] Im[var(X2)]
       0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
       1.000e+00   1.262e-02  -2.528e-03  -1.528e-04   6.379e-05   1.277e-02   1.017e-03  -1.621e-04  -2.598e-05
       2.000e+00   4.261e-02  -2.080e-02  -1.383e-03   1.773e-03   4.705e-02   4.481e-03  -2.193e-03  -4.216e-04
       3.000e+00   6.195e-02  -4.246e-02  -2.035e-03   5.261e-03   5.748e-02   4.816e-02  -9.844e-04  -5.536e-03
       4.000e+00   6.741e-02  -6.921e-02   2.448e-04   9.331e-03   2.629e-02   9.285e-02   7.930e-03  -4.883e-03
       5.000e+00  -4.060e-03  -9.878e-02   9.742e-03  -8.022e-04   3.130e-02   9.370e-02   7.800e-03  -5.866e-03
    ''').strip()


def test_write(tmpdir, traj1_coarse):
    p = tmpdir.join("traj.dat")
    traj1_coarse.write(filename=str(p))
    assert p.read() == traj1_coarse.to_str()
