import os
from os.path import join
from glob import glob
from textwrap import dedent
from qnet.misc.trajectory_data import TrajectoryData, TrajectoryParserError
from distutils import dir_util
from collections import OrderedDict
import uuid
import hashlib
import numpy as np
import qnet.misc.testing_tools
from qnet.misc.testing_tools import qsd_traj
import pytest
from pytest import fixture
# built-in fixtures: tmpdir

datadir = fixture(qnet.misc.testing_tools.datadir)

TRAJ1_SEED = 103212

traj1               = fixture(qsd_traj(datadir, 'traj1', TRAJ1_SEED))
traj1_coarse        = fixture(qsd_traj(datadir, 'traj1_coarse', TRAJ1_SEED))
traj2_10            = fixture(qsd_traj(datadir, 'traj2_10', 18322321))
traj2_10_traj1_seed = fixture(qsd_traj(datadir, 'traj2_10', TRAJ1_SEED))
traj11_20           = fixture(qsd_traj(datadir, 'traj11_20', 38324389))
traj2_coarse        = fixture(qsd_traj(datadir, 'traj2_coarse', 28324389))

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


def test_rx():
    for op_name in ['X', r'\sqrt{A}', 'A_2^(1)', 'a^{\dagger} a']:
        assert TrajectoryData._rx['op_name'].match(op_name)

    line = '# QNET Trajectory Data ID f90b9290-35ff-3d94-a215-328fe2cc139c'
    assert TrajectoryData._rx['head_ID'].match(line)
    line = '# QNET ID f90b9290-35ff-3d94-a215-328fe2cc139c'
    assert TrajectoryData._rx['head_ID'].match(line) is None
    line = '# QNET Trajectory Data ID f90b9290-35ff-3d94-a215-328fe2cc13'
    assert TrajectoryData._rx['head_ID'].match(line) is None

    line = '# Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1'
    assert TrajectoryData._rx['record'].match(line)
    line = '# Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1 ["X1", "X2"]'
    assert TrajectoryData._rx['record'].match(line)

    line = '#          t    Re[<X1>]    Im[<X1>] Re[var(X1)] Im[var(X1)]    Re[<X2>]    Im[<X2>] Re[var(X2)] Im[var(X2)]'
    assert TrajectoryData._rx['header'].match(line)
    # It must be possible to extract the col_width from the extent of the match
    assert TrajectoryData._rx['header'].match(line).end() == 12


def test_init_validation():
    dt_ok = 0.1
    ID_ok = 'd9831647-f2e7-3793-8b24-7c49c5c101a7'
    data_ok = OrderedDict([
                ('X', (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))),
                ('Y', (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)))])
    TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                   data=data_ok)
    with pytest.raises(ValueError) as excinfo:
        TrajectoryData(ID='1232', dt=dt_ok, seed=None, n_trajectories=None,
                    data=data_ok)
    assert 'badly formed hexadecimal UUID string' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        TrajectoryData(ID=ID_ok, dt=0.0, seed=None, n_trajectories=None,
                    data=data_ok)
    assert 'dt must be a value >0' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        TrajectoryData(ID=ID_ok, dt="bla", seed=None, n_trajectories=None,
                    data=data_ok)
    assert 'could not convert string' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        TrajectoryData(ID=ID_ok, dt=None, seed=None, n_trajectories=None,
                    data=data_ok)
    assert 'dt must be a float with value >0' in str(excinfo.value)
    with pytest.raises(AttributeError) as excinfo:
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=None)
    with pytest.raises(ValueError) as excinfo:
        data = OrderedDict([
                    ('X', (np.zeros(3), np.zeros(3), np.zeros(3))),
                    ('Y', (np.zeros(3), np.zeros(3), np.zeros(3)))])
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=data)
    assert 'need more than 3 values to unpack' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        data = OrderedDict([ ('X', ("a", "b", "c", "d")),
                             ('Y', ("a", "b", "c", "d"))])
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=data)
    assert 'could not convert string to float' in str(excinfo.value)
    assert TrajectoryData.col_width == 25
    data = OrderedDict([('Y'*20, data_ok['X']),])
    traj = TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                          data=data)
    assert traj.col_width == 20 + TrajectoryData._col_padding
    with pytest.raises(ValueError) as excinfo:
        data = OrderedDict([('a\tb', data_ok['X']),])
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=data)
    assert 'contains invalid characters' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        data = OrderedDict([('A[[2]', data_ok['X']),])
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=data)
    assert 'contains unbalanced brackets' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        data = OrderedDict([('A]', data_ok['X']),])
        TrajectoryData(ID=ID_ok, dt=dt_ok, seed=None, n_trajectories=None,
                    data=data)
    assert 'contains unbalanced brackets' in str(excinfo.value)


def test_from_qsd_data(traj1, datadir):
    X1_file = join(datadir, 'traj1', 'X1.out')
    X2_file = join(datadir, 'traj1', 'X2.out')
    traj2 = TrajectoryData.from_qsd_data(
                OrderedDict([('X2', X2_file), ('X1', X1_file)]), TRAJ1_SEED)
    # traj1 and traj2 are the same except that the order of the operators is
    # reversed => files are read in a differnt order
    assert traj1.ID == traj2.ID

    traj = traj1
    md5sum = lambda f: hashlib.md5(open(f,'rb').read()).hexdigest()
    md5 = "".join(sorted([md5sum(X1_file), md5sum(X2_file)]))
    assert traj.ID == str(uuid.uuid3(TrajectoryData._uuid_namespace, md5))
    assert traj.dt == 0.1
    assert traj.nt == 51
    assert traj.shape == (51, 9)
    assert len(traj.record) == 1
    assert list(traj.operators) == ['X1', 'X2']
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
    # QNET Trajectory Data ID f90b9290-35ff-3d94-a215-328fe2cc139c
    # Record f90b9290-35ff-3d94-a215-328fe2cc139c (seed 103212): 1
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
    # QNET Trajectory Data ID f90b9290-35ff-3d94-a215-328fe2cc139c
    # Record f90b9290-35ff-3d94-a215-328fe2cc139c (seed 103212): 1
    #          t    Re[<X1>]    Im[<X1>] Re[var(X1)] Im[var(X1)]    Re[<X2>]    Im[<X2>] Re[var(X2)] Im[var(X2)]
       0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00
       1.000e+00   1.262e-02  -2.528e-03  -1.528e-04   6.379e-05   1.277e-02   1.017e-03  -1.621e-04  -2.598e-05
       2.000e+00   4.261e-02  -2.080e-02  -1.383e-03   1.773e-03   4.705e-02   4.481e-03  -2.193e-03  -4.216e-04
       3.000e+00   6.195e-02  -4.246e-02  -2.035e-03   5.261e-03   5.748e-02   4.816e-02  -9.844e-04  -5.536e-03
       4.000e+00   6.741e-02  -6.921e-02   2.448e-04   9.331e-03   2.629e-02   9.285e-02   7.930e-03  -4.883e-03
       5.000e+00  -4.060e-03  -9.878e-02   9.742e-03  -8.022e-04   3.130e-02   9.370e-02   7.800e-03  -5.866e-03
    ''').strip()
    with pytest.raises(ValueError) as excinfo:
        traj1_coarse.col_width = 11
        traj1_coarse.to_str()
    assert "must be shorter than max column width" in str(excinfo.value)

    traj1_coarse.col_width = 30
    assert traj1_coarse.to_str().strip() == dedent('''
    # QNET Trajectory Data ID f90b9290-35ff-3d94-a215-328fe2cc139c
    # Record f90b9290-35ff-3d94-a215-328fe2cc139c (seed 103212): 1
    #                            t                      Re[<X1>]                      Im[<X1>]                   Re[var(X1)]                   Im[var(X1)]                      Re[<X2>]                      Im[<X2>]                   Re[var(X2)]                   Im[var(X2)]
            0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00        0.0000000000000000e+00
            1.0000000000000000e+00        1.2617600000000000e-02       -2.5277300000000002e-03       -1.5281299999999999e-04        6.3787600000000006e-05        1.2773600000000000e-02        1.0170900000000000e-03       -1.6212900000000001e-04       -2.5983600000000000e-05
            2.0000000000000000e+00        4.2612299999999999e-02       -2.0798400000000002e-02       -1.3832300000000000e-03        1.7725400000000000e-03        4.7045200000000002e-02        4.4810300000000004e-03       -2.1931699999999999e-03       -4.2162200000000002e-04
            3.0000000000000000e+00        6.1951199999999998e-02       -4.2463000000000001e-02       -2.0348499999999999e-03        5.2612700000000002e-03        5.7477100000000003e-02        4.8158100000000002e-02       -9.8441599999999989e-04       -5.5359700000000003e-03
            4.0000000000000000e+00        6.7414600000000005e-02       -6.9206400000000001e-02        2.4479599999999998e-04        9.3310400000000005e-03        2.6293199999999999e-02        9.2851000000000003e-02        7.9299799999999997e-03       -4.8827000000000002e-03
            5.0000000000000000e+00       -4.0601099999999996e-03       -9.8784499999999997e-02        9.7418899999999996e-03       -8.0215199999999999e-04        3.1303499999999998e-02        9.3702499999999994e-02        7.8002499999999999e-03       -5.8664399999999997e-03
    ''').strip()


def test_write(tmpdir, traj1_coarse):
    p = tmpdir.join("traj.dat")
    traj1_coarse.write(filename=str(p))
    assert p.read() == traj1_coarse.to_str()


def deep_eq(t1, t2):
    for col in t1.table:
        assert np.all(t1.table[col] == t2.table[col])
        assert not (t1.table[col] is t2.table[col])
    for attr in t1.__dict__:
        if attr == 'table':
            continue # handled above
        assert t1.__dict__[attr] == t2.__dict__[attr]


def test_copy(traj1):
    traj2 = traj1.copy()
    deep_eq(traj1, traj2)


def test_extend(traj1, traj2_10, traj11_20, traj1_coarse, traj2_coarse,
        traj2_10_traj1_seed):
    assert list(traj1.operators) == ['X1', 'X2']
    assert list(traj2_10.operators) == ['A2', 'X1', 'X2']
    traj1_ID = traj1.ID

    with pytest.raises(ValueError) as excinfo:
        traj1.extend(traj1)
    assert "Repeated ID" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        traj1.extend(traj2_coarse)
    assert "Extending TrajectoryData does not match dt" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        traj1.extend(traj2_10_traj1_seed)
    assert "Repeated seed" in str(excinfo.value)

    traj1_copy = traj1.copy()
    traj1.extend(traj2_10)
    assert list(traj1.operators) == ['X1', 'X2', 'A2']
    assert traj1.n_trajectories('X1') == 10
    assert traj1.n_trajectories('X2') == 10
    assert traj1.n_trajectories('A2') == 9
    assert traj1_ID != traj1.ID
    assert traj1_ID in traj1.record
    assert traj2_10.ID in traj1.record
    header_lines = traj1.to_str(show_rows=0).splitlines()
    assert header_lines[1] == '# Record d9831647-f2e7-3793-8b24-7c49c5c101a7 (seed 103212): 1 ["X1", "X2"]'
    assert header_lines[2] == '# Record fd7fadea-34a4-3225-a2d1-fe7ba8863942 (seed 18322321): 9'
    for col in (traj1._operator_cols('X1') + traj1._operator_cols('X2')):
        diff = (traj1.table[col]
                - (traj1_copy.table[col] + 9.0 * traj2_10.table[col])/10.0)
        assert np.max(np.abs(diff)) <= 1.0e-14
    for col in traj1._operator_cols('A2'):
        assert np.all(traj1.table[col] == traj2_10.table[col])
    # test the syntactic sugar
    traj1_copy += traj2_10
    deep_eq(traj1_copy, traj1)
    with pytest.raises(TypeError) as excinfo:
        traj1_copy -= traj2_10
    assert "unsupported operand" in str(excinfo.value)

    traj1.extend(traj11_20)
    header_lines = traj1.to_str(show_rows=0).splitlines()
    assert header_lines[3] == '# Record 77c14243-9a40-3bdd-9efd-7698099078ee (seed 38324389): 10'
    assert traj1.n_trajectories('X1') == 20
    assert traj1.n_trajectories('X2') == 20
    assert traj1.n_trajectories('A2') == 19
    # test the syntactic sugar
    traj_combined = traj1_copy + traj11_20
    deep_eq(traj_combined, traj1)


def test_parse_header_line():
    line = '#          t    Re[<X1>]    Im[<X1>] Re[var(X1)] Im[var(X1)]    Re[<X2>]    Im[<X2>] Re[var(X2)] Im[var(X2)]'
    fields = TrajectoryData._parse_header_line(line)
    assert fields == ['#          t', '    Re[<X1>]', '    Im[<X1>]',
                      ' Re[var(X1)]', ' Im[var(X1)]', '    Re[<X2>]',
                      '    Im[<X2>]', ' Re[var(X2)]', ' Im[var(X2)]']
    fields = TrajectoryData._parse_header_line(line, strip=True)
    assert fields == ['#          t', 'Re[<X1>]', 'Im[<X1>]',
                      'Re[var(X1)]', 'Im[var(X1)]', 'Re[<X2>]',
                      'Im[<X2>]', 'Re[var(X2)]', 'Im[var(X2)]']
    line = '#tRe[<X1>]Im[<X1>]'
    fields = TrajectoryData._parse_header_line(line)
    assert fields == ['#t', 'Re[<X1>]', 'Im[<X1>]']
    line = '# t Re[<A[1]>] Im[<A[1]>]'
    fields = TrajectoryData._parse_header_line(line, strip=True)
    assert fields == ['# t', 'Re[<A[1]>]', 'Im[<A[1]>]']
    with pytest.raises(TrajectoryParserError) as excinfo:
        line = '# t Re[<A]>] Im[<A]>]'
        fields = TrajectoryData._parse_header_line(line, strip=True)
    assert "unbalanced brackets" in str(excinfo.value)
    with pytest.raises(TrajectoryParserError) as excinfo:
        line = '# t Re[<A[1]>] Im[<A[1]>] X1 X2'
        fields = TrajectoryData._parse_header_line(line, strip=True)
    assert "trailing characters" in str(excinfo.value)


def test_get_op_names_from_header_line(traj1):
    # use of traj1 is arbitrary, we just need some instance
    line = '#          t    Re[<X1>]    Im[<X1>] Re[var(X1)] Im[var(X1)]    Re[<X2>]    Im[<X2>] Re[var(X2)] Im[var(X2)]'
    ops = traj1._get_op_names_from_header_line(line)
    assert ops == ['X1', 'X2']

    op = 'X'*20
    line = '#  t '+" ".join(TrajectoryData._operator_cols(op))
    ops = traj1._get_op_names_from_header_line(line)
    assert ops == [op, ]

    with pytest.raises(TrajectoryParserError) as excinfo:
        line = '#          t    Re[<X1>]    Im[<X1>] Re[var(X1)]'
        ops = traj1._get_op_names_from_header_line(line)
    assert "Unexpected number of columns" in str(excinfo.value)
    with pytest.raises(TrajectoryParserError) as excinfo:
        line = '#  t    Re[<X\t2>]'
        ops = traj1._get_op_names_from_header_line(line)
    assert "contains invalid characters" in str(excinfo.value)


def test_read(datadir, tmpdir, traj1, traj2_10, traj11_20):
    filename = join(datadir, 'read_input_files', 'good.dat')
    traj = TrajectoryData.read(filename)
    file_str = open(filename).read()
    assert str(traj) == file_str.rstrip()

    filename = str(tmpdir.join("traj1.dat"))
    traj1.write(filename)
    traj_r = TrajectoryData.read(filename)
    assert str(traj_r) == str(traj1)

    filename = str(tmpdir.join("traj1-20.dat"))
    traj = traj11_20 + traj2_10 + traj1
    traj.col_width = 12
    traj.write(filename)
    traj_r = TrajectoryData.read(filename)
    assert str(traj_r) == str(traj)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'no_header.dat')
        traj = TrajectoryData.read(filename)
    assert "does not define an ID" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'no_record.dat')
        traj = TrajectoryData.read(filename)
    assert "does not contain a record" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'no_col_labels.dat')
        traj = TrajectoryData.read(filename)
    assert "does not contain a header" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'no_data.dat')
        traj = TrajectoryData.read(filename)
    assert "contains no data" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'empty.dat')
        traj = TrajectoryData.read(filename)
    assert "does not define an ID" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'wrong_col_labels.dat')
        traj = TrajectoryData.read(filename)
    assert "Malformed header" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'wrong_col_labels2.dat')
        traj = TrajectoryData.read(filename)
    assert "Invalid header line" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'no_tgrid.dat')
        traj = TrajectoryData.read(filename)
    assert "missing time grid" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'missing_im.dat')
        traj = TrajectoryData.read(filename)
    assert "Unexpected number of columns" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'header_data_mismatch.dat')
        traj = TrajectoryData.read(filename)
    assert "number of data columns differs from the number indicated in "\
           "the header" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'single_row.dat')
        traj = TrajectoryData.read(filename)
    assert "Too few rows" in str(exc_info.value)

    with pytest.raises(TrajectoryParserError) as exc_info:
        filename = join(datadir, 'read_input_files', 'truncated.dat')
        traj = TrajectoryData.read(filename)
    assert "Wrong number of columns at line 3" in str(exc_info.value)

