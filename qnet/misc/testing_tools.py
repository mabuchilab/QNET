"""Collection of routines needed for testing. This includes proto-fixtures,
i.e. routines that should be imported and then turned into a fixture with the
pytest.fixture decorator.

See <https://pytest.org/latest/fixture.html>
"""
import os
from glob import glob
from collections import OrderedDict
from distutils import dir_util
from qnet.misc.trajectory_data import TrajectoryData


def datadir(tmpdir, request):
    '''Proto-fixture responsible for searching a folder with the same name of
    test module and, if available, moving all contents to a temporary directory
    so tests can use them freely.

    In any test, import the datadir routine and turn it into a fixture:
    >>> import pytest
    >>> import qnet.misc.testing_tools
    >>> datadir = pytest.fixture(qnet.misc.testing_tools.datadir)
    '''
    # http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return str(tmpdir)


def qsd_traj(datadir, folder, seed):
    """Return a proto-fixture that returns a TrajectoryData instance based on
    all the *.out file in the given folder (relative to the test datadir), and
    with the given seed.

    The returned function should be turned into a fixture:
    >>> import pytest
    >>> import qnet.misc.testing_tools
    >>> from qnet.misc.testing_tools import qsd_traj
    >>> datadir = pytest.fixture(qnet.misc.testing_tools.datadir)
    >>> traj1 = pytest.fixture(qsd_traj(datadir, 'traj1', 102121))
    """
    def proto_fixture(datadir):
        operators = OrderedDict()
        datafiles = sorted(glob(os.path.join(datadir, folder, '*.out')))
        assert len(datafiles) >0, "No files *.out in %s"%folder
        for file in datafiles:
            op_name = os.path.splitext(os.path.split(file)[1])[0]
            operators[op_name] = file
        return TrajectoryData.from_qsd_data(operators, seed=seed)
    import pytest # local import, so that qnet can be installed w/o pytest
    return proto_fixture


def fake_traj(traj_template, ID, seed):
    """Return a new trajectory that has the same data as traj_template, but a
    different ID and seed. Assumes that traj_template only has a single record
    (i.e., it was created from QSD data)"""
    assert len(traj_template.record) == 1
    orig_id = traj_template.ID
    orig_seed, ntraj, op = traj_template._record[orig_id]
    assert ID != orig_id
    assert seed != orig_seed

    traj = traj_template.copy()
    traj._ID = ID
    traj._record[ID] = (seed, ntraj, op)
    del traj._record[orig_id]
    assert (traj + traj_template).ID != traj_template.ID
    return traj

