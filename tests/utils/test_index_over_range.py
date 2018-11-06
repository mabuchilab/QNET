from qnet import IndexOverRange, IdxSym


def test_index_over_range_with_step():
    i = IdxSym('i')
    r = IndexOverRange(i, 1, 10, step=2)
    assert len(r) == 5
    assert list(r.range) == [1, 3, 5, 7, 9]
    for val in r.range:
        assert val in r
    assert 2 not in r
    assert 10 not in r


def test_index_over_range_bw():
    i = IdxSym('i')
    r = IndexOverRange(i, 10, 1, step=-2)
    assert len(r) == 5
    assert list(r.range) == [10, 8, 6, 4, 2]
    for val in r.range:
        assert val in r
    assert 5 not in r
    assert 1 not in r
