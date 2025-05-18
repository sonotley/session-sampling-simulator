import session_sampling_simulator.session_simulator as s
import pytest
import numpy as np
import warnings


@pytest.mark.parametrize(
    "spread, dist, all_equal",
    [
        [0, "lognormal", True],
        [0, "uniform", True],
        [0, "exponential", False],
        [1, "uniform", False],
        [1, "lognormal", False],
    ],
)
def test_runner(spread, dist, all_equal):
    query = s.Query(1, 20, 100, {1: 1}, spread, dist)
    runner = s.Runner(query, 100)
    assert (runner._durations == pytest.approx([20] * 100)) == all_equal


array_content = (0, 2, 2, 2, 4, 1, 1, 1, 0, 0, 0, 0, 0, 234, 544, 544, 2, 2)


@pytest.mark.parametrize(
    "arr",
    [
        np.asarray(array_content, dtype=np.int64),
        np.asarray(array_content, dtype=np.uint16),
    ],
)
def test_find_contiguous_regions(arr):
    # want to catch arithmetic warnings as errors here
    warnings.filterwarnings("error")
    regions = s.find_contiguous_regions(arr)
    warnings.resetwarnings
    assert regions == [
        (0, 0, 0),
        (1, 3, 2),
        (4, 4, 4),
        (5, 7, 1),
        (8, 12, 0),
        (13, 13, 234),
        (14, 15, 544),
        (16, 17, 2),
    ]
