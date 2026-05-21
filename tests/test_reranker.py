from convmemory.reranker import candidate_local_windows, sliding_windows


def test_sliding_windows_short_sequence():
    assert sliding_windows(num_items=3, window_size=5, stride=1) == [[0, 1, 2]]


def test_sliding_windows_normal():
    windows = sliding_windows(num_items=8, window_size=5, stride=1)

    assert [0, 1, 2, 3, 4] in windows
    assert [3, 4, 5, 6, 7] in windows


def test_candidate_local_windows_dedup():
    windows = candidate_local_windows(
        num_items=10,
        candidate_indices=[2, 2, 3],
        window_size=5,
    )

    assert windows == [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
