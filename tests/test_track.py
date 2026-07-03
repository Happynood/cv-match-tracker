from matchtracker.track import TrackRecord, filter_short_tracks, gap_fill


def _rec(frame_idx, track_id, x1=0.0):
    return TrackRecord(
        frame_idx=frame_idx,
        t_s=frame_idx * 0.1,
        track_id=track_id,
        cls="player",
        xyxy=(x1, 0.0, x1 + 10.0, 10.0),
        confidence=0.9,
    )


def test_filter_short_tracks_drops_below_threshold():
    records = [_rec(i, track_id=1) for i in range(5)] + [_rec(i, track_id=2) for i in range(20)]
    filtered = filter_short_tracks(records, min_track_len=10)
    remaining_ids = {r.track_id for r in filtered}
    assert remaining_ids == {2}
    assert len(filtered) == 20


def test_gap_fill_interpolates_within_max_gap():
    records = [_rec(0, track_id=1, x1=0.0), _rec(3, track_id=1, x1=30.0)]
    filled = gap_fill(records, max_gap=5)
    frame_indices = sorted(r.frame_idx for r in filled)
    assert frame_indices == [0, 1, 2, 3]

    by_frame = {r.frame_idx: r for r in filled}
    assert by_frame[1].xyxy[0] == 10.0
    assert by_frame[2].xyxy[0] == 20.0
    assert by_frame[1].interpolated is True
    assert by_frame[0].interpolated is False


def test_gap_fill_leaves_large_gaps_unfilled():
    records = [_rec(0, track_id=1), _rec(100, track_id=1, x1=1000.0)]
    filled = gap_fill(records, max_gap=5)
    frame_indices = sorted(r.frame_idx for r in filled)
    assert frame_indices == [0, 100]
