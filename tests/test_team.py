from matchtracker.team import canonicalize_labels, majority_vote_teams, resolve_team_labels
from matchtracker.track import TrackRecord


def test_majority_vote_teams_picks_most_common():
    votes = {1: [0, 0, 1, 0], 2: [1, 1, 0]}
    result = majority_vote_teams(votes)
    assert result == {1: 0, 2: 1}


def test_majority_vote_teams_skips_empty_votes():
    assert majority_vote_teams({1: []}) == {}


def test_canonicalize_labels_smaller_mean_x_is_team_a():
    cluster_to_tracks = {0: [1, 2], 1: [3, 4]}
    track_mean_x = {1: 80.0, 2: 90.0, 3: 10.0, 4: 20.0}
    labels = canonicalize_labels(cluster_to_tracks, track_mean_x)
    # cluster 1 has smaller mean x (15) -> "A"; cluster 0 (85) -> "B"
    assert labels == {1: "A", 0: "B"}


def _rec(track_id, x1, cls="player"):
    return TrackRecord(
        frame_idx=0,
        t_s=0.0,
        track_id=track_id,
        cls=cls,
        xyxy=(x1, 0.0, x1 + 5, 5.0),
        confidence=0.9,
    )


def test_resolve_team_labels_end_to_end():
    records = [
        _rec(1, x1=10.0),
        _rec(2, x1=20.0),
        _rec(3, x1=100.0),
        _rec(4, x1=110.0),
        _rec(99, x1=50.0, cls="referee"),
    ]
    # tracks 1,2 -> cluster 0 (left side); tracks 3,4 -> cluster 1 (right side)
    predictions = [(1, 0), (1, 0), (2, 0), (3, 1), (4, 1), (4, 1)]
    labels = resolve_team_labels(predictions, records)
    assert labels[1] == "A"
    assert labels[2] == "A"
    assert labels[3] == "B"
    assert labels[4] == "B"
    assert 99 not in labels
