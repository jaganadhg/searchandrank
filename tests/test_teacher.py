import json
from literank.teacher import add_teacher_scores, cache_teacher_scores


class StubTeacher:
    def __init__(self):
        self.calls = 0

    def score(self, queries, passages, batch_size=32):
        self.calls += 1
        return [float(len(p)) for p in passages]


def _triplets():
    return [{"query": "q", "pos": "abcd", "neg": "ab", "t_pos": 0.0, "t_neg": 0.0}]


def test_add_teacher_scores_fills_margins():
    out = add_teacher_scores(_triplets(), StubTeacher())
    assert out[0]["t_pos"] == 4.0 and out[0]["t_neg"] == 2.0


def test_cache_round_trips_and_skips_recompute(tmp_path):
    teacher = StubTeacher()
    path = tmp_path / "scores.json"
    cache_teacher_scores(_triplets(), teacher, str(path))
    assert path.exists()
    first_calls = teacher.calls
    cache_teacher_scores(_triplets(), teacher, str(path))   # should reuse file
    assert teacher.calls == first_calls
    data = json.loads(path.read_text())
    assert data[0]["t_pos"] == 4.0
