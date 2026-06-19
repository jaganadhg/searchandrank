import random
import torch
from literank.data import TripletDataset, collate_triplets, _record_to_triplet, build_eval_set


def _triplets():
    return [
        {"query": "q1", "pos": "p1", "neg": "n1", "t_pos": 2.0, "t_neg": 0.0},
        {"query": "q2", "pos": "p2", "neg": "n2", "t_pos": 1.5, "t_neg": -1.0},
    ]


def test_dataset_len_and_getitem():
    ds = TripletDataset(_triplets())
    assert len(ds) == 2
    assert ds[0]["query"] == "q1"


def test_collate_shapes_and_types():
    batch = collate_triplets(_triplets())
    assert batch["query"] == ["q1", "q2"]
    assert batch["neg"] == ["n1", "n2"]
    assert torch.allclose(batch["t_pos"], torch.tensor([2.0, 1.5]))
    assert batch["t_neg"].dtype == torch.float32


def test_record_to_triplet_valid():
    """Test that a record with one positive and one negative produces a triplet."""
    record = {
        "query": "test query",
        "passages": {
            "passage_text": ["pos_passage", "neg_passage"],
            "is_selected": [1, 0],
        },
    }
    rng = random.Random(42)
    triplet = _record_to_triplet(record, rng)
    assert triplet is not None
    assert triplet["query"] == "test query"
    assert triplet["pos"] == "pos_passage"
    assert triplet["neg"] == "neg_passage"
    assert triplet["t_pos"] == 0.0
    assert triplet["t_neg"] == 0.0


def test_record_to_triplet_no_positive():
    """Test that a record with no positive (all is_selected==0) returns None."""
    record = {
        "query": "test query",
        "passages": {
            "passage_text": ["neg1", "neg2"],
            "is_selected": [0, 0],
        },
    }
    rng = random.Random(42)
    triplet = _record_to_triplet(record, rng)
    assert triplet is None


def test_record_to_triplet_no_negative():
    """Test that a record with no negative (all is_selected==1) returns None."""
    record = {
        "query": "test query",
        "passages": {
            "passage_text": ["pos1", "pos2"],
            "is_selected": [1, 1],
        },
    }
    rng = random.Random(42)
    triplet = _record_to_triplet(record, rng)
    assert triplet is None


def test_build_eval_set_groups_by_query():
    triplets = [
        {"query": "q1", "pos": "p1", "neg": "n1", "t_pos": 2.0, "t_neg": 0.0},
        {"query": "q2", "pos": "p2", "neg": "n2", "t_pos": 1.5, "t_neg": -1.0},
    ]
    eval_set = build_eval_set(triplets)
    assert len(eval_set) == 2
    q1 = next(e for e in eval_set if e["query"] == "q1")
    assert q1["docs"] == ["p1", "n1"]
    assert q1["labels"] == [1, 0]


def test_build_eval_set_positive_wins_tie():
    # "n1" appears as a negative in the first triplet and as a positive in the
    # second triplet for the same query "q1" -> label must resolve to 1.
    triplets = [
        {"query": "q1", "pos": "p1", "neg": "n1", "t_pos": 2.0, "t_neg": 0.0},
        {"query": "q1", "pos": "n1", "neg": "p1", "t_pos": 1.0, "t_neg": 0.0},
    ]
    eval_set = build_eval_set(triplets)
    assert len(eval_set) == 1
    q1 = eval_set[0]
    labels_by_doc = dict(zip(q1["docs"], q1["labels"]))
    assert labels_by_doc["n1"] == 1
    assert labels_by_doc["p1"] == 1


def test_record_to_triplet_determinism():
    """Test that same record and seed produce identical pos/neg."""
    record = {
        "query": "test query",
        "passages": {
            "passage_text": ["pos1", "pos2", "neg1", "neg2"],
            "is_selected": [1, 1, 0, 0],
        },
    }
    rng1 = random.Random(0)
    triplet1 = _record_to_triplet(record, rng1)

    rng2 = random.Random(0)
    triplet2 = _record_to_triplet(record, rng2)

    assert triplet1["pos"] == triplet2["pos"]
    assert triplet1["neg"] == triplet2["neg"]
