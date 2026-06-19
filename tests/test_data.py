import torch
from literank.data import TripletDataset, collate_triplets


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
