from literank.config import ModelConfig, TrainConfig, DataConfig


def test_model_config_paper_defaults():
    cfg = ModelConfig()
    assert cfg.encoder_name == "distilbert-base-uncased"
    assert (cfg.query_len, cfg.doc_len) == (30, 200)
    assert (cfg.embed_dim, cfg.proj_dim) == (768, 768)
    assert (cfg.mlp1_hidden, cfg.mlp2_hidden) == (360, 2400)
    assert cfg.activation == "relu"
    assert cfg.scorer == "lite"
    assert cfg.l2_normalize is True
    assert cfg.freeze_encoder is False


def test_train_and_data_defaults():
    t = TrainConfig()
    assert t.lr == 2.8e-5
    assert t.amp is True
    assert t.checkpoint_every == 1000
    d = DataConfig()
    assert d.teacher_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert d.subset_size == 100_000


def test_train_config_early_stopping_defaults():
    t = TrainConfig()
    assert t.eval_every == 0
    assert t.patience == 3
    assert t.val_fraction == 0.1
    assert t.min_delta == 0.0
    assert t.eval_k == 10
    assert t.best_ckpt_name == "best.pt"
