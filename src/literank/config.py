from dataclasses import dataclass


@dataclass
class ModelConfig:
    encoder_name: str = "distilbert-base-uncased"
    embed_dim: int = 768        # d
    proj_dim: int = 768         # d' (< embed_dim => Small LITE)
    query_len: int = 30         # L1
    doc_len: int = 200          # L2
    mlp1_hidden: int = 360      # m1 (over query axis L1)
    mlp2_hidden: int = 2400     # m2 (over doc axis L2)
    activation: str = "relu"    # "relu" | "sigmoid" | "gelu"
    l2_normalize: bool = True
    scorer: str = "lite"        # "lite" | "maxsim"
    freeze_encoder: bool = False


@dataclass
class TrainConfig:
    lr: float = 2.8e-5
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 20_000
    kl_weight: float = 0.0
    amp: bool = True
    seed: int = 42
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    keep_last: int = 3           # max ckpt_step*.pt files to retain (older pruned); 0 = keep all
    log_every: int = 50          # steps between progress (step/loss) log lines
    eval_every: int = 0          # >0 enables periodic dev-MRR eval + early stopping
    patience: int = 3            # consecutive evals without improvement before stopping
    val_fraction: float = 0.1    # held-out fraction of triplets used for eval
    min_delta: float = 0.0       # minimum MRR gain to count as improvement
    eval_k: int = 10             # k for MRR@k
    best_ckpt_name: str = "best.pt"


@dataclass
class DataConfig:
    dataset_name: str = "microsoft/ms_marco"
    dataset_config: str = "v2.1"
    subset_size: int = 100_000
    teacher_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    teacher_max_len: int = 256
    num_dev_queries: int = 1000
