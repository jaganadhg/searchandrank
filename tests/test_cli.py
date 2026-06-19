from literank.cli import build_parser


def test_train_subcommand_parses_flags():
    p = build_parser()
    args = p.parse_args(["train", "--scorer", "maxsim", "--max-steps", "50",
                         "--proj-dim", "128", "--resume", "ck.pt"])
    assert args.command == "train"
    assert args.scorer == "maxsim"
    assert args.max_steps == 50
    assert args.proj_dim == 128
    assert args.resume == "ck.pt"


def test_train_subcommand_parses_early_stopping_flags():
    p = build_parser()
    args = p.parse_args(["train", "--eval-every", "500", "--patience", "4"])
    assert args.eval_every == 500
    assert args.patience == 4


def test_subcommands_exist():
    p = build_parser()
    for cmd in ["train", "encode", "rerank", "eval"]:
        assert p.parse_args([cmd]).command == cmd
