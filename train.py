import argparse
import configparser
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

def _read_ctx(config_path):
    parser = configparser.ConfigParser()
    path = config_path if os.path.isabs(config_path) else os.path.join(ROOT, config_path)
    if not parser.read(path):
        raise SystemExit("读不到配置: %s" % path)
    return parser.get("Training", "ctx", fallback="0").strip()

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--ctx", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--perturb_delta", type=float, default=0.1)
    parser.add_argument("--perturb_sigma", type=float, default=0.01)
    args = parser.parse_args()
    if args.multi:
        if not args.configs:
            raise SystemExit("多数据集训练没有默认数据集，请传入 --configs。")
        if not args.ctx:
            raise SystemExit("多数据集训练请显式传入 --ctx，例如 --ctx 4,5。")
        if args.config:
            raise SystemExit("多数据集只用 --configs，不要同时传 --config。")
    else:
        if args.configs:
            raise SystemExit("当前是单数据集模式。要联合训练请加上 --multi。")
        if not args.config:
            raise SystemExit("请指定 --config")
    return args

def _visible_gpus(ctx):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    from stkllm.gpu import apply_cuda_visible_devices_from_ctx

    apply_cuda_visible_devices_from_ctx(ctx)

def main():
    args = _parse()
    ctx = args.ctx or _read_ctx(args.config)
    _visible_gpus(ctx)
    from stkllm.engine import train_multi, train_single
    from stkllm.logutil import open_log

    if args.multi:
        names = [os.path.splitext(os.path.basename(path))[0] for path in args.configs]
        _, restore = open_log("multi_" + "_".join(names))
        try:
            train_multi(
                args.configs,
                ctx,
                epochs_override=args.epochs,
                consistency_weight=args.consistency_weight,
                perturb_delta=args.perturb_delta,
                perturb_sigma=args.perturb_sigma,
            )
        finally:
            restore()
        return
    name = os.path.splitext(os.path.basename(args.config))[0]
    _, restore = open_log(name)
    try:
        train_single(args.config, ctx_override=args.ctx, epochs_override=args.epochs)
    finally:
        restore()

if __name__ == "__main__":
    main()
