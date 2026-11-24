import argparse
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ctx", default=None)
    return parser.parse_args()

def main():
    args = _parse()
    import configparser

    parser = configparser.ConfigParser()
    path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    if not parser.read(path):
        raise SystemExit("读不到配置: %s" % path)
    ctx = args.ctx or parser.get("Training", "ctx", fallback="0").strip()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        from stkllm.gpu import apply_cuda_visible_devices_from_ctx

        apply_cuda_visible_devices_from_ctx(ctx)
    from stkllm.engine import test_single
    from stkllm.logutil import open_log

    name = os.path.splitext(os.path.basename(args.config))[0]
    _, restore = open_log(name + "_test")
    try:
        test_single(args.config, args.ckpt, ctx_override=args.ctx)
    finally:
        restore()

if __name__ == "__main__":
    main()
