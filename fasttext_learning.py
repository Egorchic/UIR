import argparse
import os
import time
import json
import fasttext


def parse_args():
    parser = argparse.ArgumentParser(description="Train fastText model for Bethesda corpus")
    parser.add_argument("--input", type=str, default="artifacts/corpus.txt", help="Path to corpus")
    parser.add_argument("--model", type=str, choices=["skipgram", "cbow"], default="skipgram")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--ws", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--minCount", type=int, default=2)
    parser.add_argument("--thread", type=int, default=8)
    parser.add_argument("--output", type=str, required=True, help="Output .bin model path")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    start_time = time.time()

    ft_model = fasttext.train_unsupervised(
        input=args.input,
        model=args.model,
        dim=args.dim,
        minn=args.minn,
        maxn=args.maxn,
        ws=args.ws,
        epoch=args.epoch,
        lr=args.lr,
        minCount=args.minCount,
        thread=args.thread
    )

    ft_model.save_model(args.output)

    elapsed = time.time() - start_time
    vocab_size = len(ft_model.get_words())
    model_size_bytes = os.path.getsize(args.output)

    meta = {
        "input": args.input,
        "model": args.model,
        "dim": args.dim,
        "minn": args.minn,
        "maxn": args.maxn,
        "ws": args.ws,
        "epoch": args.epoch,
        "lr": args.lr,
        "minCount": args.minCount,
        "thread": args.thread,
        "output": args.output,
        "train_time_sec": round(elapsed, 4),
        "vocab_size": vocab_size,
        "model_size_bytes": model_size_bytes
    }

    meta_path = os.path.splitext(args.output)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.output}")
    print(f"vocab size: {vocab_size}")
    print(f"train time sec: {elapsed:.4f}")
    print(f"meta saved: {meta_path}")


if __name__ == "__main__":
    main()