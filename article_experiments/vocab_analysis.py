from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import (
    DEFAULT_BETHESDA_CORPUS,
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_bethesda_corpus,
    load_eutirads_dataset,
)
from .preprocessing import tokenize_lexical


DEFAULT_SPLITS = PROJECT_ROOT / "article_results" / "splits.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "article_results" / "vocab"

def build_counter(
    texts: list[str],
) -> tuple[Counter, list[int]]:

    counter = Counter()
    lengths = []

    for text in texts:
        tokens = tokenize_lexical(text)

        counter.update(tokens)
        lengths.append(len(tokens))

    return counter, lengths


def calculate_overlap(
    source_counter: Counter,
    target_counter: Counter,
) -> dict:

    source_vocab = set(source_counter)
    target_vocab = set(target_counter)

    shared = source_vocab & target_vocab

    target_oov = target_vocab - source_vocab

    n_target_tokens = sum(target_counter.values())

    n_target_oov_tokens = sum(
        target_counter[token]
        for token in target_oov
    )

    union = source_vocab | target_vocab

    return {
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "shared_vocab_size": len(shared),

        "source_only_vocab_size": len(
            source_vocab - target_vocab
        ),

        "target_only_vocab_size": len(target_oov),

        "jaccard_vocab": (
            len(shared) / len(union)
            if union
            else 0.0
        ),

        # Какая доля уникальных слов EU-TIRADS
        # встречалась в Bethesda.
        "target_type_coverage_by_source": (
            len(shared) / len(target_vocab)
            if target_vocab
            else 0.0
        ),

        "target_type_oov_rate": (
            len(target_oov) / len(target_vocab)
            if target_vocab
            else 0.0
        ),

        # Более важная характеристика:
        # доля реальных употреблений EU-TIRADS,
        # отсутствующих в словаре Bethesda.
        "target_token_oov_rate": (
            n_target_oov_tokens / n_target_tokens
            if n_target_tokens
            else 0.0
        ),

        "target_token_coverage_by_source": (
            1.0 - (
                n_target_oov_tokens / n_target_tokens
                if n_target_tokens
                else 0.0
            )
        ),
    }


def save_top_oov(
    source_counter: Counter,
    target_counter: Counter,
    output_path: Path,
    top_n: int = 200,
) -> None:

    source_vocab = set(source_counter)

    rows = []

    total_target_tokens = sum(
        target_counter.values()
    )

    for token, count in target_counter.most_common():
        if token in source_vocab:
            continue

        rows.append(
            {
                "token": token,
                "target_count": count,
                "target_token_share": (
                    count / total_target_tokens
                    if total_target_tokens
                    else 0.0
                ),
            }
        )

        if len(rows) >= top_n:
            break

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def calculate_oov_by_class(
    df: pd.DataFrame,
    source_counter: Counter,
) -> pd.DataFrame:

    rows = []

    for label, part in df.groupby("label"):
        target_counter, lengths = build_counter(
            part["text_model"].tolist()
        )

        overlap = calculate_overlap(
            source_counter,
            target_counter,
        )

        rows.append(
            {
                "label": int(label),
                "n_objects": len(part),
                "n_tokens": sum(lengths),
                "target_vocab_size": overlap[
                    "target_vocab_size"
                ],
                "type_oov_rate": overlap[
                    "target_type_oov_rate"
                ],
                "token_oov_rate": overlap[
                    "target_token_oov_rate"
                ],
            }
        )

    return pd.DataFrame(rows)


def calculate_oov_by_split(
    df: pd.DataFrame,
    splits: pd.DataFrame,
    source_counter: Counter,
) -> pd.DataFrame:

    rows = []

    for split_id, split in splits.groupby("split_id"):

        test = split[
            split["subset"] == "test"
        ]

        test_ids = set(test["object_id"])

        target = df[
            df["object_id"].isin(test_ids)
        ]

        if len(target) != len(test):
            raise RuntimeError(
                f"Split {split_id}: "
                "не удалось сопоставить все test object_id."
            )

        target_counter, lengths = build_counter(
            target["text_model"].tolist()
        )

        overlap = calculate_overlap(
            source_counter,
            target_counter,
        )

        rows.append(
            {
                "split_id": int(split_id),
                "seed": int(test["seed"].iloc[0]),
                "n_test": len(target),
                "n_tokens": sum(lengths),
                "target_vocab_size": overlap[
                    "target_vocab_size"
                ],
                "type_oov_rate": overlap[
                    "target_type_oov_rate"
                ],
                "token_oov_rate": overlap[
                    "target_token_oov_rate"
                ],
                "token_coverage": overlap[
                    "target_token_coverage_by_source"
                ],
            }
        )

    return pd.DataFrame(rows)


def calculate_model_dictionary_overlap(
    model_path: Path,
    target_counter: Counter,
) -> dict:
    import fasttext

    model = fasttext.load_model(str(model_path))

    model_vocab = set(
        token.lower().replace("ё", "е")
        for token in model.get_words()
    )

    target_vocab = set(target_counter)

    shared = model_vocab & target_vocab
    oov = target_vocab - model_vocab

    n_target_tokens = sum(target_counter.values())

    n_oov_tokens = sum(
        target_counter[token]
        for token in oov
    )

    return {
        "model_vocab_size": len(model_vocab),
        "model_target_shared_vocab_size": len(shared),

        "model_target_type_coverage": (
            len(shared) / len(target_vocab)
            if target_vocab else 0.0
        ),

        "model_target_type_oov_rate": (
            len(oov) / len(target_vocab)
            if target_vocab else 0.0
        ),

        "model_target_token_oov_rate": (
            n_oov_tokens / n_target_tokens
            if n_target_tokens else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eutirads-data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--bethesda-corpus",
        type=Path,
        default=DEFAULT_BETHESDA_CORPUS,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--bethesda-model",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eu_df = load_eutirads_dataset(
        args.eutirads_data
    )

    bethesda_texts = load_bethesda_corpus(
        args.bethesda_corpus
    )

    eu_texts = eu_df[
        "text_model"
    ].tolist()

    source_counter, source_lengths = build_counter(
        bethesda_texts
    )

    target_counter, target_lengths = build_counter(
        eu_texts
    )

    overlap = calculate_overlap(
        source_counter,
        target_counter,
    )

    if args.bethesda_model is not None:
        model_overlap = calculate_model_dictionary_overlap(
            args.bethesda_model,
            target_counter,
        )

        overlap.update(model_overlap)

    overlap.update(
        {
            "source_n_texts": len(bethesda_texts),
            "target_n_texts": len(eu_texts),

            "source_n_tokens": sum(source_lengths),
            "target_n_tokens": sum(target_lengths),

            "source_avg_tokens_per_text": (
                sum(source_lengths)
                / len(source_lengths)
            ),

            "target_avg_tokens_per_text": (
                sum(target_lengths)
                / len(target_lengths)
            ),
        }
    )

    # JSON
    with (
        args.output_dir
        / "vocab_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            overlap,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # CSV — удобно потом вставлять в таблицу статьи.
    pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in overlap.items()
        ]
    ).to_csv(
        args.output_dir / "vocab_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_top_oov(
        source_counter,
        target_counter,
        args.output_dir
        / "top_eutirads_oov_tokens.csv",
    )

    by_class = calculate_oov_by_class(
        eu_df,
        source_counter,
    )

    by_class.to_csv(
        args.output_dir / "oov_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.splits.exists():
        splits = pd.read_csv(
            args.splits
        )

        by_split = calculate_oov_by_split(
            eu_df,
            splits,
            source_counter,
        )

        by_split.to_csv(
            args.output_dir / "oov_by_split.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("=== Bethesda -> EU-TIRADS vocabulary analysis ===")
    print()

    print(
        f"Bethesda vocabulary: "
        f"{overlap['source_vocab_size']}"
    )

    print(
        f"EU-TIRADS vocabulary: "
        f"{overlap['target_vocab_size']}"
    )

    print(
        f"Intersection: "
        f"{overlap['shared_vocab_size']}"
    )

    print(
        f"Jaccard: "
        f"{overlap['jaccard_vocab']:.4f}"
    )

    print(
        "EU-TIRADS type coverage by Bethesda: "
        f"{overlap['target_type_coverage_by_source']:.4f}"
    )

    print(
        "EU-TIRADS type OOV rate: "
        f"{overlap['target_type_oov_rate']:.4f}"
    )

    print(
        "EU-TIRADS token OOV rate: "
        f"{overlap['target_token_oov_rate']:.4f}"
    )


if __name__ == "__main__":
    main()