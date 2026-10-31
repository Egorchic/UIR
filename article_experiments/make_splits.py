from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .data import (
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    dataframe_fingerprint,
    load_eutirads_dataset,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "article_results" / "splits.csv"


def make_splits(
    df: pd.DataFrame,
    n_splits: int = 10,
    first_seed: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:

    all_rows = []

    indices = list(range(len(df)))
    labels = df["label"].to_numpy()

    for split_id in range(n_splits):
        seed = first_seed + split_id

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        train_ids = set(train_idx)
        test_ids = set(test_idx)

        if train_ids & test_ids:
            raise RuntimeError(
                f"Split {split_id}: train и test пересекаются."
            )

        if train_ids | test_ids != set(indices):
            raise RuntimeError(
                f"Split {split_id}: потеряны объекты."
            )

        for subset, subset_idx in (
            ("train", train_idx),
            ("test", test_idx),
        ):
            part = df.iloc[subset_idx]

            for row in part.itertuples(index=False):
                all_rows.append(
                    {
                        "split_id": split_id,
                        "seed": seed,
                        "object_id": row.object_id,
                        "label": int(row.label),
                        "subset": subset,
                    }
                )

    result = pd.DataFrame(all_rows)

    return result


def validate_splits(
    splits: pd.DataFrame,
    df: pd.DataFrame,
) -> None:

    all_object_ids = set(df["object_id"])

    for split_id, part in splits.groupby("split_id"):
        train = part[part["subset"] == "train"]
        test = part[part["subset"] == "test"]

        train_ids = set(train["object_id"])
        test_ids = set(test["object_id"])

        if train_ids & test_ids:
            raise RuntimeError(
                f"Split {split_id}: обнаружено пересечение train/test."
            )

        if train_ids | test_ids != all_object_ids:
            raise RuntimeError(
                f"Split {split_id}: набор объектов неполный."
            )

        labels_all = set(df["label"])
        labels_train = set(train["label"])
        labels_test = set(test["label"])

        if labels_train != labels_all:
            raise RuntimeError(
                f"Split {split_id}: не все классы присутствуют в train."
            )

        if labels_test != labels_all:
            raise RuntimeError(
                f"Split {split_id}: не все классы присутствуют в test."
            )


def save_class_distribution(
    splits: pd.DataFrame,
    path: Path,
) -> None:

    distribution = (
        splits
        .groupby(
            ["split_id", "seed", "subset", "label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_objects"})
    )

    distribution.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--first-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    df = load_eutirads_dataset(args.data)

    splits = make_splits(
        df=df,
        n_splits=args.n_splits,
        first_seed=args.first_seed,
        test_size=args.test_size,
    )

    validate_splits(splits, df)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    distribution_path = (
        args.output.parent
        / "split_class_distribution.csv"
    )

    save_class_distribution(
        splits,
        distribution_path,
    )

    manifest_path = (
        args.output.parent
        / "objects_manifest.csv"
    )

    manifest_columns = [
        column
        for column in ["object_id", "source_row", "label"]
        if column in df.columns
    ]

    df[manifest_columns].to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "n_objects": len(df),
        "n_splits": args.n_splits,
        "first_seed": args.first_seed,
        "seeds": [
            args.first_seed + i
            for i in range(args.n_splits)
        ],
        "test_size": args.test_size,
        "dataset_sha256": dataframe_fingerprint(df),
        "sklearn_version": sklearn.__version__,
    }

    metadata_path = (
        args.output.parent
        / "split_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Dataset: {len(df)} objects")
    print(f"Saved: {args.output}")
    print()

    for split_id in range(args.n_splits):
        part = splits[splits["split_id"] == split_id]

        train = part[part["subset"] == "train"]
        test = part[part["subset"] == "test"]

        print(
            f"split={split_id:02d}, "
            f"seed={args.first_seed + split_id}, "
            f"train={len(train)}, "
            f"test={len(test)}"
        )


if __name__ == "__main__":
    main()