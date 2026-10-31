from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from .data import (
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_eutirads_dataset,
)

from .metrics import calculate_metrics

from .representations.fasttext_native import (
    train_native_fasttext,
    vectorize_texts,
    build_classifier,
)

from .run_learning_curve import (
    DEFAULT_SPLITS,
    DEFAULT_OUTPUT_DIR,
    FIXED_NATIVE_FASTTEXT_PARAMS,
    load_outer_split,
    prepare_manifest,
    get_manifest_subset,
)


FRACTIONS = [
    0.75,
    1.00,
]

EPOCHS = [
    5,
    40,
]


def corpus_stats(texts):

    texts = [
        str(text).strip()
        for text in texts
    ]

    tokens = []

    for text in texts:
        tokens.extend(
            text.split()
        )

    return {
        "n_tokens": len(tokens),
        "n_unique_tokens": len(
            set(tokens)
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
        help=(
            "Если указан, запускается только один split. "
            "Если не указан — все split."
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--learning-curve-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "article_results"
            / "learning_curve_v2"
            / "native_75_100_epoch_diagnostic.csv"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Данные
    # ========================================================

    df = load_eutirads_dataset(
        args.data
    )

    splits = pd.read_csv(
        args.splits
    )

    manifest = prepare_manifest(
        df=df,
        splits=splits,
        output_dir=args.learning_curve_dir,
    )

    available_split_ids = [
        int(x)
        for x in sorted(
            splits[
                "split_id"
            ].unique()
        )
    ]

    if args.split_id is None:
        split_ids = available_split_ids
    else:

        if (
            args.split_id
            not in available_split_ids
        ):
            raise ValueError(
                f"Нет split_id={args.split_id}"
            )

        split_ids = [
            int(args.split_id)
        ]

    rows = []

    # ========================================================
    # Outer splits
    # ========================================================

    for split_id in split_ids:

        split_part = splits[
            splits["split_id"]
            == split_id
        ]

        seed = int(
            split_part[
                "seed"
            ].iloc[0]
        )

        _, outer_test = load_outer_split(
            df=df,
            splits=splits,
            split_id=split_id,
        )

        print()
        print()
        print(
            "=" * 70
        )

        print(
            f"SPLIT {split_id} | "
            f"seed={seed} | "
            f"test={len(outer_test)}"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # 75% / 100%
        # ====================================================

        for fraction in FRACTIONS:

            train_subset = get_manifest_subset(
                df=df,
                manifest=manifest,
                split_id=split_id,
                fraction=fraction,
            )

            stats = corpus_stats(
                train_subset[
                    "text_model"
                ]
            )

            print()
            print(
                f"--- fraction={fraction:.0%} "
                f"| texts={len(train_subset)} "
                f"| tokens={stats['n_tokens']} "
                f"| unique={stats['n_unique_tokens']}"
            )

            # ================================================
            # epoch=5 / epoch=40
            # ================================================

            for epoch in EPOCHS:

                ft_params = {
                    **FIXED_NATIVE_FASTTEXT_PARAMS,
                    "epoch": int(epoch),
                }

                print()
                print(
                    f"split={split_id} | "
                    f"fraction={fraction:.0%} | "
                    f"epoch={epoch}"
                )

                # --------------------------------------------
                # Native fastText
                # --------------------------------------------

                model = train_native_fasttext(
                    texts=(
                        train_subset[
                            "text_model"
                        ]
                    ),
                    params=ft_params,
                )

                vocab_size = int(
                    len(
                        model.get_words()
                    )
                )

                x_train = vectorize_texts(
                    model,
                    train_subset[
                        "text_model"
                    ],
                )

                x_test = vectorize_texts(
                    model,
                    outer_test[
                        "text_model"
                    ],
                )

                del model
                gc.collect()

                # --------------------------------------------
                # Фиксированный классификатор
                #
                # Никакого GridSearch:
                # хотим изолировать fastText.
                # --------------------------------------------

                classifier = build_classifier(
                    C=10.0,
                    class_weight=None,
                )

                classifier.fit(
                    x_train,
                    train_subset[
                        "label"
                    ],
                )

                y_pred = classifier.predict(
                    x_test
                )

                metrics = calculate_metrics(
                    outer_test[
                        "label"
                    ],
                    y_pred,
                )

                row = {
                    "split_id": int(
                        split_id
                    ),

                    "seed": int(
                        seed
                    ),

                    "fraction": float(
                        fraction
                    ),

                    "n_train_texts": int(
                        len(train_subset)
                    ),

                    "n_tokens": int(
                        stats[
                            "n_tokens"
                        ]
                    ),

                    "n_unique_tokens": int(
                        stats[
                            "n_unique_tokens"
                        ]
                    ),

                    "fasttext_epoch": int(
                        epoch
                    ),

                    "fasttext_vocab_size": (
                        vocab_size
                    ),

                    "classifier_C": 10.0,

                    "classifier_class_weight": (
                        "none"
                    ),

                    **metrics,
                }

                rows.append(
                    row
                )

                print(
                    "Macro-F1(2-5): "
                    f"{metrics['macro_f1_2_5']:.4f}"
                )

    # ========================================================
    # Сохранение
    # ========================================================

    result = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Анализ 75 -> 100
    # ========================================================

    pivot = result.pivot(
        index=[
            "split_id",
            "fasttext_epoch",
        ],
        columns="fraction",
        values="macro_f1_2_5",
    ).reset_index()

    pivot[
        "delta_100_minus_75"
    ] = (
        pivot[1.0]
        - pivot[0.75]
    )

    print()
    print()
    print(
        "=" * 80
    )

    print(
        "75% -> 100% BY SPLIT"
    )

    print(
        "=" * 80
    )

    print(
        pivot.to_string(
            index=False
        )
    )

    # ========================================================
    # Итог по epoch
    # ========================================================

    print()
    print()
    print(
        "=" * 80
    )

    print(
        "SUMMARY BY EPOCH"
    )

    print(
        "=" * 80
    )

    for epoch in EPOCHS:

        part = pivot[
            pivot[
                "fasttext_epoch"
            ]
            == epoch
        ]

        delta = part[
            "delta_100_minus_75"
        ]

        mean_75 = part[
            0.75
        ].mean()

        std_75 = part[
            0.75
        ].std(
            ddof=1
        )

        mean_100 = part[
            1.0
        ].mean()

        std_100 = part[
            1.0
        ].std(
            ddof=1
        )

        print()
        print(
            f"epoch={epoch}"
        )

        print(
            f"75%:  "
            f"{mean_75:.4f} "
            f"± {std_75:.4f}"
        )

        print(
            f"100%: "
            f"{mean_100:.4f} "
            f"± {std_100:.4f}"
        )

        print(
            "mean delta "
            "(100%-75%): "
            f"{delta.mean():+.4f}"
        )

        print(
            "100% better:",
            int(
                (
                    delta > 0
                ).sum()
            ),
            "/",
            len(delta),
        )

        print(
            "100% worse:",
            int(
                (
                    delta < 0
                ).sum()
            ),
            "/",
            len(delta),
        )

    # ========================================================
    # Сохраним pivot отдельно
    # ========================================================

    pivot_path = (
        args.output.parent
        / "native_75_100_epoch_diagnostic_pivot.csv"
    )

    pivot.to_csv(
        pivot_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Raw results -> "
        f"{args.output}"
    )

    print(
        f"Pivot -> "
        f"{pivot_path}"
    )


if __name__ == "__main__":
    main()