from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT


DEFAULT_SUMMARY_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "summary"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "summary"
)


MODEL_FILES = {
    "tfidf": "tfidf_results.csv",
    "cc_fasttext": "cc_fasttext_results.csv",
    "bethesda_fasttext": "bethesda_fasttext_results.csv",
    "native_fasttext": "native_fasttext_results.csv",
}


# Порядок задаёт направление разности:
#
# delta = model_a - model_b
#
# Поэтому положительный CI означает преимущество model_a.
PAIRWISE_COMPARISONS = [
    (
        "tfidf",
        "cc_fasttext",
    ),
    (
        "tfidf",
        "bethesda_fasttext",
    ),
    (
        "tfidf",
        "native_fasttext",
    ),
    (
        "cc_fasttext",
        "bethesda_fasttext",
    ),
    (
        "cc_fasttext",
        "native_fasttext",
    ),
    (
        "bethesda_fasttext",
        "native_fasttext",
    ),
]


def load_model_results(
    summary_dir: Path,
    model_name: str,
) -> pd.DataFrame:

    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Неизвестная модель: {model_name}"
        )

    path = (
        summary_dir
        / MODEL_FILES[model_name]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл результатов:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "model",
        "split_id",
        "seed",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name}: отсутствуют столбцы "
            f"{sorted(missing)}"
        )

    if df["split_id"].duplicated().any():
        raise ValueError(
            f"{path.name}: обнаружены "
            "повторяющиеся split_id."
        )

    return (
        df
        .sort_values("split_id")
        .reset_index(drop=True)
    )


def validate_paired_results(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    model_a: str,
    model_b: str,
):
    """
    Проверяет, что две модели действительно
    оценены на одинаковых outer split.
    """

    splits_a = (
        df_a["split_id"]
        .to_numpy()
    )

    splits_b = (
        df_b["split_id"]
        .to_numpy()
    )

    if not np.array_equal(
        splits_a,
        splits_b,
    ):
        raise ValueError(
            f"{model_a} и {model_b}: "
            "split_id не совпадают."
        )

    seeds_a = (
        df_a["seed"]
        .to_numpy()
    )

    seeds_b = (
        df_b["seed"]
        .to_numpy()
    )

    if not np.array_equal(
        seeds_a,
        seeds_b,
    ):
        raise ValueError(
            f"{model_a} и {model_b}: "
            "seed не совпадают."
        )


def paired_bootstrap(
    values_a,
    values_b,
    n_bootstrap: int,
    seed: int,
):
    """
    Парный bootstrap по outer split.

    На каждой итерации выбираются индексы split
    с возвращением. Одни и те же индексы
    применяются к обеим моделям.
    """

    a = np.asarray(
        values_a,
        dtype=float,
    )

    b = np.asarray(
        values_b,
        dtype=float,
    )

    if a.shape != b.shape:
        raise ValueError(
            "Размеры массивов не совпадают."
        )

    if a.ndim != 1:
        raise ValueError(
            "Ожидались одномерные массивы."
        )

    if len(a) < 2:
        raise ValueError(
            "Недостаточно наблюдений "
            "для bootstrap."
        )

    deltas = (
        a - b
    )

    n = len(deltas)

    rng = np.random.default_rng(
        seed
    )

    # shape:
    # (n_bootstrap, n_splits)
    indices = rng.integers(
        low=0,
        high=n,
        size=(
            n_bootstrap,
            n,
        ),
    )

    bootstrap_means = (
        deltas[
            indices
        ]
        .mean(axis=1)
    )

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [
            0.025,
            0.975,
        ],
    )

    observed_mean_delta = float(
        deltas.mean()
    )

    result = {
        "n_splits": int(n),

        "mean_a": float(
            a.mean()
        ),

        "mean_b": float(
            b.mean()
        ),

        "mean_delta": (
            observed_mean_delta
        ),

        "std_delta": float(
            deltas.std(
                ddof=1
            )
        ),

        "ci_95_low": float(
            ci_low
        ),

        "ci_95_high": float(
            ci_high
        ),

        "wins_a": int(
            np.sum(
                deltas > 0
            )
        ),

        "ties": int(
            np.sum(
                deltas == 0
            )
        ),

        "wins_b": int(
            np.sum(
                deltas < 0
            )
        ),

        # Это НЕ p-value.
        # Просто доля bootstrap-средних > 0.
        "bootstrap_fraction_gt_zero": float(
            np.mean(
                bootstrap_means > 0
            )
        ),

        "bootstrap_fraction_lt_zero": float(
            np.mean(
                bootstrap_means < 0
            )
        ),
    }

    return (
        result,
        deltas,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric",
        type=str,
        default="macro_f1_2_5",
        help=(
            "Метрика для парного bootstrap."
        ),
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.n_bootstrap <= 0:
        raise ValueError(
            "--n-bootstrap должен быть > 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Загружаем все модели один раз
    # --------------------------------------------------------

    results = {}

    for model_name in MODEL_FILES:

        df = load_model_results(
            summary_dir=args.summary_dir,
            model_name=model_name,
        )

        if args.metric not in df.columns:
            raise ValueError(
                f"{MODEL_FILES[model_name]}: "
                f"нет метрики {args.metric}"
            )

        results[
            model_name
        ] = df

    # --------------------------------------------------------
    # Проверяем количество split
    # --------------------------------------------------------

    split_counts = {
        model: len(df)
        for model, df
        in results.items()
    }

    print(
        "Split counts:"
    )

    for model, count in (
        split_counts.items()
    ):
        print(
            f"  {model}: {count}"
        )

    # --------------------------------------------------------
    # Парные сравнения
    # --------------------------------------------------------

    summary_rows = []
    delta_rows = []

    for pair_index, (
        model_a,
        model_b,
    ) in enumerate(
        PAIRWISE_COMPARISONS
    ):

        df_a = results[
            model_a
        ]

        df_b = results[
            model_b
        ]

        validate_paired_results(
            df_a=df_a,
            df_b=df_b,
            model_a=model_a,
            model_b=model_b,
        )

        values_a = (
            df_a[
                args.metric
            ]
            .to_numpy(
                dtype=float
            )
        )

        values_b = (
            df_b[
                args.metric
            ]
            .to_numpy(
                dtype=float
            )
        )

        bootstrap_seed = (
            args.seed
            + pair_index
        )

        (
            stats,
            deltas,
        ) = paired_bootstrap(
            values_a=values_a,
            values_b=values_b,
            n_bootstrap=(
                args.n_bootstrap
            ),
            seed=bootstrap_seed,
        )

        row = {
            "metric": (
                args.metric
            ),

            "model_a": (
                model_a
            ),

            "model_b": (
                model_b
            ),

            "comparison": (
                f"{model_a} - {model_b}"
            ),

            "bootstrap_seed": int(
                bootstrap_seed
            ),

            **stats,
        }

        summary_rows.append(
            row
        )

        # Сохраняем исходные парные разности,
        # чтобы их можно было проверить вручную.
        for i in range(
            len(df_a)
        ):

            delta_rows.append(
                {
                    "metric": (
                        args.metric
                    ),

                    "model_a": (
                        model_a
                    ),

                    "model_b": (
                        model_b
                    ),

                    "split_id": int(
                        df_a.loc[
                            i,
                            "split_id",
                        ]
                    ),

                    "seed": int(
                        df_a.loc[
                            i,
                            "seed",
                        ]
                    ),

                    "value_a": float(
                        values_a[i]
                    ),

                    "value_b": float(
                        values_b[i]
                    ),

                    "delta": float(
                        deltas[i]
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    deltas_df = pd.DataFrame(
        delta_rows
    )

    # --------------------------------------------------------
    # Сохраняем
    # --------------------------------------------------------

    summary_path = (
        args.output_dir
        / (
            "bootstrap_pairwise_"
            f"{args.metric}.csv"
        )
    )

    deltas_path = (
        args.output_dir
        / (
            "bootstrap_deltas_"
            f"{args.metric}.csv"
        )
    )

    config_path = (
        args.output_dir
        / (
            "bootstrap_"
            f"{args.metric}_config.json"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    deltas_df.to_csv(
        deltas_path,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "metric": (
            args.metric
        ),

        "n_bootstrap": int(
            args.n_bootstrap
        ),

        "base_seed": int(
            args.seed
        ),

        "bootstrap_unit": (
            "outer_split"
        ),

        "paired": True,

        "confidence_interval": (
            "percentile_95"
        ),

        "n_models": int(
            len(MODEL_FILES)
        ),

        "n_pairwise_comparisons": int(
            len(
                PAIRWISE_COMPARISONS
            )
        ),
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Консоль
    # --------------------------------------------------------

    print()
    print(
        "=== PAIRED BOOTSTRAP ==="
    )

    for _, row in (
        summary.iterrows()
    ):

        print()

        print(
            row["comparison"]
        )

        print(
            f"  mean A: "
            f"{row['mean_a']:.4f}"
        )

        print(
            f"  mean B: "
            f"{row['mean_b']:.4f}"
        )

        print(
            f"  delta:  "
            f"{row['mean_delta']:+.4f}"
        )

        print(
            "  95% CI: "
            f"[{row['ci_95_low']:+.4f}, "
            f"{row['ci_95_high']:+.4f}]"
        )

        print(
            "  wins: "
            f"{int(row['wins_a'])}"
            "-"
            f"{int(row['wins_b'])}"
            f" (ties="
            f"{int(row['ties'])})"
        )

    print()
    print(
        f"Summary -> {summary_path}"
    )

    print(
        f"Deltas  -> {deltas_path}"
    )

    print(
        f"Config  -> {config_path}"
    )


if __name__ == "__main__":
    main()