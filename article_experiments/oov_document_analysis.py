from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from .data import (
    DEFAULT_BETHESDA_CORPUS,
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_bethesda_corpus,
    load_eutirads_dataset,
)

from .preprocessing import (
    tokenize_lexical,
)


# ============================================================
# Пути
# ============================================================

DEFAULT_RUNS_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "runs"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "oov_document"
)


# ============================================================
# Модели
# ============================================================

DEFAULT_MODELS = [
    "bethesda_fasttext",
    "native_fasttext",
    "tfidf",
]


MODEL_LABELS = {
    "bethesda_fasttext": (
        "Bethesda fastText"
    ),

    "native_fasttext": (
        "native fastText"
    ),

    "tfidf": (
        "TF-IDF"
    ),

    # Добавим после пункта 6.
    "bethesda_no_subword": (
        "Bethesda fastText "
        "без subword"
    ),

    "bethesda_ablation_no_subword": (
        "Без subword"
    ),

    "bethesda_ablation_2_5": (
        "Subword 2–5"
    ),

    "bethesda_ablation_3_6": (
        "Subword 3–6"
    ),

    "bethesda_ablation_4_7": (
        "Subword 4–7"
    ),
}


# ============================================================
# Словарь Bethesda
# ============================================================

def build_source_vocab(
    texts: list[str],
) -> set[str]:
    """
    Формирует лексический словарь исходного
    Bethesda-корпуса.

    Важно:
    здесь используется та же tokenization,
    что и в vocab_analysis.py.

    Это соответствует ранее полученному
    token OOV ≈ 52.4%.
    """

    vocab = set()

    for text in texts:

        tokens = tokenize_lexical(
            text
        )

        vocab.update(
            tokens
        )

    return vocab


# ============================================================
# OOV каждого документа
# ============================================================

def calculate_document_oov(
    df: pd.DataFrame,
    source_vocab: set[str],
) -> pd.DataFrame:
    """
    Для каждого EU-TIRADS документа:

        OOV rate =
            число токенов вне словаря Bethesda
            ---------------------------------
            общее число лексических токенов
    """

    rows = []

    for _, row in df.iterrows():

        tokens = tokenize_lexical(
            row["text_model"]
        )

        n_tokens = len(
            tokens
        )

        n_oov = sum(
            token not in source_vocab
            for token in tokens
        )

        if n_tokens == 0:

            oov_rate = np.nan

        else:

            oov_rate = (
                n_oov
                / n_tokens
            )

        rows.append(
            {
                "object_id": (
                    row[
                        "object_id"
                    ]
                ),

                "label": int(
                    row[
                        "label"
                    ]
                ),

                "n_tokens": int(
                    n_tokens
                ),

                "n_oov_tokens": int(
                    n_oov
                ),

                "oov_rate": float(
                    oov_rate
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if (
        result[
            "oov_rate"
        ]
        .isna()
        .any()
    ):
        bad = result[
            result[
                "oov_rate"
            ].isna()
        ]

        raise ValueError(
            "Обнаружены документы без "
            "лексических токенов:\n"
            f"{bad[['object_id', 'label']]}"
        )

    return result


# ============================================================
# Группы OOV
# ============================================================

def add_oov_groups(
    document_oov: pd.DataFrame,
    n_groups: int,
):
    """
    Группы формируются по квантилям OOV
    на всей фиксированной выборке EU-TIRADS.

    Это предпочтительнее жёстких интервалов
    0-20, 20-40, ...:
    в каждой группе получается достаточно
    объектов для оценки качества.

    Одинаковые значения OOV не разделяются
    искусственно между соседними группами.
    """

    result = document_oov.copy()

    groups = pd.qcut(
        result[
            "oov_rate"
        ],
        q=n_groups,
        duplicates="drop",
    )

    result[
        "oov_group_id"
    ] = (
        groups
        .cat
        .codes
        .astype(int)
    )

    actual_n_groups = (
        result[
            "oov_group_id"
        ]
        .nunique()
    )

    if actual_n_groups < 2:
        raise RuntimeError(
            "Не удалось сформировать "
            "минимум две OOV-группы."
        )

    if (
        actual_n_groups
        != n_groups
    ):
        print(
            "WARNING: из-за совпадающих "
            "квантилей сформировано "
            f"{actual_n_groups}, "
            f"а не {n_groups} групп."
        )

    return result


# ============================================================
# Описание групп
# ============================================================

def describe_oov_groups(
    document_oov: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for group_id, part in (
        document_oov
        .groupby(
            "oov_group_id"
        )
    ):

        row = {
            "oov_group_id": int(
                group_id
            ),

            "n_objects": int(
                len(part)
            ),

            "oov_min": float(
                part[
                    "oov_rate"
                ].min()
            ),

            "oov_mean": float(
                part[
                    "oov_rate"
                ].mean()
            ),

            "oov_median": float(
                part[
                    "oov_rate"
                ].median()
            ),

            "oov_max": float(
                part[
                    "oov_rate"
                ].max()
            ),

            "mean_tokens_per_document": float(
                part[
                    "n_tokens"
                ].mean()
            ),
        }

        for label in [
            1,
            2,
            3,
            4,
            5,
        ]:

            row[
                f"n_class_{label}"
            ] = int(
                (
                    part[
                        "label"
                    ]
                    == label
                ).sum()
            )

        row[
            "oov_group_label"
        ] = (
            f"{row['oov_min'] * 100:.0f}"
            "–"
            f"{row['oov_max'] * 100:.0f}%"
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "oov_group_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Predictions
# ============================================================

def load_model_predictions(
    model_name: str,
    runs_dir: Path,
    document_oov: pd.DataFrame,
) -> pd.DataFrame:
    """
    Читает уже существующие predictions.csv.

    Ничего заново не обучается.
    """

    model_dir = (
        runs_dir
        / model_name
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Не найден каталог модели:\n"
            f"{model_dir}"
        )

    prediction_files = sorted(
        model_dir.glob(
            "split_*/predictions.csv"
        )
    )

    if not prediction_files:
        raise FileNotFoundError(
            "Не найдены predictions.csv "
            f"для {model_name}."
        )

    rows = []

    for path in prediction_files:

        split_name = (
            path
            .parent
            .name
        )

        split_id = int(
            split_name
            .replace(
                "split_",
                "",
            )
        )

        pred = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        required = {
            "object_id",
            "y_true",
            "y_pred",
        }

        missing = (
            required
            - set(
                pred.columns
            )
        )

        if missing:
            raise ValueError(
                f"{path}: отсутствуют "
                f"столбцы {sorted(missing)}"
            )

        merged = pred.merge(
            document_oov,
            on="object_id",
            how="left",
            validate="one_to_one",
        )

        if (
            merged[
                "oov_rate"
            ]
            .isna()
            .any()
        ):
            raise RuntimeError(
                f"{model_name}, split={split_id}: "
                "не удалось найти OOV "
                "для части объектов."
            )

        if not np.array_equal(
            merged[
                "y_true"
            ].to_numpy(
                dtype=int
            ),
            merged[
                "label"
            ].to_numpy(
                dtype=int
            ),
        ):
            raise RuntimeError(
                f"{model_name}, split={split_id}: "
                "y_true не совпадает "
                "с label датасета."
            )

        merged[
            "split_id"
        ] = split_id

        merged[
            "model"
        ] = model_name

        merged[
            "is_correct"
        ] = (
            merged[
                "y_true"
            ]
            == merged[
                "y_pred"
            ]
        )

        rows.append(
            merged
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Метрики внутри OOV-группы
# ============================================================

def calculate_group_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = predictions.groupby(
        [
            "model",
            "split_id",
            "oov_group_id",
        ]
    )

    for (
        model_name,
        split_id,
        group_id,
    ), part in grouped:

        y_true = (
            part[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )

        y_pred = (
            part[
                "y_pred"
            ]
            .to_numpy(
                dtype=int
            )
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=[
                2,
                3,
                4,
                5,
            ],
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        row = {
            "model": (
                model_name
            ),

            "split_id": int(
                split_id
            ),

            "oov_group_id": int(
                group_id
            ),

            "n_objects": int(
                len(part)
            ),

            "mean_oov_rate": float(
                part[
                    "oov_rate"
                ].mean()
            ),

            "macro_f1_2_5": float(
                macro_f1
            ),

            "accuracy": float(
                accuracy
            ),

            "error_rate": float(
                1.0
                - accuracy
            ),
        }

        for label in [
            1,
            2,
            3,
            4,
            5,
        ]:

            row[
                f"support_{label}"
            ] = int(
                (
                    y_true
                    == label
                ).sum()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Среднее по 10 внешним split
# ============================================================

def summarize_group_metrics(
    metrics: pd.DataFrame,
    group_info: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        metrics
        .groupby(
            [
                "model",
                "oov_group_id",
            ]
        )
        .agg(
            n_splits=(
                "split_id",
                "nunique",
            ),

            mean_n_test_objects=(
                "n_objects",
                "mean",
            ),

            macro_f1_mean=(
                "macro_f1_2_5",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1_2_5",
                "std",
            ),

            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            error_rate_mean=(
                "error_rate",
                "mean",
            ),

            error_rate_std=(
                "error_rate",
                "std",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        group_info[
            [
                "oov_group_id",
                "oov_group_label",
                "oov_min",
                "oov_mean",
                "oov_max",
            ]
        ],
        on="oov_group_id",
        how="left",
        validate="many_to_one",
    )

    return summary


# ============================================================
# Bethesda - native delta
# ============================================================

def calculate_bethesda_native_delta(
    metrics: pd.DataFrame,
    group_info: pd.DataFrame,
):
    """
    Парная разность рассчитывается внутри
    одного split и одной OOV-группы.

    Здесь НЕ строим доверительный интервал
    и не трактуем 10 split как независимые
    наблюдения.

    Сохраняем только mean/std разности
    как описательную характеристику.
    """

    pivot = (
        metrics
        .pivot_table(
            index=[
                "split_id",
                "oov_group_id",
            ],
            columns="model",
            values="macro_f1_2_5",
        )
        .reset_index()
    )

    required = {
        "bethesda_fasttext",
        "native_fasttext",
    }

    if not required.issubset(
        pivot.columns
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    pivot[
        "delta_bethesda_minus_native"
    ] = (
        pivot[
            "bethesda_fasttext"
        ]
        - pivot[
            "native_fasttext"
        ]
    )

    delta_summary = (
        pivot
        .groupby(
            "oov_group_id"
        )
        .agg(
            n_pairs=(
                "delta_bethesda_minus_native",
                "count",
            ),

            delta_mean=(
                "delta_bethesda_minus_native",
                "mean",
            ),

            delta_std=(
                "delta_bethesda_minus_native",
                "std",
            ),

            delta_min=(
                "delta_bethesda_minus_native",
                "min",
            ),

            delta_max=(
                "delta_bethesda_minus_native",
                "max",
            ),
        )
        .reset_index()
    )

    delta_summary = (
        delta_summary
        .merge(
            group_info[
                [
                    "oov_group_id",
                    "oov_group_label",
                    "oov_mean",
                ]
            ],
            on="oov_group_id",
            how="left",
            validate="one_to_one",
        )
    )

    return (
        pivot,
        delta_summary,
    )


# ============================================================
# Графики
# ============================================================

def plot_macro_f1(
    summary: pd.DataFrame,
    group_info: pd.DataFrame,
    output_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8.8, 5.6)
    )

    group_ids = (
        group_info[
            "oov_group_id"
        ]
        .tolist()
    )

    x = np.arange(
        len(group_ids)
    )

    for model_name in (
        summary[
            "model"
        ]
        .drop_duplicates()
    ):

        part = (
            summary[
                summary[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "oov_group_id"
            )
        )

        ax.errorbar(
            x,
            part[
                "macro_f1_mean"
            ],
            yerr=part[
                "macro_f1_std"
            ],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=(
                MODEL_LABELS.get(
                    model_name,
                    model_name,
                )
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        group_info[
            "oov_group_label"
        ]
    )

    ax.set_xlabel(
        "Доля OOV-токенов в документе"
    )

    ax.set_ylabel(
        "Macro-F1 для классов EU-TIRADS 2–5"
    )

    ax.set_title(
        "Зависимость качества от лексического "
        "расхождения с Bethesda"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / "oov_macro_f1.png"
    )

    svg_path = (
        output_dir
        / "oov_macro_f1.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"saved -> {png_path}"
    )


def plot_error_rate(
    summary: pd.DataFrame,
    group_info: pd.DataFrame,
    output_dir: Path,
):
    """
    Дополнительный график.

    Если в какой-то OOV-группе окажется
    недостаточная поддержка отдельных классов,
    доля ошибок остаётся интерпретируемой
    на уровне объектов.
    """

    fig, ax = plt.subplots(
        figsize=(8.8, 5.6)
    )

    x = np.arange(
        len(
            group_info
        )
    )

    for model_name in (
        summary[
            "model"
        ]
        .drop_duplicates()
    ):

        part = (
            summary[
                summary[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "oov_group_id"
            )
        )

        ax.errorbar(
            x,
            (
                part[
                    "error_rate_mean"
                ]
                * 100
            ),
            yerr=(
                part[
                    "error_rate_std"
                ]
                * 100
            ),
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=(
                MODEL_LABELS.get(
                    model_name,
                    model_name,
                )
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        group_info[
            "oov_group_label"
        ]
    )

    ax.set_xlabel(
        "Доля OOV-токенов в документе"
    )

    ax.set_ylabel(
        "Доля ошибок, %"
    )

    ax.set_title(
        "Частота ошибок при росте OOV"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / "oov_error_rate.png"
    )

    svg_path = (
        output_dir
        / "oov_error_rate.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"saved -> {png_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--bethesda-corpus",
        type=Path,
        default=DEFAULT_BETHESDA_CORPUS,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--n-groups",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Данные
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "DOCUMENT-LEVEL OOV ANALYSIS"
    )

    print(
        "=" * 60
    )

    eu_df = load_eutirads_dataset(
        args.data
    )

    bethesda_texts = (
        load_bethesda_corpus(
            args.bethesda_corpus
        )
    )

    # ========================================================
    # Словарь Bethesda
    # ========================================================

    print()
    print(
        "Building Bethesda vocabulary..."
    )

    source_vocab = (
        build_source_vocab(
            bethesda_texts
        )
    )

    print(
        "Bethesda lexical vocabulary: "
        f"{len(source_vocab)}"
    )

    # ========================================================
    # OOV каждого документа
    # ========================================================

    document_oov = (
        calculate_document_oov(
            df=eu_df,
            source_vocab=(
                source_vocab
            ),
        )
    )

    global_token_oov = (
        document_oov[
            "n_oov_tokens"
        ].sum()
        /
        document_oov[
            "n_tokens"
        ].sum()
    )

    print()
    print(
        "Global token OOV: "
        f"{global_token_oov:.4f} "
        f"({global_token_oov * 100:.2f}%)"
    )

    document_oov = (
        add_oov_groups(
            document_oov,
            n_groups=(
                args.n_groups
            ),
        )
    )

    group_info = (
        describe_oov_groups(
            document_oov
        )
    )

    # ========================================================
    # Сохраняем OOV документов
    # ========================================================

    document_oov.to_csv(
        args.output_dir
        / "document_oov.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_info.to_csv(
        args.output_dir
        / "oov_group_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=== OOV GROUPS ==="
    )

    print(
        group_info[
            [
                "oov_group_id",
                "oov_group_label",
                "n_objects",
                "oov_mean",
                "n_class_2",
                "n_class_3",
                "n_class_4",
                "n_class_5",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Predictions
    # ========================================================

    prediction_frames = []

    for model_name in (
        args.models
    ):

        print()
        print(
            f"Loading predictions: "
            f"{model_name}"
        )

        prediction_frames.append(
            load_model_predictions(
                model_name=(
                    model_name
                ),

                runs_dir=(
                    args.runs_dir
                ),

                document_oov=(
                    document_oov
                ),
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        args.output_dir
        / "predictions_with_oov.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Метрики
    # ========================================================

    metrics = (
        calculate_group_metrics(
            predictions
        )
    )

    metrics.to_csv(
        args.output_dir
        / "oov_metrics_by_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        summarize_group_metrics(
            metrics=metrics,
            group_info=group_info,
        )
    )

    summary.to_csv(
        args.output_dir
        / "oov_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Bethesda - native
    # ========================================================

    (
        delta_by_split,
        delta_summary,
    ) = (
        calculate_bethesda_native_delta(
            metrics=metrics,
            group_info=group_info,
        )
    )

    if not delta_by_split.empty:

        delta_by_split.to_csv(
            args.output_dir
            / "bethesda_native_delta_by_split.csv",
            index=False,
            encoding="utf-8-sig",
        )

        delta_summary.to_csv(
            args.output_dir
            / "bethesda_native_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Графики
    # ========================================================

    plot_macro_f1(
        summary=summary,
        group_info=group_info,
        output_dir=(
            args.output_dir
        ),
    )

    plot_error_rate(
        summary=summary,
        group_info=group_info,
        output_dir=(
            args.output_dir
        ),
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "=== MACRO-F1 BY OOV GROUP ==="
    )

    print(
        summary
        .sort_values(
            [
                "model",
                "oov_group_id",
            ]
        )[
            [
                "model",
                "oov_group_label",
                "n_splits",
                "macro_f1_mean",
                "macro_f1_std",
                "error_rate_mean",
            ]
        ]
        .to_string(
            index=False
        )
    )

    if not delta_summary.empty:

        print()
        print(
            "=== BETHESDA - NATIVE ==="
        )

        print(
            delta_summary[
                [
                    "oov_group_label",
                    "n_pairs",
                    "delta_mean",
                    "delta_std",
                    "delta_min",
                    "delta_max",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 60
    )

    print(
        f"Results -> "
        f"{args.output_dir}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()