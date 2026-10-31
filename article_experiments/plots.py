from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Пути
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "article_results"
)

SUMMARY_DIR = (
    RESULTS_DIR
    / "summary"
)

LEARNING_CURVE_DIR = (
    RESULTS_DIR
    / "learning_curve"
)

LEARNING_CURVE_V2_DIR = (
    RESULTS_DIR
    / "learning_curve_v2"
)

VOCAB_DIR = (
    RESULTS_DIR
    / "vocab"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


# ============================================================
# Модели
# ============================================================

MODEL_FILES = {
    "tfidf": (
        SUMMARY_DIR
        / "tfidf_results.csv"
    ),

    "cc_fasttext": (
        SUMMARY_DIR
        / "cc_fasttext_results.csv"
    ),

    "bethesda_fasttext": (
        SUMMARY_DIR
        / "bethesda_fasttext_results.csv"
    ),

    "rubiobert": (
        SUMMARY_DIR
        / "rubiobert_results.csv"
    ),

    "rubert": (
        SUMMARY_DIR
        / "rubert_results.csv"
    ),

    "native_fasttext": (
        SUMMARY_DIR
        / "native_fasttext_results.csv"
    ),
}


MODEL_ORDER = [
    "tfidf",
    "cc_fasttext",
    "bethesda_fasttext",
    "rubiobert",
    "rubert",
    "native_fasttext",
]


MODEL_LABELS = {
    "tfidf": (
        "TF-IDF + LR"
    ),

    "cc_fasttext": (
        "cc.ru.300 fastText"
    ),

    "bethesda_fasttext": (
        "Bethesda fastText"
    ),

    "rubiobert": (
        "RuBioBERT (frozen)"
    ),

    "rubert": (
        "RuBERT (frozen)"
    ),

    "native_fasttext": (
        "fastText с нуля"
    ),
}


SHORT_MODEL_LABELS = {
    "tfidf": "TF-IDF",

    "cc_fasttext": (
        "cc.ru.300"
    ),

    "bethesda_fasttext": (
        "Bethesda FT"
    ),

    "rubiobert": (
        "RuBioBERT"
    ),

    "rubert": (
        "RuBERT"
    ),

    "native_fasttext": (
        "native FT"
    ),
}


# ============================================================
# Learning curve
# ============================================================

LEARNING_CURVE_FILES = {
    "tfidf": (
        LEARNING_CURVE_DIR
        / "tfidf_learning_curve.csv"
    ),

    "cc_fasttext": (
        LEARNING_CURVE_DIR
        / "cc_fasttext_learning_curve.csv"
    ),

    "bethesda_fasttext": (
        LEARNING_CURVE_DIR
        / "bethesda_fasttext_learning_curve.csv"
    ),

    # Для native fastText используем именно v2,
    # т.к. это финальная восстановленная версия.
    "native_fasttext": (
        LEARNING_CURVE_V2_DIR
        / "native_fasttext_learning_curve.csv"
    ),
}


LEARNING_CURVE_ORDER = [
    "tfidf",
    "cc_fasttext",
    "bethesda_fasttext",
    "native_fasttext",
]


# ============================================================
# Общие настройки
# ============================================================

METRIC = (
    "macro_f1_2_5"
)

DPI = 300


def configure_matplotlib():
    """
    Общий стиль всех рисунков.

    Не задаём нестандартные шрифты,
    чтобы код одинаково работал
    на разных компьютерах.
    """

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def require_file(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден необходимый файл:\n"
            f"{path}"
        )


def read_csv(
    path: Path,
) -> pd.DataFrame:

    require_file(
        path
    )

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def save_figure(
    fig,
    name: str,
):
    """
    Сохраняет сразу PNG и SVG.

    PNG удобно вставлять куда угодно.
    SVG предпочтительнее для Word,
    если версия Word его поддерживает:
    график остаётся векторным.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURES_DIR
        / f"{name}.png"
    )

    svg_path = (
        FIGURES_DIR
        / f"{name}.svg"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
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

    print(
        f"saved -> {svg_path}"
    )


# ============================================================
# Загрузка результатов моделей
# ============================================================

def load_all_model_results():
    results = {}

    for model_name in MODEL_ORDER:

        path = MODEL_FILES[
            model_name
        ]

        df = read_csv(
            path
        )

        required_columns = {
            "split_id",
            METRIC,
            "f1_2",
            "f1_3",
            "f1_4",
            "f1_5",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{path.name}: "
                f"нет столбцов "
                f"{sorted(missing)}"
            )

        results[
            model_name
        ] = df

    return results


# ============================================================
# Рисунок 1
# Основное сравнение моделей
# ============================================================

def plot_model_comparison(
    results,
):
    """
    Средний Macro-F1(2-5) ± стандартное
    отклонение по 10 outer split.

    Используем point + error bars, а не обычные
    столбцы: так корректнее визуализировать
    небольшие различия между моделями.
    """

    rows = []

    for model_name in MODEL_ORDER:

        values = (
            results[
                model_name
            ][
                METRIC
            ]
        )

        rows.append(
            {
                "model": model_name,

                "label": (
                    MODEL_LABELS[
                        model_name
                    ]
                ),

                "mean": (
                    values.mean()
                ),

                "std": (
                    values.std(
                        ddof=1
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = (
        summary
        .sort_values(
            "mean",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 5.6)
    )

    y = np.arange(
        len(summary)
    )

    ax.errorbar(
        summary["mean"],
        y,
        xerr=summary["std"],
        fmt="o",
        capsize=4,
        markersize=7,
        linewidth=1.5,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        summary[
            "label"
        ]
    )

    ax.set_xlabel(
        "Macro-F1 для классов EU-TIRADS 2–5"
    )

    ax.set_title(
        "Сравнение методов представления текста"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Диапазон выбран для лучшей читаемости.
    # Это point plot, а не bar chart,
    # поэтому начинать ось с нуля необязательно.
    min_value = (
        summary["mean"]
        - summary["std"]
    ).min()

    max_value = (
        summary["mean"]
        + summary["std"]
    ).max()

    margin = 0.025

    ax.set_xlim(
        max(
            0.0,
            min_value - margin,
        ),
        min(
            1.0,
            max_value + margin,
        ),
    )

    # Подписываем средние значения.
    for i, row in (
        summary.iterrows()
    ):
        ax.text(
            row["mean"] + 0.004,
            i + 0.12,
            f"{row['mean']:.3f}",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        "01_model_comparison",
    )


# ============================================================
# Рисунок 2
# Кривые обучения
# ============================================================

def plot_learning_curves():
    fig, ax = plt.subplots(
        figsize=(9.2, 5.8)
    )

    for model_name in (
        LEARNING_CURVE_ORDER
    ):

        df = read_csv(
            LEARNING_CURVE_FILES[
                model_name
            ]
        )

        grouped = (
            df
            .groupby(
                "fraction"
            )[
                METRIC
            ]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
            .reset_index()
            .sort_values(
                "fraction"
            )
        )

        x = (
            grouped[
                "fraction"
            ]
            * 100
        )

        ax.errorbar(
            x,
            grouped[
                "mean"
            ],
            yerr=grouped[
                "std"
            ],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=(
                MODEL_LABELS[
                    model_name
                ]
            ),
        )

    ax.set_xlabel(
        "Доля обучающей выборки, %"
    )

    ax.set_ylabel(
        "Macro-F1 для классов EU-TIRADS 2–5"
    )

    ax.set_title(
        "Кривые обучения"
    )

    ax.set_xticks(
        [
            10,
            25,
            50,
            75,
            100,
        ]
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "02_learning_curves",
    )


# ============================================================
# Вспомогательная функция для bootstrap-графиков
# ============================================================

def human_comparison_label(
    model_a: str,
    model_b: str,
):
    return (
        f"{SHORT_MODEL_LABELS[model_a]}"
        " − "
        f"{SHORT_MODEL_LABELS[model_b]}"
    )


def draw_bootstrap_forest(
    df: pd.DataFrame,
    title: str,
    filename: str,
):
    """
    Forest plot:
    точка = средняя парная разность Macro-F1,
    линия = 95% bootstrap CI.

    Если интервал пересекает 0,
    устойчивое преимущество не подтверждается.
    """

    df = df.copy()

    df[
        "label"
    ] = [
        human_comparison_label(
            row.model_a,
            row.model_b,
        )
        for row
        in df.itertuples()
    ]

    # Чтобы верхняя строка была первой.
    df = (
        df
        .reset_index(
            drop=True
        )
    )

    n = len(
        df
    )

    fig_height = max(
        4.5,
        0.43 * n + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.2,
            fig_height,
        )
    )

    y = np.arange(
        n
    )[::-1]

    left_error = (
        df[
            "mean_delta"
        ]
        - df[
            "ci_95_low"
        ]
    )

    right_error = (
        df[
            "ci_95_high"
        ]
        - df[
            "mean_delta"
        ]
    )

    ax.errorbar(
        df[
            "mean_delta"
        ],
        y,
        xerr=np.vstack(
            [
                left_error,
                right_error,
            ]
        ),
        fmt="o",
        capsize=3,
        markersize=6,
        linewidth=1.4,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df[
            "label"
        ]
    )

    ax.set_xlabel(
        "Разность Macro-F1: модель A − модель B"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename,
    )


# ============================================================
# Рисунок 3
# Ключевые bootstrap-сравнения
# ============================================================

def plot_bootstrap_key_comparisons():
    path = (
        SUMMARY_DIR
        / "bootstrap_pairwise_macro_f1_2_5.csv"
    )

    df = read_csv(
        path
    )

    # Сравнения, которые напрямую отвечают
    # на основные исследовательские вопросы.
    key_pairs = [
        (
            "tfidf",
            "cc_fasttext",
        ),

        (
            "cc_fasttext",
            "bethesda_fasttext",
        ),

        (
            "bethesda_fasttext",
            "native_fasttext",
        ),

        (
            "rubiobert",
            "rubert",
        ),

        (
            "bethesda_fasttext",
            "rubiobert",
        ),

        (
            "rubert",
            "native_fasttext",
        ),
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in key_pairs:

        match = df[
            (
                df["model_a"]
                == model_a
            )
            &
            (
                df["model_b"]
                == model_b
            )
        ]

        if len(match) != 1:
            raise RuntimeError(
                "Не найдено однозначное "
                "bootstrap-сравнение: "
                f"{model_a} - {model_b}"
            )

        rows.append(
            match.iloc[0]
        )

    selected = pd.DataFrame(
        rows
    )

    draw_bootstrap_forest(
        df=selected,
        title=(
            "Парные различия моделей: "
            "95% bootstrap-интервалы"
        ),
        filename=(
            "03_bootstrap_key_comparisons"
        ),
    )


# ============================================================
# Рисунок 4
# F1 по отдельным классам
# ============================================================

def plot_classwise_f1(
    results,
):
    """
    Показывает, на каких категориях EU-TIRADS
    возникает основная разница между моделями.

    Класс 1 не отображается:
    в наборе всего 3 таких объекта, и он
    исключён из основной Macro-F1(2-5).
    """

    classes = [
        2,
        3,
        4,
        5,
    ]

    fig, ax = plt.subplots(
        figsize=(10.2, 6.0)
    )

    for model_name in (
        MODEL_ORDER
    ):

        means = []

        for cls in classes:

            column = (
                f"f1_{cls}"
            )

            means.append(
                results[
                    model_name
                ][
                    column
                ].mean()
            )

        ax.plot(
            classes,
            means,
            marker="o",
            linewidth=1.5,
            label=(
                MODEL_LABELS[
                    model_name
                ]
            ),
        )

    ax.set_xticks(
        classes
    )

    ax.set_xlabel(
        "Класс EU-TIRADS"
    )

    ax.set_ylabel(
        "Средний F1"
    )

    ax.set_title(
        "Качество классификации по отдельным классам"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "04_classwise_f1",
    )


# ============================================================
# Рисунок 5
# OOV по классам
# ============================================================

def plot_oov_by_class():
    path = (
        VOCAB_DIR
        / "oov_by_class.csv"
    )

    df = read_csv(
        path
    )

    # Класс 1 исключаем из графика:
    # там только 3 объекта.
    df = (
        df[
            df[
                "label"
            ].isin(
                [
                    2,
                    3,
                    4,
                    5,
                ]
            )
        ]
        .sort_values(
            "label"
        )
        .copy()
    )

    labels = (
        df[
            "label"
        ]
        .astype(str)
        .tolist()
    )

    type_oov = (
        df[
            "type_oov_rate"
        ]
        .to_numpy()
        * 100
    )

    token_oov = (
        df[
            "token_oov_rate"
        ]
        .to_numpy()
        * 100
    )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(8.8, 5.6)
    )

    ax.bar(
        x - width / 2,
        type_oov,
        width,
        label=(
            "OOV среди уникальных слов"
        ),
    )

    ax.bar(
        x + width / 2,
        token_oov,
        width,
        label=(
            "OOV среди словоупотреблений"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "Класс EU-TIRADS"
    )

    ax.set_ylabel(
        "Доля OOV, %"
    )

    ax.set_title(
        "Лексическое расхождение Bethesda и EU-TIRADS"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "05_oov_by_class",
    )


# ============================================================
# Рисунок 6
# Распределение классов в датасете
# ============================================================

def plot_class_distribution():
    path = (
        RESULTS_DIR
        / "objects_manifest.csv"
    )

    df = read_csv(
        path
    )

    counts = (
        df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    labels = [
        str(x)
        for x in counts.index
    ]

    values = (
        counts.to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(8.4, 5.4)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_xlabel(
        "Класс EU-TIRADS"
    )

    ax.set_ylabel(
        "Количество текстов"
    )

    ax.set_title(
        "Распределение классов в выборке EU-TIRADS"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            (
                bar.get_x()
                + bar.get_width()
                / 2
            ),
            value,
            str(
                int(value)
            ),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    save_figure(
        fig,
        "06_class_distribution",
    )


# ============================================================
# Приложение A
# Все 15 bootstrap-сравнений
# ============================================================

def plot_bootstrap_all():
    path = (
        SUMMARY_DIR
        / "bootstrap_pairwise_macro_f1_2_5.csv"
    )

    df = read_csv(
        path
    )

    draw_bootstrap_forest(
        df=df,
        title=(
            "Все парные сравнения моделей: "
            "95% bootstrap-интервалы"
        ),
        filename=(
            "appendix_bootstrap_all"
        ),
    )


# ============================================================
# Приложение B
# Диагностика аномалии native fastText
# ============================================================

def plot_native_epoch_diagnostic():
    path = (
        LEARNING_CURVE_V2_DIR
        / "native_75_100_epoch_diagnostic.csv"
    )

    df = read_csv(
        path
    )

    required = {
        "fraction",
        "fasttext_epoch",
        METRIC,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name}: "
            f"нет столбцов "
            f"{sorted(missing)}"
        )

    grouped = (
        df
        .groupby(
            [
                "fasttext_epoch",
                "fraction",
            ]
        )[
            METRIC
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(8.6, 5.6)
    )

    for epoch in sorted(
        grouped[
            "fasttext_epoch"
        ].unique()
    ):

        part = (
            grouped[
                grouped[
                    "fasttext_epoch"
                ]
                == epoch
            ]
            .sort_values(
                "fraction"
            )
        )

        ax.errorbar(
            (
                part[
                    "fraction"
                ]
                * 100
            ),
            part[
                "mean"
            ],
            yerr=part[
                "std"
            ],
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=(
                f"epoch={int(epoch)}"
            ),
        )

    ax.set_xticks(
        [
            75,
            100,
        ]
    )

    ax.set_xlabel(
        "Доля обучающей выборки, %"
    )

    ax.set_ylabel(
        "Macro-F1 для классов EU-TIRADS 2–5"
    )

    ax.set_title(
        "Диагностика зависимости native fastText "
        "от числа эпох"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "appendix_native_fasttext_epoch_diagnostic",
    )


# ============================================================
# Таблица с числами, использованными на основном графике
# ============================================================

def save_model_summary_table(
    results,
):
    """
    Это не рисунок, но удобно сохранить рядом
    точные mean/std для последующей таблицы статьи.
    """

    rows = []

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_f1_2_5",
        "weighted_f1",
    ]

    for model_name in MODEL_ORDER:

        df = results[
            model_name
        ]

        row = {
            "model": (
                model_name
            ),

            "display_name": (
                MODEL_LABELS[
                    model_name
                ]
            ),
        }

        for metric in metrics:

            row[
                f"{metric}_mean"
            ] = (
                df[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = (
                df[
                    metric
                ].std(
                    ddof=1
                )
            )

        for cls in [
            2,
            3,
            4,
            5,
        ]:

            row[
                f"f1_{cls}_mean"
            ] = (
                df[
                    f"f1_{cls}"
                ].mean()
            )

            row[
                f"f1_{cls}_std"
            ] = (
                df[
                    f"f1_{cls}"
                ].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    path = (
        FIGURES_DIR
        / "model_summary_for_article.csv"
    )

    summary.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"saved -> {path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    configure_matplotlib()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 60
    )

    print(
        "BUILDING ARTICLE FIGURES"
    )

    print(
        "=" * 60
    )

    results = (
        load_all_model_results()
    )

    print()
    print(
        "[1/8] Model comparison"
    )

    plot_model_comparison(
        results
    )

    print()
    print(
        "[2/8] Learning curves"
    )

    plot_learning_curves()

    print()
    print(
        "[3/8] Key bootstrap comparisons"
    )

    plot_bootstrap_key_comparisons()

    print()
    print(
        "[4/8] Class-wise F1"
    )

    plot_classwise_f1(
        results
    )

    print()
    print(
        "[5/8] OOV by class"
    )

    plot_oov_by_class()

    print()
    print(
        "[6/8] Dataset class distribution"
    )

    plot_class_distribution()

    print()
    print(
        "[7/8] Full bootstrap plot"
    )

    plot_bootstrap_all()

    print()
    print(
        "[8/8] Native fastText diagnostic"
    )

    plot_native_epoch_diagnostic()

    print()
    print(
        "Saving numerical summary"
    )

    save_model_summary_table(
        results
    )

    print()
    print(
        "=" * 60
    )

    print(
        "DONE"
    )

    print(
        f"Figures -> {FIGURES_DIR}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()