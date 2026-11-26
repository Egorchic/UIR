from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Пути
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS = PROJECT_ROOT / "article_results"

OUTPUT_DIR = (
    RESULTS
    / "figures_color_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Общий стиль
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)


# Спокойная палитра.
COLORS = {
    "blue": "#377EB8",
    "orange": "#FF7F00",
    "green": "#4DAF4A",
    "red": "#E41A1C",
    "purple": "#984EA3",
    "gray": "#666666",
}


def save_figure(fig, name: str):
    png = OUTPUT_DIR / f"{name}.png"
    svg = OUTPUT_DIR / f"{name}.svg"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"saved -> {png}")
    print(f"saved -> {svg}")


# ============================================================
# РИС. 1
# Ключевые парные различия моделей
# ============================================================

def plot_bootstrap():
    path = (
        RESULTS
        / "summary"
        / "bootstrap_pairwise_macro_f1_2_5.csv"
    )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    comparisons = [
        (
            "bethesda_fasttext",
            "native_fasttext",
            "Bethesda fastText − native fastText",
            COLORS["green"],
        ),
        (
            "rubiobert",
            "rubert",
            "RuBioBERT − RuBERT",
            COLORS["purple"],
        ),
        (
            "cc_fasttext",
            "bethesda_fasttext",
            "cc.ru.300 − Bethesda fastText",
            COLORS["orange"],
        ),
    ]

    rows = []

    for model_a, model_b, label, color in comparisons:
        part = df[
            (df["model_a"] == model_a)
            & (df["model_b"] == model_b)
        ]

        if len(part) != 1:
            raise RuntimeError(
                f"Не найден результат: "
                f"{model_a} - {model_b}"
            )

        row = part.iloc[0]

        rows.append(
            {
                "label": label,
                "delta": row["mean_delta"],
                "low": row["ci_95_low"],
                "high": row["ci_95_high"],
                "color": color,
            }
        )

    # Переворачиваем, чтобы первая строка была сверху.
    rows = rows[::-1]

    fig, ax = plt.subplots(
        figsize=(7.0, 3.5)
    )

    for i, row in enumerate(rows):
        delta = row["delta"]

        left_error = (
            delta
            - row["low"]
        )

        right_error = (
            row["high"]
            - delta
        )

        ax.errorbar(
            delta,
            i,
            xerr=np.array(
                [
                    [left_error],
                    [right_error],
                ]
            ),
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=2,
            color=row["color"],
            ecolor=row["color"],
        )

    ax.axvline(
        0,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_yticks(
        range(len(rows))
    )

    ax.set_yticklabels(
        [row["label"] for row in rows]
    )

    ax.set_xlabel(
        "Разность Macro-F1(2–5)"
    )

    ax.set_title(
        "Ключевые парные различия моделей"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "01_bootstrap_key_color",
    )


# ============================================================
# РИС. 2
# Learning curves
# ============================================================

def load_learning_curve(path: Path):
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    summary = (
        df
        .groupby(
            "fraction"
        )[
            "macro_f1_2_5"
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

    summary[
        "percent"
    ] = (
        summary[
            "fraction"
        ]
        * 100
    )

    return summary


def plot_learning_curves():
    curves = [
        (
            "TF-IDF",
            RESULTS
            / "learning_curve"
            / "tfidf_learning_curve.csv",
            COLORS["blue"],
            "o",
        ),
        (
            "cc.ru.300",
            RESULTS
            / "learning_curve"
            / "cc_fasttext_learning_curve.csv",
            COLORS["orange"],
            "s",
        ),
        (
            "Bethesda fastText",
            RESULTS
            / "learning_curve"
            / "bethesda_fasttext_learning_curve.csv",
            COLORS["green"],
            "^",
        ),
        (
            "native fastText",
            RESULTS
            / "learning_curve_v2"
            / "native_fasttext_learning_curve.csv",
            COLORS["red"],
            "D",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5)
    )

    for (
        label,
        path,
        color,
        marker,
    ) in curves:

        summary = (
            load_learning_curve(
                path
            )
        )

        ax.errorbar(
            summary[
                "percent"
            ],
            summary[
                "mean"
            ],
            yerr=summary[
                "std"
            ],
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=3,
        )

    ax.set_xlabel(
        "Доля целевой обучающей выборки, %"
    )

    ax.set_ylabel(
        "Macro-F1(2–5)"
    )

    ax.set_title(
        "Зависимость качества от объёма целевых данных"
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
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "02_learning_curves_color",
    )


# ============================================================
# РИС. 3
# Subword + OOV
# ============================================================

def plot_subword_oov():
    ablation_path = (
        RESULTS
        / "subword_ablation"
        / "subword_ablation_summary.csv"
    )

    oov_path = (
        RESULTS
        / "oov_document"
        / "oov_metrics_summary.csv"
    )

    ablation = pd.read_csv(
        ablation_path,
        encoding="utf-8-sig",
    )

    oov = pd.read_csv(
        oov_path,
        encoding="utf-8-sig",
    )

    order = [
        "Без subword",
        "2–5",
        "3–6",
        "4–7",
    ]

    ablation = (
        ablation
        .set_index(
            "display_label"
        )
        .loc[
            order
        ]
        .reset_index()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.2)
    )

    # --------------------------------------------------------
    # (a) Общий ablation
    # --------------------------------------------------------

    ax = axes[0]

    bar_colors = [
        COLORS["red"],
        COLORS["green"],
        COLORS["blue"],
        COLORS["orange"],
    ]

    x = np.arange(
        len(
            ablation
        )
    )

    bars = ax.bar(
        x,
        ablation[
            "macro_f1_2_5_mean"
        ],
        yerr=ablation[
            "macro_f1_2_5_std"
        ],
        capsize=4,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        order
    )

    ax.set_xlabel(
        "Диапазон символьных n-грамм"
    )

    ax.set_ylabel(
        "Macro-F1(2–5)"
    )

    ax.set_title(
        "(а) Ablation subword-механизма"
    )

    ax.grid(
        axis="y",
        alpha=0.22,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    for bar, value in zip(
        bars,
        ablation[
            "macro_f1_2_5_mean"
        ],
    ):
        ax.text(
            (
                bar.get_x()
                + bar.get_width() / 2
            ),
            (
                bar.get_height()
                + 0.012
            ),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # (b) OOV: 2–5 vs no subword
    # --------------------------------------------------------

    ax = axes[1]

    model_settings = [
        (
            "bethesda_ablation_2_5",
            "Subword 2–5",
            COLORS["green"],
            "o",
        ),
        (
            "bethesda_ablation_no_subword",
            "Без subword",
            COLORS["red"],
            "s",
        ),
    ]

    for (
        model_name,
        label,
        color,
        marker,
    ) in model_settings:

        part = (
            oov[
                oov[
                    "model"
                ]
                == model_name
            ]
            .sort_values(
                "oov_group_id"
            )
        )

        ax.errorbar(
            part[
                "oov_group_id"
            ],
            part[
                "macro_f1_mean"
            ],
            yerr=part[
                "macro_f1_std"
            ],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=3,
            label=label,
        )

    labels = (
        oov[
            oov[
                "model"
            ]
            == "bethesda_ablation_2_5"
        ]
        .sort_values(
            "oov_group_id"
        )[
            "oov_group_label"
        ]
        .tolist()
    )

    ax.set_xticks(
        range(
            len(
                labels
            )
        )
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "Доля OOV-токенов в документе"
    )

    ax.set_ylabel(
        "Macro-F1(2–5)"
    )

    ax.set_title(
        "(б) Качество при различном OOV"
    )

    ax.grid(
        alpha=0.22,
    )

    ax.legend(
        frameon=False,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout(
        w_pad=2.0
    )

    save_figure(
        fig,
        "03_subword_oov_color",
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Building colored figures..."
    )

    plot_bootstrap()

    plot_learning_curves()

    plot_subword_oov()

    print()
    print(
        f"Done -> {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()