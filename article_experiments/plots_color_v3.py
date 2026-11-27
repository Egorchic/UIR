from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Пути
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS = PROJECT_ROOT / "article_results"

OUTPUT_DIR = RESULTS / "figures_color_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Стиль
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "lines.linewidth": 2.0,
    }
)

COLORS = {
    "tfidf": "#1f77b4",       # синий
    "cc": "#ff7f0e",          # оранжевый
    "bethesda": "#2ca02c",    # зелёный
    "native": "#d62728",      # красный
    "rubiobert": "#9467bd",   # фиолетовый
    "gray": "#7f7f7f",
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
# Вспомогательные функции
# ============================================================

def load_csv(path: Path):
    return pd.read_csv(path, encoding="utf-8-sig")


def summarize_learning_curve(path: Path):
    df = load_csv(path)

    summary = (
        df.groupby("fraction")["macro_f1_2_5"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("fraction")
    )

    summary["percent"] = summary["fraction"] * 100
    return summary


# ============================================================
# Рис. 1. Learning curve
# ============================================================

def plot_learning_curves():
    curves = [
        (
            "TF-IDF",
            RESULTS / "learning_curve" / "tfidf_learning_curve.csv",
            COLORS["tfidf"],
            "o",
            "-",
        ),
        (
            "cc.ru.300",
            RESULTS / "learning_curve" / "cc_fasttext_learning_curve.csv",
            COLORS["cc"],
            "s",
            "--",
        ),
        (
            "Bethesda FT",
            RESULTS / "learning_curve" / "bethesda_fasttext_learning_curve.csv",
            COLORS["bethesda"],
            "^",
            "-",
        ),
        (
            "native FT",
            RESULTS / "learning_curve_v2" / "native_fasttext_learning_curve.csv",
            COLORS["native"],
            "d",
            ":",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for label, path, color, marker, linestyle in curves:
        summary = summarize_learning_curve(path)

        ax.errorbar(
            summary["percent"],
            summary["mean"],
            yerr=summary["std"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=8,
            linewidth=2.2,
            capsize=4,
        )

    ax.set_xlabel("Доля обучающей выборки, %")
    ax.set_ylabel("Macro-F1(2–5)")

    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_ylim(0.47, 0.76)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout()

    save_figure(fig, "01_learning_curve_color")


# ============================================================
# Рис. 2. Bootstrap / forest plot
# ============================================================

def plot_bootstrap():
    path = RESULTS / "summary" / "bootstrap_pairwise_macro_f1_2_5.csv"
    df = load_csv(path)

    comparisons = [
        (
            "bethesda_fasttext",
            "native_fasttext",
            "Bethesda FT − native FT",
            COLORS["bethesda"],
        ),
        (
            "rubiobert",
            "rubert",
            "RuBioBERT − RuBERT",
            COLORS["rubiobert"],
        ),
        (
            "cc_fasttext",
            "bethesda_fasttext",
            "cc.ru.300 − Bethesda FT",
            COLORS["cc"],
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
                f"Не найдено сравнение: {model_a} - {model_b}"
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

    # порядок как на скриншоте: сверху Bethesda-native
    rows = rows

    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    y_positions = np.arange(len(rows))[::-1]

    for y, row in zip(y_positions, rows):
        delta = row["delta"]
        left = delta - row["low"]
        right = row["high"] - delta

        ax.errorbar(
            delta,
            y,
            xerr=np.array([[left], [right]]),
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            markersize=8,
            elinewidth=2.2,
            capsize=6,
            capthick=2.0,
        )

    ax.axvline(
        0,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.8,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows])

    ax.set_xlabel("Разность Macro-F1(2–5)")
    ax.set_xlim(-0.01, 0.16)

    ax.grid(axis="x", alpha=0.25)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout()

    save_figure(fig, "02_bootstrap_key_color")


# ============================================================
# Рис. 3. Subword + OOV
# ============================================================

def plot_subword_oov():
    ablation_path = RESULTS / "subword_ablation" / "subword_ablation_summary.csv"
    oov_path = RESULTS / "oov_document" / "oov_metrics_summary.csv"

    ablation = load_csv(ablation_path)
    oov = load_csv(oov_path)

    order = ["Без subword", "2–5", "3–6", "4–7"]

    ablation = (
        ablation.set_index("display_label")
        .loc[order]
        .reset_index()
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.2, 8.2),
        gridspec_kw={"height_ratios": [1.0, 1.1]},
    )

    # ---------------------- (a) ----------------------
    ax = axes[0]

    x = np.arange(len(order))

    color_map = {
        "Без subword": COLORS["native"],
        "2–5": COLORS["bethesda"],
        "3–6": COLORS["tfidf"],
        "4–7": COLORS["cc"],
    }

    for i, row in ablation.iterrows():
        label = row["display_label"]
        ax.errorbar(
            i,
            row["macro_f1_2_5_mean"],
            yerr=row["macro_f1_2_5_std"],
            fmt="o",
            color=color_map[label],
            ecolor=color_map[label],
            markersize=10,
            elinewidth=2.2,
            capsize=5,
            capthick=2.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["без\nsubword", "2–5", "3–6", "4–7"])
    ax.set_ylabel("Macro-F1(2–5)")
    ax.set_title("(а) Диапазон символьных n-грамм", loc="left")
    ax.set_ylim(0.32, 0.72)

    ax.grid(axis="y", alpha=0.25)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # ---------------------- (б) ----------------------
    ax = axes[1]

    settings = [
        (
            "bethesda_ablation_2_5",
            "subword 2–5",
            COLORS["bethesda"],
            "o",
            "-",
        ),
        (
            "bethesda_ablation_no_subword",
            "без subword",
            COLORS["native"],
            "s",
            "--",
        ),
    ]

    for model_name, label, color, marker, linestyle in settings:
        part = (
            oov[oov["model"] == model_name]
            .sort_values("oov_group_id")
        )

        ax.errorbar(
            part["oov_group_id"],
            part["macro_f1_mean"],
            yerr=part["macro_f1_std"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=8,
            linewidth=2.0,
            capsize=4,
        )

    labels = (
        oov[oov["model"] == "bethesda_ablation_2_5"]
        .sort_values("oov_group_id")["oov_group_label"]
        .tolist()
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Доля OOV-токенов, %")
    ax.set_ylabel("Macro-F1(2–5)")
    ax.set_title("(б) Качество по OOV-группам", loc="left")
    ax.set_ylim(0.28, 0.81)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout(h_pad=1.2)

    save_figure(fig, "03_subword_oov_color")


# ============================================================
# Main
# ============================================================

def main():
    print("Building colored figures with article-like scale...")

    plot_learning_curves()
    plot_bootstrap()
    plot_subword_oov()

    print()
    print(f"Done -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()