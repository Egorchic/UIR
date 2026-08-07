import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = "experiments/results.csv"
PLOTS_DIR = "plots"

MACRO_F1_YMIN = 0.90
MACRO_F1_YMAX = 0.935


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    df = pd.read_csv(RESULTS_CSV)

    numeric_cols = [
        "dim", "minn", "maxn", "C",
        "accuracy", "macro_f1", "weighted_f1",
        "embedding_time_sec", "train_time_sec", "predict_time_sec"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Удаляем дубли конфигураций
    df = df.sort_values("run_name").drop_duplicates(
        subset=["fasttext_model", "dim", "minn", "maxn", "class_weight", "C"],
        keep="first"
    ).reset_index(drop=True)

    return df


def annotate_points(ax, x, y, fmt="{:.4f}", dy=0.0006):
    for xi, yi in zip(x, y):
        ax.text(xi, yi + dy, fmt.format(yi), ha="center", va="bottom", fontsize=9)


def save_line_plot(x, y, xlabel, ylabel, title, filename, ylim=None):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.plot(x, y, marker="o")
    annotate_points(ax, x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


def save_dot_plot(labels, values, xlabel, ylabel, title, filename, ylim=None):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    x = list(range(len(labels)))
    ax.scatter(x, values, s=90)

    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.0006, f"{yi:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


def plot_dim_experiments(df):
    subset = df[
        (df["fasttext_model"] == "skipgram") &
        (df["minn"] == 2) &
        (df["maxn"] == 5) &
        (df["class_weight"] == "balanced") &
        (df["C"] == 1.0)
    ].copy()

    subset = subset.sort_values("dim")

    if subset.empty:
        print("Нет данных для графиков по размерности.")
        return

    save_line_plot(
        x=subset["dim"].tolist(),
        y=subset["macro_f1"].tolist(),
        xlabel="Размерность вектора",
        ylabel="Macro-F1",
        title="Влияние размерности fastText на Macro-F1",
        filename="macro_f1_vs_dim.png",
        ylim=(MACRO_F1_YMIN, MACRO_F1_YMAX)
    )

    save_line_plot(
        x=subset["dim"].tolist(),
        y=subset["embedding_time_sec"].tolist(),
        xlabel="Размерность вектора",
        ylabel="Время построения эмбеддингов, с",
        title="Влияние размерности fastText на время построения эмбеддингов",
        filename="embedding_time_vs_dim.png"
    )


def plot_model_comparison(df):
    subset = df[
        (df["dim"] == 200) &
        (df["minn"] == 2) &
        (df["maxn"] == 5) &
        (df["class_weight"] == "balanced") &
        (df["C"] == 1.0) &
        (df["fasttext_model"].isin(["skipgram", "cbow"]))
    ].copy()

    # фиксируем порядок
    order = ["cbow", "skipgram"]
    subset["order"] = subset["fasttext_model"].map({name: i for i, name in enumerate(order)})
    subset = subset.sort_values("order")

    if subset.empty:
        print("Нет данных для сравнения skipgram и cbow.")
        return

    save_dot_plot(
        labels=subset["fasttext_model"].tolist(),
        values=subset["macro_f1"].tolist(),
        xlabel="Архитектура fastText",
        ylabel="Macro-F1",
        title="Сравнение skipgram и cbow по Macro-F1",
        filename="skipgram_vs_cbow_macro_f1.png",
        ylim=(MACRO_F1_YMIN, MACRO_F1_YMAX)
    )


def plot_subword_comparison(df):
    subset = df[
        (df["fasttext_model"] == "skipgram") &
        (df["dim"] == 200) &
        (df["class_weight"] == "balanced") &
        (df["C"] == 1.0) &
        (
            ((df["minn"] == 0) & (df["maxn"] == 0)) |
            ((df["minn"] == 2) & (df["maxn"] == 5))
        )
    ].copy()

    if subset.empty:
        print("Нет данных для сравнения subword и no-subword.")
        return

    subset["subword_mode"] = subset.apply(
        lambda row: "без subword" if row["minn"] == 0 and row["maxn"] == 0 else "с subword",
        axis=1
    )
    subset["order"] = subset["subword_mode"].map({"без subword": 0, "с subword": 1})
    subset = subset.sort_values("order")

    save_dot_plot(
        labels=subset["subword_mode"].tolist(),
        values=subset["macro_f1"].tolist(),
        xlabel="Режим построения признаков",
        ylabel="Macro-F1",
        title="Влияние subword n-грамм на Macro-F1",
        filename="subword_vs_nosubword_macro_f1.png",
        ylim=(MACRO_F1_YMIN, MACRO_F1_YMAX)
    )


def plot_class_weight_comparison(df):
    subset = df[
        (df["fasttext_model"] == "skipgram") &
        (df["dim"] == 200) &
        (df["minn"] == 2) &
        (df["maxn"] == 5) &
        (df["C"] == 1.0) &
        (df["class_weight"].isin(["none", "balanced"]))
    ].copy()

    if subset.empty:
        print("Нет данных для сравнения class_weight.")
        return

    subset["class_weight_label"] = subset["class_weight"].replace({
        "none": "без балансировки",
        "balanced": "balanced"
    })
    subset["order"] = subset["class_weight_label"].map({
        "без балансировки": 0,
        "balanced": 1
    })
    subset = subset.sort_values("order")

    save_dot_plot(
        labels=subset["class_weight_label"].tolist(),
        values=subset["macro_f1"].tolist(),
        xlabel="Параметр class_weight",
        ylabel="Macro-F1",
        title="Влияние балансировки классов на Macro-F1",
        filename="class_weight_macro_f1.png",
        ylim=(MACRO_F1_YMIN, MACRO_F1_YMAX)
    )


def plot_c_comparison(df):
    subset = df[
        (df["fasttext_model"] == "skipgram") &
        (df["dim"] == 200) &
        (df["minn"] == 2) &
        (df["maxn"] == 5) &
        (df["class_weight"] == "none") &
        (df["C"].isin([0.1, 1.0, 10.0]))
    ].copy()

    subset = subset.sort_values("C")

    if subset.empty:
        print("Нет данных для сравнения C.")
        return

    save_line_plot(
        x=subset["C"].tolist(),
        y=subset["macro_f1"].tolist(),
        xlabel="Параметр C",
        ylabel="Macro-F1",
        title="Влияние параметра C на Macro-F1",
        filename="macro_f1_vs_C.png",
        ylim=(MACRO_F1_YMIN, MACRO_F1_YMAX)
    )


def main():
    ensure_plots_dir()
    df = load_results()

    plot_dim_experiments(df)
    plot_model_comparison(df)
    plot_subword_comparison(df)
    plot_class_weight_comparison(df)
    plot_c_comparison(df)

    print(f"Графики сохранены в папку: {PLOTS_DIR}")


if __name__ == "__main__":
    main()