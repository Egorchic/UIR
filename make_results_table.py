import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = "experiments/results.csv"
OUTPUT_DIR = "experiments/summary"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    return df


def make_pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    pretty_df = df.copy()

    pretty_df["Архитектура"] = pretty_df["fasttext_model"]
    pretty_df["Размерность"] = pretty_df["dim"].astype("Int64")
    pretty_df["Subword"] = pretty_df.apply(
        lambda row: "без subword" if row["minn"] == 0 and row["maxn"] == 0 else f"{int(row['minn'])}-{int(row['maxn'])}",
        axis=1
    )
    pretty_df["Балансировка"] = pretty_df["class_weight"].replace({
        "balanced": "balanced",
        "none": "нет"
    })
    pretty_df["C"] = pretty_df["C"].round(3)

    pretty_df["Accuracy"] = pretty_df["accuracy"].round(4)
    pretty_df["Macro-F1"] = pretty_df["macro_f1"].round(4)
    pretty_df["Weighted-F1"] = pretty_df["weighted_f1"].round(4)
    pretty_df["Время эмбеддингов, с"] = pretty_df["embedding_time_sec"].round(3)
    pretty_df["Время обучения LR, с"] = pretty_df["train_time_sec"].round(3)
    pretty_df["Время предсказания, с"] = pretty_df["predict_time_sec"].round(4)

    cols = [
        "run_name",
        "Архитектура",
        "Размерность",
        "Subword",
        "Балансировка",
        "C",
        "Accuracy",
        "Macro-F1",
        "Weighted-F1",
        "Время эмбеддингов, с",
        "Время обучения LR, с",
        "Время предсказания, с"
    ]

    pretty_df = pretty_df[cols].rename(columns={"run_name": "Конфигурация"})
    pretty_df = pretty_df.sort_values(["Macro-F1", "Accuracy"], ascending=[False, False]).reset_index(drop=True)

    return pretty_df


def save_csv_and_excel(df: pd.DataFrame):
    csv_path = os.path.join(OUTPUT_DIR, "results_pretty.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, "results_pretty.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Не удалось сохранить xlsx: {e}")

    print(f"Сохранено: {csv_path}")
    print(f"Сохранено: {xlsx_path}")


def save_top_configs(df: pd.DataFrame, top_n: int = 5):
    top_df = df.head(top_n)
    top_csv = os.path.join(OUTPUT_DIR, f"top_{top_n}_configs.csv")
    top_df.to_csv(top_csv, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {top_csv}")


def dataframe_to_png(df: pd.DataFrame, filename: str, title: str):
    # Возьмем только разумное число строк для картинки
    display_df = df.copy()

    n_rows, n_cols = display_df.shape

    fig_width = max(12, n_cols * 1.4)
    fig_height = max(4, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.2)

    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Сохранено: {out_path}")


def main():
    ensure_output_dir()

    raw_df = load_results()
    pretty_df = make_pretty_columns(raw_df)

    save_csv_and_excel(pretty_df)
    save_top_configs(pretty_df, top_n=5)

    # Полная таблица
    dataframe_to_png(
        df=pretty_df,
        filename="results_table_full.png",
        title="Сводная таблица результатов экспериментов"
    )

    # Топ-5
    dataframe_to_png(
        df=pretty_df.head(5),
        filename="results_table_top5.png",
        title="Лучшие конфигурации по значению Macro-F1"
    )


if __name__ == "__main__":
    main()