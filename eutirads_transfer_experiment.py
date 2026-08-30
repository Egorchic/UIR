import argparse
import json
import os
import re
import time
from pathlib import Path

import fasttext
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Предобработка
# ============================================================

RU_STOPWORDS = {
    "и", "а", "но", "или", "ли", "же", "то",
    "в", "во", "на", "у", "к", "ко", "о", "об", "от", "до",
    "по", "из", "за", "под", "над", "при", "для", "без",
    "с", "со", "про", "через",
    "это", "как", "так", "что", "чтобы", "где", "когда",
    "там", "тут", "все", "всё", "этот", "эта", "эти",
    "тот", "та", "те"
}

JUNK_TOKENS = {"nan", "none", "null"}


def preprocess_text(text: str) -> str:
    """
    Предобработка, максимально близкая к исходному проекту УИР.

    В clean_strict явные EU-TIRADS-метки уже удалены.
    Здесь выполняем дальнейшую нормализацию текста.
    """
    text = str(text).lower()

    # Служебные элементы
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", " ", text)
    text = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", text)
    text = re.sub(r"[№#]\s*\w+", " ", text)

    # Дополнительная защита от утечки EU-TIRADS
    text = re.sub(
        r"(?i)\b[eе]\s*u\s*[-–—]?\s*tirads\b(?:\s*[1-5])?",
        " ",
        text
    )

    # В исходном проекте числа удалялись
    text = re.sub(r"\b\d+\b", " ", text)

    # Оставляем только русские/латинские буквы и пробелы
    text = re.sub(r"[^a-zа-яё\s]", " ", text)

    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if len(token) > 1
        and token not in RU_STOPWORDS
        and token not in JUNK_TOKENS
    ]

    return " ".join(tokens)


# ============================================================
# Работа с датасетом
# ============================================================

def load_dataset(path: str, sheet_name: str = "clean_strict"):
    df = pd.read_excel(path, sheet_name=sheet_name)

    required_columns = {
        "source_row",
        "input_raw",
        "input_clean",
        "label",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"В датасете отсутствуют необходимые колонки: {sorted(missing)}"
        )

    df = df.copy()

    # Метки EU-TIRADS
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([1, 2, 3, 4, 5])].copy()
    df["label"] = df["label"].astype(int)

    # Предобработка текста
    df["text_model"] = (
        df["input_clean"]
        .fillna("")
        .astype(str)
        .apply(preprocess_text)
    )

    # Удаляем пустые тексты после preprocessing
    before = len(df)

    df = df[
        df["text_model"].str.len() > 0
    ].copy()

    removed_empty = before - len(df)

    if removed_empty:
        print(
            f"Удалено пустых после preprocessing: "
            f"{removed_empty}"
        )

    # --------------------------------------------------------
    # Удаляем точные дубликаты
    # --------------------------------------------------------
    #
    # Если после preprocessing получился одинаковый текст
    # с одинаковой меткой, оставляем только один экземпляр.
    #

    before = len(df)

    df = df.drop_duplicates(
        subset=["text_model", "label"]
    ).copy()

    removed_duplicates = before - len(df)

    print(
        f"Удалено точных дубликатов text_model + label: "
        f"{removed_duplicates}"
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# Разбиение train / test
# ============================================================

def make_split(df, random_state=42):
    indices = np.arange(len(df))
    y = df["label"].to_numpy()

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    return train_idx, test_idx



# ============================================================
# fastText embeddings
# ============================================================

def build_embeddings(texts, ft_model):
    vectors = []

    for text in texts:
        vector = ft_model.get_sentence_vector(text)
        vectors.append(vector.astype(np.float32))

    return np.vstack(vectors)


# ============================================================
# Графики
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    output_path,
    normalize=None
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format=".2f" if normalize else "d")

    if normalize:
        ax.set_title("Normalized confusion matrix")
    else:
        ax.set_title("Confusion matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return cm


# ============================================================
# Основной эксперимент
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "EU-TIRADS classification using frozen Bethesda fastText embeddings"
        )
    )

    parser.add_argument(
        "--data",
        default="usidatanew_cleaned.xlsx"
    )

    parser.add_argument(
        "--sheet",
        default="clean_strict"
    )

    parser.add_argument(
        "--ft-model",
        required=True,
        help="Путь к старой Bethesda fastText .bin модели"
    )

    parser.add_argument(
        "--run-name",
        default="eutirads_transfer_bethesda_ft"
    )

    parser.add_argument(
        "--output-dir",
        default="experiments_eutirads"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000
    )

    args = parser.parse_args()

    class_weight = (
        None
        if args.class_weight == "none"
        else "balanced"
    )

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EU-TIRADS TRANSFER EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    df = load_dataset(
        args.data,
        sheet_name=args.sheet
    )

    print(f"Всего объектов: {len(df)}")
    print("\nРаспределение классов:")
    print(df["label"].value_counts().sort_index())

    # Сохраняем фактически использованный датасет
    df.to_csv(
        run_dir / "dataset_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 2. Train / test split
    # --------------------------------------------------------

    train_idx, test_idx = make_split(
        df,
        random_state=args.random_state
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"\nTrain: {len(train_df)}")
    print(f"Test:  {len(test_df)}")

    print("\nTrain classes:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTest classes:")
    print(test_df["label"].value_counts().sort_index())

    # Сохраняем разбиение
    split_df = df[
        [
            "source_row",
            "label",
        ]
    ].copy()

    split_df["split"] = "train"
    split_df.loc[test_idx, "split"] = "test"

    split_df.to_csv(
        run_dir / "split.csv",
        index=False,
        encoding="utf-8-sig"
    )

    split_df["split"] = "train"
    split_df.loc[test_idx, "split"] = "test"

    split_df.to_csv(
        run_dir / "split.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 3. Загружаем ЗАМОРОЖЕННЫЙ Bethesda fastText
    # --------------------------------------------------------

    print("\nЗагрузка fastText...")
    ft_model = fasttext.load_model(args.ft_model)

    print(
        f"Размерность fastText: "
        f"{ft_model.get_dimension()}"
    )

    # --------------------------------------------------------
    # 4. Embeddings
    # --------------------------------------------------------

    embedding_start = time.time()

    X = build_embeddings(
        df["text_model"].to_numpy(),
        ft_model
    )

    embedding_time = time.time() - embedding_start

    y = df["label"].to_numpy()

    print(
        f"Embeddings: {X.shape}, "
        f"{embedding_time:.2f} sec"
    )

    np.save(
        run_dir / "X_embeddings.npy",
        X
    )

    np.save(
        run_dir / "y_labels.npy",
        y
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # --------------------------------------------------------
    # 5. Новый EU-TIRADS классификатор
    # --------------------------------------------------------

    classifier = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "lr",
            LogisticRegression(
                C=args.C,
                solver="lbfgs",
                max_iter=args.max_iter,
                class_weight=class_weight
            )
        )
    ])

    train_start = time.time()

    classifier.fit(
        X_train,
        y_train
    )

    train_time = time.time() - train_start

    # --------------------------------------------------------
    # 6. Predictions
    # --------------------------------------------------------

    predict_start = time.time()

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    predict_time = time.time() - predict_start

    classes = classifier.named_steps["lr"].classes_

    # --------------------------------------------------------
    # 7. Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0
    )

    # --------------------------------------------------------
    # 8. Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes]
    )

    cm_df.to_csv(
        run_dir / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        classes,
        run_dir / "confusion_matrix.png"
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        classes,
        run_dir / "confusion_matrix_normalized.png",
        normalize="true"
    )

    # --------------------------------------------------------
    # 9. Classification report
    # --------------------------------------------------------

    report_df = pd.DataFrame(
        report_dict
    ).T

    report_df.to_csv(
        run_dir / "classification_report.csv",
        encoding="utf-8-sig"
    )

    with open(
        run_dir / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_text)

    # --------------------------------------------------------
    # 10. Predictions для анализа ошибок
    # --------------------------------------------------------

    predictions = test_df[
        [
            "source_row",
            "input_raw",
            "input_clean",
            "text_model",
        ]
    ].copy()

    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    predictions["correct"] = (
        predictions["y_true"] == predictions["y_pred"]
    )

    for i, cls in enumerate(classes):
        predictions[f"proba_{cls}"] = y_proba[:, i]

    predictions.to_excel(
        run_dir / "predictions.xlsx",
        index=False
    )

    # Отдельно сохраняем ошибки
    errors = predictions[
        ~predictions["correct"]
    ].copy()

    errors.to_excel(
        run_dir / "errors.xlsx",
        index=False
    )

    # --------------------------------------------------------
    # 11. Class distributions
    # --------------------------------------------------------

    distribution = pd.DataFrame({
        "all": df["label"].value_counts().sort_index(),
        "train": train_df["label"].value_counts().sort_index(),
        "test": test_df["label"].value_counts().sort_index(),
    }).fillna(0).astype(int)

    distribution.to_csv(
        run_dir / "class_distribution.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 12. Сохраняем классификатор
    # --------------------------------------------------------

    joblib.dump(
        classifier,
        run_dir / "eutirads_lr.joblib"
    )

    with open(
        run_dir / "classes.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            [int(c) for c in classes],
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 13. Общие результаты эксперимента
    # --------------------------------------------------------

    metrics = {
        "experiment": "Bethesda fastText -> EU-TIRADS classifier",

        "dataset": str(args.data),
        "sheet": args.sheet,

        "fasttext_model": str(args.ft_model),
        "embedding_dimension": int(ft_model.get_dimension()),

        "random_state": args.random_state,

        "C": args.C,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,

        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),

        "embedding_time_sec": round(
            embedding_time, 4
        ),

        "train_time_sec": round(
            train_time, 4
        ),

        "predict_time_sec": round(
            predict_time, 4
        ),

        "accuracy": round(
            float(accuracy), 4
        ),

        "balanced_accuracy": round(
            float(balanced_accuracy), 4
        ),

        "macro_precision": round(
            float(macro_precision), 4
        ),

        "macro_recall": round(
            float(macro_recall), 4
        ),

        "macro_f1": round(
            float(macro_f1), 4
        ),

        "weighted_f1": round(
            float(weighted_f1), 4
        ),
    }

    with open(
        run_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 14. Итог
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro precision:   {macro_precision:.4f}")
    print(f"Macro recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print("\nClassification report:")
    print(report_text)

    print(f"\nРезультаты сохранены в:")
    print(run_dir)


if __name__ == "__main__":
    main()