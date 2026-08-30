import argparse
import json
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_embeddings(texts, ft_model):
    vectors = []

    for text in texts:
        vector = ft_model.get_sentence_vector(str(text))
        vectors.append(vector.astype(np.float32))

    return np.vstack(vectors)


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    output_path,
    normalize=None,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format=".2f" if normalize else "d",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Папка первого эксперимента Bethesda -> EU-TIRADS",
    )

    parser.add_argument(
        "--output-dir",
        default="experiments_eutirads",
    )

    parser.add_argument(
        "--run-name",
        default="native_eutirads_ft",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. Загружаем ТОЧНО тот же датасет и split
    # ========================================================

    df = pd.read_csv(
        baseline_dir / "dataset_used.csv"
    )

    split_df = pd.read_csv(
        baseline_dir / "split.csv"
    )

    # Соединяем по исходной строке Excel
    df = df.merge(
        split_df[["source_row", "split"]],
        on="source_row",
        how="inner",
    )

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("=" * 60)
    print("EU-TIRADS NATIVE FASTTEXT EXPERIMENT")
    print("=" * 60)

    print(f"Train: {len(train_df)}")
    print(f"Test:  {len(test_df)}")

    print("\nTrain classes:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nTest classes:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # 2. Формируем corpus ТОЛЬКО ИЗ TRAIN
    # ========================================================

    corpus_path = run_dir / "corpus_train.txt"

    with open(
        corpus_path,
        "w",
        encoding="utf-8",
    ) as f:
        for text in train_df["text_model"]:
            text = str(text).strip()

            if text:
                f.write(text + "\n")

    # ========================================================
    # 3. Обучаем новый fastText с нуля
    # ========================================================

    print("\nОбучение fastText на EU-TIRADS train...")

    start = time.time()

    ft_model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=200,
        minn=2,
        maxn=5,
    )

    ft_train_time = time.time() - start

    print(
        f"fastText обучен за "
        f"{ft_train_time:.2f} сек."
    )

    ft_model.save_model(
        str(run_dir / "eutirads_fasttext.bin")
    )

    # ========================================================
    # 4. Получаем embeddings
    # ========================================================

    embedding_start = time.time()

    X_train = build_embeddings(
        train_df["text_model"],
        ft_model,
    )

    X_test = build_embeddings(
        test_df["text_model"],
        ft_model,
    )

    embedding_time = (
        time.time() - embedding_start
    )

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(
        f"X_train: {X_train.shape}"
    )
    print(
        f"X_test:  {X_test.shape}"
    )

    # ========================================================
    # 5. Новый классификатор
    # ========================================================

    class_weight = (
        None
        if args.class_weight == "none"
        else "balanced"
    )

    classifier = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "lr",
            LogisticRegression(
                C=args.C,
                solver="lbfgs",
                max_iter=args.max_iter,
                class_weight=class_weight,
            ),
        ),
    ])

    start = time.time()

    classifier.fit(
        X_train,
        y_train,
    )

    lr_train_time = time.time() - start

    # ========================================================
    # 6. Prediction
    # ========================================================

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    classes = classifier.named_steps[
        "lr"
    ].classes_

    # ========================================================
    # 7. Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # Дополнительно считаем macro-F1
    # только по классам 2-5
    mask_2_5 = np.isin(
        y_test,
        [2, 3, 4, 5],
    )

    macro_f1_2_5 = f1_score(
        y_test[mask_2_5],
        y_pred[mask_2_5],
        labels=[2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )

    # ========================================================
    # 8. Сохраняем результаты
    # ========================================================

    metrics = {
        "experiment":
            "EU-TIRADS fastText -> EU-TIRADS classifier",

        "fasttext_source":
            "EU-TIRADS train only",

        "fasttext_model":
            "skipgram",

        "dim": 200,
        "minn": 2,
        "maxn": 5,

        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),

        "accuracy": float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "macro_f1_classes_2_5":
            float(macro_f1_2_5),

        "weighted_f1":
            float(weighted_f1),

        "fasttext_train_time_sec":
            float(ft_train_time),

        "embedding_time_sec":
            float(embedding_time),

        "lr_train_time_sec":
            float(lr_train_time),
    }

    with open(
        run_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        run_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    pd.DataFrame(
        report_dict
    ).T.to_csv(
        run_dir / "classification_report.csv",
        encoding="utf-8-sig",
    )

    # Confusion matrix
    save_confusion_matrix(
        y_test,
        y_pred,
        classes,
        run_dir / "confusion_matrix.png",
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        classes,
        run_dir /
        "confusion_matrix_normalized.png",
        normalize="true",
    )

    # Predictions
    predictions = test_df[
        [
            "source_row",
            "input_raw",
            "input_clean",
            "text_model",
            "label",
        ]
    ].copy()

    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred

    predictions["correct"] = (
        predictions["y_true"]
        ==
        predictions["y_pred"]
    )

    for i, cls in enumerate(classes):
        predictions[
            f"proba_{cls}"
        ] = y_proba[:, i]

    predictions.to_excel(
        run_dir / "predictions.xlsx",
        index=False,
    )

    predictions[
        ~predictions["correct"]
    ].to_excel(
        run_dir / "errors.xlsx",
        index=False,
    )

    # Сохраняем LR
    joblib.dump(
        classifier,
        run_dir / "eutirads_lr.joblib",
    )

    # ========================================================
    # 9. Console
    # ========================================================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:          "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro precision:   "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall:      "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1:.4f}"
    )

    print(
        f"Macro F1 (2-5):    "
        f"{macro_f1_2_5:.4f}"
    )

    print(
        f"Weighted F1:       "
        f"{weighted_f1:.4f}"
    )

    print("\nClassification report:")
    print(report_text)

    print(
        f"\nРезультаты сохранены в: "
        f"{run_dir}"
    )


if __name__ == "__main__":
    main()