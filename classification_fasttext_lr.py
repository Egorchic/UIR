import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import fasttext

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate Logistic Regression on fastText embeddings")
    parser.add_argument("--data", type=str, default="dataset_clean.csv")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fastText .bin model")
    parser.add_argument("--run-name", type=str, required=True, help="Experiment run name")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Directory to save experiment results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight", type=str, default="balanced", help='"balanced" or "none"')
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--use-scaler", action="store_true", default=True)
    return parser.parse_args()


def normalize_class_weight(value: str):
    if value.lower() == "none":
        return None
    if value.lower() == "balanced":
        return "balanced"
    raise ValueError("class-weight must be 'balanced' or 'none'")


def load_dataset(path: str):
    df = pd.read_csv(
        path,
        header=None,
        skiprows=1,
        names=["bethesda_label", "text_clean"],
        encoding="utf-8"
    )
    df.columns = ["bethesda_label", "text_clean"]
    df["text_clean"] = df["text_clean"].astype(str).str.strip()
    df = df[df["text_clean"].str.len() > 0].copy()
    return df


def build_embeddings(df: pd.DataFrame, ft_model):
    X = np.vstack(
        df["text_clean"].map(lambda t: ft_model.get_sentence_vector(t).astype(np.float32)).to_numpy()
    )
    y = df["bethesda_label"].astype(str).to_numpy()
    return X, y


def main():
    args = parse_args()
    class_weight = normalize_class_weight(args.class_weight)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # загрузка данных
    df = load_dataset(args.data)

    # загрузка fastText
    ft = fasttext.load_model(args.model_path)

    emb_start = time.time()
    X, y = build_embeddings(df, ft)
    emb_time = time.time() - emb_start

    np.save(os.path.join(run_dir, "X_embeddings.npy"), X)
    np.save(os.path.join(run_dir, "y_labels.npy"), y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    steps = []
    if args.use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "lr",
        LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight=class_weight,
            C=args.C
        )
    ))

    clf = Pipeline(steps)

    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start

    pred_start = time.time()
    pred = clf.predict(X_test)
    pred_time = time.time() - pred_start

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")
    report_dict = classification_report(y_test, pred, output_dict=True)
    report_text = classification_report(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    classes = clf.named_steps["lr"].classes_.tolist()
    per_class = precision_recall_fscore_support(y_test, pred, labels=classes, zero_division=0)

    per_class_metrics = []
    for i, cls in enumerate(classes):
        per_class_metrics.append({
            "class": cls,
            "precision": round(float(per_class[0][i]), 4),
            "recall": round(float(per_class[1][i]), 4),
            "f1": round(float(per_class[2][i]), 4),
            "support": int(per_class[3][i])
        })

    joblib.dump(clf, os.path.join(run_dir, "bethesda_lr.joblib"))
    with open(os.path.join(run_dir, "bethesda_classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(run_dir, "confusion_matrix.npy"), cm)

    metrics = {
        "run_name": args.run_name,
        "data": args.data,
        "fasttext_model_path": args.model_path,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "class_weight": args.class_weight,
        "C": args.C,
        "max_iter": args.max_iter,
        "use_scaler": args.use_scaler,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "embedding_time_sec": round(emb_time, 4),
        "train_time_sec": round(train_time, 4),
        "predict_time_sec": round(pred_time, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(report_dict["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report_dict["macro avg"]["recall"]), 4),
        "per_class_metrics": per_class_metrics
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Run: {args.run_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(report_text)
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()