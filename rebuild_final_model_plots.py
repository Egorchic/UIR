import os
import json
import joblib
import numpy as np
import pandas as pd
import fasttext
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA


PLOTS_DIR = "plots"
MODELS_DIR = "models"

DATA_PATH = "dataset_clean.csv"
FASTTEXT_MODEL_PATH = "models/ft_skipgram_dim200_minn2_maxn5.bin"

CLASS_ORDER = ["I", "II", "III", "IV", "V", "VI"]


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


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


def plot_confusion_matrix_counts(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Матрица ошибок итоговой модели (абсолютные значения)"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_normalized(cm, classes, out_path):
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, where=cm_sum != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Нормированная матрица ошибок итоговой модели"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_true_vs_pred(y_true, y_pred, classes, out_path):
    true_counts = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts.values, width, label="Истинные классы")
    plt.bar(x + width / 2, pred_counts.values, width, label="Предсказанные классы")

    plt.xticks(x, classes)
    plt.xlabel("Класс Bethesda")
    plt.ylabel("Количество объектов")
    plt.title("Сравнение распределений истинных и предсказанных классов")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pca(X_test, y_test, classes, out_path):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 6))
    for cls in classes:
        mask = (y_test == cls)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.7, label=cls)

    plt.xlabel("Первая главная компонента")
    plt.ylabel("Вторая главная компонента")
    plt.title("PCA-визуализация тестовой выборки (истинные классы)")
    plt.legend(title="Классы")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    # 1. Данные
    df = load_dataset(DATA_PATH)

    # 2. fastText
    ft = fasttext.load_model(FASTTEXT_MODEL_PATH)

    # 3. Эмбеддинги
    X, y = build_embeddings(df, ft)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Финальный классификатор:
    # class_weight=None, C=1.0
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=3000,
            class_weight=None,
            C=1.0
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 6. Метрики
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report_text = classification_report(y_test, y_pred)

    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Weighted-F1:", round(weighted_f1, 4))
    print(report_text)

    # 7. Сохраняем модель и классы финальной конфигурации
    joblib.dump(clf, os.path.join(MODELS_DIR, "bethesda_lr_final.joblib"))
    with open(os.path.join(MODELS_DIR, "bethesda_classes_final.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_ORDER, f, ensure_ascii=False, indent=2)

    with open(os.path.join(PLOTS_DIR, "final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # 8. Матрица ошибок
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    plot_confusion_matrix_counts(
        cm=cm,
        classes=CLASS_ORDER,
        out_path=os.path.join(PLOTS_DIR, "cm_counts.png")
    )

    plot_confusion_matrix_normalized(
        cm=cm,
        classes=CLASS_ORDER,
        out_path=os.path.join(PLOTS_DIR, "cm_normalized.png")
    )

    plot_true_vs_pred(
        y_true=y_test,
        y_pred=y_pred,
        classes=CLASS_ORDER,
        out_path=os.path.join(PLOTS_DIR, "bar_true_vs_pred.png")
    )

    plot_pca(
        X_test=X_test,
        y_test=y_test,
        classes=CLASS_ORDER,
        out_path=os.path.join(PLOTS_DIR, "pca_test_true.png")
    )

    print(f"Графики итоговой модели сохранены в папку: {PLOTS_DIR}")


if __name__ == "__main__":
    main()