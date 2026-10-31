from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ALL_LABELS = [1, 2, 3, 4, 5]
PRIMARY_LABELS = [2, 3, 4, 5]


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Единый набор метрик для всех моделей статьи.

    Основная метрика:
        macro_f1_2_5

    Класс 1 исключается только из основной дополнительной
    macro-метрики из-за крайне малого числа объектов.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=ALL_LABELS,
            average="macro",
            zero_division=0,
        ),

        "macro_f1_2_5": f1_score(
            y_true,
            y_pred,
            labels=PRIMARY_LABELS,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=ALL_LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    for label in ALL_LABELS:
        metrics[f"precision_{label}"] = precision_score(
            y_true,
            y_pred,
            labels=[label],
            average="macro",
            zero_division=0,
        )

        metrics[f"recall_{label}"] = recall_score(
            y_true,
            y_pred,
            labels=[label],
            average="macro",
            zero_division=0,
        )

        metrics[f"f1_{label}"] = f1_score(
            y_true,
            y_pred,
            labels=[label],
            average="macro",
            zero_division=0,
        )

    return metrics