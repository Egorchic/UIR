from __future__ import annotations

from pathlib import Path

import fasttext
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_fasttext_model(
    model_path: str | Path,
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найдена fastText-модель:\n{model_path}"
        )

    return fasttext.load_model(
        str(model_path)
    )


def vectorize_texts(
    model,
    texts,
) -> np.ndarray:
    """
    Получает sentence vector для каждого текста.

    Модель fastText заморожена и не обучается
    на EU-TIRADS.
    """

    vectors = []

    for text in texts:
        text = str(text).replace("\n", " ").strip()

        vector = model.get_sentence_vector(
            text
        )

        vectors.append(vector)

    return np.asarray(
        vectors,
        dtype=np.float32,
    )


def build_classifier_pipeline() -> Pipeline:
    """
    StandardScaler обучается только внутри train/CV,
    поэтому утечки статистик test здесь нет.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                ),
            ),
        ]
    )


def tune_fasttext_classifier(
    x_train: np.ndarray,
    y_train,
    seed: int,
):
    """
    Подбираются только параметры классификатора.

    Само Bethesda fastText-пространство
    остаётся неизменным.
    """

    scorer = make_scorer(
        f1_score,
        labels=[2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )

    inner_cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed,
    )

    pipeline = build_classifier_pipeline()

    param_grid = {
        "classifier__C": [
            0.1,
            1.0,
            10.0,
        ],

        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )

    search.fit(
        x_train,
        y_train,
    )

    return search


def get_fasttext_metadata(model) -> dict:
    """
    Сохраняет параметры исходной fastText-модели,
    если текущая версия Python-библиотеки
    позволяет их прочитать.
    """

    metadata = {
        "fasttext_dimension": int(
            model.get_dimension()
        ),
        "fasttext_vocab_size": len(
            model.get_words()
        ),
    }

    try:
        args = model.get_args()

        fields = [
            "model",
            "dim",
            "ws",
            "epoch",
            "minCount",
            "minn",
            "maxn",
            "neg",
            "wordNgrams",
            "lr",
        ]

        for field in fields:
            value = getattr(
                args,
                field,
                None,
            )

            if value is None:
                continue

            # Некоторые fastText-поля могут быть enum.
            if isinstance(
                value,
                (str, int, float, bool)
            ):
                metadata[
                    f"fasttext_{field}"
                ] = value
            else:
                metadata[
                    f"fasttext_{field}"
                ] = str(value)

    except Exception:
        # Не критично: основные параметры модели
        # всё равно сохраняются выше.
        pass

    return metadata