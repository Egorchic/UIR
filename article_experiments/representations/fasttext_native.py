from __future__ import annotations

import tempfile
from itertools import product
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Сетка параметров
# ============================================================

FASTTEXT_MODELS = [
    "skipgram",
    "cbow",
]

FASTTEXT_DIMS = [
    100,
    200,
    300,
]

FASTTEXT_SUBWORDS = [
    (0, 0),   # без subword
    (2, 5),
    (3, 6),
]

FASTTEXT_EPOCHS = [
    5,
    20,
]

FASTTEXT_WS = [
    5,
    10,
]

CLASSIFIER_C = [
    0.1,
    1.0,
    10.0,
]

CLASSIFIER_CLASS_WEIGHT = [
    None,
    "balanced",
]


# ============================================================
# Работа с корпусом
# ============================================================

def write_fasttext_corpus(
    texts,
    path: str | Path,
) -> None:

    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for text in texts:
            text = (
                str(text)
                .replace("\n", " ")
                .strip()
            )

            if text:
                f.write(
                    text + "\n"
                )


def train_native_fasttext(
    texts,
    params: dict,
):
    """
    Обучает fastText только на переданных текстах.

    В nested CV сюда передаётся inner-train.
    После выбора параметров — полный outer-train.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        corpus_path = (
            Path(tmp_dir)
            / "corpus.txt"
        )

        write_fasttext_corpus(
            texts,
            corpus_path,
        )

        model = fasttext.train_unsupervised(
            input=str(corpus_path),

            model=params["model"],

            dim=int(
                params["dim"]
            ),

            ws=int(
                params["ws"]
            ),

            epoch=int(
                params["epoch"]
            ),

            minn=int(
                params["minn"]
            ),

            maxn=int(
                params["maxn"]
            ),

            # Маленький целевой корпус:
            # не выбрасываем редкие слова.
            minCount=1,

            # Для большей воспроизводимости.
            thread=1,
        )

    return model


# ============================================================
# Векторизация
# ============================================================

def vectorize_texts(
    model,
    texts,
) -> np.ndarray:

    vectors = []

    for text in texts:

        text = (
            str(text)
            .replace("\n", " ")
            .strip()
        )

        vectors.append(
            model.get_sentence_vector(
                text
            )
        )

    return np.asarray(
        vectors,
        dtype=np.float32,
    )


# ============================================================
# Классификатор
# ============================================================

def build_classifier(
    C: float,
    class_weight,
) -> Pipeline:

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
                    C=float(C),
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def macro_f1_2_5(
    y_true,
    y_pred,
) -> float:

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=[2, 3, 4, 5],
            average="macro",
            zero_division=0,
        )
    )


# ============================================================
# Сетка fastText
# ============================================================

def generate_fasttext_configs():

    configs = []

    for (
        model,
        dim,
        subword,
        epoch,
        ws,
    ) in product(
        FASTTEXT_MODELS,
        FASTTEXT_DIMS,
        FASTTEXT_SUBWORDS,
        FASTTEXT_EPOCHS,
        FASTTEXT_WS,
    ):

        minn, maxn = subword

        configs.append(
            {
                "model": model,
                "dim": int(dim),
                "minn": int(minn),
                "maxn": int(maxn),
                "epoch": int(epoch),
                "ws": int(ws),
            }
        )

    return configs


# ============================================================
# Nested tuning
# ============================================================

def tune_native_fasttext(
    train_texts,
    train_labels,
    seed: int,
):
    """
    Совместно подбирает:
    - архитектуру fastText;
    - размерность;
    - subword;
    - число эпох;
    - размер контекстного окна;
    - C логистической регрессии;
    - class_weight.

    ВАЖНО:
    fastText в каждом inner fold обучается
    только на inner-train.

    Тексты inner-validation не участвуют
    даже в unsupervised-обучении представления.
    """

    texts = np.asarray(
        list(train_texts),
        dtype=object,
    )

    labels = np.asarray(
        list(train_labels),
    )

    inner_cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed,
    )

    folds = list(
        inner_cv.split(
            texts,
            labels,
        )
    )

    fasttext_configs = (
        generate_fasttext_configs()
    )

    rows = []

    total_configs = (
        len(fasttext_configs)
        * len(CLASSIFIER_C)
        * len(CLASSIFIER_CLASS_WEIGHT)
    )

    print(
        f"Native fastText: "
        f"{len(fasttext_configs)} embedding configs, "
        f"{total_configs} combined configs"
    )

    # --------------------------------------------------------
    # Перебор fastText-конфигураций
    # --------------------------------------------------------

    for config_id, ft_params in enumerate(
        fasttext_configs
    ):

        # Для каждого fastText config сначала
        # строим embeddings двух inner folds.
        fold_embeddings = []

        for fold_id, (
            inner_train_idx,
            inner_valid_idx,
        ) in enumerate(folds):

            inner_train_texts = (
                texts[
                    inner_train_idx
                ]
            )

            inner_valid_texts = (
                texts[
                    inner_valid_idx
                ]
            )

            model = train_native_fasttext(
                texts=inner_train_texts,
                params=ft_params,
            )

            x_inner_train = (
                vectorize_texts(
                    model,
                    inner_train_texts,
                )
            )

            x_inner_valid = (
                vectorize_texts(
                    model,
                    inner_valid_texts,
                )
            )

            fold_embeddings.append(
                {
                    "train_idx": (
                        inner_train_idx
                    ),

                    "valid_idx": (
                        inner_valid_idx
                    ),

                    "x_train": (
                        x_inner_train
                    ),

                    "x_valid": (
                        x_inner_valid
                    ),
                }
            )

            del model

        # ----------------------------------------------------
        # На готовых embeddings перебираем LR
        # ----------------------------------------------------

        for C in CLASSIFIER_C:

            for class_weight in (
                CLASSIFIER_CLASS_WEIGHT
            ):

                fold_scores = []

                for fold_id, fold in enumerate(
                    fold_embeddings
                ):

                    classifier = (
                        build_classifier(
                            C=C,
                            class_weight=(
                                class_weight
                            ),
                        )
                    )

                    y_inner_train = labels[
                        fold["train_idx"]
                    ]

                    y_inner_valid = labels[
                        fold["valid_idx"]
                    ]

                    classifier.fit(
                        fold["x_train"],
                        y_inner_train,
                    )

                    y_pred = (
                        classifier.predict(
                            fold["x_valid"]
                        )
                    )

                    score = macro_f1_2_5(
                        y_inner_valid,
                        y_pred,
                    )

                    fold_scores.append(
                        score
                    )

                row = {
                    "config_id": (
                        config_id
                    ),

                    **ft_params,

                    "classifier_C": (
                        float(C)
                    ),

                    "classifier_class_weight": (
                        class_weight
                    ),

                    "fold_0_score": float(
                        fold_scores[0]
                    ),

                    "fold_1_score": float(
                        fold_scores[1]
                    ),

                    "mean_score": float(
                        np.mean(
                            fold_scores
                        )
                    ),

                    "std_score": float(
                        np.std(
                            fold_scores,
                            ddof=1,
                        )
                    ),
                }

                rows.append(
                    row
                )

        print(
            f"fastText config "
            f"{config_id + 1}/"
            f"{len(fasttext_configs)} "
            f"completed"
        )

    cv_results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Выбор лучшей совместной конфигурации
    # --------------------------------------------------------

    best_index = (
        cv_results[
            "mean_score"
        ].idxmax()
    )

    best_row = (
        cv_results
        .loc[best_index]
    )

    best_params = {
        "model": str(
            best_row["model"]
        ),

        "dim": int(
            best_row["dim"]
        ),

        "minn": int(
            best_row["minn"]
        ),

        "maxn": int(
            best_row["maxn"]
        ),

        "epoch": int(
            best_row["epoch"]
        ),

        "ws": int(
            best_row["ws"]
        ),

        "classifier_C": float(
            best_row[
                "classifier_C"
            ]
        ),

        "classifier_class_weight": (
            None
            if pd.isna(
                best_row[
                    "classifier_class_weight"
                ]
            )
            else str(
                best_row[
                    "classifier_class_weight"
                ]
            )
        ),
    }

    best_score = float(
        best_row[
            "mean_score"
        ]
    )

    return (
        best_params,
        best_score,
        cv_results,
    )


# ============================================================
# Финальное обучение outer split
# ============================================================

def run_native_fasttext(
    train_texts,
    train_labels,
    test_texts,
    seed: int,
):
    """
    Полный эксперимент одного outer split:

    1. nested tuning;
    2. выбор параметров;
    3. fastText на полном outer-train;
    4. classifier на полном outer-train;
    5. предсказание outer-test.
    """

    (
        best_params,
        best_inner_score,
        cv_results,
    ) = tune_native_fasttext(
        train_texts=train_texts,
        train_labels=train_labels,
        seed=seed,
    )

    print()
    print(
        "Best native fastText params:"
    )

    print(
        best_params
    )

    # --------------------------------------------------------
    # Переобучаем representation на полном outer train
    # --------------------------------------------------------

    fasttext_params = {
        "model": (
            best_params["model"]
        ),

        "dim": (
            best_params["dim"]
        ),

        "minn": (
            best_params["minn"]
        ),

        "maxn": (
            best_params["maxn"]
        ),

        "epoch": (
            best_params["epoch"]
        ),

        "ws": (
            best_params["ws"]
        ),
    }

    model = train_native_fasttext(
        texts=train_texts,
        params=fasttext_params,
    )

    x_train = vectorize_texts(
        model,
        train_texts,
    )

    x_test = vectorize_texts(
        model,
        test_texts,
    )

    # --------------------------------------------------------
    # Финальный classifier
    # --------------------------------------------------------

    classifier = build_classifier(
        C=best_params[
            "classifier_C"
        ],

        class_weight=best_params[
            "classifier_class_weight"
        ],
    )

    classifier.fit(
        x_train,
        train_labels,
    )

    y_pred = classifier.predict(
        x_test
    )

    metadata = {
        "fasttext_vocab_size": int(
            len(
                model.get_words()
            )
        ),

        "embedding_dimension": int(
            model.get_dimension()
        ),
    }

    return {
        "classifier": classifier,
        "predictions": y_pred,

        "best_params": (
            best_params
        ),

        "best_inner_score": (
            best_inner_score
        ),

        "cv_results": (
            cv_results
        ),

        "metadata": metadata,
    }