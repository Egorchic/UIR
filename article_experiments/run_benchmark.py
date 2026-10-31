from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .data import (
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_eutirads_dataset,
)
from .metrics import calculate_metrics

from .representations.fasttext_pretrained import (
    get_fasttext_metadata,
    load_fasttext_model,
    tune_fasttext_classifier,
    vectorize_texts,
)

from .representations.fasttext_native import (
    run_native_fasttext,
)

from .representations.tfidf import tune_tfidf

from .representations.transformer import (
    get_or_create_embeddings as get_or_create_transformer_embeddings,
)


# ============================================================
# Пути
# ============================================================

DEFAULT_SPLITS = (
    PROJECT_ROOT
    / "article_results"
    / "splits.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "runs"
)


DEFAULT_FASTTEXT_MODELS = {
    "bethesda_fasttext": (
        PROJECT_ROOT
        / "models"
        / "best_model"
        / "ft_skipgram_dim200_minn2_maxn5.bin"
    ),

    "cc_fasttext": (
        PROJECT_ROOT
        / "models"
        / "pretrained"
        / "cc.ru.300.bin"
    ),
}


PRETRAINED_FASTTEXT_MODELS = {
    "bethesda_fasttext",
    "cc_fasttext",
}

TRANSFORMER_MODELS = {
    "rubert": (
        "DeepPavlov/rubert-base-cased"
    ),

    "rubiobert": (
        "alexyalunin/RuBioBERT"
    ),
}


DEFAULT_TRANSFORMER_CACHE_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "transformer_embeddings"
)


# ============================================================
# Вспомогательные функции
# ============================================================

def to_json_compatible(value):
    """
    Рекурсивно преобразует numpy-типы, Path и другие
    потенциально проблемные объекты в JSON-совместимый вид.
    """

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value


def load_split(
    df: pd.DataFrame,
    splits: pd.DataFrame,
    split_id: int,
):
    """
    Восстанавливает train/test по object_id
    для заданного внешнего разбиения.
    """

    split = splits[
        splits["split_id"] == split_id
    ].copy()

    if split.empty:
        raise ValueError(
            f"split_id={split_id} отсутствует."
        )

    train_info = split[
        split["subset"] == "train"
    ]

    test_info = split[
        split["subset"] == "test"
    ]

    train_ids = set(
        train_info["object_id"]
    )

    test_ids = set(
        test_info["object_id"]
    )

    if train_ids & test_ids:
        raise RuntimeError(
            f"Split {split_id}: train/test пересекаются."
        )

    train = df[
        df["object_id"].isin(train_ids)
    ].copy()

    test = df[
        df["object_id"].isin(test_ids)
    ].copy()

    if len(train) != len(train_info):
        raise RuntimeError(
            f"Split {split_id}: "
            "не все train object_id найдены."
        )

    if len(test) != len(test_info):
        raise RuntimeError(
            f"Split {split_id}: "
            "не все test object_id найдены."
        )

    return train, test


def count_tokens(texts) -> int:
    """
    Число whitespace-токенов.
    Используется для журналирования объёма данных.
    """

    return sum(
        len(str(text).split())
        for text in texts
    )


def get_fasttext_representation_name(
    model_name: str,
) -> str:

    if model_name == "bethesda_fasttext":
        return "frozen_bethesda_fasttext"

    if model_name == "cc_fasttext":
        return "frozen_cc_ru_300_fasttext"

    raise ValueError(
        f"Неизвестная pretrained fastText модель: "
        f"{model_name}"
    )


# ============================================================
# Сохранение результатов одного запуска
# ============================================================

def save_run(
    result: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    split_id: int,
    seed: int,
    output_dir: Path,
):
    run_dir = (
        output_dir
        / model_name
        / f"split_{split_id:02d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "model": model_name,

        "split_id": int(split_id),
        "seed": int(seed),

        "n_train_texts": int(len(train)),
        "n_test_texts": int(len(test)),

        "n_train_tokens": int(
            count_tokens(
                train["text_model"]
            )
        ),

        "n_test_tokens": int(
            count_tokens(
                test["text_model"]
            )
        ),

        # Для текущих TF-IDF и frozen-fastText моделей
        # эти параметры неприменимы.
        # Они понадобятся позднее для native fastText/BERT.
        "epochs": None,
        "batch_size": None,

        "best_inner_macro_f1_2_5": (
            float(
                result["best_inner_score"]
            )
        ),

        "best_params": result[
            "best_params"
        ],

        "sklearn_version": (
            sklearn.__version__
        ),
    }

    extra_config = result.get(
        "extra_config",
        {},
    )

    config.update(
        extra_config
    )

    config = to_json_compatible(
        config
    )

    metrics = to_json_compatible(
        result["metrics"]
    )

    # ----------------------
    # config.json
    # ----------------------

    with (
        run_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------
    # metrics.json
    # ----------------------

    with (
        run_dir / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------
    # predictions.csv
    # ----------------------

    predictions = pd.DataFrame(
        {
            "object_id": (
                test["object_id"]
                .to_numpy()
            ),

            "y_true": (
                test["label"]
                .to_numpy()
            ),

            "y_pred": np.asarray(
                result["predictions"]
            ),
        }
    )

    predictions.to_csv(
        run_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------
    # cv_results.csv
    # ----------------------

    cv_results = result.get(
        "cv_results"
    )

    if cv_results is not None:

        if not isinstance(
                cv_results,
                pd.DataFrame,
        ):
            cv_results = pd.DataFrame(
                cv_results
            )

        cv_results.to_csv(
            run_dir / "cv_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    else:

        search = result.get(
            "search"
        )

        if search is not None:
            cv_results = pd.DataFrame(
                search.cv_results_
            )

            cv_results.to_csv(
                run_dir / "cv_results.csv",
                index=False,
                encoding="utf-8-sig",
            )

    return run_dir


# ============================================================
# Основной запуск
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "tfidf",
            "bethesda_fasttext",
            "cc_fasttext",
            "native_fasttext",
            "rubert",
            "rubiobert",
        ],
        required=True,
    )

    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
        help=(
            "Если не указан, запускаются "
            "все доступные outer split."
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--fasttext-model",
        type=Path,
        default=None,
        help=(
            "Необязательный явный путь к .bin. "
            "Если отсутствует, используется стандартный "
            "путь для выбранной pretrained fastText модели."
        ),
    )

    parser.add_argument(
        "--transformer-cache-dir",
        type=Path,
        default=DEFAULT_TRANSFORMER_CACHE_DIR,
    )

    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--transformer-max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--transformer-device",
        type=str,
        default=None,
        help=(
            "Например cpu, cuda или cuda:0. "
            "По умолчанию CUDA используется "
            "автоматически, если доступна."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Данные
    # ========================================================

    df = load_eutirads_dataset(
        args.data
    )

    if not args.splits.exists():
        raise FileNotFoundError(
            f"Не найден файл split:\n"
            f"{args.splits}\n"
            "Сначала запусти make_splits.py."
        )

    splits = pd.read_csv(
        args.splits
    )

    required_split_columns = {
        "split_id",
        "seed",
        "object_id",
        "label",
        "subset",
    }

    missing = (
        required_split_columns
        - set(splits.columns)
    )

    if missing:
        raise ValueError(
            "В splits.csv отсутствуют столбцы: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # Pretrained fastText
    # ========================================================

    all_fasttext_embeddings = None
    fasttext_metadata = None
    fasttext_model_path = None

    if args.model in PRETRAINED_FASTTEXT_MODELS:

        if args.fasttext_model is not None:
            fasttext_model_path = (
                args.fasttext_model
            )
        else:
            fasttext_model_path = (
                DEFAULT_FASTTEXT_MODELS[
                    args.model
                ]
            )

        if not fasttext_model_path.exists():
            raise FileNotFoundError(
                "Не найдена fastText-модель:\n"
                f"{fasttext_model_path}\n\n"
                "Можно указать другой путь через "
                "--fasttext-model."
            )

        print()
        print(
            "Loading fastText model:"
        )
        print(
            fasttext_model_path
        )

        fasttext_model = (
            load_fasttext_model(
                fasttext_model_path
            )
        )

        fasttext_metadata = (
            get_fasttext_metadata(
                fasttext_model
            )
        )

        print()
        print(
            "Vectorizing entire "
            "EU-TIRADS dataset..."
        )

        all_fasttext_embeddings = (
            vectorize_texts(
                fasttext_model,
                df["text_model"],
            )
        )

        print(
            "Embeddings shape: "
            f"{all_fasttext_embeddings.shape}"
        )

        if (
            len(all_fasttext_embeddings)
            != len(df)
        ):
            raise RuntimeError(
                "Число embeddings не совпадает "
                "с числом объектов."
            )

        # Для cc.ru.300 это особенно полезно:
        # бинарная модель занимает много памяти,
        # а после векторизации больше не нужна.
        del fasttext_model
        gc.collect()

    # ============================================================
    # Frozen Transformer embeddings
    # ============================================================

    all_transformer_embeddings = None
    transformer_metadata = None
    transformer_model_id = None

    if args.model in TRANSFORMER_MODELS:

        if "input_clean" not in df.columns:
            raise ValueError(
                "Для transformer-эксперимента "
                "в dataset_used.csv необходим "
                "столбец input_clean."
            )

        transformer_texts = (
            df[
                "input_clean"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if (
                transformer_texts
                == ""
        ).any():
            raise ValueError(
                "В input_clean обнаружены "
                "пустые тексты."
            )

        transformer_model_id = (
            TRANSFORMER_MODELS[
                args.model
            ]
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"Preparing frozen {args.model}"
        )

        print(
            f"Model: {transformer_model_id}"
        )

        print(
            "Input column: input_clean"
        )

        print(
            "=" * 60
        )

        (
            all_transformer_embeddings,
            transformer_metadata,
        ) = (
            get_or_create_transformer_embeddings(
                model_key=args.model,

                model_id=(
                    transformer_model_id
                ),

                object_ids=(
                    df[
                        "object_id"
                    ].tolist()
                ),

                texts=(
                    transformer_texts.tolist()
                ),

                cache_dir=(
                    args.transformer_cache_dir
                ),

                batch_size=(
                    args.transformer_batch_size
                ),

                max_length=(
                    args.transformer_max_length
                ),

                device=(
                    args.transformer_device
                ),
            )
        )

        print(
            "Embeddings shape: "
            f"{all_transformer_embeddings.shape}"
        )

    # ========================================================
    # Какие split запускать
    # ========================================================

    available_split_ids = [
        int(x)
        for x in sorted(
            splits["split_id"].unique()
        )
    ]

    if args.split_id is None:
        split_ids = (
            available_split_ids
        )

    else:
        requested_split_id = int(
            args.split_id
        )

        if (
            requested_split_id
            not in available_split_ids
        ):
            raise ValueError(
                f"split_id={requested_split_id} "
                "отсутствует.\n"
                f"Доступны: {available_split_ids}"
            )

        split_ids = [
            requested_split_id
        ]

    # ========================================================
    # Основной цикл
    # ========================================================

    summary_rows = []

    for split_id in split_ids:

        split_id = int(
            split_id
        )

        split_part = splits[
            splits["split_id"]
            == split_id
        ]

        seed_values = (
            split_part["seed"]
            .unique()
        )

        if len(seed_values) != 1:
            raise RuntimeError(
                f"Split {split_id}: "
                "найдено несколько seed."
            )

        seed = int(
            seed_values[0]
        )

        train, test = load_split(
            df=df,
            splits=splits,
            split_id=split_id,
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"{args.model} | "
            f"split={split_id} | "
            f"seed={seed}"
        )

        print(
            "=" * 60
        )

        # ====================================================
        # TF-IDF
        # ====================================================

        if args.model == "tfidf":

            search = tune_tfidf(
                x_train=train[
                    "text_model"
                ],
                y_train=train[
                    "label"
                ],
                seed=seed,
            )

            y_pred = search.predict(
                test["text_model"]
            )

            result = {
                "model": (
                    search.best_estimator_
                ),

                "search": search,

                "predictions": y_pred,

                "metrics": calculate_metrics(
                    test["label"],
                    y_pred,
                ),

                "best_params": (
                    search.best_params_
                ),

                "best_inner_score": float(
                    search.best_score_
                ),

                "extra_config": {
                    "representation": (
                        "tfidf"
                    ),

                    "representation_fit_scope": (
                        "outer_train_only"
                    ),
                },
            }

        # ====================================================
        # Frozen pretrained fastText:
        # Bethesda или cc.ru.300
        # ====================================================

        elif (
            args.model
            in PRETRAINED_FASTTEXT_MODELS
        ):

            if (
                all_fasttext_embeddings
                is None
            ):
                raise RuntimeError(
                    "fastText embeddings "
                    "не были рассчитаны."
                )

            # Индексы train/test являются исходными
            # индексами строк df, поэтому ими можно
            # безопасно индексировать матрицу embeddings.
            train_indices = (
                train.index.to_numpy(
                    dtype=int
                )
            )

            test_indices = (
                test.index.to_numpy(
                    dtype=int
                )
            )

            x_train = (
                all_fasttext_embeddings[
                    train_indices
                ]
            )

            x_test = (
                all_fasttext_embeddings[
                    test_indices
                ]
            )

            search = (
                tune_fasttext_classifier(
                    x_train=x_train,
                    y_train=train[
                        "label"
                    ],
                    seed=seed,
                )
            )

            y_pred = search.predict(
                x_test
            )

            representation = (
                get_fasttext_representation_name(
                    args.model
                )
            )

            result = {
                "model": (
                    search.best_estimator_
                ),

                "search": search,

                "predictions": y_pred,

                "metrics": calculate_metrics(
                    test["label"],
                    y_pred,
                ),

                "best_params": (
                    search.best_params_
                ),

                "best_inner_score": float(
                    search.best_score_
                ),

                "extra_config": {
                    "representation": (
                        representation
                    ),

                    "fasttext_model_path": str(
                        fasttext_model_path
                    ),

                    "fasttext_model_file": (
                        fasttext_model_path.name
                    ),

                    "fasttext_training_on_eutirads": (
                        False
                    ),

                    "embedding_dimension": int(
                        all_fasttext_embeddings
                        .shape[1]
                    ),

                    **(
                        fasttext_metadata
                        or {}
                    ),
                },
            }

        # ====================================================
        # Frozen Transformer:
        # RuBERT или RuBioBERT
        # ====================================================

        elif (
            args.model
            in TRANSFORMER_MODELS
        ):

            if (
                all_transformer_embeddings
                is None
            ):
                raise RuntimeError(
                    "Transformer embeddings "
                    "не были рассчитаны."
                )

            train_indices = (
                train.index.to_numpy(
                    dtype=int
                )
            )

            test_indices = (
                test.index.to_numpy(
                    dtype=int
                )
            )

            x_train = (
                all_transformer_embeddings[
                    train_indices
                ]
            )

            x_test = (
                all_transformer_embeddings[
                    test_indices
                ]
            )

            # Используем тот же классификатор,
            # что и для frozen fastText:
            # StandardScaler + LogisticRegression
            search = (
                tune_fasttext_classifier(
                    x_train=x_train,
                    y_train=train[
                        "label"
                    ],
                    seed=seed,
                )
            )

            y_pred = search.predict(
                x_test
            )

            result = {
                "model": (
                    search.best_estimator_
                ),

                "search": search,

                "predictions": (
                    y_pred
                ),

                "metrics": (
                    calculate_metrics(
                        test[
                            "label"
                        ],
                        y_pred,
                    )
                ),

                "best_params": (
                    search.best_params_
                ),

                "best_inner_score": float(
                    search.best_score_
                ),

                "extra_config": {
                    "representation": (
                        f"frozen_{args.model}_"
                        "mean_pooling"
                    ),

                    "representation_fit_scope": (
                        "pretrained_encoder_frozen"
                    ),

                    "transformer_input_column": (
                        "input_clean"
                    ),

                    "transformer_model_id": (
                        transformer_model_id
                    ),

                    "transformer_training_on_eutirads": (
                        False
                    ),

                    "transformer_finetuning_on_eutirads": (
                        False
                    ),

                    "embedding_dimension": int(
                        all_transformer_embeddings
                        .shape[1]
                    ),

                    "n_train_input_clean_tokens": int(
                        count_tokens(
                            train[
                                "input_clean"
                            ]
                        )
                    ),

                    "n_test_input_clean_tokens": int(
                        count_tokens(
                            test[
                                "input_clean"
                            ]
                        )
                    ),

                    **(
                        transformer_metadata
                        or {}
                    ),
                },
            }

        elif args.model == "native_fasttext":

            native_result = (
                run_native_fasttext(
                    train_texts=(
                        train["text_model"]
                    ),

                    train_labels=(
                        train["label"]
                    ),

                    test_texts=(
                        test["text_model"]
                    ),

                    seed=seed,
                )
            )

            y_pred = (
                native_result[
                    "predictions"
                ]
            )

            best_params = (
                native_result[
                    "best_params"
                ]
            )

            result = {
                "model": (
                    native_result[
                        "classifier"
                    ]
                ),

                "predictions": (
                    y_pred
                ),

                "metrics": (
                    calculate_metrics(
                        test["label"],
                        y_pred,
                    )
                ),

                "best_params": (
                    best_params
                ),

                "best_inner_score": (
                    native_result[
                        "best_inner_score"
                    ]
                ),

                "cv_results": (
                    native_result[
                        "cv_results"
                    ]
                ),

                "extra_config": {
                    "representation": (
                        "native_eutirads_fasttext"
                    ),

                    "representation_fit_scope": (
                        "outer_train_only"
                    ),

                    "fasttext_training_on_eutirads": (
                        True
                    ),

                    "epochs": int(
                        best_params[
                            "epoch"
                        ]
                    ),

                    # У fastText здесь нет параметра
                    # mini-batch в привычном смысле.
                    "batch_size": None,

                    "fasttext_model": (
                        best_params[
                            "model"
                        ]
                    ),

                    "fasttext_dim": int(
                        best_params[
                            "dim"
                        ]
                    ),

                    "fasttext_minn": int(
                        best_params[
                            "minn"
                        ]
                    ),

                    "fasttext_maxn": int(
                        best_params[
                            "maxn"
                        ]
                    ),

                    "fasttext_ws": int(
                        best_params[
                            "ws"
                        ]
                    ),

                    "fasttext_minCount": 1,

                    "fasttext_thread": 1,

                    **native_result[
                        "metadata"
                    ],
                },
            }

        else:
            raise ValueError(
                f"Не реализована модель: "
                f"{args.model}"
            )

        # ====================================================
        # Сохранение run
        # ====================================================

        run_dir = save_run(
            result=result,
            train=train,
            test=test,
            model_name=args.model,
            split_id=split_id,
            seed=seed,
            output_dir=args.output_dir,
        )

        row = {
            "model": args.model,
            "split_id": int(split_id),
            "seed": int(seed),
            **result["metrics"],
        }

        summary_rows.append(
            to_json_compatible(row)
        )

        print(
            "best inner Macro-F1(2-5): "
            f"{result['best_inner_score']:.4f}"
        )

        print(
            "test Macro-F1(2-5):       "
            f"{result['metrics']['macro_f1_2_5']:.4f}"
        )

        print(
            "test Accuracy:            "
            f"{result['metrics']['accuracy']:.4f}"
        )

        print(
            f"saved -> {run_dir}"
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_dir = (
        args.output_dir.parent
        / "summary"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Полный запуск сохраняется как основной
    # результат модели.
    if args.split_id is None:

        summary_path = (
            summary_dir
            / f"{args.model}_results.csv"
        )

    # Одиночный тестовый запуск НЕ перезаписывает
    # результаты десяти split.
    else:

        summary_path = (
            summary_dir
            / (
                f"{args.model}_"
                f"split_{args.split_id:02d}"
                f"_result.csv"
            )
        )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Summary saved -> "
        f"{summary_path}"
    )

    # ========================================================
    # Итоговые метрики
    # ========================================================

    metrics_to_report = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_f1_2_5",
        "weighted_f1",
    ]

    if len(summary) > 1:

        print()
        print(
            "=== MEAN OVER SPLITS ==="
        )

        for metric in metrics_to_report:

            mean = summary[
                metric
            ].mean()

            std = summary[
                metric
            ].std(
                ddof=1
            )

            print(
                f"{metric}: "
                f"{mean:.4f} "
                f"± {std:.4f}"
            )

    else:

        print()
        print(
            "=== SINGLE SPLIT RESULT ==="
        )

        for metric in metrics_to_report:
            print(
                f"{metric}: "
                f"{summary[metric].iloc[0]:.4f}"
            )


if __name__ == "__main__":
    main()