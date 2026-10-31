from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_eutirads_dataset,
)
from .metrics import calculate_metrics
from .representations.fasttext_native import (
    train_native_fasttext,
    vectorize_texts as vectorize_native_texts,
)
from .representations.fasttext_pretrained import (
    get_fasttext_metadata,
    load_fasttext_model,
    tune_fasttext_classifier,
    vectorize_texts as vectorize_pretrained_texts,
)
from .representations.tfidf import (
    tune_tfidf,
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
    / "learning_curve_v2"
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


FRACTIONS = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# Фиксированная конфигурация native fastText
# ============================================================

FIXED_NATIVE_FASTTEXT_PARAMS = {
    "model": "skipgram",
    "dim": 200,
    "minn": 2,
    "maxn": 5,
    "epoch": 5,
    "ws": 5,
}


LEARNING_CURVE_PROTOCOL = (
    "v2_fixed_native_fasttext_"
    "classifier_tuned_per_fraction"
)


# ============================================================
# Вспомогательные функции
# ============================================================

def to_json_compatible(value):

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


def load_outer_split(
    df: pd.DataFrame,
    splits: pd.DataFrame,
    split_id: int,
):
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
            f"split={split_id}: "
            "train/test пересекаются."
        )

    train = df[
        df["object_id"].isin(
            train_ids
        )
    ].copy()

    test = df[
        df["object_id"].isin(
            test_ids
        )
    ].copy()

    if len(train) != len(train_info):
        raise RuntimeError(
            f"split={split_id}: "
            "не все train object_id найдены."
        )

    if len(test) != len(test_info):
        raise RuntimeError(
            f"split={split_id}: "
            "не все test object_id найдены."
        )

    return train, test


# ============================================================
# Вложенные train subsets
# ============================================================

def make_nested_train_subset(
    outer_train: pd.DataFrame,
    fraction: float,
    seed: int,
):
    """
    Класс 1 сохраняется полностью:
        в outer train всего 2 объекта.

    Классы 2-5 уменьшаются детерминированно.

    Для одного split используется один случайный
    порядок внутри каждого класса, поэтому:

        D10 ⊂ D25 ⊂ D50 ⊂ D75 ⊂ D100
    """

    if not (
        0 < fraction <= 1
    ):
        raise ValueError(
            "fraction должна быть в (0, 1]."
        )

    selected_indices = []

    # ----------------------------------------
    # Class 1
    # ----------------------------------------

    class_1 = outer_train[
        outer_train["label"] == 1
    ]

    selected_indices.extend(
        class_1.index.tolist()
    )

    # ----------------------------------------
    # Classes 2-5
    # ----------------------------------------

    for label in [
        2,
        3,
        4,
        5,
    ]:

        group = (
            outer_train[
                outer_train["label"] == label
            ]
            .sort_values(
                "object_id"
            )
        )

        if group.empty:
            raise RuntimeError(
                f"Нет объектов класса {label}."
            )

        # Один и тот же порядок для всех fractions
        # конкретного split и класса.
        rng = np.random.default_rng(
            seed
            + label * 10000
        )

        order = rng.permutation(
            len(group)
        )

        n_select = int(
            np.ceil(
                len(group)
                * fraction
            )
        )

        # Нужно минимум 2 для 2-fold inner CV.
        n_select = max(
            2,
            n_select,
        )

        n_select = min(
            n_select,
            len(group),
        )

        selected = group.iloc[
            order[:n_select]
        ]

        selected_indices.extend(
            selected.index.tolist()
        )

    subset = (
        outer_train.loc[
            selected_indices
        ]
        .copy()
        .sort_index()
    )

    return subset


# ============================================================
# Manifest
# ============================================================

def create_manifest(
    df: pd.DataFrame,
    splits: pd.DataFrame,
):
    rows = []

    for split_id in sorted(
        splits["split_id"].unique()
    ):

        split_id = int(
            split_id
        )

        part = splits[
            splits["split_id"]
            == split_id
        ]

        seed = int(
            part["seed"].iloc[0]
        )

        outer_train, _ = (
            load_outer_split(
                df=df,
                splits=splits,
                split_id=split_id,
            )
        )

        for fraction in FRACTIONS:

            subset = (
                make_nested_train_subset(
                    outer_train=outer_train,
                    fraction=fraction,
                    seed=seed,
                )
            )

            for _, row in (
                subset.iterrows()
            ):

                rows.append(
                    {
                        "split_id": (
                            split_id
                        ),

                        "seed": (
                            seed
                        ),

                        "fraction": (
                            float(
                                fraction
                            )
                        ),

                        "object_id": (
                            row[
                                "object_id"
                            ]
                        ),

                        "label": int(
                            row[
                                "label"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def prepare_manifest(
    df,
    splits,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "learning_curve_subsets.csv"
    )

    new_manifest = (
        create_manifest(
            df=df,
            splits=splits,
        )
    )

    if path.exists():

        old_manifest = pd.read_csv(
            path
        )

        columns = [
            "split_id",
            "seed",
            "fraction",
            "object_id",
            "label",
        ]

        old_sorted = (
            old_manifest[
                columns
            ]
            .sort_values(
                columns
            )
            .reset_index(
                drop=True
            )
        )

        new_sorted = (
            new_manifest[
                columns
            ]
            .sort_values(
                columns
            )
            .reset_index(
                drop=True
            )
        )

        if not old_sorted.equals(
            new_sorted
        ):
            raise RuntimeError(
                "Существующий manifest "
                "не совпадает с текущим."
            )

        return old_manifest

    new_manifest.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return new_manifest


def get_manifest_subset(
    df,
    manifest,
    split_id,
    fraction,
):
    mask = (
        (
            manifest["split_id"]
            == split_id
        )
        &
        np.isclose(
            manifest["fraction"],
            fraction,
        )
    )

    part = manifest[
        mask
    ]

    ids = set(
        part["object_id"]
    )

    subset = df[
        df["object_id"].isin(
            ids
        )
    ].copy()

    if len(subset) != len(part):
        raise RuntimeError(
            "Не удалось восстановить subset."
        )

    return subset


# ============================================================
# Пути run
# ============================================================

def fraction_name(
    fraction: float,
):
    return (
        f"fraction_"
        f"{int(round(fraction * 100)):03d}"
    )


def get_run_dir(
    output_dir,
    model_name,
    split_id,
    fraction,
):
    return (
        output_dir
        / model_name
        / f"split_{split_id:02d}"
        / fraction_name(
            fraction
        )
    )


# ============================================================
# Resume
# ============================================================

def load_existing_point(
    run_dir: Path,
):

    metrics_path = (
        run_dir
        / "metrics.json"
    )

    config_path = (
        run_dir
        / "config.json"
    )

    if not (
        metrics_path.exists()
        and config_path.exists()
    ):
        return None

    with config_path.open(
        encoding="utf-8",
    ) as f:

        config = json.load(
            f
        )

    if (
        config.get(
            "learning_curve_protocol"
        )
        != LEARNING_CURVE_PROTOCOL
    ):
        return None

    with metrics_path.open(
        encoding="utf-8",
    ) as f:

        metrics = json.load(
            f
        )

    return (
        metrics,
        config,
    )


# ============================================================
# Save point
# ============================================================

def save_point(
    output_dir,
    model_name,
    split_id,
    seed,
    fraction,
    train,
    test,
    result,
):

    run_dir = get_run_dir(
        output_dir=output_dir,
        model_name=model_name,
        split_id=split_id,
        fraction=fraction,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = {
        str(int(label)): int(count)
        for label, count
        in (
            train[
                "label"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    config = {
        "learning_curve_protocol": (
            LEARNING_CURVE_PROTOCOL
        ),

        "model": (
            model_name
        ),

        "split_id": int(
            split_id
        ),

        "seed": int(
            seed
        ),

        "fraction": float(
            fraction
        ),

        "n_train_texts": int(
            len(train)
        ),

        "n_test_texts": int(
            len(test)
        ),

        "train_class_counts": (
            class_counts
        ),

        "class_1_policy": (
            "all_outer_train_class_1_objects_kept"
        ),

        "primary_metric": (
            "macro_f1_2_5"
        ),

        "best_inner_macro_f1_2_5": (
            float(
                result[
                    "best_inner_score"
                ]
            )
        ),

        "best_params": (
            result[
                "best_params"
            ]
        ),
    }

    config.update(
        result.get(
            "extra_config",
            {},
        )
    )

    config = to_json_compatible(
        config
    )

    metrics = to_json_compatible(
        result[
            "metrics"
        ]
    )

    # ----------------------------------------
    # JSON
    # ----------------------------------------

    with (
        run_dir
        / "config.json"
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

    with (
        run_dir
        / "metrics.json"
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

    # ----------------------------------------
    # Predictions
    # ----------------------------------------

    predictions = pd.DataFrame(
        {
            "object_id": (
                test[
                    "object_id"
                ].to_numpy()
            ),

            "y_true": (
                test[
                    "label"
                ].to_numpy()
            ),

            "y_pred": (
                np.asarray(
                    result[
                        "predictions"
                    ]
                )
            ),
        }
    )

    predictions.to_csv(
        run_dir
        / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------
    # CV results
    # ----------------------------------------

    search = result.get(
        "search"
    )

    if search is not None:

        pd.DataFrame(
            search.cv_results_
        ).to_csv(
            run_dir
            / "cv_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return run_dir


# ============================================================
# Main
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
        ],
        required=True,
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
    )

    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # Данные
    # ========================================================

    df = load_eutirads_dataset(
        args.data
    )

    splits = pd.read_csv(
        args.splits
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = prepare_manifest(
        df=df,
        splits=splits,
        output_dir=args.output_dir,
    )

    # ========================================================
    # Pretrained embeddings:
    # один раз на весь датасет
    # ========================================================

    all_embeddings = None
    fasttext_metadata = {}
    fasttext_model_path = None

    if (
        args.model
        in PRETRAINED_FASTTEXT_MODELS
    ):

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

        if not (
            fasttext_model_path.exists()
        ):
            raise FileNotFoundError(
                f"Не найдена модель:\n"
                f"{fasttext_model_path}"
            )

        print()
        print(
            f"Loading fastText: "
            f"{fasttext_model_path}"
        )

        ft_model = (
            load_fasttext_model(
                fasttext_model_path
            )
        )

        fasttext_metadata = (
            get_fasttext_metadata(
                ft_model
            )
        )

        print(
            "Vectorizing entire dataset..."
        )

        all_embeddings = (
            vectorize_pretrained_texts(
                ft_model,
                df["text_model"],
            )
        )

        print(
            f"Embeddings shape: "
            f"{all_embeddings.shape}"
        )

        del ft_model
        gc.collect()

    # ========================================================
    # Split IDs
    # ========================================================

    available_split_ids = [
        int(x)
        for x in sorted(
            splits[
                "split_id"
            ].unique()
        )
    ]

    if args.split_id is None:

        split_ids = (
            available_split_ids
        )

    else:

        split_id = int(
            args.split_id
        )

        if (
            split_id
            not in available_split_ids
        ):
            raise ValueError(
                f"Нет split_id="
                f"{split_id}"
            )

        split_ids = [
            split_id
        ]

    summary_rows = []

    # ========================================================
    # Main loop
    # ========================================================

    for split_id in split_ids:

        split_part = splits[
            splits["split_id"]
            == split_id
        ]

        seed = int(
            split_part[
                "seed"
            ].iloc[0]
        )

        _, outer_test = (
            load_outer_split(
                df=df,
                splits=splits,
                split_id=split_id,
            )
        )

        for fraction in FRACTIONS:

            train_subset = (
                get_manifest_subset(
                    df=df,
                    manifest=manifest,
                    split_id=split_id,
                    fraction=fraction,
                )
            )

            run_dir = get_run_dir(
                output_dir=(
                    args.output_dir
                ),
                model_name=(
                    args.model
                ),
                split_id=split_id,
                fraction=fraction,
            )

            print()
            print(
                "=" * 70
            )

            print(
                f"{args.model} | "
                f"split={split_id} | "
                f"fraction={fraction:.0%}"
            )

            print(
                f"train size="
                f"{len(train_subset)}"
            )

            print(
                "classes="
                f"{train_subset['label']}"
                ".value_counts()"
                ".sort_index()"
                ".to_dict()}"
            )

            print(
                "=" * 70
            )

            # =================================================
            # Resume
            # =================================================

            if not args.overwrite:

                existing = (
                    load_existing_point(
                        run_dir
                    )
                )

                if existing is not None:

                    metrics, config = (
                        existing
                    )

                    print(
                        "Already calculated "
                        "-> skip"
                    )

                    summary_rows.append(
                        {
                            "model": (
                                args.model
                            ),

                            "split_id": (
                                split_id
                            ),

                            "seed": (
                                seed
                            ),

                            "fraction": (
                                fraction
                            ),

                            "n_train_texts": (
                                len(
                                    train_subset
                                )
                            ),

                            **metrics,
                        }
                    )

                    continue

            # =================================================
            # TF-IDF
            # =================================================

            if args.model == "tfidf":

                search = tune_tfidf(
                    x_train=(
                        train_subset[
                            "text_model"
                        ]
                    ),

                    y_train=(
                        train_subset[
                            "label"
                        ]
                    ),

                    seed=seed,
                )

                y_pred = (
                    search.predict(
                        outer_test[
                            "text_model"
                        ]
                    )
                )

                result = {
                    "search": (
                        search
                    ),

                    "predictions": (
                        y_pred
                    ),

                    "metrics": (
                        calculate_metrics(
                            outer_test[
                                "label"
                            ],
                            y_pred,
                        )
                    ),

                    "best_params": (
                        search.best_params_
                    ),

                    "best_inner_score": (
                        float(
                            search.best_score_
                        )
                    ),

                    "extra_config": {
                        "representation": (
                            "tfidf"
                        ),

                        "hyperparameter_policy": (
                            "tuned_per_fraction"
                        ),
                    },
                }

            # =================================================
            # Frozen pretrained fastText
            # =================================================

            elif (
                args.model
                in PRETRAINED_FASTTEXT_MODELS
            ):

                train_indices = (
                    train_subset.index
                    .to_numpy(
                        dtype=int
                    )
                )

                test_indices = (
                    outer_test.index
                    .to_numpy(
                        dtype=int
                    )
                )

                x_train = (
                    all_embeddings[
                        train_indices
                    ]
                )

                x_test = (
                    all_embeddings[
                        test_indices
                    ]
                )

                search = (
                    tune_fasttext_classifier(
                        x_train=x_train,
                        y_train=(
                            train_subset[
                                "label"
                            ]
                        ),
                        seed=seed,
                    )
                )

                y_pred = (
                    search.predict(
                        x_test
                    )
                )

                if (
                    args.model
                    == "bethesda_fasttext"
                ):
                    representation = (
                        "frozen_bethesda_fasttext"
                    )

                else:
                    representation = (
                        "frozen_cc_ru_300_fasttext"
                    )

                result = {
                    "search": (
                        search
                    ),

                    "predictions": (
                        y_pred
                    ),

                    "metrics": (
                        calculate_metrics(
                            outer_test[
                                "label"
                            ],
                            y_pred,
                        )
                    ),

                    "best_params": (
                        search.best_params_
                    ),

                    "best_inner_score": (
                        float(
                            search.best_score_
                        )
                    ),

                    "extra_config": {
                        "representation": (
                            representation
                        ),

                        "hyperparameter_policy": (
                            "classifier_tuned_per_fraction"
                        ),

                        "fasttext_model_path": (
                            str(
                                fasttext_model_path
                            )
                        ),

                        **fasttext_metadata,
                    },
                }

            # =================================================
            # Native fastText — FAST VERSION
            # =================================================

            elif (
                args.model
                == "native_fasttext"
            ):

                print(
                    "Training ONE native "
                    "fastText model..."
                )

                print(
                    "Fixed params:"
                )

                print(
                    FIXED_NATIVE_FASTTEXT_PARAMS
                )

                # ---------------------------------------------
                # 1 fastText training only
                # ---------------------------------------------

                native_model = (
                    train_native_fasttext(
                        texts=(
                            train_subset[
                                "text_model"
                            ]
                        ),

                        params=(
                            FIXED_NATIVE_FASTTEXT_PARAMS
                        ),
                    )
                )

                # ---------------------------------------------
                # Embeddings
                # ---------------------------------------------

                x_train = (
                    vectorize_native_texts(
                        native_model,
                        train_subset[
                            "text_model"
                        ],
                    )
                )

                x_test = (
                    vectorize_native_texts(
                        native_model,
                        outer_test[
                            "text_model"
                        ],
                    )
                )

                vocab_size = int(
                    len(
                        native_model.get_words()
                    )
                )

                del native_model
                gc.collect()

                # ---------------------------------------------
                # Только дешёвый tuning LR
                # ---------------------------------------------

                search = (
                    tune_fasttext_classifier(
                        x_train=x_train,
                        y_train=(
                            train_subset[
                                "label"
                            ]
                        ),
                        seed=seed,
                    )
                )

                y_pred = (
                    search.predict(
                        x_test
                    )
                )

                best_params = {
                    **FIXED_NATIVE_FASTTEXT_PARAMS,

                    "classifier_C": (
                        float(
                            search.best_params_[
                                "classifier__C"
                            ]
                        )
                    ),

                    "classifier_class_weight": (
                        search.best_params_[
                            "classifier__class_weight"
                        ]
                    ),
                }

                result = {
                    "search": (
                        search
                    ),

                    "predictions": (
                        y_pred
                    ),

                    "metrics": (
                        calculate_metrics(
                            outer_test[
                                "label"
                            ],
                            y_pred,
                        )
                    ),

                    "best_params": (
                        best_params
                    ),

                    "best_inner_score": (
                        float(
                            search.best_score_
                        )
                    ),

                    "extra_config": {
                        "representation": (
                            "native_eutirads_fasttext"
                        ),

                        "fasttext_training_on_eutirads": (
                            True
                        ),

                        "fasttext_hyperparameter_policy": (
                            "fixed_from_full_benchmark"
                        ),

                        "classifier_hyperparameter_policy": (
                            "tuned_per_fraction"
                        ),

                        "fasttext_selection_basis": (
                            "most_frequent_and_stable_"
                            "configuration_across_"
                            "10_outer_splits"
                        ),

                        "fasttext_model": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "model"
                            ]
                        ),

                        "fasttext_dim": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "dim"
                            ]
                        ),

                        "fasttext_minn": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "minn"
                            ]
                        ),

                        "fasttext_maxn": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "maxn"
                            ]
                        ),

                        "fasttext_epoch": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "epoch"
                            ]
                        ),

                        "fasttext_ws": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "ws"
                            ]
                        ),

                        "fasttext_minCount": (
                            1
                        ),

                        "fasttext_vocab_size": (
                            vocab_size
                        ),

                        "embedding_dimension": (
                            FIXED_NATIVE_FASTTEXT_PARAMS[
                                "dim"
                            ]
                        ),

                        "batch_size": None,
                    },
                }

            else:

                raise ValueError(
                    f"Неизвестная модель: "
                    f"{args.model}"
                )

            # =================================================
            # Save
            # =================================================

            run_dir = save_point(
                output_dir=(
                    args.output_dir
                ),

                model_name=(
                    args.model
                ),

                split_id=(
                    split_id
                ),

                seed=(
                    seed
                ),

                fraction=(
                    fraction
                ),

                train=(
                    train_subset
                ),

                test=(
                    outer_test
                ),

                result=(
                    result
                ),
            )

            metrics = (
                result[
                    "metrics"
                ]
            )

            print(
                "Macro-F1(2-5): "
                f"{metrics['macro_f1_2_5']:.4f}"
            )

            print(
                f"saved -> "
                f"{run_dir}"
            )

            summary_rows.append(
                {
                    "model": (
                        args.model
                    ),

                    "split_id": (
                        split_id
                    ),

                    "seed": (
                        seed
                    ),

                    "fraction": (
                        fraction
                    ),

                    "n_train_texts": (
                        len(
                            train_subset
                        )
                    ),

                    **metrics,
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        args.output_dir
        / (
            f"{args.model}_"
            f"learning_curve.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=== LEARNING CURVE SUMMARY ==="
    )

    grouped = (
        summary.groupby(
            "fraction"
        )[
            "macro_f1_2_5"
        ]
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
    )

    for fraction, row in (
        grouped.iterrows()
    ):

        print(
            f"{fraction:.0%}: "
            f"{row['mean']:.4f} "
            f"± {row['std']:.4f} "
            f"(n={int(row['count'])})"
        )

    print()
    print(
        f"Summary -> "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()