from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import (
    DEFAULT_BETHESDA_CORPUS,
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_eutirads_dataset,
)

from .metrics import (
    calculate_metrics,
)

from .representations.fasttext_pretrained import (
    get_fasttext_metadata,
    tune_fasttext_classifier,
)

from .run_benchmark import (
    load_split,
    save_run,
)


# ============================================================
# Пути
# ============================================================

DEFAULT_SPLITS = (
    PROJECT_ROOT
    / "article_results"
    / "splits.csv"
)

DEFAULT_RUNS_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "runs"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "article_results"
    / "subword_ablation"
)

DEFAULT_MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "subword_ablation"
)


# ============================================================
# Общие параметры Bethesda fastText
# ============================================================

FASTTEXT_COMMON_PARAMS = {
    "model": "skipgram",
    "dim": 200,
    "ws": 5,
    "epoch": 20,
    "lr": 0.05,
    "minCount": 2,

    # Для ablation используем один поток:
    # все варианты обучаются в максимально
    # одинаковом режиме.
    "thread": 1,
}


# ============================================================
# Варианты ablation
# ============================================================

ABLATION_VARIANTS = {
    "bethesda_ablation_no_subword": {
        "label": "Без subword",
        "minn": 0,
        "maxn": 0,
    },

    "bethesda_ablation_2_5": {
        "label": "2–5",
        "minn": 2,
        "maxn": 5,
    },

    "bethesda_ablation_3_6": {
        "label": "3–6",
        "minn": 3,
        "maxn": 6,
    },

    "bethesda_ablation_4_7": {
        "label": "4–7",
        "minn": 4,
        "maxn": 7,
    },
}


# ============================================================
# Обучение source fastText
# ============================================================

def train_or_load_model(
    variant_name: str,
    variant: dict,
    corpus_path: Path,
    models_dir: Path,
    force_retrain: bool,
):
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / f"{variant_name}.bin"
    )

    metadata_path = (
        models_dir
        / f"{variant_name}_meta.json"
    )

    if (
        model_path.exists()
        and not force_retrain
    ):
        print()
        print(
            f"Loading existing model: "
            f"{model_path}"
        )

        model = fasttext.load_model(
            str(model_path)
        )

        return (
            model,
            model_path,
        )

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Не найден Bethesda corpus:\n"
            f"{corpus_path}"
        )

    params = {
        **FASTTEXT_COMMON_PARAMS,
        "minn": int(
            variant["minn"]
        ),
        "maxn": int(
            variant["maxn"]
        ),
    }

    print()
    print(
        "=" * 60
    )

    print(
        f"Training {variant_name}"
    )

    print(
        f"minn={params['minn']}, "
        f"maxn={params['maxn']}"
    )

    print(
        "=" * 60
    )

    model = (
        fasttext.train_unsupervised(
            input=str(
                corpus_path
            ),

            model=params[
                "model"
            ],

            dim=params[
                "dim"
            ],

            ws=params[
                "ws"
            ],

            epoch=params[
                "epoch"
            ],

            lr=params[
                "lr"
            ],

            minCount=params[
                "minCount"
            ],

            minn=params[
                "minn"
            ],

            maxn=params[
                "maxn"
            ],

            thread=params[
                "thread"
            ],
        )
    )

    model.save_model(
        str(model_path)
    )

    metadata = {
        "variant": (
            variant_name
        ),

        "display_label": (
            variant["label"]
        ),

        "corpus": str(
            corpus_path
        ),

        **params,

        "vocab_size": len(
            model.get_words()
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"saved -> {model_path}"
    )

    return (
        model,
        model_path,
    )


# ============================================================
# Векторизация
# ============================================================

def vectorize_with_subword(
    model,
    texts,
) -> np.ndarray:
    """
    Обычная fastText-векторизация.

    OOV-слова могут получать вектор
    через символьные n-граммы.
    """

    vectors = []

    for text in texts:

        text = (
            str(text)
            .replace(
                "\n",
                " ",
            )
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


def vectorize_without_subword(
    model,
    texts,
) -> tuple[np.ndarray, dict]:
    """
    В варианте без subword неизвестные слова
    явно исключаются.

    Это делает обработку OOV однозначной:
    неизвестное слово не получает вектор
    и не участвует в sentence representation.

    Если после удаления OOV слов текст пуст,
    используется нулевой вектор.
    """

    model_vocab = set(
        model.get_words()
    )

    dimension = int(
        model.get_dimension()
    )

    vectors = []

    n_documents = 0
    n_empty_after_oov = 0

    n_tokens_total = 0
    n_tokens_kept = 0
    n_tokens_removed = 0

    for text in texts:

        tokens = (
            str(text)
            .replace(
                "\n",
                " ",
            )
            .strip()
            .split()
        )

        n_documents += 1

        n_tokens_total += len(
            tokens
        )

        known_tokens = [
            token
            for token in tokens
            if token in model_vocab
        ]

        n_tokens_kept += len(
            known_tokens
        )

        n_tokens_removed += (
            len(tokens)
            - len(known_tokens)
        )

        if known_tokens:

            filtered_text = (
                " ".join(
                    known_tokens
                )
            )

            vector = (
                model
                .get_sentence_vector(
                    filtered_text
                )
            )

        else:

            n_empty_after_oov += 1

            vector = np.zeros(
                dimension,
                dtype=np.float32,
            )

        vectors.append(
            vector
        )

    stats = {
        "n_documents": int(
            n_documents
        ),

        "n_empty_after_oov": int(
            n_empty_after_oov
        ),

        "n_tokens_total": int(
            n_tokens_total
        ),

        "n_tokens_kept": int(
            n_tokens_kept
        ),

        "n_tokens_removed": int(
            n_tokens_removed
        ),

        "removed_token_rate": float(
            n_tokens_removed
            / n_tokens_total
            if n_tokens_total
            else 0.0
        ),
    }

    return (
        np.asarray(
            vectors,
            dtype=np.float32,
        ),
        stats,
    )


def vectorize_variant(
    model,
    texts,
    variant: dict,
):
    if (
        int(
            variant["minn"]
        )
        == 0
        and
        int(
            variant["maxn"]
        )
        == 0
    ):

        return (
            vectorize_without_subword(
                model,
                texts,
            )
        )

    embeddings = (
        vectorize_with_subword(
            model,
            texts,
        )
    )

    stats = {
        "n_documents": int(
            len(embeddings)
        ),

        "n_empty_after_oov": 0,

        "n_tokens_total": None,

        "n_tokens_kept": None,

        "n_tokens_removed": None,

        "removed_token_rate": None,
    }

    return (
        embeddings,
        stats,
    )


# ============================================================
# Один ablation-вариант
# ============================================================

def run_variant(
    variant_name: str,
    variant: dict,
    model,
    model_path: Path,
    df: pd.DataFrame,
    splits: pd.DataFrame,
    runs_dir: Path,
):
    print()
    print(
        f"Vectorizing EU-TIRADS: "
        f"{variant_name}"
    )

    (
        embeddings,
        oov_handling_stats,
    ) = (
        vectorize_variant(
            model=model,
            texts=df[
                "text_model"
            ],
            variant=variant,
        )
    )

    print(
        "Embeddings shape: "
        f"{embeddings.shape}"
    )

    if (
        len(embeddings)
        != len(df)
    ):
        raise RuntimeError(
            "Число embeddings "
            "не совпадает с датасетом."
        )

    if (
        variant["minn"] == 0
        and variant["maxn"] == 0
    ):

        print()
        print(
            "No-subword OOV handling:"
        )

        print(
            "removed token rate: "
            f"{oov_handling_stats['removed_token_rate']:.4f}"
        )

        print(
            "empty documents after OOV removal: "
            f"{oov_handling_stats['n_empty_after_oov']}"
        )

    result_rows = []

    split_ids = sorted(
        int(x)
        for x in (
            splits[
                "split_id"
            ].unique()
        )
    )

    for split_id in split_ids:

        split_info = (
            splits[
                splits[
                    "split_id"
                ]
                == split_id
            ]
        )

        seed = int(
            split_info[
                "seed"
            ].iloc[0]
        )

        train, test = (
            load_split(
                df=df,
                splits=splits,
                split_id=split_id,
            )
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
            embeddings[
                train_indices
            ]
        )

        x_test = (
            embeddings[
                test_indices
            ]
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"{variant_name} | "
            f"split={split_id} | "
            f"seed={seed}"
        )

        print(
            "=" * 60
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

        y_pred = (
            search.predict(
                x_test
            )
        )

        metrics = (
            calculate_metrics(
                test[
                    "label"
                ],
                y_pred,
            )
        )

        ft_metadata = (
            get_fasttext_metadata(
                model
            )
        )

        result = {
            "model": (
                search.best_estimator_
            ),

            "search": (
                search
            ),

            "predictions": (
                y_pred
            ),

            "metrics": (
                metrics
            ),

            "best_params": (
                search.best_params_
            ),

            "best_inner_score": float(
                search.best_score_
            ),

            "extra_config": {
                "representation": (
                    variant_name
                ),

                "representation_fit_scope": (
                    "bethesda_source_corpus"
                ),

                "source_model_path": (
                    str(
                        model_path
                    )
                ),

                "fasttext_model": (
                    FASTTEXT_COMMON_PARAMS[
                        "model"
                    ]
                ),

                "fasttext_dim": int(
                    FASTTEXT_COMMON_PARAMS[
                        "dim"
                    ]
                ),

                "fasttext_ws": int(
                    FASTTEXT_COMMON_PARAMS[
                        "ws"
                    ]
                ),

                "epochs": int(
                    FASTTEXT_COMMON_PARAMS[
                        "epoch"
                    ]
                ),

                "fasttext_lr": float(
                    FASTTEXT_COMMON_PARAMS[
                        "lr"
                    ]
                ),

                "fasttext_minCount": int(
                    FASTTEXT_COMMON_PARAMS[
                        "minCount"
                    ]
                ),

                "fasttext_thread": int(
                    FASTTEXT_COMMON_PARAMS[
                        "thread"
                    ]
                ),

                "fasttext_minn": int(
                    variant[
                        "minn"
                    ]
                ),

                "fasttext_maxn": int(
                    variant[
                        "maxn"
                    ]
                ),

                "batch_size": None,

                "oov_handling": (
                    "skip_unknown_tokens"
                    if (
                        variant[
                            "minn"
                        ]
                        == 0
                        and
                        variant[
                            "maxn"
                        ]
                        == 0
                    )
                    else
                    (
                        "fasttext_character_"
                        "subword_composition"
                    )
                ),

                **oov_handling_stats,

                **ft_metadata,
            },
        }

        run_dir = (
            save_run(
                result=result,
                train=train,
                test=test,
                model_name=(
                    variant_name
                ),
                split_id=(
                    split_id
                ),
                seed=seed,
                output_dir=(
                    runs_dir
                ),
            )
        )

        row = {
            "variant": (
                variant_name
            ),

            "display_label": (
                variant[
                    "label"
                ]
            ),

            "minn": int(
                variant[
                    "minn"
                ]
            ),

            "maxn": int(
                variant[
                    "maxn"
                ]
            ),

            "split_id": int(
                split_id
            ),

            "seed": int(
                seed
            ),

            "best_inner_macro_f1_2_5": float(
                search.best_score_
            ),

            **{
                key: value
                for key, value
                in metrics.items()
                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
            },
        }

        result_rows.append(
            row
        )

        print(
            f"best inner Macro-F1(2-5): "
            f"{search.best_score_:.4f}"
        )

        print(
            "test Macro-F1(2-5):       "
            f"{metrics['macro_f1_2_5']:.4f}"
        )

        print(
            "test Accuracy:            "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"saved -> {run_dir}"
        )

    return pd.DataFrame(
        result_rows
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    results: pd.DataFrame,
):
    rows = []

    for (
        variant_name,
        part,
    ) in results.groupby(
        "variant"
    ):

        first = (
            part.iloc[0]
        )

        row = {
            "variant": (
                variant_name
            ),

            "display_label": (
                first[
                    "display_label"
                ]
            ),

            "minn": int(
                first[
                    "minn"
                ]
            ),

            "maxn": int(
                first[
                    "maxn"
                ]
            ),

            "n_splits": int(
                len(part)
            ),
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_f1_2_5",
            "weighted_f1",
        ]:

            if metric not in (
                part.columns
            ):
                continue

            row[
                f"{metric}_mean"
            ] = float(
                part[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                part[
                    metric
                ].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    order = {
        name: i
        for i, name in enumerate(
            ABLATION_VARIANTS
        )
    }

    summary[
        "_order"
    ] = (
        summary[
            "variant"
        ]
        .map(
            order
        )
    )

    summary = (
        summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# График
# ============================================================

def plot_summary(
    summary: pd.DataFrame,
    output_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8.0, 5.4)
    )

    x = np.arange(
        len(summary)
    )

    ax.errorbar(
        x,
        summary[
            "macro_f1_2_5_mean"
        ],
        yerr=summary[
            "macro_f1_2_5_std"
        ],
        fmt="o",
        capsize=5,
        markersize=8,
        linewidth=1.5,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        summary[
            "display_label"
        ]
    )

    ax.set_xlabel(
        "Диапазон символьных n-грамм"
    )

    ax.set_ylabel(
        "Macro-F1 для классов EU-TIRADS 2–5"
    )

    ax.set_title(
        "Ablation study subword-механизма "
        "Bethesda fastText"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for i, row in (
        summary.iterrows()
    ):

        ax.text(
            i,
            (
                row[
                    "macro_f1_2_5_mean"
                ]
                + 0.008
            ),
            (
                f"{row['macro_f1_2_5_mean']:.3f}"
            ),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    png_path = (
        output_dir
        / "subword_ablation.png"
    )

    svg_path = (
        output_dir
        / "subword_ablation.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"saved -> {png_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_EUTIRADS_DATA,
    )

    parser.add_argument(
        "--bethesda-corpus",
        type=Path,
        default=DEFAULT_BETHESDA_CORPUS,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Переобучить source fastText, "
            "даже если .bin уже существует."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Dataset / splits
    # ========================================================

    df = (
        load_eutirads_dataset(
            args.data
        )
    )

    if not args.splits.exists():
        raise FileNotFoundError(
            f"Не найден splits.csv:\n"
            f"{args.splits}"
        )

    splits = pd.read_csv(
        args.splits
    )

    # ========================================================
    # Все варианты
    # ========================================================

    all_results = []

    for (
        variant_name,
        variant,
    ) in ABLATION_VARIANTS.items():

        (
            model,
            model_path,
        ) = (
            train_or_load_model(
                variant_name=(
                    variant_name
                ),

                variant=variant,

                corpus_path=(
                    args.bethesda_corpus
                ),

                models_dir=(
                    args.models_dir
                ),

                force_retrain=(
                    args.force_retrain
                ),
            )
        )

        variant_results = (
            run_variant(
                variant_name=(
                    variant_name
                ),

                variant=variant,

                model=model,

                model_path=(
                    model_path
                ),

                df=df,

                splits=splits,

                runs_dir=(
                    args.runs_dir
                ),
            )
        )

        all_results.append(
            variant_results
        )

        del model

        gc.collect()

    # ========================================================
    # Итоговые таблицы
    # ========================================================

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results_path = (
        args.output_dir
        / "subword_ablation_by_split.csv"
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        build_summary(
            results
        )
    )

    summary_path = (
        args.output_dir
        / "subword_ablation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_summary(
        summary=summary,
        output_dir=(
            args.output_dir
        ),
    )

    print()
    print(
        "=" * 60
    )

    print(
        "SUBWORD ABLATION SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        summary[
            [
                "display_label",
                "n_splits",
                "macro_f1_2_5_mean",
                "macro_f1_2_5_std",
                "accuracy_mean",
                "accuracy_std",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Results -> "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()