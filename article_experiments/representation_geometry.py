from __future__ import annotations

import argparse
import json
from pathlib import Path

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data import (
    DEFAULT_EUTIRADS_DATA,
    PROJECT_ROOT,
    load_eutirads_dataset,
)

from .representations.fasttext_native import (
    train_native_fasttext,
    vectorize_texts as vectorize_native_texts,
)

from .representations.fasttext_pretrained import (
    load_fasttext_model,
    vectorize_texts as vectorize_pretrained_texts,
)

from .representations.transformer import (
    get_or_create_embeddings as get_or_create_transformer_embeddings,
)

from .run_benchmark import (
    load_split,
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
    / "representation_geometry"
)

DEFAULT_BETHESDA_MODEL = (
    PROJECT_ROOT
    / "models"
    / "best_model"
    / "ft_skipgram_dim200_minn2_maxn5.bin"
)

DEFAULT_TRANSFORMER_CACHE = (
    PROJECT_ROOT
    / "article_results"
    / "transformer_embeddings"
)

DEFAULT_NATIVE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "geometry"
)


# ============================================================
# Transformer models
# ============================================================

TRANSFORMER_MODELS = {
    "rubert": (
        "DeepPavlov/rubert-base-cased"
    ),

    "rubiobert": (
        "alexyalunin/RuBioBERT"
    ),
}


MODEL_LABELS = {
    "bethesda_fasttext": (
        "Bethesda fastText"
    ),

    "native_fasttext": (
        "native fastText"
    ),

    "rubert": (
        "RuBERT"
    ),

    "rubiobert": (
        "RuBioBERT"
    ),
}


MODEL_ORDER = [
    "bethesda_fasttext",
    "native_fasttext",
    "rubert",
    "rubiobert",
]


CLASSES = [
    2,
    3,
    4,
    5,
]


# ============================================================
# Native fastText
# ============================================================

def load_native_best_params(
    runs_dir: Path,
    split_id: int,
):
    """
    Берёт уже выбранную nested-CV конфигурацию
    native fastText для данного outer split.
    """

    config_path = (
        runs_dir
        / "native_fasttext"
        / f"split_{split_id:02d}"
        / "config.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Не найден config native fastText:\n"
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    best_params = config[
        "best_params"
    ]

    required = {
        "model",
        "dim",
        "minn",
        "maxn",
        "epoch",
        "ws",
    }

    missing = (
        required
        - set(
            best_params
        )
    )

    if missing:
        raise ValueError(
            "В native config нет параметров: "
            f"{sorted(missing)}"
        )

    return {
        "model": (
            best_params[
                "model"
            ]
        ),

        "dim": int(
            best_params[
                "dim"
            ]
        ),

        "minn": int(
            best_params[
                "minn"
            ]
        ),

        "maxn": int(
            best_params[
                "maxn"
            ]
        ),

        "epoch": int(
            best_params[
                "epoch"
            ]
        ),

        "ws": int(
            best_params[
                "ws"
            ]
        ),
    }


def get_or_train_native_model(
    train_texts,
    params: dict,
    split_id: int,
    model_dir: Path,
):
    """
    Для геометрического анализа native fastText
    восстанавливается на outer-train выбранного split.

    Это не меняет основной benchmark.
    """

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / f"native_split_{split_id:02d}.bin"
    )

    metadata_path = (
        model_dir
        / f"native_split_{split_id:02d}.json"
    )

    if model_path.exists():

        print()
        print(
            "Loading cached native geometry model:"
        )

        print(
            model_path
        )

        return (
            fasttext.load_model(
                str(
                    model_path
                )
            ),
            model_path,
        )

    print()
    print(
        "Training native fastText "
        f"for split {split_id}:"
    )

    print(
        params
    )

    model = (
        train_native_fasttext(
            texts=train_texts,
            params=params,
        )
    )

    model.save_model(
        str(
            model_path
        )
    )

    metadata = {
        "split_id": int(
            split_id
        ),

        "fit_scope": (
            "outer_train_only"
        ),

        "n_train_texts": int(
            len(
                train_texts
            )
        ),

        **params,
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
# Transformer embeddings
# ============================================================

def get_transformer_embeddings(
    model_key: str,
    df: pd.DataFrame,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    device: str | None,
):
    model_id = (
        TRANSFORMER_MODELS[
            model_key
        ]
    )

    if "input_clean" not in df.columns:
        raise ValueError(
            "Для Transformer нужен "
            "столбец input_clean."
        )

    texts = (
        df[
            "input_clean"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    embeddings, metadata = (
        get_or_create_transformer_embeddings(
            model_key=(
                model_key
            ),

            model_id=(
                model_id
            ),

            object_ids=(
                df[
                    "object_id"
                ].tolist()
            ),

            texts=texts,

            cache_dir=(
                cache_dir
            ),

            batch_size=(
                batch_size
            ),

            max_length=(
                max_length
            ),

            device=device,
        )
    )

    return (
        embeddings,
        metadata,
    )


# ============================================================
# Silhouette
# ============================================================

def calculate_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
):
    """
    Silhouette считается в исходной размерности.

    Для embedding-space используем cosine distance.
    """

    if len(
        np.unique(
            labels
        )
    ) < 2:

        raise ValueError(
            "Для silhouette нужно "
            "минимум два класса."
        )

    return float(
        silhouette_score(
            embeddings,
            labels,
            metric="cosine",
        )
    )


# ============================================================
# Cosine similarity между классами
# ============================================================

def calculate_class_cosine_matrix(
    embeddings: np.ndarray,
    labels: np.ndarray,
):
    """
    Cell(i, j) =
    среднее cosine similarity между всеми
    документами классов i и j.

    На диагонали self-similarity документа
    с самим собой исключается.
    """

    full_similarity = (
        cosine_similarity(
            embeddings
        )
    )

    matrix = np.zeros(
        (
            len(CLASSES),
            len(CLASSES),
        ),
        dtype=float,
    )

    for i, class_a in enumerate(
        CLASSES
    ):

        indices_a = np.where(
            labels
            == class_a
        )[0]

        for j, class_b in enumerate(
            CLASSES
        ):

            indices_b = np.where(
                labels
                == class_b
            )[0]

            block = (
                full_similarity[
                    np.ix_(
                        indices_a,
                        indices_b,
                    )
                ]
            )

            if (
                class_a
                == class_b
            ):

                # Убираем similarity объекта
                # с самим собой (=1).
                mask = (
                    ~np.eye(
                        len(
                            indices_a
                        ),
                        dtype=bool,
                    )
                )

                values = block[
                    mask
                ]

            else:

                values = (
                    block.ravel()
                )

            if len(
                values
            ) == 0:

                value = np.nan

            else:

                value = float(
                    values.mean()
                )

            matrix[
                i,
                j,
            ] = value

    return matrix


# ============================================================
# t-SNE
# ============================================================

def calculate_tsne(
    embeddings: np.ndarray,
    seed: int,
):
    """
    Для визуализации сначала L2-нормируем
    векторы.

    Тогда евклидово расстояние t-SNE
    ближе связано с cosine geometry.
    """

    x = normalize(
        embeddings,
        norm="l2",
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=seed,
    )

    return tsne.fit_transform(
        x
    )


# ============================================================
# Plot t-SNE
# ============================================================

def plot_tsne_all(
    projections: dict,
    labels: np.ndarray,
    output_dir: Path,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            10.5,
            9.0,
        ),
    )

    axes = axes.ravel()

    for ax, model_name in zip(
        axes,
        MODEL_ORDER,
    ):

        projection = (
            projections[
                model_name
            ]
        )

        for label in CLASSES:

            mask = (
                labels
                == label
            )

            ax.scatter(
                projection[
                    mask,
                    0,
                ],

                projection[
                    mask,
                    1,
                ],

                s=18,
                alpha=0.70,
                label=(
                    f"EU-TIRADS {label}"
                ),
            )

        ax.set_title(
            MODEL_LABELS[
                model_name
            ]
        )

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )

    handles, legend_labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        "t-SNE скрытых представлений "
        "одинаковых объектов EU-TIRADS"
    )

    fig.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.96,
        ]
    )

    png_path = (
        output_dir
        / "tsne_spaces.png"
    )

    svg_path = (
        output_dir
        / "tsne_spaces.svg"
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
# Cosine matrix figure
# ============================================================

def plot_cosine_matrices(
    cosine_matrices: dict,
    output_dir: Path,
):
    all_values = np.concatenate(
        [
            matrix.ravel()
            for matrix
            in cosine_matrices.values()
        ]
    )

    finite = all_values[
        np.isfinite(
            all_values
        )
    ]

    vmin = float(
        finite.min()
    )

    vmax = float(
        finite.max()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            10.0,
            8.5,
        ),
    )

    axes = axes.ravel()

    image = None

    for ax, model_name in zip(
        axes,
        MODEL_ORDER,
    ):

        matrix = (
            cosine_matrices[
                model_name
            ]
        )

        image = ax.imshow(
            matrix,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ax.set_title(
            MODEL_LABELS[
                model_name
            ]
        )

        ax.set_xticks(
            np.arange(
                len(
                    CLASSES
                )
            )
        )

        ax.set_yticks(
            np.arange(
                len(
                    CLASSES
                )
            )
        )

        ax.set_xticklabels(
            CLASSES
        )

        ax.set_yticklabels(
            CLASSES
        )

        ax.set_xlabel(
            "Класс"
        )

        ax.set_ylabel(
            "Класс"
        )

        for i in range(
            len(
                CLASSES
            )
        ):

            for j in range(
                len(
                    CLASSES
                )
            ):

                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.85,
        label=(
            "Среднее cosine similarity"
        ),
    )

    fig.suptitle(
        "Косинусное сходство объектов "
        "между категориями EU-TIRADS"
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.91,
        wspace=0.28,
        hspace=0.28,
    )

    png_path = (
        output_dir
        / "class_cosine_similarity.png"
    )

    svg_path = (
        output_dir
        / "class_cosine_similarity.svg"
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
# Macro-F1 основного benchmark
# ============================================================

def load_split_macro_f1(
    runs_dir: Path,
    model_name: str,
    split_id: int,
):
    path = (
        runs_dir
        / model_name
        / f"split_{split_id:02d}"
        / "metrics.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Не найдены metrics:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metrics = json.load(
            file
        )

    return float(
        metrics[
            "macro_f1_2_5"
        ]
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-id",
        type=int,
        default=0,
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
        "--bethesda-model",
        type=Path,
        default=DEFAULT_BETHESDA_MODEL,
    )

    parser.add_argument(
        "--transformer-cache-dir",
        type=Path,
        default=DEFAULT_TRANSFORMER_CACHE,
    )

    parser.add_argument(
        "--native-model-dir",
        type=Path,
        default=DEFAULT_NATIVE_MODEL_DIR,
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
    )

    parser.add_argument(
        "--tsne-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Dataset / split
    # ========================================================

    df = load_eutirads_dataset(
        args.data
    )

    splits = pd.read_csv(
        args.splits
    )

    train, test = (
        load_split(
            df=df,
            splits=splits,
            split_id=(
                args.split_id
            ),
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "REPRESENTATION GEOMETRY"
    )

    print(
        f"split={args.split_id}"
    )

    print(
        "=" * 60
    )

    print(
        f"outer train: {len(train)}"
    )

    print(
        f"outer test:  {len(test)}"
    )

    # --------------------------------------------------------
    # Исключаем класс 1 только из geometry analysis
    # --------------------------------------------------------

    analysis_test = (
        test[
            test[
                "label"
            ].isin(
                CLASSES
            )
        ]
        .copy()
    )

    labels = (
        analysis_test[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    test_indices = (
        analysis_test
        .index
        .to_numpy(
            dtype=int
        )
    )

    print()
    print(
        "Geometry objects "
        "(classes 2-5): "
        f"{len(analysis_test)}"
    )

    print(
        "Class distribution:"
    )

    print(
        analysis_test[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    representations = {}

    # ========================================================
    # Bethesda fastText
    # ========================================================

    print()
    print(
        "Preparing Bethesda fastText..."
    )

    bethesda_model = (
        load_fasttext_model(
            args.bethesda_model
        )
    )

    representations[
        "bethesda_fasttext"
    ] = (
        vectorize_pretrained_texts(
            bethesda_model,
            analysis_test[
                "text_model"
            ],
        )
    )

    del bethesda_model

    # ========================================================
    # Native fastText
    # ========================================================

    print()
    print(
        "Preparing native fastText..."
    )

    native_params = (
        load_native_best_params(
            runs_dir=(
                args.runs_dir
            ),

            split_id=(
                args.split_id
            ),
        )
    )

    (
        native_model,
        native_model_path,
    ) = (
        get_or_train_native_model(
            train_texts=(
                train[
                    "text_model"
                ].tolist()
            ),

            params=(
                native_params
            ),

            split_id=(
                args.split_id
            ),

            model_dir=(
                args.native_model_dir
            ),
        )
    )

    representations[
        "native_fasttext"
    ] = (
        vectorize_native_texts(
            native_model,
            analysis_test[
                "text_model"
            ],
        )
    )

    del native_model

    # ========================================================
    # Transformers
    # ========================================================

    for model_key in [
        "rubert",
        "rubiobert",
    ]:

        print()
        print(
            f"Preparing {model_key}..."
        )

        (
            all_embeddings,
            metadata,
        ) = (
            get_transformer_embeddings(
                model_key=(
                    model_key
                ),

                df=df,

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

        representations[
            model_key
        ] = (
            all_embeddings[
                test_indices
            ]
        )

    # ========================================================
    # Проверка размеров
    # ========================================================

    for (
        model_name,
        embeddings,
    ) in representations.items():

        if (
            len(
                embeddings
            )
            != len(
                analysis_test
            )
        ):
            raise RuntimeError(
                f"{model_name}: "
                "размер embeddings "
                "не совпадает с test."
            )

        print(
            f"{model_name}: "
            f"{embeddings.shape}"
        )

    # ========================================================
    # Silhouette + cosine
    # ========================================================

    summary_rows = []

    cosine_matrices = {}

    for model_name in (
        MODEL_ORDER
    ):

        embeddings = (
            representations[
                model_name
            ]
        )

        silhouette = (
            calculate_silhouette(
                embeddings=(
                    embeddings
                ),

                labels=labels,
            )
        )

        cosine_matrix = (
            calculate_class_cosine_matrix(
                embeddings=(
                    embeddings
                ),

                labels=labels,
            )
        )

        cosine_matrices[
            model_name
        ] = cosine_matrix

        macro_f1 = (
            load_split_macro_f1(
                runs_dir=(
                    args.runs_dir
                ),

                model_name=(
                    model_name
                ),

                split_id=(
                    args.split_id
                ),
            )
        )

        summary_rows.append(
            {
                "model": (
                    model_name
                ),

                "display_name": (
                    MODEL_LABELS[
                        model_name
                    ]
                ),

                "split_id": int(
                    args.split_id
                ),

                "n_objects": int(
                    len(
                        analysis_test
                    )
                ),

                "embedding_dim": int(
                    embeddings.shape[
                        1
                    ]
                ),

                "cosine_silhouette": float(
                    silhouette
                ),

                "macro_f1_2_5": float(
                    macro_f1
                ),
            }
        )

        cosine_df = pd.DataFrame(
            cosine_matrix,
            index=[
                f"true_{x}"
                for x in CLASSES
            ],
            columns=[
                f"class_{x}"
                for x in CLASSES
            ],
        )

        cosine_df.to_csv(
            args.output_dir
            / (
                f"{model_name}_"
                "class_cosine_similarity.csv"
            ),
            encoding="utf-8-sig",
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        args.output_dir
        / "geometry_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # t-SNE
    # ========================================================

    projections = {}

    for model_name in (
        MODEL_ORDER
    ):

        print()
        print(
            f"t-SNE: {model_name}"
        )

        projection = (
            calculate_tsne(
                embeddings=(
                    representations[
                        model_name
                    ]
                ),

                seed=(
                    args.tsne_seed
                ),
            )
        )

        projections[
            model_name
        ] = projection

        projection_df = pd.DataFrame(
            {
                "object_id": (
                    analysis_test[
                        "object_id"
                    ]
                    .to_numpy()
                ),

                "label": labels,

                "x": (
                    projection[
                        :,
                        0
                    ]
                ),

                "y": (
                    projection[
                        :,
                        1
                    ]
                ),
            }
        )

        projection_df.to_csv(
            args.output_dir
            / (
                f"{model_name}_"
                "tsne.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Figures
    # ========================================================

    plot_tsne_all(
        projections=(
            projections
        ),

        labels=labels,

        output_dir=(
            args.output_dir
        ),
    )

    plot_cosine_matrices(
        cosine_matrices=(
            cosine_matrices
        ),

        output_dir=(
            args.output_dir
        ),
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "GEOMETRY SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        summary[
            [
                "display_name",
                "embedding_dim",
                "cosine_silhouette",
                "macro_f1_2_5",
            ]
        ]
        .sort_values(
            "macro_f1_2_5",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Native geometry model -> "
        f"{native_model_path}"
    )

    print()
    print(
        f"Results -> "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()