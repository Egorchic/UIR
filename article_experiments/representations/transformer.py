from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


# ============================================================
# Вспомогательные функции
# ============================================================

def _resolve_device(
    torch,
    requested_device: str | None,
) -> str:

    if requested_device is not None:
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def _text_fingerprint(
    object_ids,
    texts,
) -> str:
    """
    Проверяет, что cache относится именно к текущей
    последовательности объектов и текстов.
    """

    hasher = hashlib.sha256()

    for object_id, text in zip(
        object_ids,
        texts,
    ):
        line = (
            f"{object_id}\t{text}\n"
        )

        hasher.update(
            line.encode(
                "utf-8"
            )
        )

    return hasher.hexdigest()


def _get_token_lengths(
    tokenizer,
    texts,
) -> np.ndarray:
    """
    Длины текстов в subword-токенах BERT
    до truncation.
    """

    texts = [
        str(text)
        for text in texts
    ]

    try:

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )

        lengths = encoded[
            "length"
        ]

        return np.asarray(
            lengths,
            dtype=int,
        )

    except Exception:

        # Более медленный, но универсальный fallback.
        lengths = [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in texts
        ]

        return np.asarray(
            lengths,
            dtype=int,
        )


# ============================================================
# Frozen Transformer embeddings
# ============================================================

def encode_frozen_transformer(
    model_id: str,
    texts,
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
):
    """
    Получает одно фиксированное представление для текста.

    Encoder полностью заморожен:
    model.eval() + torch.inference_mode().

    Представление текста:
    mean pooling последнего hidden state.

    Padding, [CLS], [SEP] и другие special tokens
    в среднее не включаются.
    """

    try:
        import torch

        import transformers

        from transformers import (
            AutoModel,
            AutoTokenizer,
        )

    except ImportError as exc:
        raise ImportError(
            "Для transformer-эксперимента нужны "
            "torch и transformers.\n"
            "Установи их перед запуском."
        ) from exc

    texts = [
        str(text).strip()
        for text in texts
    ]

    if any(
        text == ""
        for text in texts
    ):
        raise ValueError(
            "Для Transformer обнаружены "
            "пустые тексты."
        )

    resolved_device = (
        _resolve_device(
            torch,
            device,
        )
    )

    print()
    print(
        f"Loading tokenizer: {model_id}"
    )

    try:
        tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
            )
        )

    except Exception:

        tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
            )
        )

    print(
        f"Loading model: {model_id}"
    )

    model = (
        AutoModel.from_pretrained(
            model_id
        )
    )

    model.to(
        resolved_device
    )

    model.eval()

    print(
        f"Device: {resolved_device}"
    )

    print(
        f"Texts: {len(texts)}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Max length: {max_length}"
    )

    # ========================================================
    # Статистика длины
    # ========================================================

    token_lengths = (
        _get_token_lengths(
            tokenizer,
            texts,
        )
    )

    n_truncated = int(
        (
            token_lengths
            > max_length
        ).sum()
    )

    print()
    print(
        "Tokenizer length stats:"
    )

    print(
        f"mean={token_lengths.mean():.2f}"
    )

    print(
        f"median={np.median(token_lengths):.2f}"
    )

    print(
        f"p95={np.percentile(token_lengths, 95):.2f}"
    )

    print(
        f"p99={np.percentile(token_lengths, 99):.2f}"
    )

    print(
        f"max={token_lengths.max()}"
    )

    print(
        f"truncated={n_truncated}"
        f"/{len(texts)}"
    )

    # ========================================================
    # Embeddings
    # ========================================================

    vectors = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(texts),
        )

        batch_texts = (
            texts[
                start:end
            ]
        )

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        special_tokens_mask = (
            encoded.pop(
                "special_tokens_mask"
            )
            .to(
                resolved_device
            )
        )

        encoded = {
            key: value.to(
                resolved_device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():

            output = model(
                **encoded
            )

            hidden = (
                output.last_hidden_state
            )

            attention_mask = (
                encoded[
                    "attention_mask"
                ]
                .unsqueeze(-1)
                .to(
                    hidden.dtype
                )
            )

            non_special_mask = (
                (
                    1
                    - special_tokens_mask
                )
                .unsqueeze(-1)
                .to(
                    hidden.dtype
                )
            )

            pooling_mask = (
                attention_mask
                * non_special_mask
            )

            vector_sum = (
                (
                    hidden
                    * pooling_mask
                )
                .sum(
                    dim=1
                )
            )

            denominator = (
                pooling_mask
                .sum(
                    dim=1
                )
                .clamp(
                    min=1.0
                )
            )

            pooled = (
                vector_sum
                / denominator
            )

        vectors.append(
            pooled
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        print(
            f"\rEncoded "
            f"{end}/{len(texts)}",
            end="",
            flush=True,
        )

    print()

    embeddings = np.vstack(
        vectors
    )

    config = model.config

    metadata = {
        "transformer_model_id": (
            model_id
        ),

        "transformer_model_class": (
            model.__class__.__name__
        ),

        "transformer_tokenizer_class": (
            tokenizer.__class__.__name__
        ),

        "transformer_hidden_size": int(
            embeddings.shape[1]
        ),

        "transformer_num_hidden_layers": (
            int(
                getattr(
                    config,
                    "num_hidden_layers",
                    -1,
                )
            )
        ),

        "transformer_max_position_embeddings": (
            int(
                getattr(
                    config,
                    "max_position_embeddings",
                    -1,
                )
            )
        ),

        "transformer_pooling": (
            "mean_last_hidden_"
            "excluding_special_tokens"
        ),

        "transformer_max_length": int(
            max_length
        ),

        "transformer_inference_batch_size": int(
            batch_size
        ),

        "transformer_device": (
            resolved_device
        ),

        "transformer_frozen": True,

        "transformer_finetuning_on_eutirads": (
            False
        ),

        "transformer_n_texts": int(
            len(texts)
        ),

        "transformer_mean_token_length": float(
            token_lengths.mean()
        ),

        "transformer_p95_token_length": float(
            np.percentile(
                token_lengths,
                95,
            )
        ),

        "transformer_p99_token_length": float(
            np.percentile(
                token_lengths,
                99,
            )
        ),

        "transformer_max_token_length": int(
            token_lengths.max()
        ),

        "transformer_n_truncated": (
            n_truncated
        ),

        "transformer_truncation_rate": float(
            n_truncated
            / len(texts)
        ),

        "torch_version": (
            torch.__version__
        ),

        "transformers_version": (
            transformers.__version__
        ),
    }

    commit_hash = getattr(
        config,
        "_commit_hash",
        None,
    )

    if commit_hash is not None:

        metadata[
            "transformer_commit_hash"
        ] = str(
            commit_hash
        )

    return (
        embeddings,
        metadata,
    )


# ============================================================
# Cache
# ============================================================

def get_or_create_embeddings(
    model_key: str,
    model_id: str,
    object_ids,
    texts,
    cache_dir: str | Path,
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
):
    """
    Transformer дорогой, поэтому embeddings всех 3086 текстов
    рассчитываются один раз.

    После этого 10 outer split используют одну и ту же
    замороженную матрицу признаков.
    """

    cache_dir = Path(
        cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fingerprint = (
        _text_fingerprint(
            object_ids,
            texts,
        )
    )

    cache_stem = (
        f"{model_key}_"
        f"mean_"
        f"maxlen{max_length}"
    )

    embeddings_path = (
        cache_dir
        / f"{cache_stem}.npy"
    )

    metadata_path = (
        cache_dir
        / f"{cache_stem}.json"
    )

    # ========================================================
    # Existing cache
    # ========================================================

    if (
        embeddings_path.exists()
        and metadata_path.exists()
    ):

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        valid_cache = (
            metadata.get(
                "transformer_model_id"
            )
            == model_id
            and metadata.get(
                "input_fingerprint"
            )
            == fingerprint
            and int(
                metadata.get(
                    "transformer_max_length",
                    -1,
                )
            )
            == int(
                max_length
            )
        )

        if valid_cache:

            print()
            print(
                "Loading cached "
                "Transformer embeddings:"
            )

            print(
                embeddings_path
            )

            embeddings = np.load(
                embeddings_path
            )

            if (
                len(embeddings)
                != len(texts)
            ):
                raise RuntimeError(
                    "Размер cached embeddings "
                    "не совпадает с датасетом."
                )

            metadata = dict(
                metadata
            )

            metadata[
                "transformer_cache_hit"
            ] = True

            return (
                embeddings,
                metadata,
            )

    # ========================================================
    # Compute
    # ========================================================

    embeddings, metadata = (
        encode_frozen_transformer(
            model_id=model_id,
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    )

    metadata[
        "input_fingerprint"
    ] = fingerprint

    metadata[
        "transformer_cache_hit"
    ] = False

    np.save(
        embeddings_path,
        embeddings,
    )

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

    print()
    print(
        "Transformer embeddings saved:"
    )

    print(
        embeddings_path
    )

    return (
        embeddings,
        metadata,
    )