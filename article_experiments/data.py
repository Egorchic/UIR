from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EUTIRADS_DATA = (
    PROJECT_ROOT
    / "experiments_eutirads"
    / "transfer_bethesda_ft"
    / "dataset_used.csv"
)

DEFAULT_BETHESDA_CORPUS = PROJECT_ROOT / "artifacts" / "corpus.txt"


EXPECTED_EUTIRADS_SIZE = 3086
EXPECTED_EUTIRADS_COUNTS = {
    1: 3,
    2: 770,
    3: 1719,
    4: 433,
    5: 161,
}


def load_eutirads_dataset(
    path: str | Path = DEFAULT_EUTIRADS_DATA,
    validate_expected: bool = True,
) -> pd.DataFrame:
    """
    Загружает замороженный набор EU-TIRADS,
    использованный в предыдущем эксперименте.

    Ожидается dataset_used.csv после:
    - очистки;
    - удаления утечки метки;
    - предобработки;
    - удаления дубликатов text_model + label.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл EU-TIRADS:\n{path}\n\n"
            "Передай путь через --data или положи dataset_used.csv "
            "в стандартный каталог эксперимента."
        )

    df = pd.read_csv(path)

    required = {"text_model", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"В {path} отсутствуют обязательные столбцы: {sorted(missing)}.\n"
            "Для статьи нужно использовать именно dataset_used.csv "
            "из старого эксперимента, а не исходный clean_strict."
        )

    df = df.copy()

    df["text_model"] = df["text_model"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if (df["text_model"] == "").any():
        raise ValueError("В dataset_used.csv обнаружены пустые text_model.")

    # Стабильный идентификатор объекта.
    # Лучше использовать source_row, если он сохранился.
    if "source_row" in df.columns:
        source_row = pd.to_numeric(df["source_row"], errors="raise").astype(int)
        df["object_id"] = source_row.map(lambda x: f"EU_{x:06d}")
    else:
        df["object_id"] = [
            f"EU_{i:06d}"
            for i in range(len(df))
        ]

    if df["object_id"].duplicated().any():
        duplicated = df.loc[
            df["object_id"].duplicated(keep=False),
            "object_id",
        ].tolist()

        raise ValueError(
            "object_id должен быть уникальным. "
            f"Найдены повторы: {duplicated[:10]}"
        )

    if validate_expected:
        validate_current_eutirads(df)

    return df.reset_index(drop=True)


def validate_current_eutirads(df: pd.DataFrame) -> None:
    """
    Проверяет, что используется именно выборка,
    на которой были получены результаты практики.
    """

    if len(df) != EXPECTED_EUTIRADS_SIZE:
        raise ValueError(
            f"Ожидалось {EXPECTED_EUTIRADS_SIZE} объектов, "
            f"получено {len(df)}."
        )

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if counts != EXPECTED_EUTIRADS_COUNTS:
        raise ValueError(
            "Распределение классов отличается от зафиксированного.\n"
            f"Ожидалось: {EXPECTED_EUTIRADS_COUNTS}\n"
            f"Получено:  {counts}"
        )


def load_bethesda_corpus(
    path: str | Path = DEFAULT_BETHESDA_CORPUS,
) -> list[str]:
    """
    Загружает тот корпус Bethesda, который использовался
    для обучения исходного fastText-пространства.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден Bethesda-корпус:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        texts = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if not texts:
        raise ValueError("Bethesda corpus.txt пуст.")

    return texts


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    SHA-256 выборки. Нужен, чтобы позднее проверить,
    что все эксперименты запускались на одних данных.
    """

    canonical = (
        df[["object_id", "label", "text_model"]]
        .sort_values("object_id")
        .astype(str)
        .to_csv(index=False)
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()