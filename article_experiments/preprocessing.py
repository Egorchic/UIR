from __future__ import annotations

import re


# Для анализа словаря рассматриваем именно словоформы.
# Числа исключаем: в EU-TIRADS много размеров узлов,
# и они искусственно раздували бы OOV.
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def normalize_word(word: str) -> str:
    """
    Минимальная нормализация словоформы.
    Лемматизацию намеренно не выполняем:
    нас интересует фактический лексический overlap корпусов.
    """

    return (
        word
        .lower()
        .replace("ё", "е")
    )


def tokenize_lexical(text: str) -> list[str]:
    """
    Извлекает только лексические токены.
    Используется исключительно для анализа словарей/OOV.
    """

    if not isinstance(text, str):
        text = str(text)

    return [
        normalize_word(token)
        for token in WORD_RE.findall(text)
    ]