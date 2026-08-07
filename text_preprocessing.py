import pandas as pd
import re
import os

pd.set_option('display.max_columns', None)

# Указываем полный путь к файлу
file_path = "dataset.xlsx"

# Проверяем, существует ли файл
if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Содержимое директории: {os.listdir('.')}")
    exit(1)

# Чтение данных
print(f"Чтение файла: {file_path}")
df = pd.read_excel(
    file_path,
    names=['name', 'desc'],
    skiprows=2
)

print(f"Загружено строк: {len(df)}")

# 1) Склеиваем текст из всех колонок (устойчиво, даже если колонок много)
df = df.fillna("")
df["text_raw"] = (
    df.astype(str)
      .agg(" ".join, axis=1)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)

# 2) Достаём метку Bethesda (I–VI), как и раньше, но компактнее
beth = r"\b[ВB][еe][тt][нh][еe][сs][дd][аa]\b"          # Bethesda с лат/кир смешением
num  = r"(VI|IV|V|III|II|I)(?![IVXLCDM])"               # I..VI

patterns = [
    rf"(?is){beth}.{{0,300}}?\b{num}",
    rf"(?is){beth}.{{0,800}}?\bкатегор\w*.{{0,80}}?\b{num}",
    rf"(?i)\b{num}.{{0,80}}?диагностическ\w*\s+категор\w*",
    rf"(?i)диагностическ\w*\s+категор\w*.{{0,80}}?\b{num}",
    rf"(?i)\bкатегор\w*.{{0,80}}?\b{num}",
]

label = None
for p in patterns:
    extracted = df["text_raw"].str.extract(p, expand=False)
    label = extracted if label is None else label.fillna(extracted)

df["Bethesda_label"] = label.str.upper()

# 3) Выкидываем строки без метки Bethesda
df = df[df["Bethesda_label"].notna()].copy()

# минимальный список стоп-слов (можно расширять)
RU_STOPWORDS = {
    "и","а","но","или","ли","же","то","не","ни",
    "в","во","на","у","к","ко","о","об","от","до","по","из","за","под","над","при","для","без","с","со","про","через",
    "это","как","так","что","чтобы","где","когда","там","тут","все","всё","этот","эта","эти","тот","та","те"
}
JUNK_TOKENS = {"nan","none","null"}

# 4) Функция чистки текста
def clean_text(s: str) -> str:
    s = s.lower()

    # убрать явные "служебные" штуки
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)                     # ссылки
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)                # email
    s = re.sub(r"\+?\d[\d\s\-\(\)]{8,}\d", " ", s)                   # телефоны (грубо)
    s = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", s)       # даты
    s = re.sub(r"[№#]\s*\w+", " ", s)                                # №123, #ABC

    # убрать всё, что связано с Bethesda, чтобы не обучаться "по метке"
    s = re.sub(rf"(?is){beth}.{{0,300}}?\b{num}\b", " ", s)          # bethesda ... IV
    s = re.sub(rf"(?is){beth}", " ", s)                              # само слово bethesda
    s = re.sub(r"(?i)диагностическ\w*\s+категор\w*", " ", s)         # "диагностическая категория"
    s = re.sub(r"(?i)\bкатегор\w*\b", " ", s)                        # "категория" (часто только про Bethesda)

    # опционально: убрать все числа (часто это мусор для эмбеддингов)
    s = re.sub(r"\b\d+\b", " ", s)

    # оставить только буквы (рус/лат) и пробелы
    s = re.sub(r"[^a-zа-яё\s]", " ", s)

    # нормализовать пробелы
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    tokens = [
        t for t in tokens
        if len(t) > 1  # убрать одиночные буквы типа "п", "л", "в"
           and t not in RU_STOPWORDS  # убрать предлоги/союзы
           and t not in JUNK_TOKENS  # убрать мусорные токены
    ]

    return " ".join(tokens)

df["text_clean"] = df["text_raw"].apply(clean_text)

# 5) убрать совсем пустые после чистки
df = df[df["text_clean"].str.len() > 0].copy()

print("Осталось строк:", len(df))
print(df["Bethesda_label"].value_counts().sort_index())

# если нужно сохранить:
# df.to_excel("dataset_clean.xlsx", index=False)
df[["Bethesda_label", "text_clean"]].to_csv("dataset_clean.csv", index=False, encoding="utf-8")