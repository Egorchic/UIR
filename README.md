# UIR

## Структура проекта

```text
UIR/
├── artifacts/
│   └── corpus.txt
│
├── experiments/
│   └── ...
│
├── experiments_eutirads/
│   └── ...
│
├── models/
│   └── ...
│
├── plots/
│   └── ...
│
├── text_preprocessing.py
├── corpus_preprocessing.py
├── fasttext_learning.py
├── classification_fasttext_lr.py
├── experiment_runner.py
├── plot_experiment_results.py
├── make_results_table.py
├── rebuild_final_model_plots.py
├── eutirads_transfer_experiment.py
├── eutirads_native_experiment.py
└── .gitignore
```

## Назначение файлов

### `text_preprocessing.py`

Подготавливает исходный датасет Bethesda к дальнейшей работе.

Входной файл:

```text
dataset.xlsx
```

Результат:

```text
dataset_clean.csv
```

---

### `corpus_preprocessing.py`

Формирует текстовый корпус из подготовленного датасета.

Вход:

```text
dataset_clean.csv
```

Результат:

```text
artifacts/corpus.txt
```

---

### `fasttext_learning.py`

Обучает fastText-модель на подготовленном корпусе.

Параметры модели можно задавать через аргументы командной строки.

Пример запуска:

```bash
python fasttext_learning.py \
    --model skipgram \
    --dim 200 \
    --minn 2 \
    --maxn 5 \
    --output models/ft_skipgram_dim200_minn2_maxn5.bin
```

Результат сохраняется в каталоге:

```text
models/
```

---

### `classification_fasttext_lr.py`

Загружает подготовленный датасет и fastText-модель, строит векторные представления текстов и обучает классификатор Bethesda.

Пример запуска:

```bash
python classification_fasttext_lr.py \
    --model-path models/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name example_run \
    --class-weight none \
    --C 1.0
```

Результаты запуска сохраняются в:

```text
experiments/example_run/
```

---

### `experiment_runner.py`

Автоматически запускает серию экспериментов с различными параметрами.

Скрипт:

* при необходимости обучает fastText-модели;
* запускает классификатор;
* сохраняет результаты каждого запуска;
* формирует общую таблицу результатов.

Запуск:

```bash
python experiment_runner.py
```

Результаты сохраняются в:

```text
experiments/
```

и в:

```text
experiments/results.csv
```

---

### `plot_experiment_results.py`

Строит графики по результатам экспериментов из:

```text
experiments/results.csv
```

Запуск:

```bash
python plot_experiment_results.py
```

Графики сохраняются в:

```text
plots/
```

---

### `make_results_table.py`

Формирует сводные таблицы результатов экспериментов.

Запуск:

```bash
python make_results_table.py
```

Результаты сохраняются в:

```text
experiments/summary/
```

---

### `rebuild_final_model_plots.py`

Повторно запускает выбранную итоговую конфигурацию Bethesda и строит итоговые графики и таблицы для анализа результатов.

Запуск:

```bash
python rebuild_final_model_plots.py
```

Результаты сохраняются в:

```text
plots/
```

---

### `eutirads_transfer_experiment.py`

Проводит эксперимент по переносу ранее обученного на текстах Bethesda пространства fastText на задачу классификации EU-TIRADS.

Скрипт:

* загружает очищенный датасет EU-TIRADS;
* выполняет дополнительную предобработку и удаление точных дубликатов;
* формирует стратифицированное разбиение train/test;
* получает векторы текстов с помощью фиксированной Bethesda fastText-модели;
* обучает новый `StandardScaler + LogisticRegression` для классов EU-TIRADS;
* рассчитывает метрики и сохраняет предсказания, ошибки и матрицы ошибок.

Пример запуска:

```bash
python eutirads_transfer_experiment.py \
    --data usidatanew_cleaned.xlsx \
    --ft-model models/best_model/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name transfer_bethesda_ft
```

Вариант с автоматической балансировкой классов:

```bash
python eutirads_transfer_experiment.py \
    --data usidatanew_cleaned.xlsx \
    --ft-model models/best_model/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name transfer_bethesda_ft_balanced \
    --class-weight balanced
```

Результаты сохраняются в:

```text
experiments_eutirads/<run_name>/
```

Среди сохраняемых файлов:

```text
dataset_used.csv
split.csv
X_embeddings.npy
y_labels.npy
metrics.json
classification_report.txt
classification_report.csv
confusion_matrix.csv
confusion_matrix.png
confusion_matrix_normalized.png
predictions.xlsx
errors.xlsx
class_distribution.csv
eutirads_lr.joblib
classes.json
```

---

### `eutirads_native_experiment.py`

Обучает новое пространство fastText непосредственно на обучающей части EU-TIRADS и сравнивает его с переносом пространства Bethesda.

Скрипт использует `dataset_used.csv` и `split.csv` из первого эксперимента, поэтому обе модели проверяются на одном и том же разбиении.

fastText обучается только на train-части:

```text
model = skipgram
dim = 200
minn = 2
maxn = 5
```

Пример запуска:

```bash
python eutirads_native_experiment.py \
    --baseline-dir experiments_eutirads/transfer_bethesda_ft \
    --run-name native_eutirads_ft
```

Вариант с автоматической балансировкой классов:

```bash
python eutirads_native_experiment.py \
    --baseline-dir experiments_eutirads/transfer_bethesda_ft \
    --run-name native_eutirads_ft_balanced \
    --class-weight balanced
```

Результаты сохраняются в:

```text
experiments_eutirads/<run_name>/
```

В том числе сохраняются:

```text
corpus_train.txt
eutirads_fasttext.bin
eutirads_lr.joblib
metrics.json
classification_report.txt
classification_report.csv
confusion_matrix.png
confusion_matrix_normalized.png
predictions.xlsx
errors.xlsx
```

---

## Эксперимент переноса Bethesda → EU-TIRADS

Цель дополнительного эксперимента — проверить, насколько скрытое векторное пространство, сформированное на цитологических текстах Bethesda, переносится на связанную задачу классификации эхографических описаний EU-TIRADS.

Сравниваются два варианта:

```text
EU-TIRADS text
    ↓
Bethesda fastText (frozen)
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
EU-TIRADS class
```

и

```text
EU-TIRADS text
    ↓
EU-TIRADS fastText (trained on train only)
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
EU-TIRADS class
```

Для корректного сравнения используются одинаковые:

* train/test-разбиение;
* размерность fastText `dim=200`;
* архитектура `skipgram`;
* subword-параметры `minn=2`, `maxn=5`;
* классификатор Logistic Regression.

### Итоговые результаты

| Векторное пространство | Accuracy | Macro-F1 | Macro-F1, классы 2–5 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Bethesda, без балансировки | 0.7314 | 0.5271 | 0.6580 | 0.7283 |
| EU-TIRADS, без балансировки | 0.6586 | 0.4124 | 0.5157 | 0.6354 |
| Bethesda, balanced | 0.6861 | 0.5009 | 0.6280 | 0.6945 |
| EU-TIRADS, balanced | 0.6019 | 0.4380 | 0.5482 | 0.6153 |

Полученные результаты показывают, что при ограниченном объеме данных EU-TIRADS ранее обученное пространство Bethesda оказалось эффективнее обучения fastText с нуля на новом небольшом корпусе.

Категория EU-TIRADS 1 после удаления точных дубликатов содержит только три уникальных объекта, поэтому дополнительно анализируется Macro-F1 по классам 2–5.

---

## Каталоги

### `artifacts/`

Содержит промежуточные текстовые данные, используемые при обучении моделей Bethesda.

### `experiments/`

Содержит результаты отдельных экспериментальных запусков Bethesda.

Каждый эксперимент хранится в отдельном каталоге:

```text
experiments/<run_name>/
```

### `experiments_eutirads/`

Предназначен для результатов экспериментов переноса Bethesda → EU-TIRADS и обучения пространства EU-TIRADS с нуля.

Каждый запуск хранится в отдельном каталоге:

```text
experiments_eutirads/<run_name>/
```

### `models/`

Предназначен для хранения обученных fastText-моделей.

Для эксперимента переноса требуется ранее обученная Bethesda-модель, например:

```text
models/best_model/ft_skipgram_dim200_minn2_maxn5.bin
```

### `plots/`

Содержит построенные графики и визуализации результатов.

---

## Необходимые локальные файлы

Для полного запуска исходной части проекта Bethesda необходим файл:

```text
dataset.xlsx
```

После запуска `text_preprocessing.py` будет создан:

```text
dataset_clean.csv
```

Для экспериментов EU-TIRADS необходим подготовленный файл:

```text
usidatanew_cleaned.xlsx
```

Ожидаемый лист:

```text
clean_strict
```

Основные используемые столбцы:

```text
source_row
input_raw
input_clean
label
```

Также для эксперимента переноса требуется ранее обученная Bethesda fastText-модель:

```text
models/best_model/ft_skipgram_dim200_minn2_maxn5.bin
```

---

## Установка

Рекомендуется использовать виртуальное окружение.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Установка основных зависимостей:

```bash
pip install numpy pandas scikit-learn fasttext joblib matplotlib openpyxl
```

На Windows при проблемах с установкой `fasttext` можно использовать:

```bash
pip install fasttext-wheel
```

---

## Рекомендуемый порядок запуска

### Эксперименты Bethesda

#### 1. Подготовка данных

```bash
python text_preprocessing.py
```

#### 2. Формирование корпуса

```bash
python corpus_preprocessing.py
```

#### 3. Запуск экспериментов

```bash
python experiment_runner.py
```

#### 4. Построение графиков

```bash
python plot_experiment_results.py
```

#### 5. Формирование сводных таблиц

```bash
python make_results_table.py
```

#### 6. Построение итоговых графиков

```bash
python rebuild_final_model_plots.py
```

---

### Эксперименты EU-TIRADS

#### 1. Перенос пространства Bethesda

```bash
python eutirads_transfer_experiment.py \
    --data usidatanew_cleaned.xlsx \
    --ft-model models/best_model/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name transfer_bethesda_ft
```

#### 2. Обучение пространства EU-TIRADS с нуля

```bash
python eutirads_native_experiment.py \
    --baseline-dir experiments_eutirads/transfer_bethesda_ft \
    --run-name native_eutirads_ft
```

#### 3. Эксперименты с балансировкой классов

```bash
python eutirads_transfer_experiment.py \
    --data usidatanew_cleaned.xlsx \
    --ft-model models/best_model/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name transfer_bethesda_ft_balanced \
    --class-weight balanced
```

```bash
python eutirads_native_experiment.py \
    --baseline-dir experiments_eutirads/transfer_bethesda_ft \
    --run-name native_eutirads_ft_balanced \
    --class-weight balanced
```

---

## Быстрый запуск отдельной конфигурации Bethesda

Если полный экспериментальный цикл не требуется, можно отдельно обучить модель:

```bash
python fasttext_learning.py \
    --model skipgram \
    --dim 200 \
    --minn 2 \
    --maxn 5 \
    --output models/ft_skipgram_dim200_minn2_maxn5.bin
```

а затем запустить классификатор:

```bash
python classification_fasttext_lr.py \
    --model-path models/ft_skipgram_dim200_minn2_maxn5.bin \
    --run-name example_run \
    --class-weight none \
    --C 1.0
```

Результат будет сохранен в:

```text
experiments/example_run/
```
