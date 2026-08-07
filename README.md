# UIR

Репозиторий содержит программную часть учебно-исследовательской работы.

## Структура проекта

```text
UIR/
├── artifacts/
│   └── corpus.txt
│
├── experiments/
│   └── ...
│
├── models/
│   └── .gitkeep
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
└── .gitignore
```

## Назначение файлов

### `text_preprocessing.py`

Подготавливает исходный датасет к дальнейшей работе.

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

Загружает подготовленный датасет и fastText-модель, строит векторные представления текстов и обучает классификатор.

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

Повторно запускает выбранную итоговую конфигурацию и строит итоговые графики и таблицы для анализа результатов.

Запуск:

```bash
python rebuild_final_model_plots.py
```

Результаты сохраняются в:

```text
plots/
```

---

## Каталоги

### `artifacts/`

Содержит промежуточные текстовые данные, используемые при обучении моделей.

### `experiments/`

Содержит результаты отдельных экспериментальных запусков.

Каждый эксперимент хранится в отдельном каталоге:

```text
experiments/<run_name>/
```

### `models/`

Предназначен для хранения обученных моделей.

Сами модели в репозиторий не загружены из-за большого объема файлов. При необходимости они могут быть обучены заново с помощью `fasttext_learning.py` или `experiment_runner.py`.

### `plots/`

Содержит построенные графики и визуализации результатов.

---

## Необходимые локальные файлы

Для полного запуска проекта необходим исходный файл:

```text
dataset.xlsx
```

Его необходимо поместить в корневую директорию репозитория.

После запуска `text_preprocessing.py` будет создан:

```text
dataset_clean.csv
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

---

## Рекомендуемый порядок запуска

Для полного запуска с исходного датасета:

### 1. Подготовка данных

```bash
python text_preprocessing.py
```

### 2. Формирование корпуса

```bash
python corpus_preprocessing.py
```

### 3. Запуск экспериментов

```bash
python experiment_runner.py
```

### 4. Построение графиков

```bash
python plot_experiment_results.py
```

### 5. Формирование сводных таблиц

```bash
python make_results_table.py
```

### 6. Построение итоговых графиков

```bash
python rebuild_final_model_plots.py
```

---

## Быстрый запуск отдельной конфигурации

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
