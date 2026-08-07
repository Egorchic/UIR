import csv
import json
import os
import subprocess
import sys


PYTHON = sys.executable

FASTTEXT_SCRIPT = "fasttext_learning.py"
CLASSIFIER_SCRIPT = "classification_fasttext_lr.py"

MODELS_DIR = "models"
EXPERIMENTS_DIR = "experiments"
RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, "results.csv")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def run_command(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_fasttext(model_type, dim, minn, maxn):
    model_name = f"ft_{model_type}_dim{dim}_minn{minn}_maxn{maxn}.bin"
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        cmd = [
            PYTHON, FASTTEXT_SCRIPT,
            "--model", model_type,
            "--dim", str(dim),
            "--minn", str(minn),
            "--maxn", str(maxn),
            "--output", model_path
        ]
        run_command(cmd)
    else:
        print(f"fastText model already exists: {model_path}")

    return model_path


def run_classifier(run_name, model_path, class_weight="balanced", C=1.0):
    cmd = [
        PYTHON, CLASSIFIER_SCRIPT,
        "--model-path", model_path,
        "--run-name", run_name,
        "--output-dir", EXPERIMENTS_DIR,
        "--class-weight", class_weight,
        "--C", str(C)
    ]
    run_command(cmd)

    metrics_path = os.path.join(EXPERIMENTS_DIR, run_name, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_result(row, file_exists):
    fieldnames = [
        "run_name", "fasttext_model", "dim", "minn", "maxn",
        "class_weight", "C", "accuracy", "macro_f1", "weighted_f1",
        "embedding_time_sec", "train_time_sec", "predict_time_sec"
    ]

    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ensure_dirs()

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    file_exists = False

    experiments = []

    # 1) dim = 100 / 200 / 300
    for dim in [100, 200, 300]:
        experiments.append({
            "model": "skipgram",
            "dim": dim,
            "minn": 2,
            "maxn": 5,
            "class_weight": "balanced",
            "C": 1.0,
            "run_name": f"skipgram_dim{dim}_subword_balanced_C1"
        })

    # 2) skipgram vs cbow
    for model_type in ["skipgram", "cbow"]:
        experiments.append({
            "model": model_type,
            "dim": 200,
            "minn": 2,
            "maxn": 5,
            "class_weight": "balanced",
            "C": 1.0,
            "run_name": f"{model_type}_dim200_subword_balanced_C1"
        })

    # 3) minn=0,maxn=0 vs minn=2,maxn=5
    for minn, maxn, tag in [(0, 0, "nosubword"), (2, 5, "subword")]:
        experiments.append({
            "model": "skipgram",
            "dim": 200,
            "minn": minn,
            "maxn": maxn,
            "class_weight": "balanced",
            "C": 1.0,
            "run_name": f"skipgram_dim200_{tag}_balanced_C1"
        })

    # 4) class_weight=None vs balanced
    for cw in ["none", "balanced"]:
        experiments.append({
            "model": "skipgram",
            "dim": 200,
            "minn": 2,
            "maxn": 5,
            "class_weight": cw,
            "C": 1.0,
            "run_name": f"skipgram_dim200_subword_{cw}_C1"
        })

    # 5) C=0.1 / 1 / 10 для balanced
    for c_val in [0.1, 1.0, 10.0]:
        experiments.append({
            "model": "skipgram",
            "dim": 200,
            "minn": 2,
            "maxn": 5,
            "class_weight": "balanced",
            "C": c_val,
            "run_name": f"skipgram_dim200_subword_balanced_C{str(c_val).replace('.', '_')}"
        })

    # 6) C=0.1 / 1 / 10 для none
    for c_val in [0.1, 1.0, 10.0]:
        experiments.append({
            "model": "skipgram",
            "dim": 200,
            "minn": 2,
            "maxn": 5,
            "class_weight": "none",
            "C": c_val,
            "run_name": f"skipgram_dim200_subword_none_C{str(c_val).replace('.', '_')}"
        })

    seen = set()
    unique_experiments = []
    for exp in experiments:
        if exp["run_name"] not in seen:
            unique_experiments.append(exp)
            seen.add(exp["run_name"])

    for exp in unique_experiments:
        model_path = train_fasttext(
            model_type=exp["model"],
            dim=exp["dim"],
            minn=exp["minn"],
            maxn=exp["maxn"]
        )

        metrics = run_classifier(
            run_name=exp["run_name"],
            model_path=model_path,
            class_weight=exp["class_weight"],
            C=exp["C"]
        )

        row = {
            "run_name": exp["run_name"],
            "fasttext_model": exp["model"],
            "dim": exp["dim"],
            "minn": exp["minn"],
            "maxn": exp["maxn"],
            "class_weight": exp["class_weight"],
            "C": exp["C"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "embedding_time_sec": metrics["embedding_time_sec"],
            "train_time_sec": metrics["train_time_sec"],
            "predict_time_sec": metrics["predict_time_sec"]
        }

        append_result(row, file_exists)
        file_exists = True

    print(f"All experiments finished. Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()