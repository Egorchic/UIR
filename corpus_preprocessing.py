import pandas as pd
import os

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("dataset_clean.csv", header=None, encoding="utf-8", sep=",")
df.columns = ["bethesda_label", "text_clean"]

# на всякий случай уберём пустые строки
df["text_clean"] = df["text_clean"].astype(str).str.strip()
df = df[df["text_clean"].str.len() > 0].copy()

corpus_path = "artifacts/corpus.txt"
df["text_clean"].to_csv(corpus_path, index=False, header=False)

print("saved:", corpus_path, "docs:", len(df))