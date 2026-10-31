import pandas as pd

path = (
    "article_results/"
    "learning_curve_v2/"
    "native_fasttext_learning_curve.csv"
)

df = pd.read_csv(path)

pivot = df.pivot(
    index="split_id",
    columns="fraction",
    values="macro_f1_2_5",
)

print("=== NATIVE FASTTEXT BY SPLIT ===")
print(pivot.to_string())

print("\n=== 75% -> 100% ===")

pivot["delta_100_minus_75"] = (
    pivot[1.0] - pivot[0.75]
)

print(
    pivot[
        [
            0.75,
            1.0,
            "delta_100_minus_75",
        ]
    ].to_string()
)

print()
print(
    "Mean delta:",
    pivot[
        "delta_100_minus_75"
    ].mean()
)

print(
    "100% better:",
    (
        pivot[
            "delta_100_minus_75"
        ] > 0
    ).sum()
)

print(
    "100% worse:",
    (
        pivot[
            "delta_100_minus_75"
        ] < 0
    ).sum()
)