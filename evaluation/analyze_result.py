import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("results")

summary_file = RESULTS_DIR / "summary_report.csv"

df = pd.read_csv(summary_file)

# -----------------------------
# Compute improvement over baseline
# -----------------------------
baseline = df[df["method"] == "tesseract_baseline"]

records = []

for _, base_row in baseline.iterrows():

    lang = base_row["language"]
    cat = base_row["category"]

    subset = df[(df["language"] == lang) & (df["category"] == cat)]

    for _, row in subset.iterrows():

        if row["method"] == "tesseract_baseline":
            continue

        cer_improve = ((base_row["avg_cer"] - row["avg_cer"]) / base_row["avg_cer"]) * 100
        wer_improve = ((base_row["avg_wer"] - row["avg_wer"]) / base_row["avg_wer"]) * 100

        records.append({
            "language": lang,
            "category": cat,
            "method": row["method"],
            "CER_improvement_%": round(cer_improve,2),
            "WER_improvement_%": round(wer_improve,2)
        })

imp_df = pd.DataFrame(records)

# save improvement table
imp_csv = RESULTS_DIR / "improvement_report.csv"
imp_df.to_csv(imp_csv, index=False)

print("\nImprovement over baseline:")
print(imp_df)


# -----------------------------
# Graph 1: Overall CER by method
# -----------------------------
overall = df.groupby("method")["avg_cer"].mean()

plt.figure(figsize=(6,4))
overall.plot(kind="bar")

plt.title("Average CER by OCR Method")
plt.ylabel("Character Error Rate (lower is better)")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "cer_by_method.png")
plt.close()


# -----------------------------
# Graph 2: CER by category
# -----------------------------
pivot_cat = df.pivot_table(
    index="category",
    columns="method",
    values="avg_cer"
)

pivot_cat.plot(kind="bar", figsize=(10,6))

plt.title("CER by Image Category")
plt.ylabel("Character Error Rate")
plt.xlabel("Category")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(RESULTS_DIR / "cer_by_category.png")
plt.close()


# -----------------------------
# Graph 3: CER Heatmap
# -----------------------------
heatmap_data = df.pivot_table(
    index="category",
    columns="method",
    values="avg_cer"
)

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")

plt.title("OCR CER Heatmap")

plt.tight_layout()

plt.savefig(RESULTS_DIR / "cer_heatmap.png")
plt.close()


print("\nGraphs saved to:", RESULTS_DIR)