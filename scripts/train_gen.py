from datasets import load_dataset
import csv
from pathlib import Path
import random
import re
import pandas as pd 
import matplotlib.pyplot as plt

def find_relation_in_pair(premise_lower: str, hypothesis_lower: str):
    for rel in relations:
        if rel in premise_lower or rel in hypothesis_lower:
            return rel
    return None

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\.+$", "", text)  # remove one or more periods at the end
    return text


relations = ["is a child of",
             "partially overlaps with",
             "is a type of",
             "is a sibling of",
             "is an equivalent name of",
             "is located in",
             "is the mother of",
             "is the father of",
             "possibly the same as",
             "has part(s) that are instances of",
             "led to",
             "is different from",
             "is part of",
             "caused by"]

symmetry = {"is a child of": 0,
             "partially overlaps with": 1,
             "is a type of": 0,
             "is a sibling of": 1,
             "is an equivalent name of": 1,
             "is located in": 0,
             "is the mother of": 0,
             "is the father of": 0,
             "possibly the same as": 1,
             "has part(s) that are instances of": 0,
             "led to": 0,
             "is different from": 1,
             "is part of": 0,
             "caused by": 0}


DATASET_NAME = "MoyYuan/Asymmetricity-Text-NLI"
TARGET_PER_CLASS = 5_000

ds_train = load_dataset(DATASET_NAME, split="train")
df_train = ds_train.to_pandas()

for col in ("premise", "hypothesis"):
    if col in df_train.columns:
        df_train[col] = df_train[col].astype(str)
        df_train[col + "_lower"] = df_train[col].str.lower()
    else:
        raise ValueError(f"Column '{col}' not found in df_train")

df_train["found_relation"] = df_train.apply(
    lambda r: find_relation_in_pair(r["premise_lower"], r["hypothesis_lower"]), axis=1
)

matches = df_train[df_train["found_relation"].notna()].copy()
matches["sym_flag"] = matches["found_relation"].map(symmetry)
matches = matches[matches["sym_flag"].notna()]  
matches["symmetry"] = matches["sym_flag"].map({0: "asymmetric", 1: "symmetric"})


n_sym = min(5000, (matches["sym_flag"] == 1).sum())
n_asym = min(5000, (matches["sym_flag"] == 0).sum())

sample_sym = matches[matches["sym_flag"] == 1].sample(n=n_sym, random_state=42)
sample_asym = matches[matches["sym_flag"] == 0].sample(n=n_asym, random_state=42)

sampled = pd.concat([sample_sym, sample_asym], axis=0)

out_df = pd.DataFrame({
    "textid": sampled.index.astype(str),
    "text": sampled["premise"].apply(clean_text),
    "pair": sampled["hypothesis"].apply(clean_text),
    "relation": sampled["found_relation"],
    "symmetry": sampled["symmetry"]
})

out_df["label"] = out_df["symmetry"].map({
    "symmetric": "entailment",
    "asymmetric": "neutral"
})

out_path = Path("../data/sym_train.tsv")
out_df.to_csv(out_path, sep="\t", index=False)

print(f"Wrote {len(out_df)} rows to {out_path}")
print(out_df.head())
print("\nCounts:", out_df["symmetry"].value_counts().to_dict())


relation_counts = out_df["relation"].value_counts().reset_index()
relation_counts.columns = ["relation", "count"]

plt.figure(figsize=(10, 6))
plt.barh(relation_counts["relation"], relation_counts["count"], color="skyblue", edgecolor="black")
plt.xlabel("Count", fontsize=12)
plt.ylabel("Relation", fontsize=12)
plt.title("Distribution of Relations in Sampled Data", fontsize=14, weight="bold")
plt.gca().invert_yaxis()  
plt.tight_layout()

chart_path = Path("../data/symmetry_train_relation_distribution.png")
plt.savefig(chart_path, dpi=300)
plt.close()
