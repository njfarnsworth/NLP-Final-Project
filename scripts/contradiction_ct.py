import csv


file_path = "../predictions/sym_ood_agg.tsv"

# Counter
count = 0

# Open the file and read as tab-separated
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    
    for row in reader:
        if row["target"].strip().lower() == "neutral":
            count += 1

print(f'Number of "contradiction" entries in target column: {count}')
