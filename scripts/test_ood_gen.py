import re
import random
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from SPARQLWrapper import SPARQLWrapper, JSON


TRAIN_TSV = Path("../data/sym_train.tsv") 
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


CATEGORIES = {
    "person": "Q5",           # spouse of, collaborated with, teacher of, older than
    "organization": "Q43229", # collaborated with
    "city": "Q515",           # adjacent to, north of
    "river": "Q4022",         # intersects
    "mountain": "Q8502",      # greater than (elevation)
}

PATTERN = re.compile(r"^(Q\d+)\s+(.*?)\s+(Q\d+)\.?$")

def parse_qids_from_delex_sentence(s):
    s = str(s or "").strip()
    m = PATTERN.match(s)
    if not m:
        return None, None, None
    return m.group(1), m.group(2).strip(), m.group(3)

def batched(iterable, n):
    lst = list(iterable)
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def classify_qids_by_type(qids, categories, batch_size=200):
    """
    Return {category: set(qids)} for input qids whose P31/P279* matches categories[type_qid].
    """
    out = {c: set() for c in categories}
    if not qids:
        return out

    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql", agent="wd-qid-filter/1.0")
    endpoint.setReturnFormat(JSON)

    type_qids = list(categories.values())
    type_values = " ".join(f"wd:{t}" for t in type_qids)
    type_to_cat = {t: c for c, t in categories.items()}

    for batch in batched(sorted(qids), batch_size):
        item_values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT ?item ?matchedType WHERE {{
          VALUES ?item {{ {item_values} }}
          VALUES ?matchedType {{ {type_values} }}
          ?item wdt:P31/wdt:P279* ?matchedType .
        }}
        """
        endpoint.setQuery(query)
        res = endpoint.query().convert()
        for b in res["results"]["bindings"]:
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            t_qid = b["matchedType"]["value"].rsplit("/", 1)[-1]
            cat = type_to_cat.get(t_qid)
            if cat:
                out[cat].add(qid)
    return out

def fetch_labels_for_qids(qids, batch_size=200, lang="en"):
    """
    Return {qid: label_lang} for given qids (falls back to QID if missing label).
    """
    labels = {}
    if not qids:
        return labels

    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql", agent="wd-labels/1.0")
    endpoint.setReturnFormat(JSON)

    for batch in batched(sorted(qids), batch_size):
        values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          VALUES ?item {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
        }}
        """
        endpoint.setQuery(query)
        res = endpoint.query().convert()
        for b in res["results"]["bindings"]:
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            label = b.get("itemLabel", {}).get("value", qid)
            labels[qid] = label
    return labels


train_df = pd.read_csv(TRAIN_TSV, sep="\t", dtype={"textid": str})
if "textid" not in train_df.columns:
    raise ValueError("Your TSV must include a 'textid' column (string).")

textids = set(train_df["textid"].astype(str))
print(f"Loaded {len(textids)} TextIDs from {TRAIN_TSV}")


delex_df = load_dataset("MoyYuan/Asymmetricity-Delex-NLI", split="train").to_pandas().reset_index()
delex_df["textid"] = delex_df["index"].astype(str)
delex_df.drop(columns=["index"], inplace=True)

delex_text_col = "premise" if "premise" in delex_df.columns else ("text" if "text" in delex_df.columns else None)
if delex_text_col is None:
    raise ValueError("Could not find a delex text column (looked for 'premise' or 'text').")


delex_subset = delex_df[delex_df["textid"].isin(textids)].copy()
print(f"Filtered delex to {len(delex_subset)} rows matching your TextIDs.")


qids = set()
n_parsed = 0
for s in delex_subset[delex_text_col].astype(str):
    subj, reltxt, obj = parse_qids_from_delex_sentence(s)
    if subj and obj:
        qids.update([subj, obj])
        n_parsed += 1

print(f"Parsed QIDs from {n_parsed} delex rows; {len(qids)} unique QIDs collected.")

classified = classify_qids_by_type(qids, CATEGORIES, batch_size=200)

# ----------------------------
# 6) Sample up to 100 per category (from YOUR QIDs), fetch labels, and show diagnostics
# ----------------------------
sampled_qids_by_cat = {}
for cat, ids in classified.items():
    ids = sorted(list(ids))
    k = min(100, len(ids))
    sampled_qids_by_cat[cat] = random.sample(ids, k)

all_sampled_qids = {qid for ids in sampled_qids_by_cat.values() for qid in ids}
qid_to_label = fetch_labels_for_qids(all_sampled_qids, batch_size=200, lang="en")

sampled = {
    cat: [(qid, qid_to_label.get(qid, qid)) for qid in sampled_qids_by_cat.get(cat, [])]
    for cat in CATEGORIES
}


relations = {
    "is the spouse of": True,     
    "collaborated with": True,
    "is adjacent to": True,
    "Intersects": True,
    "is equivalent to": True,
    "is the teacher of": False,    
    "is a subset of": False,       
    "is greater than": False,
    "is older than": False,
    "is north of": False,
}


relation_to_categories = {
    "is the spouse of":       [("person", "person")],
    "collaborated with":      [("person", "person"), ("organization", "organization")],
    "is adjacent to":         [("city", "city"), ("river", "river"), ("mountain", "mountain")],
    "Intersects":             [("river", "river")],
    "is equivalent to":       [("person", "person"), ("organization", "organization"),
                               ("city", "city"), ("river", "river"), ("mountain", "mountain")],
    "is the teacher of":      [("person", "person")],
    "is a subset of":       [("organization", "organization")], 
    "is greater than":        [("mountain", "mountain")],  
    "is older than":          [("person", "person")],
    "is north of":            [("city", "city"), ("river", "river"), ("mountain", "mountain")],
}

def make_pairs(labels_list, max_pairs=None, allow_self=False):
    """
    From a list like [(QID, 'Label'), ...] produce (A_label, B_label) pairs.
    """
    names = [label for _, label in labels_list]
    pairs = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if not allow_self and i == j:
                continue
            # unordered for symmetric semantics; but we’ll still emit AB vs BA per your format rule
            if a != b:
                pairs.append((a, b))
    # Deduplicate unordered pairs (A,B) ~ (B,A) to keep dataset small
    seen = set()
    uniq = []
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    if max_pairs is not None:
        random.shuffle(uniq)
        uniq = uniq[:max_pairs]
    return uniq

# Build all (A,B, relation, symmetric_bool) rows from your sampled Wikidata labels
rows = []
for rel, is_sym in relations.items():
    cats = relation_to_categories.get(rel, [])
    if not cats:
        continue  # skip relations we didn't map to categories
    for (catA, catB) in cats:
        listA = sampled.get(catA, [])
        listB = sampled.get(catB, [])
        if not listA or not listB:
            continue
        if catA == catB:
            pairs_AB = make_pairs(listA, max_pairs=100, allow_self=False)
        else:
            namesA = [lbl for _, lbl in listA]
            namesB = [lbl for _, lbl in listB]
            pairs_AB = [(a, b) for a in namesA for b in namesB]
            # keep it small
            random.shuffle(pairs_AB)
            pairs_AB = pairs_AB[:100]

        for A, B in pairs_AB:
            rows.append((A, B, rel, "symmetric" if is_sym else "asymmetric"))

def norm(s: str) -> str:
    return s.lower().replace(".", "").strip()

output_file = "/home/nfarnsworth/NLP-Final-Project/data/sym_test_ood.tsv"

with open(output_file, "w", encoding="utf-8") as f:
    # new header
    f.write("textid\ttext\tpair\trelation\tsymmetry\ttarget\n")

    textid = 1
    for A, B, rel, sym in rows:
        s1 = f"{A} {rel} {B}"
        s2 = f"{B} {rel} {A}"
        t1 = norm(s1)   
        t2 = norm(s2)

        if sym == "symmetric":
            # AB vs BA = entailment
            f.write(f"{textid}\t{t1}\t{t2}\t{rel}\tsymmetric\tentailment\n")
        else:
            f.write(f"{textid}\t{t1}\t{t2}\t{rel}\tasymmetric\tneutral\n")
        textid += 1
   

print(f"✅ Saved {textid-1} rows to {output_file}")
