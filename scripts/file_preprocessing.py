"""
Small file covnerting the moNLI dataset into NLPScholar friendly format
"""

import json
import pandas as pd
import numpy as np

def convert_jsonl(in_fpath:str):
    rows = []
    cols = ['textid','text','pair','weak','strong','label']
    with open(in_fpath,"r") as file:
        idx = 1
        for line in file:
            dct = json.loads(line)
            if dct['gold_label'] == 'entailment':
                strong = dct['sentence1_lex']
                weak = dct['sentence2_lex']
            else:
                strong = dct['sentence2_lex']
                weak = dct['sentence1_lex']
            row = [idx,dct['sentence1'],dct['sentence2'],weak,strong,dct['gold_label']]
            rows.append(row)
            idx += 1
    df = pd.DataFrame(rows,columns=cols)
    return df



def split_data(df:pd.DataFrame, out:str,split:tuple):
    """
    Split data into train, valid, and test splits maintaining the grouping of templated sentences
    This is a choice that ensures that there is no overlap in sentences between test, train, and valid data
    """
    train, val, test = split
    group_col = 'strong'

    groups = df['strong'].unique() # split on templates
    np.random.seed(42)
    np.random.shuffle(groups)

    n_groups = len(groups)
    train_end = int(train * n_groups)
    val_end = int((train + val) * n_groups)

    # Assign group sets
    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    # Subset the data
    df_train = df[df[group_col].isin(train_groups)]
    df_val = df[df[group_col].isin(val_groups)]
    df_test = df[df[group_col].isin(test_groups)]
    df_test = df_test.rename(columns={'label':'target'})

    # Derive file names
    for name, subset in zip(["train", "val", "test"], [df_train, df_val, df_test]):
        if not subset.empty:
            subset.to_csv(f"../data/{out}_{name}.tsv",sep="\t", index=False)

    return df_train, df_val, df_test



def main():
    #negative_train = convert_jsonl("../data/nmonli_train.jsonl")
    #split negative train into train and valid
    #split_data(negative_train,"negative",(.9,.1,.0))
    
    #negative_test = convert_jsonl("../data/nmonli_test.jsonl")
    #split_data(negative_test,"negative",(0,0,1))
    
    #pmonli = convert_jsonl("../data/pmonli.jsonl")
    #plit_data(pmonli,"positive",(.7,.15,.15))

    #synth_pos = pd.read_csv("../data/positive_synthetic.tsv",sep="\t")
    #synth_pos_cleaned = synth_pos[~synth_pos['text'].str.contains("not")]
    #synth_pos_cleaned['label'] = synth_pos_cleaned['label'].replace({"entailment":"neutral","neutral":"entailment"})
    #split_data(synth_pos_cleaned,"synth-pos-2",(.8,.2,0))


    #synth_neg = pd.read_csv("../data/negative_synthetic.tsv",sep="\t")
    #split_data(synth_neg,"synth-neg",(.8,.2,0))


    #create mega data set for cross evaluation, train on everything
    data_files = ['negative_train.tsv','negative_val.tsv','positive_train.tsv','positive_val.tsv','synth-neg_train.tsv','synth-neg_val.tsv','synth-pos_train.tsv','synth-pos_val.tsv']
    dfs = []
    for file in data_files:
        dfs.append(pd.read_csv(f"../data/{file}",sep="\t"))
    
    df = pd.concat(dfs,ignore_index=True)
    df = split_data(df,"mono-data",(.8,.2,0))
    #df.to_csv("../data/mono_data.tsv",index=False,sep="\t")





main()