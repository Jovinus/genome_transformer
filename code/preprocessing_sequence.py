# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# %%
def k_mer_generator(input_sequence, k):
    
    seq_leng = len(input_sequence)
    
    num_move = (seq_leng - k ) + 1
    
    k_mers = []
    
    for i in range(num_move):
        k_mers.append(input_sequence[i:i+k])
        
    k_mers = " ".join(k_mers)
    
    return k_mers

# %%
label_dict = {'Homo_sapiens':0, 'Gorilla_gorilla':1}

# %%
if __name__ == '__main__':
    df_orig = pd.read_csv("../data/train.csv", usecols=['id', 'genome_sequence', 'species'])
    df_orig = df_orig.assign(max_seq_len = lambda x: x['genome_sequence'].apply(len),
                             unique_neuc = lambda x: x['genome_sequence'].apply(lambda y: sorted(list(set(y)))),
                             unique_neuc_num = lambda x: x['unique_neuc'].apply(len),
                             k_mers = lambda x: x['genome_sequence'].apply(lambda y: k_mer_generator(input_sequence=y, k=6).upper()), 
                             label = lambda x: x['species'].map(label_dict)
                             )
    
    df_orig = df_orig.query("max_seq_len == 80")
    
    df_orig[['k_mers', 'label']].to_csv("../data/train_prop.tsv", index=False, sep='\t')
    
    df_orig = pd.read_csv("../data/test.csv", usecols=['id', 'genome_sequence'])   
    df_orig = df_orig.assign(k_mers = lambda x: x['genome_sequence'].apply(lambda y: k_mer_generator(input_sequence=y, k=6).upper()))
    df_orig[['id', 'k_mers']].to_csv("../data/test_prop.tsv", index=False, sep='\t')
# %%
