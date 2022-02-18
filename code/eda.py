# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# %%
df_orig = pd.read_csv("../data/train.csv", usecols=['id', 'genome_sequence', 'species'])
# %%
df_orig = df_orig.assign(max_seq_len = lambda x: x['genome_sequence'].apply(len),
                         unique_neuc = lambda x: x['genome_sequence'].apply(lambda y: sorted(list(set(y)))),
                         unique_neuc_num = lambda x: x['unique_neuc'].apply(len))
# %%
df_orig['max_seq_len'].value_counts()

# %%
display(df_orig.head())
# %%
