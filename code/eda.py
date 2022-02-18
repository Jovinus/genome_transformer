# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# %%
df_orig = pd.read_csv("../data/train.csv", usecols=['id', 'genome_sequence', 'species'])
df_orig.head()
# %%
