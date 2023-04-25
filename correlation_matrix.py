"""
This code computes Pearson correlation coefficient
It returns a color-coded correlation matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## IMPORTING DATA

df = pd.read_csv('datasets/RawMaterials_and_Industries.csv', index_col=0)
df = df.iloc[:,9:] ## keeping only stationary series

## CORRELATION MATRIX

df_corr = df.corr()
df_corr = df_corr.iloc[3:,:3]

## EXPLORING

fig, ax = plt.subplots()
for i in range(len(df_corr)):
    for j in range(len(df_corr.columns)):
        value = round(df_corr.iloc[i].iat[j], 3)
        ax.text(j, i, str(value), va='center', ha='center', fontsize=15)
for i in range(len(df_corr.columns)):
    ax.vlines(i+0.5, -1, len(df_corr), color='k', lw=0.9)
for i in range(len(df_corr)):
    ax.hlines(i+0.5, -1, len(df_corr.columns), color='k', lw=0.9)
xticks = [-1,0,1,2]
xdic = {
    0: 'Brent',
    1: 'HH',
    2: 'NEX'
}
yticks = [-1,0,1,2,3,4,5]
ydic = {
    0: 'S&P 500',
    1: 'Industrials',
    2: 'Financials',
    3: 'IT',
    4: 'Utilities',
    5: 'Materials'
}
xlabels = [xdic.get(t, xticks[i]) for i,t in enumerate(xticks)]
ax.set_xticklabels(xlabels, rotation=0, fontsize=15)
ylabels = [ydic.get(t, yticks[i]) for i,t in enumerate(yticks)]
ax.set_yticklabels(ylabels, fontsize=15)
ax.matshow(df_corr, cmap=plt.cm.Blues, alpha=0.5, aspect=2/5)

plt.savefig('graphs/corr_matrix.pdf', bbox_inches = 'tight')
