"""
This code prints out a color coded matrix with P-values of Granger causality test (H_0=NO G.-causality)
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## matrix for fuels Granger causing industries

df1 = pd.read_csv('datasets/granger_pvalues_1.csv', index_col=0)

fig, ax = plt.subplots()
for i in range(len(df1)):
    for j in range(len(df1.columns)):
        value = round(df1.iloc[i].iat[j], 3)
        ax.text(j, i, str(value), va='center', ha='center', fontsize=15)
for i in range(len(df1.columns)):
    ax.vlines(i+0.5, -1, len(df1), color='k', lw=0.9)
for i in range(len(df1)):
    ax.hlines(i+0.5, -1, len(df1.columns), color='k', lw=0.9)
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
cmap = plt.cm.Blues
cmap_reversed = cmap.reversed()
ax.matshow(df1, cmap=cmap_reversed, alpha=0.5, aspect=2/5)

plt.savefig('graphs/granger_matrix_1.pdf', bbox_inches = 'tight')

## matrix for industries Granger causing fuels 

df2 = pd.read_csv('datasets/granger_pvalues_2.csv', index_col=0)

fig, ax = plt.subplots()
for i in range(len(df2)):
    for j in range(len(df2.columns)):
        value = round(df2.iloc[i].iat[j], 3)
        ax.text(j, i, str(value), va='center', ha='center', fontsize=15)
for i in range(len(df2.columns)):
    ax.vlines(i+0.5, -1, len(df2), color='k', lw=0.9)
for i in range(len(df2)):
    ax.hlines(i+0.5, -1, len(df2.columns), color='k', lw=0.9)
yticks = [-1,0,1,2]
ydic = {
    0: 'Brent',
    1: 'HH',
    2: 'NEX'
}
xticks = [-1,0,1,2,3,4,5]
xdic = {
    0: 'S&P 500',
    1: 'Industrials',
    2: 'Financials',
    3: 'IT',
    4: 'Utilities',
    5: 'Materials'
}
xlabels = [xdic.get(t, xticks[i]) for i,t in enumerate(xticks)]
ax.set_xticklabels(xlabels, rotation=45, fontsize=15)
ylabels = [ydic.get(t, yticks[i]) for i,t in enumerate(yticks)]
ax.set_yticklabels(ylabels, fontsize=15)
cmap = plt.cm.Blues
cmap_reversed = cmap.reversed()
ax.matshow(df2, cmap=cmap_reversed, alpha=0.5, aspect=4/5)

plt.savefig('graphs/granger_matrix_2.pdf', bbox_inches = 'tight')

