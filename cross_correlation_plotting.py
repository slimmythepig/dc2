
import pandas as pd
import matplotlib.pyplot as plt

## IMPORTING RESULTS

df = pd.read_csv('datasets/cross_correlation_results_1.csv', index_col=0)
df['Brent'] = df['Brent'].apply(eval)
df['Henry Hub'] = df['Henry Hub'].apply(eval)
df['NEX'] = df['NEX'].apply(eval) 
df['S&P 500'] = df['S&P 500'].apply(eval)
df['Industrials'] = df['Industrials'].apply(eval)
df['Financials'] = df['Financials'].apply(eval)
df['IT'] = df['IT'].apply(eval)
df['Utilities'] = df['Utilities'].apply(eval)
df['Materials'] = df['Materials'].apply(eval)

## PLOTTING 

fig, axs = plt.subplots(6, 3, sharex=True, sharey=True, figsize=(5, 9))
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.supxlabel('LAGS (DAYS)', fontsize=16, y=0.05)
fig.supylabel('LAGGED CROSS CORRELATION', fontsize=16, x=-0.02)

xticks = [-40, 0, 40]

### BRENT column

cc_test = df['Brent'].loc['Industrials']
axs[0,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[0,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[0,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[0,0].set_xticks(xticks)
axs[0,0].tick_params(axis='both', labelsize=12)
axs[0,0].set_xlabel('Brent', fontsize=16, labelpad=-98)

cc_test = df['Brent'].loc['Financials']
axs[1,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[1,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[1,0].tick_params(axis='y', labelsize=12)
axs[1,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Brent'].loc['IT']
axs[2,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[2,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[2,0].tick_params(axis='y', labelsize=12)
axs[2,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Brent'].loc['Utilities']
axs[3,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[3,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[3,0].tick_params(axis='y', labelsize=12)
axs[3,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Brent'].loc['Materials']
axs[4,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[4,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[4,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[4,0].tick_params(axis='y', labelsize=12)

cc_test = df['Brent'].loc['S&P 500']
axs[5,0].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[5,0].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[5,0].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[5,0].tick_params(axis='both', labelsize=12)

### HENRY HUB column

cc_test = df['Henry Hub'].loc['Industrials']
axs[0,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[0,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[0,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[0,1].set_xlabel('HH', fontsize=16, labelpad=-98)

cc_test = df['Henry Hub'].loc['Financials']
axs[1,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[1,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[1,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Henry Hub'].loc['IT']
axs[2,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[2,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[2,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Henry Hub'].loc['Utilities']
axs[3,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[3,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[3,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Henry Hub'].loc['Materials']
axs[4,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[4,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[4,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')

cc_test = df['Henry Hub'].loc['S&P 500']
axs[5,1].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[5,1].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[5,1].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[5,1].tick_params(axis='x', labelsize=12)

### NEX column

cc_test = df['NEX'].loc['Industrials']
axs[0,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[0,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[0,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[0,2].set_ylabel('Industrials', fontsize=16, labelpad=-94, rotation=270)
axs[0,2].set_xlabel('NEX', fontsize=16, labelpad=-98)

cc_test = df['NEX'].loc['Financials']
axs[1,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[1,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[1,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[1,2].set_ylabel('Financials', fontsize=16, labelpad=-94, rotation=270)

cc_test = df['NEX'].loc['IT']
axs[2,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[2,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[2,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[2,2].set_ylabel('IT', fontsize=16, labelpad=-94, rotation=270)

cc_test = df['NEX'].loc['Utilities']
axs[3,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[3,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[3,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[3,2].set_ylabel('Utilities', fontsize=16, labelpad=-94, rotation=270)

cc_test = df['NEX'].loc['Materials']
axs[4,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[4,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[4,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[4,2].set_ylabel('Materials', fontsize=16, labelpad=-94, rotation=270)

cc_test = df['NEX'].loc['S&P 500']
axs[5,2].plot(cc_test[0], cc_test[1], lw=2, color='b')
axs[5,2].plot(cc_test[2], cc_test[3], lw=2, color='r')
axs[5,2].vlines(0, -0.6, 0.6, lw=1, ls='--', color='k')
axs[5,2].set_ylabel('S&P 500', fontsize=16, labelpad=-94, rotation=270)
axs[5,2].tick_params(axis='x', labelsize=12)

plt.savefig('graphs/cross_correlations.pdf', bbox_inches = 'tight')
