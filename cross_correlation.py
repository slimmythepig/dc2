"""
This code computes cross correlation coefficient at different lags
It returns a matrix in which each cell contains 4 lists:
1) negative lags (second timeseries shifted leftward respect to the first timeseries)
2) correlation coefficient corresponding to lag in first array
3) postive lags (second timeseries shifted rightward respect to the first timeseries)
4) correlation coefficient corresponding to lag in third array
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

## IMPORTING DATA

df = pd.read_csv('datasets/RawMaterials_and_Industries.csv', index_col=0)

## CROSS CORRELATION ANALYSIS

def CrossCorrelationAnalysis(X, Y, maxlag, dt=1, filename='plot_name.pdf'):
    result_X, result_Y = adfuller(X), adfuller(Y)
    if (result_X[1] > 0.05 or result_Y[1] > 0.05):
        print("Time Series are not stationary")
        return
    else:
        ## we consider X(t+lag), Y(t)
        ## so that for lag<0 X preceeds Y and vice-versa.
        dim = len(X)
        lags = np.arange(-maxlag, maxlag+1, 1)
        lagged_cross_corr = []
        for lag in lags:
            XY = [(X[t+lag], Y[t]) for t in range(dim-np.abs(lag))]
            X_lagged = [el[0] for el in XY]
            Y_lagged = [el[1] for el in XY]
            lagged_cross_corr.append(stats.pearsonr(X_lagged, Y_lagged)[0])
        lags_min = [l for l in lags if l<=0]
        lags_maj = [l for l in lags if l>=0]
        lcc_min = [lagged_cross_corr[i] for i in range(len(lags)) if lags[i]<=0]
        lcc_maj = [lagged_cross_corr[i] for i in range(len(lags)) if lags[i]>=0]
        return [lags_min, lcc_min, lags_maj, lcc_maj]
   
df_cc = pd.DataFrame(
    columns=['Brent', 'Henry Hub', 'NEX', 'S&P 500', 'Industrials', 'Financials', 'IT', 'Utilities', 'Materials'],
    index=['Brent', 'Henry Hub', 'NEX', 'S&P 500', 'Industrials', 'Financials', 'IT', 'Utilities', 'Materials'])

maxlag = 50

df_cc['Brent'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(Brent)'], maxlag)
df_cc['Brent'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Brent'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(NEX)'], maxlag)
df_cc['Brent'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Brent'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Brent'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Brent'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Brent'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Brent'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(Brent)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['Henry Hub'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(Brent)'], maxlag)
df_cc['Henry Hub'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Henry Hub'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(NEX)'], maxlag)
df_cc['Henry Hub'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Henry Hub'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Henry Hub'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Henry Hub'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Henry Hub'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Henry Hub'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(Henry Hub)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['NEX'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(Brent)'], maxlag)
df_cc['NEX'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['NEX'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(NEX)'], maxlag)
df_cc['NEX'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500)'], maxlag)
df_cc['NEX'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['NEX'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['NEX'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['NEX'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['NEX'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(NEX)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['S&P 500'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(Brent)'], maxlag)
df_cc['S&P 500'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['S&P 500'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(NEX)'], maxlag)
df_cc['S&P 500'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500)'], maxlag)
df_cc['S&P 500'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['S&P 500'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['S&P 500'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['S&P 500'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['S&P 500'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['Industrials'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(Brent)'], maxlag)
df_cc['Industrials'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Industrials'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(NEX)'], maxlag)
df_cc['Industrials'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Industrials'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Industrials'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Industrials'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Industrials'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Industrials'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Industrials)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['Financials'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(Brent)'], maxlag)
df_cc['Financials'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Financials'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(NEX)'], maxlag)
df_cc['Financials'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Financials'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Financials'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Financials'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Financials'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Financials'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Financials)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['IT'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(Brent)'], maxlag)
df_cc['IT'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['IT'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(NEX)'], maxlag)
df_cc['IT'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500)'], maxlag)
df_cc['IT'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['IT'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['IT'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['IT'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['IT'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 IT)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['Utilities'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(Brent)'], maxlag)
df_cc['Utilities'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Utilities'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(NEX)'], maxlag)
df_cc['Utilities'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Utilities'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Utilities'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Utilities'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Utilities'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Utilities'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Utilities)'], df['DIFF(S&P 500 Materials)'], maxlag)

df_cc['Materials'].loc['Brent'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(Brent)'], maxlag)
df_cc['Materials'].loc['Henry Hub'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(Henry Hub)'], maxlag)
df_cc['Materials'].loc['NEX'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(NEX)'], maxlag)
df_cc['Materials'].loc['S&P 500'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500)'], maxlag)
df_cc['Materials'].loc['Industrials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500 Industrials)'], maxlag)
df_cc['Materials'].loc['Financials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500 Financials)'], maxlag)
df_cc['Materials'].loc['IT'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500 IT)'], maxlag)
df_cc['Materials'].loc['Utilities'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500 Utilities)'], maxlag)
df_cc['Materials'].loc['Materials'] = CrossCorrelationAnalysis(df['DIFF(S&P 500 Materials)'], df['DIFF(S&P 500 Materials)'], maxlag)

## EXPORTING DATA

df_cc.to_csv('datasets/cross_correlation_results_1.csv')