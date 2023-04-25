"""
Code for testing industries Granger-causing fuels
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

## IMPORTING DATA

df = pd.read_csv('datasets/RawMaterials_and_Industries.csv', index_col=0)

## DATAFRAME TO STORE RESULTS

df_granger_1 = pd.DataFrame(
    columns=['S&P500_x', 'Industrials_x', 'Financials_x', 'IT_x', 'Utilities_x', 'Materials_x'],
    index=['Brent_y', 'HenryHub_y', 'NEX_y']
)

## FUNCTIONS

### Selecting the VAR order p by computing the Bayesian information criterion

def select_p(train):
    bic = [] ## Bayesian information criterion
    model = VAR(train) 
    p = np.arange(1,60)
    for i in p:
        result = model.fit(i) 
        bic.append(result.bic)
    p = bic.index(min(bic))+1 ## excluding lag=0
    return p

### Granger causality test

def granger_pavalue(data):
    """
    data: two-column pandas dataframe containing two timeseries
    This function tests null hypothesis that second column does NOT Granger-cause first column
    """
    p = select_p(data) ## picking best lag
    granger_res = grangercausalitytests(data, p, verbose=False)
    p_values = [round(granger_res[i+1][0]['ssr_ftest'][1], 4) for i in range(p)] ## extracting p-values
    min_p_value = np.min(p_values)
    return min_p_value

## RUNNING THE TESTS

### S&P 500 column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500)']]
df_granger_1['S&P500_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500)']]
df_granger_1['S&P500_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500)']]
df_granger_1['S&P500_x'].loc['NEX_y'] = granger_pavalue(df_test)

### Industrials column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500 Industrials)']]
df_granger_1['Industrials_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500 Industrials)']]
df_granger_1['Industrials_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500 Industrials)']]
df_granger_1['Industrials_x'].loc['NEX_y'] = granger_pavalue(df_test)

### Financials column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500 Financials)']]
df_granger_1['Financials_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500 Financials)']]
df_granger_1['Financials_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500 Financials)']]
df_granger_1['Financials_x'].loc['NEX_y'] = granger_pavalue(df_test)

### IT column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500 IT)']]
df_granger_1['IT_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500 IT)']]
df_granger_1['IT_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500 IT)']]
df_granger_1['IT_x'].loc['NEX_y'] = granger_pavalue(df_test)

### Utilities column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500 Utilities)']]
df_granger_1['Utilities_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500 Utilities)']]
df_granger_1['Utilities_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500 Utilities)']]
df_granger_1['Utilities_x'].loc['NEX_y'] = granger_pavalue(df_test) 

### Materials column

df_test = df[['DIFF(Brent)', 'DIFF(S&P 500 Materials)']]
df_granger_1['Materials_x'].loc['Brent_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(Henry Hub)', 'DIFF(S&P 500 Materials)']]
df_granger_1['Materials_x'].loc['HenryHub_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(NEX)', 'DIFF(S&P 500 Materials)']]
df_granger_1['Materials_x'].loc['NEX_y'] = granger_pavalue(df_test) 

df_granger_1.to_csv('datasets/granger_pvalues_2.csv')