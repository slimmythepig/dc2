"""
Code for testing fuels Granger-causing industries
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
    columns=['Brent_x', 'HenryHub_x', 'NEX_x'],
    index=['S&P500_y', 'Industrials_y', 'Financials_y', 'IT_y', 'Utilities_y', 'Materials_y']
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

### Brent column

df_test = df[['DIFF(S&P 500)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['S&P500_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Industrials)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['Industrials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Financials)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['Financials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 IT)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['IT_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Utilities)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['Utilities_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Materials)', 'DIFF(Brent)']]
df_granger_1['Brent_x'].loc['Materials_y'] = granger_pavalue(df_test)

### Henry Hub column

df_test = df[['DIFF(S&P 500)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['S&P500_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Industrials)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['Industrials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Financials)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['Financials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 IT)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['IT_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Utilities)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['Utilities_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Materials)', 'DIFF(Henry Hub)']]
df_granger_1['HenryHub_x'].loc['Materials_y'] = granger_pavalue(df_test)

### NEX column

df_test = df[['DIFF(S&P 500)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['S&P500_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Industrials)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['Industrials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Financials)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['Financials_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 IT)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['IT_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Utilities)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['Utilities_y'] = granger_pavalue(df_test)
df_test = df[['DIFF(S&P 500 Materials)', 'DIFF(NEX)']]
df_granger_1['NEX_x'].loc['Materials_y'] = granger_pavalue(df_test)

## EXPORTING DATA

df_granger_1.to_csv('datasets/granger_pvalues_1.csv')