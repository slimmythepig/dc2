
import pandas as pd
import yfinance as yf
import datetime as dt

start = dt.datetime(2008, 1, 1)
end = dt.datetime(2020, 1, 1)

## COLLECTING DATA FROM YAHOO FINANCE

BRENT = yf.download('BZ=F', start, end) ## Brent Crude Oil Price Index (Oil)
BRENT = BRENT.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
BRENT = BRENT.rename(columns={'Adj Close':'Brent'})

HENRYHUB = yf.download('NG=F', start, end) ## Henry Hub Natural Gas Futures (Natural Gas)
HENRYHUB = HENRYHUB.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
HENRYHUB = HENRYHUB.rename(columns={'Adj Close':'Henry Hub'})

NEX = yf.download('^NEX', start, end) ## Newcastle Export Index (Coal)
NEX = NEX.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
NEX = NEX.rename(columns={'Adj Close':'NEX'})

SP500 = yf.download('^GSPC', start, end)
SP500 = SP500.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500 = SP500.rename(columns={'Adj Close':'S&P 500'})

SP500_industrials = yf.download('^SP500-20', start, end)
SP500_industrials = SP500_industrials.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500_industrials = SP500_industrials.rename(columns={'Adj Close':'S&P 500 Industrials'})

SP500_financials = yf.download('^SP500-40', start, end)
SP500_financials = SP500_financials.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500_financials = SP500_financials.rename(columns={'Adj Close':'S&P 500 Financials'})

SP500_it = yf.download('^SP500-45', start, end)
SP500_it = SP500_it.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500_it = SP500_it.rename(columns={'Adj Close':'S&P 500 IT'})

SP500_utilities = yf.download('^SP500-55', start, end)
SP500_utilities = SP500_utilities.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500_utilities = SP500_utilities.rename(columns={'Adj Close':'S&P 500 Utilities'})

SP500_materials = yf.download('^SP500-15', start, end)
SP500_materials = SP500_materials.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'})
SP500_materials = SP500_materials.rename(columns={'Adj Close':'S&P 500 Materials'})

OBJS = [
    BRENT['Brent'],
    HENRYHUB['Henry Hub'],
    NEX['NEX'],
    SP500['S&P 500'],
    SP500_industrials['S&P 500 Industrials'],
    SP500_financials['S&P 500 Financials'],
    SP500_it['S&P 500 IT'],
    SP500_utilities['S&P 500 Utilities'],
    SP500_materials['S&P 500 Materials']
]
df = pd.concat(OBJS,axis=1)
df = df.dropna()

 ## DIFFERENTIATING IN ORDER TO GET STATIONARY TIMESERIES

def diff(X_t, dt):
    N = len(X_t)
    diff_X_t = []
    for i in range(N):
        if i==0:
            diff = (X_t[i+1]-X_t[i])/dt
            diff_X_t.append(diff)
        
        elif i==N-1:
            diff = (X_t[i]-X_t[i-1])/dt
            diff_X_t.append(diff)
            
        else:
            diff_fw = (X_t[i+1]-X_t[i])/dt
            diff_bw = (X_t[i]-X_t[i-1])/dt
            diff = 0.5*(diff_fw+diff_bw)
            diff_X_t.append(diff)
    return diff_X_t

df['DIFF(Brent)'] = diff(df['Brent'], 1)
df['DIFF(Henry Hub)'] = diff(df['Henry Hub'], 1)
df['DIFF(NEX)'] = diff(df['NEX'], 1)
df['DIFF(S&P 500)'] = diff(df['S&P 500'], 1)
df['DIFF(S&P 500 Industrials)'] = diff(df['S&P 500 Industrials'], 1)
df['DIFF(S&P 500 Financials)'] = diff(df['S&P 500 Financials'], 1)
df['DIFF(S&P 500 IT)'] = diff(df['S&P 500 IT'], 1)
df['DIFF(S&P 500 Utilities)'] = diff(df['S&P 500 Utilities'], 1)
df['DIFF(S&P 500 Materials)'] = diff(df['S&P 500 Materials'], 1)

## EXPORTING DATA

df.to_csv('datasets/RawMaterials_and_Industries.csv', index=True)