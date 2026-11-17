import yfinance as yf
import numpy as np
tickers = ['^GSPC', '^STOXX50E', '^N225', '^FTSE', '000001.SS']
df = yf.download(
    tickers = tickers,
    start = '2010-01-01',
    end = '2026-01-01',
    auto_adjust= True
)['Close']
return_sp500 = 100*np.log(df['^GSPC']).dropna().diff().dropna()
return_eustoxx = 100*np.log(df['^STOXX50E']).dropna().diff().dropna()
return_nikkei = 100*np.log(df['^N225']).dropna().diff().dropna()
return_ftse = 100*np.log(df['^FTSE']).dropna().diff().dropna()
return_sse = 100*np.log(df['000001.SS']).dropna().diff().dropna()

return_sp500.to_csv('data/sp500_returns.csv')
return_eustoxx.to_csv('data/eustoxx_returns.csv')
return_nikkei.to_csv('data/nikkei_returns.csv')
return_ftse.to_csv('data/ftse_returns.csv')
return_sse.to_csv('data/sse_returns.csv')