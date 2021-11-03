import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

from get_data.static import RESTfulProcessor

resp = RESTfulProcessor(verbose=False)
df_train, df_test = resp.process_data()

df_all = pd.concat([df_train, df_test], ignore_index=True)

sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']
df_model = df_all.loc[df_all['ticker'].isin(sample_tickers)]

cutoff = datetime.datetime.strptime('2021-08-01', '%Y-%m-%d').date()
df_train = df_model[df_model['date'] < cutoff].copy()
df_test = df_model[df_model['date'] >= cutoff].copy()

df_ts = df_train.loc[df_train['ticker'].isin(['AAPL']), ['date', 'close_pct_chg']].copy()
df_ts.sort_values(by='date', inplace=True)

# unit test/stationary
fig, ax = plt.subplots()
sns.relplot(x='date', y='close_pct_chg', data=df_ts, kind='line')
plt.show()

## D-F test
result = adfuller(df_ts['close_pct_chg'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ACF - decide AR(p) lag term
acf = plot_acf(df_ts['close_pct_chg'].dropna())
acf.show()
# PACF
pacf = plot_pacf(df_ts['close_pct_chg'].dropna())
pacf.show()

# ARIMA model implementation
model = ARIMA(endog=df_ts['close_pct_chg'].dropna().values, order=(1, 0, 1))
result = model.fit()
result.summary()

# diagnostics tool
pd = result.plot_diagnostics(figsize=(10, 6))
pd.show()
