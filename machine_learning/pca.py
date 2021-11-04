import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

from get_data.static import RESTfulProcessor

resp = RESTfulProcessor(verbose=False)
df_train, df_test = resp.process_data()

df_all = pd.concat([df_train, df_test], ignore_index=True)

sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']
df_model = df_all[df_all['ticker'].isin(sample_tickers)]
df_ohe = pd.get_dummies(df_model['ticker'], drop_first=True)
df_model = df_model.join(df_ohe)

df_model = df_model[[
    'date', 'ticker', 'after_hours', 'high', 'low', 'open', 'close', 'pre_market', 'volume', 'last_after_hours',
    'after_hours_chg', 'after_hours_pct_chg', 'last_pre_market', 'pre_market_chg', 'pre_market_pct_chg',
    'last_volume', 'volume_chg', 'volume_pct_chg', 'last_open', 'open_chg', 'open_pct_chg', 'last_close',
    'close_chg', 'close_pct_chg', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA'
]]
df_model.dropna(inplace=True)

cutoff = datetime.datetime.strptime('2021-08-01', '%Y-%m-%d').date()
df_train = df_model[df_model['date'] < cutoff].copy()
df_test = df_model[df_model['date'] >= cutoff].copy()

# use mean/std to normalize values
ss = StandardScaler()
df_train_scale = df_train.copy()
df_train_scale.drop(columns=['ticker', 'date'], inplace=True)
df_train_scale = pd.DataFrame(ss.fit_transform(df_train_scale), columns=df_train_scale.columns)


# PCA and key metrics
pca = PCA()
components = pca.fit_transform(df_train_scale) # component value
variance = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_
loadings = pca.components_.T * np.sqrt(variance)
cum_variance = np.cumsum(variance_ratio)
