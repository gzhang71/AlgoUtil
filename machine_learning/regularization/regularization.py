import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

from get_data.static import RESTfulProcessor

resp = RESTfulProcessor(verbose=False)
df_train, df_test = resp.process_data()

df_all = pd.concat([df_train, df_test], ignore_index=True)

sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']
df_model = df_all.loc[
    df_all['ticker'].isin(sample_tickers),
    ['last_close', 'date', 'pre_market', 'volume', 'close', 'ticker']
]

df_ohe = pd.get_dummies(df_model['ticker'], drop_first=True)
df_model = df_model.join(df_ohe)
df_model.drop(columns='ticker', inplace=True)
df_model.dropna(inplace=True)

cutoff = datetime.datetime.strptime('2021-08-01', '%Y-%m-%d').date()
df_train = df_model[df_model['date'] < cutoff].copy()
df_test = df_model[df_model['date'] >= cutoff].copy()

x_cols = ['last_close', 'pre_market', 'AMZN', 'MSFT']

# Regularization
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression

linear = LinearRegression().fit(X=df_train[x_cols], y=df_train['close'])
linear.score(X=df_test[x_cols], y=df_test['close'])

lasso = Lasso(alpha=1, max_iter=100000).fit(X=df_train[x_cols], y=df_train['close'])
lasso.score(X=df_test[x_cols], y=df_test['close'])

ridge = Ridge(alpha=1).fit(X=df_train[x_cols], y=df_train['close'])
ridge.score(X=df_test[x_cols], y=df_test['close'])

elastic_net = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=100000).fit(X=df_train[x_cols], y=df_train['close'])
elastic_net.score(X=df_test[x_cols], y=df_test['close'])

# Compare between regularization result vs linear regression
