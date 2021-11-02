import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

from get_data.static import RESTfulProcessor

resp = RESTfulProcessor(verbose=False)
df_train, df_test = resp.process_data()

sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']
df_train_reg = df_train.loc[
    df_train['ticker'].isin(sample_tickers),
    ['last_close', 'pre_market', 'volume', 'close', 'ticker']
]

df_ohe = pd.get_dummies(df_train_reg['ticker'], drop_first=True)
df_train_reg = df_train_reg.join(df_ohe)
df_train_reg.drop(columns='ticker', inplace=True)

x_cols = ['last_close', 'pre_market', 'AMZN', 'MSFT']
Y = df_train_reg['close']
X = sm.add_constant(df_train_reg[x_cols])

model = sm.OLS(endog=Y, exog=X, missing='drop')
result1 = model.fit()
df_train_reg['predict'] = result1.predict(X)
df_train_reg['r'] = result1.resid
result1.summary2()

# remove intercept
import statsmodels.formula.api as smf
results = smf.ols('close ~ last_close + pre_market + np.log(volume)', data=df_train_reg).fit()
results.summary()

df_test_reg = df_test.loc[
    df_test['ticker'].isin(sample_tickers),
    ['last_close', 'pre_market', 'volume', 'close', 'ticker']
]

df_test_reg = df_test_reg.join(pd.get_dummies(df_test_reg['ticker'], drop_first=True))
df_test_reg.drop(columns='ticker', inplace=True)
X_new = sm.add_constant(df_test_reg[x_cols])
result1.predict(X_new)

# sklearn approach
# logistic regression
from sklearn.linear_model import LogisticRegression, LinearRegression
df_train_reg2 = df_train_reg.dropna().copy()

linear_model = LinearRegression().fit(X=df_train_reg2[x_cols], y=df_train_reg2['close'])
logistic_model = LogisticRegression(random_state=0, penalty='l2').fit(X=df_train_reg2[x_cols], y=df_train_reg2['GOOG'])

# Regularization
from sklearn.linear_model import Lasso, Ridge

lasso = Lasso(alpha=1).fit(X=df_train_reg2[x_cols], y=df_train_reg2['close'])
lasso.score(X=df_test_reg[x_cols], y=df_test_reg['close'])

ridge = Ridge(alpha=1).fit(X=df_train_reg2[x_cols], y=df_train_reg2['close'])
ridge.score(X=df_test_reg[x_cols], y=df_test_reg['close'])