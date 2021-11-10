import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor


random_seed = 147
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
X_cols = ['after_hours', 'high', 'low', 'open', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']

xgb_model = GradientBoostingRegressor(max_depth=3, n_estimators=30, warm_start=True, random_state=random_seed)
xgb_model = xgb_model.fit(X=df_train[X_cols], y=df_train['close'])
df_test['pred'] = xgb_model.predict(X=df_test[X_cols])

lgb_model = LGBMRegressor(max_depth=3, n_estimators=30, random_state=random_seed)
lgb_model = lgb_model.fit(X=df_train[X_cols], y=df_train['close'])
df_test['pred'] = lgb_model.predict(X=df_test[X_cols])
