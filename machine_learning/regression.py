import pandas as pd
import numpy as np

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

# statsmodels.api regression
import statsmodels.api as sm

x_cols = ['last_close', 'pre_market', 'AMZN', 'MSFT', 'volume']
Y = df_train_reg['close']
X = sm.add_constant(df_train_reg[x_cols])

model = sm.OLS(endog=Y, exog=X, missing='drop')
result = model.fit()
df_train_reg['predict'] = result.predict(X)
df_train_reg['r'] = result.resid
print(result.summary2())  # vs result.summary()

# remove intercept
import statsmodels.formula.api as smf

result2 = smf.ols('close ~ last_close + pre_market + np.log(volume)', data=df_train_reg).fit()
print(result2.summary())
print(result2.summary2())

# Out of Sample Prediction
df_test_reg = df_test.loc[
    df_test['ticker'].isin(sample_tickers),
    ['last_close', 'pre_market', 'volume', 'close', 'ticker']
]
df_test_reg = df_test_reg.join(pd.get_dummies(df_test_reg['ticker'], drop_first=True))
df_test_reg.drop(columns='ticker', inplace=True)
X_new = sm.add_constant(df_test_reg[x_cols])
df_test_reg['predict'] = result2.predict(X_new)
df_test_reg['r'] = df_test_reg['close'] - df_test_reg['predict']
