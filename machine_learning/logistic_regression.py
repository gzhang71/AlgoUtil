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

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X=df_train[x_cols], y=df_train['GOOG'])
df_test['predict'] = logistic_model.predict_proba(X=df_test[x_cols])[:, 1]
df_train['predict'] = logistic_model.predict_proba(X=df_train[x_cols])[:, 1]

ls_auc = roc_auc_score(y_true=df_train['GOOG'], y_score=df_train['predict'])
n1_auc = roc_auc_score(y_true=df_train['GOOG'], y_score=[0] * df_train.shape[0])
n2_auc = roc_auc_score(y_true=df_train['GOOG'], y_score=[1] * df_train.shape[0])
n3_auc = roc_auc_score(y_true=df_train['GOOG'], y_score=np.random.rand(df_train.shape[0], 1))

# Calculate
# roc_curve(y_test[:, i], y_score[:, i])
ls_fpr, ls_tpr, ls_thr = roc_curve(y_true=df_train['GOOG'], y_score=df_train['predict'])
n1_fpr, n1_tpr, n1_thr = roc_curve(y_true=df_train['GOOG'], y_score=[0] * df_train.shape[0])
n2_fpr, n2_tpr, n2_thr = roc_curve(y_true=df_train['GOOG'], y_score=[1] * df_train.shape[0])
n3_fpr, n3_tpr, n3_thr = roc_curve(y_true=df_train['GOOG'], y_score=np.random.rand(df_train.shape[0], 1))

# plot the roc curve for the model
fig, ax = plt.subplots()
ax.plot(ls_fpr, ls_tpr, label='P1')
ax.plot(n1_fpr, n1_tpr, label='N1')
ax.plot(n2_fpr, n2_tpr, label='N2')
ax.plot(n3_fpr, n3_tpr, label='N3')
ax.set_xlabel('False Positive Ratio')
ax.set_xlabel('True Positive Ratio')
ax.legend()
fig.show()

# precision-recall curve
from sklearn.metrics import precision_recall_curve

ls_prec, ls_recall, _ = precision_recall_curve(y_true=df_train['GOOG'], probas_pred=df_train['predict'])
n3_prec, n3_recall, _ = precision_recall_curve(y_true=df_train['GOOG'], probas_pred=np.random.rand(df_train.shape[0], 1))

fig, ax = plt.subplots()
ax.plot(ls_prec, ls_recall, label='P1')
ax.plot(n3_prec, n3_recall, label='N3')
ax.set_xlabel('Precision')
ax.set_xlabel('Recall')
ax.legend()
fig.show()

