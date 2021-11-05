import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import shap

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
# Decision Tree
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor().fit(df_train[X_cols], df_train['close'])
print('untuned score: ', clf.score(df_train[X_cols], df_train['close']))
clf = DecisionTreeRegressor(splitter='random', max_depth=5, min_samples_split=16, min_samples_leaf=10).fit(
    df_train[X_cols], df_train['close'])
print('tuned score: ', clf.score(df_train[X_cols], df_train['close']))
fig = plt.figure(figsize=(50, 30))
_ = tree.plot_tree(clf, feature_names=X_cols, class_names='close', filled=True)
fig.show()

# SHAP
shap.initjs()
explainer = shap.TreeExplainer(clf, df_train[X_cols])
# shap_values = explainer(df_train[X_cols])

shap_values = explainer.shap_values(df_test[X_cols])[0]
p = shap.force_plot(explainer.expected_value, shap_values, df_test[X_cols].iloc[0].values)

shap_values = explainer.shap_values(df_test[X_cols])
shap.summary_plot(shap_values, df_test[X_cols], plot_type='bar')

# visualize the first prediction's explanation
shap_values = explainer(df_test[X_cols])
shap.plots.waterfall(shap_values[0])

# Decision plot
explainer = shap.TreeExplainer(clf, df_train[X_cols])
shap_values = explainer.shap_values(df_test[X_cols])
shap.decision_plot(explainer.expected_value, shap_values[[1, 76]], df_test[X_cols].iloc[[1, 76]])

# feature importance
importance = clf.feature_importances_
# summarize feature importance
fig2, ax2 = plt.subplots(figsize=(10, 6))
for f, s in zip(X_cols, importance):
    print('Feature: {f}, Score: {s}'.format(f=f, s=s))
# plot feature importance
ax2 = sns.barplot(x=importance, y=X_cols)
fig2.show()

# ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor

extra_trees = ExtraTreesRegressor().fit(df_train[X_cols], df_train['close'])
print('untuned score: ', extra_trees.score(df_train[X_cols], df_train['close']))
extra_trees = ExtraTreesRegressor(max_depth=23, min_samples_split=16, min_samples_leaf=5, n_jobs=-1).fit(
    df_train[X_cols], df_train['close'])
print('tuned score: ', extra_trees.score(df_train[X_cols], df_train['close']))

for f, s in zip(X_cols, extra_trees.feature_importances_):
    print('Feature: {f}, Score: {s}'.format(f=f, s=s))

# Random Forest
seed = 4657514
from sklearn.ensemble import RandomForestRegressor

rf_mod = RandomForestRegressor(n_estimators=150, min_samples_leaf=15, oob_score=True, random_state=seed, ccp_alpha=0.08)
rf_mod.fit(df_train[X_cols].values, df_train['close'].values)
print('untuned score: ', rf_mod.score(df_train[X_cols].values, df_train['close'].values))

rf_weighted = RandomForestRegressor(n_estimators=150, min_samples_leaf=15, oob_score=True, random_state=seed,
                                    ccp_alpha=0.08)
rf_weighted.fit(df_train[X_cols].values, df_train['close'].values,
                sample_weight=np.maximum(1 / df_train['close'].values, 1))
print('weighted score: ', rf_weighted.score(df_train[X_cols].values, df_train['close'].values))

for f, s in zip(X_cols, rf_mod.feature_importances_):
    print('Feature: {f}, Score: {s:.4f}'.format(f=f, s=s))
