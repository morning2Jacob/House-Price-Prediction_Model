# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:56:48 2023

@author: Jacob
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectFpr, SelectKBest, f_regression

#####################################################################################
#Wrapper Method (Forward) === Linear Regression -- 1
#####################################################################################

X_train = df_train.drop(['price'], axis=1)
y_train = df_train['price']
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS

sfs = SFS(
    estimator=LinearRegression(),
    n_features_to_select = 8,
    direction = 'forward',
    scoring='r2',
    cv=10,
    n_jobs=4
    )

sfs.fit(X_train, y_train)
SFS_feat_1 = sfs.get_feature_names_out()
print(SFS_feat_1)

#R2 0.7791031941528552
col = list(SFS_feat_1)
X_test = df_test[col]
model = LinearRegression()
model.fit(X_train[col], y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Wrapper Method (Forward) === Linear Regression -- 2
#####################################################################################

X_train = df_train[['age', 'plain', 'room', 'elevator', 'build_type_透天厝',
                    'district_北屯區', 'deal_year_101', 'deal_year_102']]

from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn import metrics

sfs = SFS(
    estimator=LinearRegression(),
    n_features_to_select = 7,
    direction = 'forward',
    scoring='r2',
    cv=10,
    n_jobs=4
    )

sfs.fit(X_train, y_train)
SFS_feat_2 = sfs.get_feature_names_out()
print(SFS_feat_2)

#R2 -- 0.7731156277637288
col = list(SFS_feat_2)
X_test = df_test[col]
model = LinearRegression()
model.fit(X_train[col], y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Wrapper Method (Forward) === Random Forest Regressor -- 1
#####################################################################################

X_train = df_train.drop(['price'], axis=1)
y_train = df_train['price']

from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor

sfs = SFS(
    estimator=RandomForestRegressor(random_state=42),
    n_features_to_select = 8,
    direction = 'forward',
    scoring='r2',
    cv=10,
    n_jobs=7
    )

sfs.fit(X_train, y_train)
SFS_feat_3 = sfs.get_feature_names_out()
print(SFS_feat_3)

#R2 -- 0.9159347287205043
col = list(SFS_feat_3)
model = RandomForestRegressor(random_state=42)
model.fit(X_train[col], y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Wrapper Method (Forward) === Random Forest Regressor -- 2
#####################################################################################

X_train = df_train[['age', 'longitude', 'latitude', 'plain', 'total_floor',
                    'deal_year_109', 'deal_year_110', 'deal_year_111']]
y_train = df_train['price']

from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor

sfs = SFS(
    estimator=RandomForestRegressor(random_state=42),
    n_features_to_select = 7,
    direction = 'forward',
    scoring='r2',
    cv=10,
    n_jobs=7
    )

sfs.fit(X_train, y_train)
SFS_feat_4 = sfs.get_feature_names_out()
print(SFS_feat_4)

#R2 -- 0.9152307017271177
col = list(SFS_feat_4)
X_test = df_test[col]
model = RandomForestRegressor(random_state=42)
model.fit(X_train[col], y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Pearson's Correlation
#####################################################################################

from scipy.stats import pearsonr
Pearson_corr = []
feature1 = []
feature2 = []
for i in df_train.columns:
    for k in df_train.columns:
        if i == k:
            continue
        else:
            corr = pearsonr(df_train[i], df_train[k])
            feature1.append(i)
            feature2.append(k)
            Pearson_corr.append(corr[0])

Pearson_correlation = pd.DataFrame({'feature1': feature1,
                                   'feature2': feature2,
                                   'correlation': Pearson_corr})

#-----------------------------------------------------------------------------------------------------------------------------------------
#Check Feature vs Target Pearson's Correlation
#-----------------------------------------------------------------------------------------------------------------------------------------

Pearson_correlation[(Pearson_correlation['correlation']>0.7) | (Pearson_correlation['correlation']<-0.7)]

Pearson_correlation[(Pearson_correlation['feature1']=='room') & (Pearson_correlation['feature2']=='price')]

Pearson_correlation[(Pearson_correlation['feature1']=='bathroom') & (Pearson_correlation['feature2']=='price')]

#-----------------------------------------------------------------------------------------------------------------------------------------
#Check Feature vs Feature Pearson's Correlation (0.2 / 0.3 / 0.5)
#-----------------------------------------------------------------------------------------------------------------------------------------

dt = Pearson_correlation[Pearson_correlation['feature1']=='price'].sort_values(by='correlation', ignore_index=True)

column_02 = []
for i in dt[(dt['correlation']<-0.2) | (dt['correlation']>0.2)]['feature2']:
    column_02.append(i)
    
column_03 = []
for i in dt[(dt['correlation']<-0.3) | (dt['correlation']>0.3)]['feature2']:
        column_03.append(i)
        
column_05 = []
for i in dt[(dt['correlation']<-0.5) | (dt['correlation']>0.5)]['feature2']:
        column_05.append(i)
        
X_train_02 = df_train[column_02].drop(['room', 'build_type_透天厝'], axis=1)
X_train_03 = df_train[column_03].drop(['room'], axis=1)
X_train_05 = df_train[column_05].drop(['room'], axis=1)

#####################################################################################
#Invert MinMaxScaler
#####################################################################################

df_train[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']] = mms.inverse_transform(
    df_train[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']])

df_test[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']] = mms.inverse_transform(
    df_test[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']])

#####################################################################################
#Embedd Method (Linear Regression) -- X_train_02
#####################################################################################

from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import SelectFromModel

regressor = LinearRegression()
model = SelectFromModel(regressor)
model.fit(X_train_02, y_train)
selected_feat = X_train_02.columns[(model.get_support())]
print(selected_feat)

#R2 -- 0.7119472371761304
col = list(selected_feat)
X_test = df_test[col]
model = LinearRegression()
model.fit(X_train_02[col], y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Embedd Method (RandomForestRegressor) -- X_train_02
#####################################################################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

regressor = RandomForestRegressor(random_state=42)
model = SelectFromModel(regressor)
model.fit(X_train_02, y_train)
importances02 = list(model.estimator_.feature_importances_)
print(importances02)

#Plot Feature Importance -- [0.06917634674982095, 0.0003230663083090765, 0.0011241894918578926, 0.0046134600051946865, 0.030031528087602254, 0.010796295465787756, 0.014949990190758605, 0.8689851237006687]
feat_list = list(X_train_02.columns)
list1, list2 = (list(t) for t in zip(*sorted(zip(importances02, feat_list))))
x_value = list(range(len(importances02)))

plt.figure(figsize=(12, 6))
plt.barh(x_value, list1, linewidth=1.2)
plt.title("XGBoost Regressor Feature Importance (Pearson's Correlation between ±0.2 and ±1)")
plt.yticks(x_value, list2, rotation='horizontal')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

#Remove two lowest Feature Importance Columns ['build_type_套房', 'build_type_公寓']
del list2[0:2]

#Tuning Hyperparameters -- {'max_depth': 80, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 1000}
col = list2
from sklearn.model_selection import GridSearchCV
regressor = RandomForestRegressor(random_state=42)
param_grid = {'max_depth': [80, 110],
             'max_features' : [2, 3],
             'min_samples_leaf': [3, 4, 5],
             'min_samples_split': [8, 10, 12],
             'n_estimators': [100, 1000]}

search02 = GridSearchCV(regressor, param_grid, cv=5, n_jobs=-1, verbose=2).fit(
    X_train_02[col], y_train)
print('The best hyperparameters are:', search02.best_params_)

#R2 -- 0.8229172356476869
col = list(X_train_02.columns)
X_test = df_test[col]
model = RandomForestRegressor(max_depth=80,
                              max_features=3, 
                              min_samples_leaf=3,
                              min_samples_split=8, 
                              n_estimators=1000, 
                              random_state=42)
model.fit(X_train_02, y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Embedd Method (XGBoost Regressor) -- X_train_02
#####################################################################################

import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

regressor = xgb.XGBRegressor(random_state=42, eval_metric='rmsle')
model = SelectFromModel(regressor)
model.fit(X_train_02, y_train)
importances02 = list(model.estimator_.feature_importances_)
print(importances02)

#Plot Feature Importance -- [0.056645032, 0.018225675, 0.044099007, 0.0322636, 0.17305557, 0.017357891, 0.028899953, 0.62945324]
feat_list = list(X_train_02.columns)
list1, list2 = (list(t) for t in zip(*sorted(zip(importances02, feat_list))))
x_value = list(range(len(importances02)))

plt.figure(figsize=(12, 6))
plt.barh(x_value, list1, linewidth=1.2)
plt.title("XGBoost Regressor Feature Importance (Pearson's Correlation between ±0.2 and ±1)")
plt.yticks(x_value, list2, rotation='horizontal')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

#Tuning Hyperparameters -- {'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 700}
from sklearn.model_selection import GridSearchCV
regressor = xgb.XGBRegressor(random_state=42, eval_metric='rmsle')
param_grid = {'max_depth': [4, 5, 6],
             'n_estimators': [500, 600, 700],
             'learning_rate': [0.01, 0.015, 0.03]}

search02 = GridSearchCV(regressor, param_grid, cv=5, n_jobs=7, verbose=2).fit(
    X_train_02, y_train)
print('The best hyperparameters are:', search02.best_params_)

#R2 -- 0.7990446157874477
col = list(X_train_02.columns)
X_test = df_test[col]
model = xgb.XGBRegressor(learning_rate=0.03, max_depth=6, random_state=42,
                             n_estimators=700, eval_metric='rmsle')
model.fit(X_train_02, y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Embedd Method (XGBoost Regressor) -- X_train_03
#####################################################################################

import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

regressor = xgb.XGBRegressor(random_state=42, eval_metric='rmsle')
model = SelectFromModel(regressor)
model.fit(X_train_03, y_train)
importances03 = list(model.estimator_.feature_importances_)
print(importances03)

#Plot Feature Importance -- [0.08230023, 0.03766588, 0.054536324, 0.82549757]
feat_list = list(X_train_03.columns)
list1, list2 = (list(t) for t in zip(*sorted(zip(importances03, feat_list))))
x_value = list(range(len(importances03)))

plt.figure(figsize=(12, 6))
plt.barh(x_value, list1, linewidth=1.2)
plt.title("XGBoost Feature Importance (Pearson's Correlation between ±0.3 and ±1)")
plt.yticks(x_value, list2, rotation='horizontal')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

#Tuning Hyperparameters -- {'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 700}
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [4, 5, 6],
             'n_estimators': [500, 600, 700],
             'learning_rate': [0.01, 0.015, 0.03]}

search03 = GridSearchCV(regressor, param_grid, cv=5, n_jobs=7, verbose=2,
                        error_score='raise').fit(X_train_03, y_train)
print('The best hyperparameters are:', search03.best_params_)

#R2 -- .7869128507407567
col = list(X_train_03.columns)
X_test = df_test[col]
model = xgb.XGBRegressor(learning_rate=0.03, max_depth=6, random_state=42,
                             n_estimators=700, eval_metric='rmsle')
model.fit(X_train_03, y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))

#####################################################################################
#Pearson's Correlation ±0.5 (XGBoost Regressor) -- X_train_05
#####################################################################################

import xgboost as xgb
regressor = xgb.XGBRegressor(random_state=42, eval_metric='rmsle')

#Tuning Hyperparameters -- {'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 700}
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [4, 5, 6],
             'n_estimators': [500, 600, 700],
             'learning_rate': [0.01, 0.015, 0.03]}


search05 = GridSearchCV(regressor, param_grid, cv=5, n_jobs=7, verbose=2,
                        error_score='raise').fit(X_train_05, y_train)
print('The best hyperparameters are:', search05.best_params_)

#R2 -- 0.7469485735427872
col = list(X_train_05.columns)
X_test = df_test[col]
model = xgb.XGBRegressor(learning_rate=0.03, max_depth=6, random_state=42,
                             n_estimators=700, eval_metric='rmsle')
model.fit(X_train_05, y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("r2 score is: {}".format(r2))