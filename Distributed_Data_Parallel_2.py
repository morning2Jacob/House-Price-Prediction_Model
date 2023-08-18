# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:56:09 2023

@author: Jacob
"""
import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Jaocb Kai\\Documents\\Uni_Python\\Clean_RealEstates')
df = df.drop(['Unnamed: 0'], axis=1)
df = df[(df['main_purpose']=='住家') | (df['main_purpose']=='住商')]
df = df.drop(['address', 'road', 'layout', 'unit_price', 'deal_date', 'floor', 'highest_floor', 'parking_unit'], axis=1)

#####################################################################################
##Train, validation, and test split
#####################################################################################

X = df.drop(['price'], axis=1)
y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

#####################################################################################
#Fill nan
#####################################################################################

df_train['age'] = df_train['age'].fillna(df_train['age'].median())
df_test['age'] = df_test['age'].fillna(df_test['age'].median())

#####################################################################################
#Filter outliers
#####################################################################################

num_list = ['age', 'plain', 'room', 'bathroom', 'price']

def outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lowerbound = Q1 - 3*IQR
    upperbound = Q3 + 3*IQR
    
    ls = df[(df[column] >= lowerbound) & (df[column] <= upperbound)]
    return ls
    
def livingroom_outliers(df):
    df = df[df['livingroom'] <= df['livingroom'].quantile(0.99)]
    return df


for i in num_list:
    df_train = outliers(df_train, i)
df_train = livingroom_outliers(df_train)
    
for i in num_list:
    df_test = outliers(df_test, i)
df_test = livingroom_outliers(df_test)

#####################################################################################
#OneHotEncoding
#####################################################################################

df_train = pd.get_dummies(df_train, columns=['build_type', 'main_purpose', 'district', 'deal_year', 'MRT'])
df_test = pd.get_dummies(df_test, columns=['build_type', 'main_purpose', 'district', 'deal_year', 'MRT'])

#####################################################################################
#OrdinalEncoding
#####################################################################################

df_train['total_floor'] = df_train['total_floor'].map(lambda x: 8 if x=='全' else x)
df_train['total_floor'] = df_train['total_floor'].astype(int)

df_test['total_floor'] = df_test['total_floor'].map(lambda x: 8 if x=='全' else x)
df_test['total_floor'] = df_test['total_floor'].astype(int)

#####################################################################################
#MinMaxScaler
#####################################################################################

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_train[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']] = mms.fit_transform(
    df_train[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']])

df_test[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']] = mms.fit_transform(
    df_test[['age', 'longitude', 'latitude', 'plain', 'room', 'livingroom', 'bathroom', 'total_floor']])

#####################################################################################
#y_test
#####################################################################################

y_test = df_test['price']
y_train = df_train['price']