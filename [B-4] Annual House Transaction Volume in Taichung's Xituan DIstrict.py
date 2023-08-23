# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 02:34:53 2023

@author: Jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')

df_xituan = df[df.district=='西屯區']
df_xituan = df_xituan.groupby('deal_year')['MRT_nearby'].count().reset_index()
df_xituan['amount'] = df_xituan.MRT_nearby
df_xituan = df_xituan.drop('MRT_nearby', axis=1)

x=df_xituan.deal_year
y=df_xituan.amount
x_text_coordinate = [year+0.1 for year in df_xituan.deal_year]
y_text_coordinate = [amount+20 for amount in df_xituan.amount]

plt.figure(figsize=(12, 6))
plt.plot(x, y)
for x_text, y_text, s in zip(x_text_coordinate, y_text_coordinate, df_xituan.amount):
    plt.text(x_text, y_text, s, color='#61677A')
plt.title('西屯區交易變化量(交易時間101-111年)', fontsize=18)
plt.xlabel('時間(年)', fontsize=16)
plt.ylabel('交易數(筆)', fontsize=16)
plt.savefig("C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\House_Price_Prediction_Module_&_Data_Visualization\\Annual House Transaction Volume in Taichung's Xituan District.png", dpi=100)
plt.show()