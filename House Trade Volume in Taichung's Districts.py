# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:43:38 2023

@author: Jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')
df = df.drop(['Unnamed: 0'], axis=1)
df = df[(df['main_purpose']=='住家') | (df['main_purpose']=='住商')]
df_dist =  df.groupby('district')['price'].count().sort_values(ascending=False).reset_index()
df_dist['amount'] = df_dist['price']
df_dist = df_dist.drop('price', axis=1)

amount = df_dist['amount'].tolist()
labels = [f'{dist}\n{count}件' for dist, count in zip(df_dist['district'], df_dist['amount'])]
plt.figure(figsize=(12, 6))
squarify.plot(sizes=amount,
              label=labels,
              color=seaborn.color_palette('YlOrBr_r', len(amount)),
              alpha=0.8,
              pad=1,
              text_kwargs={'fontsize': 14,
                           'color': 'black'})
plt.title('台中各地區物件數量', fontsize=18, color='black')
plt.axis('off')
plt.tight_layout()
plt.savefig("C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\House_Price_Prediction_Module_&_Data_Visualization\\House Trade Volume in Taichung's Districts.png", dpi=100)
plt.show()