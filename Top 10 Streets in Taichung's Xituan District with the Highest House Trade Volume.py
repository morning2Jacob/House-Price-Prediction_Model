# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 04:12:44 2023

@author: Jacob
"""

import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')
df_xituan = df[df.district=='西屯區']
df_xituan = df_xituan.groupby('road')['MRT_nearby'].count().sort_values(ascending=False).reset_index()
xituan_top10 = df_xituan.head(10)
xituan_top10['amount'] = xituan_top10['MRT_nearby']
xituan_top10 = xituan_top10.drop('MRT_nearby', axis=1)
amount = xituan_top10['amount'].tolist()

labels = [f'{road}\n{count}件' for road, count in zip(xituan_top10['road'], xituan_top10['amount'])]
plt.figure(figsize=(10, 6))
squarify.plot(sizes=amount,
              label=labels,
              color=seaborn.color_palette('plasma_r', len(amount)),
              alpha=1,
              ec = 'gray',
              text_kwargs={'fontsize': 10,
                           'color': 'white'})
plt.title('台中西屯區前十筆最大交易量的路段', fontsize=18, color='black')
plt.axis('off')

# color bar
# create dummy invisible image with a color map
img = plt.imshow([xituan_top10.amount], cmap='plasma_r')
img.set_visible(False)
plt.colorbar(img, orientation="vertical", shrink=.96)

plt.savefig("C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\House_Price_Prediction_Module_&_Data_Visualization\\Top 10 Streets in Taichung's Xituan District with the Highest House Trade Volume.png", dpi=100)
plt.show()
