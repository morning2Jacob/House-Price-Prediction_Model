# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 05:24:05 2023

@author: Jacob
"""

import pandas as pd

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')
df = df.drop(['Unnamed: 0'], axis=1)

df = df[df['district']=='西屯區']
df = df[(df['main_purpose']=='住家') | (df['main_purpose']=='住商')]
df = df[df['manager'] == 1]
df = df[df['MRT_nearby']==1]

import folium as fl
from folium import plugins

house_map = fl.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                    zoom_start=7, 
                    control_scale=True)
marker_cluster = plugins.MarkerCluster().add_to(house_map)
for name, row in df.iterrows():
    fl.Marker(location=[row['latitude'], row['longitude']], 
              popup=row['address'], 
              tooltip=row['unit_price'],
              icon=fl.Icon(icon='house', prefix='fa')
              ).add_to(marker_cluster)
    
house_map.save('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\資料處理&視覺化\\house_site.html')