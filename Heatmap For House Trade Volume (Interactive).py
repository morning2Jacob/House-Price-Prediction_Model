# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:23:38 2023

@author: Jacob
"""

import pandas as pd
import pandas as pd
import plotly.express as px

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')

df_map = df.groupby(['longitude', 'latitude'])['price'].count().reset_index()

# Data with latitude/longitude and values
fig = px.density_mapbox(df, lat = 'latitude', lon = 'longitude', z = 'price',
                        radius = 8,
                        center = dict(lat = df_map['latitude'].mean(), lon = df_map['longitude'].mean()),
                        zoom = 10,
                        mapbox_style = 'open-street-map')

fig.show()