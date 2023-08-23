# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:40:49 2023

@author: Jacob
"""

import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\project_2023\\Taichung_RealEstates')

df = df.drop_duplicates()
df = df.drop(['community', 'build_share1', 'build_share2', 'note'], axis=1)
df = df.dropna(subset=['build_type', 'main_purpose', 'layout'])

df = df[(df['build_type']!='其他') &
        (df['build_type']!='工廠') & 
        (df['build_type']!='廠辦') &
        (df['build_type']!='農舍') &
        (df['build_type']!='倉庫')
        ]

dct_type = {'住宅大樓(11層含以上有電梯)':'大樓',
       '透天厝':'透天厝',
       '華廈(10層含以下有電梯)':'華廈',
       '套房(1房(1廳)1衛)':'套房',
       '公寓(5樓含以下無電梯)':'公寓',
       '店面（店舖)':'店面',    
       '辦公商業大樓':'商辦',      
       }
df['build_type']=df['build_type'].map(dct_type)

dct = {'住家用':'住家',
       '住商用':'住商',
       '商業用':'商業',
       '辦公用':'商業',
       '商辦用':'商業',
       '住商辦用':'住商',
       '工業用':'工業',
       '住工用':'工業', 
       '工商用':'工業', 
       '農業用':'其他',    
       '其他':'其他',    
       '見其他登記事項':'其他'      
       }
df['main_purpose']=df['main_purpose'].map(dct) 

df['price'] = df['price'].str.replace(',', '').astype(float)
df['plain'] = df['plain'].str.replace(',', '').astype(float)

#------------------------------------------------------------------------------------
#Extract the road name & drop the rows which got the problem on extracting
#Due to the numbers of rows were small comparing with the scale of the whole data set, I decided to drop them out
#------------------------------------------------------------------------------------

import re
road_list = []
index_of_road_problem = []
df = df.reset_index()
df = df.drop(['index'], axis=1)
for index, row in df.iterrows():
    #這裡的0到9不是一般的數字，而是中文的數字(FF10, FF11, FF12...)，就跟括弧一樣。不能用[0-9]，所以要一個一個列出
    t = re.findall('#([\u4e00-\u9fff]+)[０, １, ２, ３, ４, ５, ６, ７, ８, ９]', row[1])
    road_list.extend(t)
    #因為type(t)是list
    if t == []:
        index_of_road_problem.append(index)
    else:
        continue
    
df = df.drop(index_of_road_problem, axis=0)
road_series = pd.Series(road_list)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = pd.concat([df, road_series], axis=1)
df = df.rename(columns={0:'road'})

#Extracting layout
def extract_layout(column, index):
    numbers_of_list = []
    for values in column:
        t = re.findall('[0-9]+', values)
        numbers_of_list.append(int(t[index]))
    numbers_of_series = pd.Series(numbers_of_list)
    return numbers_of_series

numbers_of_room = extract_layout(df['layout'], 0)
number_of_livingroom = extract_layout(df['layout'], 1)
numbers_of_bathroom = extract_layout(df['layout'], 2)
df_layout = pd.DataFrame({'room' : numbers_of_room,
                          'livingroom' : number_of_livingroom,
                          'bathroom' : numbers_of_bathroom})
df = pd.concat([df, df_layout], axis=1)

#------------------------------------------------------------------------------------
#Extracting floors
#------------------------------------------------------------------------------------

floor_list = []
floor_nan_index_list = []
for index, values in zip(df.index, df['floor']):
    floor1 = re.findall('^(.*)/', values)
    for floor in floor1:
        #因為有些樓層的標示原本是像'十二層,/三十層'
        #可以在re.findall之後將list轉乘np.array，再用np.unique()找出來
        #如果要查出該問題row的index，可以用np.where()查詢
        floor = floor.rstrip(',')
        #裡面有的值不是NaN，而是''，所以直接用boolean filter out，再drop it out
        #原本的值 e.g. '/十層'
        if floor == '':
            floor_nan_index_list.append(index)
        else:
            floor_list.append(floor)
floor_series = pd.Series(floor_list)

        
df = df.drop(floor_nan_index_list, axis=0)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = pd.concat([df, floor_series], axis=1)
df = df.drop('floor', axis=1)
df = df.rename(columns={0:'floor'})

#------------------------------------------------------------------------------------
#Extracting deal year
#------------------------------------------------------------------------------------

df['deal_year']=df['deal_date'].apply(lambda x:x[0:3])

#Create total floor column
total_floor_list = []
for i in df['floor']:
    if '全' in i:
        total_floor_list.append('全')
    else:
        total_floor_list.append(i.count('層'))
total_floor_series = pd.Series(total_floor_list)
df = pd.concat([df, total_floor_series], axis=1)
df = df.rename(columns={0:'total_floor'})

#------------------------------------------------------------------------------------
#Create highest floor column
#------------------------------------------------------------------------------------

highest_floor_list = []
problem_floor = []
floor_list = ['四十層', '三十九層', '三十八層', '三十七層', '三十六層', '三十五層', '三十四層', '三十三層', '三十二層', '三十一層',
              '三十層', '二十九層', '二十八層', '二十七層', '二十六層', '二十五層', '二十四層', '二十三層', '二十二層', '二十一層',
              '二十層', '十九層', '十八層', '十七層', '十六層', '十五層', '十四層', '十三層', '十二層', '十一層',
              '十層', '九層', '八層', '七層', '六層', '五層', '四層', '三層', '二層', '一層']
for i, index in zip(df['floor'], df.index):
    if '全' in i:
        highest_floor_list.append('全')
    else:
        for h in floor_list:
            if h in i:
                highest_floor_list.append(h)
                break
            else:
                if h == '一層':
                    highest_floor_list.append('無')
                    problem_floor.append(index)
highest_floor_series = pd.Series(highest_floor_list)        
df = pd.concat([df, highest_floor_series], axis=1)
df = df.rename(columns={0:'highest_floor'})

#------------------------------------------------------------------------------------
#Caculate distances of MRT stations nearby
#------------------------------------------------------------------------------------

df_MRT = pd.read_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\project_2023\\MRT_Coordinates_Taichung.csv')
df_MRT = df_MRT.rename(columns={'經度':'longitude', '緯度':'latitude', '站名':'station'})

from math import radians, sin, cos, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

MRT_list = []
MRT_YN_list = []
run_time = 0
for lon1, lat1 in zip(df['longitude'], df['latitude']):
    run_time += 1
    for lon2, lat2, station in zip(df_MRT['longitude'], df_MRT['latitude'], df_MRT['station']):
        distance = haversine(lon1, lat1, lon2, lat2)
        if distance < 0.5:
            MRT_list.append(station)
            MRT_YN_list.append(1)
            break
        else:
            continue
    if len(MRT_list) != run_time:
        MRT_list.append('無')
        MRT_YN_list.append(0)
    else:
        continue
    
MRT_series = pd.Series(MRT_list)
MRT_YN_series = pd.Series(MRT_YN_list)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = pd.concat([df, MRT_series, MRT_YN_series], axis=1)
df = df.rename(columns={0:'MRT', 1:'MRT_nearby'})

#------------------------------------------------------------------------------------
#datetime
#------------------------------------------------------------------------------------

month_problem_index = []
for date, index in zip(df['deal_date'], df.index):
    if date[4:6]=='00':
        month_problem_index.append(index)
    else:
        continue
    
df = df.drop(month_problem_index, axis=0)
    
    
d_list = []
for i in range(len(df['deal_date'])):
    d=df['deal_date'].iloc[i].replace(df['deal_date'].iloc[i][0:3], str(int(df['deal_date'].iloc[i][0:3]) + 1911))
    d_list.append(d)
d_series = pd.Series(d_list)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = pd.concat([df, d_series], axis=1)
df = df.drop(['deal_date'], axis=1)
df = df.rename(columns={0:'deal_date'})
df.isna().sum()
df['deal_date'] = pd.to_datetime(df['deal_date'], format='%Y/%m/%d').dt.date

#------------------------------------------------------------------------------------
#Elevator
#------------------------------------------------------------------------------------

elevator_list = []
for elevator in df['elevator']:
    if elevator == '無':
        elevator_list.append(0)
    if elevator == '有':
        elevator_list.append(1)
elevator_series = pd.Series(elevator_list)
df = pd.concat([df, elevator_series], axis=1)
df = df.drop(['elevator'], axis=1)
df = df.rename(columns={0:'elevator'})

#------------------------------------------------------------------------------------
#Manager
#------------------------------------------------------------------------------------

df['manager'] = df['manager'].map(lambda x: 0 if x=='無' else 1)

#------------------------------------------------------------------------------------
#Longitude & Latitude
#------------------------------------------------------------------------------------

df = df[(df['longitude']>120) & (df['longitude']<122)
        & (df['latitude']>22) & (df['latitude']<25)
        ]

#------------------------------------------------------------------------------------
#Save as csv
#------------------------------------------------------------------------------------

df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\Clean_RealEstates')