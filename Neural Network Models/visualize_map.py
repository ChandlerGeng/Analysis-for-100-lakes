import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.basemap import Basemap

# source: https://blog.csdn.net/pengranxindong/article/details/79136486
# https://matplotlib.org/stable/api/markers_api.html

# get data

# open_path = 'D:/Lancaster-MSC/Satellite-Image/predict_numpy/data&graph(final)/total.csv'
open_path = 'D:/Lancaster-MSC/Satellite-Image/predict_numpy/data&graph(final)/total-var.csv'
data = []
for i in range(104):
  data.append([])

a = csv.reader(open(open_path,'r'))
show=[]
for i in a:
    show.append(i)

for i in range(len(show)):
    for j in range(len(show[0])):
        data[i].append(eval(show[i][j]))

data = np.array(data)

# draw map
fig = plt.figure(figsize=(24, 20), facecolor='cornsilk') # window size
m = Basemap()  # initialise map

def drawMap():

    m.drawmapboundary(fill_color='white')  # ocean

    #'darkolivegreen'  'coral'

    m.fillcontinents(color='gray',    # land colour
                    lake_color='gray',    #  lake colour

                     )
    m.drawcoastlines()  # coastline
    m.drawcountries()  # borders

def set_lonlat(_m, lon_list, lat_list, lon_labels, lat_labels, lonlat_size):

    lon_dict = _m.drawmeridians(lon_list, labels=lon_labels, color='grey', fontsize=lonlat_size)
    lat_dict = _m.drawparallels(lat_list, labels=lat_labels, color='grey', fontsize=lonlat_size)
    lon_list = []
    lat_list = []
    for lon_key in lon_dict.keys():
        try:
            lon_list.append(lon_dict[lon_key][1][0].get_position()[0])
        except:
            continue

    for lat_key in lat_dict.keys():
        try:
            lat_list.append(lat_dict[lat_key][1][0].get_position()[1])
        except:
            continue
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_yticks(lat_list)
    ax.set_xticks(lon_list)
    ax.tick_params(labelcolor='none')

drawMap()
set_lonlat(m, range(0, 360, 30), range(-90, 90, 30), [0, 0, 1, 0], [1, 0, 0, 0], 12)

# for i in range(len(data)):
#     # unchanged
#     if 500 > data[i][0] > -500:
#         m.plot(data[i][2],data[i][1], marker='.', color='white')
#     # increase
#     if data[i][0] >= 500:
#         m.plot(data[i][2], data[i][1], marker='v', color='red')
#     # decrease
#     if -500 >= data[i][0]:
#         m.plot(data[i][2], data[i][1], marker='*', color='blue')

# for i in range(len(data)):
#     # unchanged
#     if 500 > data[i][0] > -500:
#         m.plot(data[i][2],data[i][1], marker='.', color='blue')
#     # increase
#     if 1500 > data[i][0] >= 500:
#         m.plot(data[i][2], data[i][1], marker='.', color='LemonChiffon')
#     if 2500 > data[i][0] >= 1500:
#         m.plot(data[i][2], data[i][1], marker='.', color='Khaki')
#     if data[i][0] >= 2500:
#         m.plot(data[i][2], data[i][1], marker='.', color='Yellow')
#     # decrease
#     if -500 >= data[i][0] > -1500:
#         m.plot(data[i][2], data[i][1], marker='.', color='Salmon')
#     if -1500 >= data[i][0] > -2500:
#         m.plot(data[i][2], data[i][1], marker='.', color='IndianRed')
#     if -2500 >= data[i][0]:
#         m.plot(data[i][2], data[i][1], marker='.', color='red')

# variation
for i in range(len(data)):
    # unchanged
    if 0.05 > data[i][0] > -0.05:
        m.plot(data[i][2],data[i][1], marker='.', color='blue')
    # increase
    if 0.2 > data[i][0] >= 0.05:
        m.plot(data[i][2], data[i][1], marker='.', color='LemonChiffon')
    if 0.35 > data[i][0] >= 0.2:
        m.plot(data[i][2], data[i][1], marker='.', color='Khaki')
    if data[i][0] >= 0.35:
        m.plot(data[i][2], data[i][1], marker='.', color='Yellow')
    # decrease
    if -0.05 >= data[i][0] > -0.2:
        m.plot(data[i][2], data[i][1], marker='.', color='Salmon')
    if -0.2 >= data[i][0] > -0.35:
        m.plot(data[i][2], data[i][1], marker='.', color='IndianRed')
    if -0.35 >= data[i][0]:
        m.plot(data[i][2], data[i][1], marker='.', color='red')

plt.savefig('D:/Lancaster-MSC/Satellite-Image/predict_numpy/data&graph(final)/world_gray_var.png')
plt.show()
