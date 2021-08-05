#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:37:40 2021

@author: juanmalagon
"""

from sklearn.cluster import AgglomerativeClustering # For HAC clustering
import scipy.cluster.hierarchy as sch # For HAC Denddrogram

import matplotlib.pyplot as plt # for data visualization
import plotly.graph_objects as go # for data visualization

import pandas as pd # for data manipulation
from geopy.geocoders import Nominatim # for getting city coordinates
from progressbar import ProgressBar # for displaying progress 
import time # for adding time delays

# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]

# For other columns with missing values, fill them in with column mean
df=df.fillna(df.mean())

# Add spaces between multiple words in location names
df['Location2']=df['Location'].str.replace( r"([A-Z])", r" \1").str.strip()
# Update Location for Pearce RAAF so it can be found by geolocator
df['Location2']=df['Location2'].apply(lambda x: 'Pearce, Bullsbrook' if x=='Pearce R A A F' else x)

# Show a snaphsot of data
df

# Create a list of unique locations (cities)
loc_list=list(df.Location2.unique())

geolocator = Nominatim(user_agent="add-your-agent-name")
country ="Australia"
loc_res=[]

pbar=ProgressBar() # This will help us to show the progress of our iteration
for city in pbar(loc_list):
    loc = geolocator.geocode(city+','+ country)
    res = [city, loc.latitude, loc.longitude]
    loc_res = loc_res + [res]
    time.sleep(1) # sleep for 1 second before submitting the next query

# Add locations to a dataframe
df_loc=pd.DataFrame(loc_res, columns=['Loc', 'Latitude', 'Longitude'])

# Show data
df_loc

# Create a figure
fig = go.Figure(data=go.Scattergeo(
        lat=df_loc['Latitude'],
        lon=df_loc['Longitude'],
        hovertext=df_loc['Loc'], 
        mode = 'markers',
        marker_color = 'black',
        ))

# Update layout so we can zoom in on Australia
fig.update_layout(
        width=980,
        height=720,
        margin={"r":0,"t":10,"l":0,"b":10},
        geo = dict(
            scope='world',
            projection_type='miller',
            landcolor = "rgb(250, 250, 250)",
            center=dict(lat=-25.69839, lon=139.8813), # focus point
            projection_scale=6 # zoom in on
        ),
    )
fig.show()

# Select attributes
X = df_loc[['Latitude', 'Longitude']]

# Create a figure
plt.figure(figsize=(16,9), dpi=300)

# Create linkage
Z = sch.linkage(X, method='average', optimal_ordering=True) # note we use method='average'

# Specify cluster colors
sch.set_link_color_palette(['red', '#34eb34', 'blue', '#ae34eb'])

# Draw a dendrogram
sch.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=list(df_loc['Loc']), 
               color_threshold=14.55, above_threshold_color='black')
 
# Add horizontal line
plt.axhline(y=14.55, c='grey', lw=1, linestyle='dashed')

# Show the plot
plt.show()

# Select attributes
X = df_loc[['Latitude', 'Longitude']]

# Create a figure
plt.figure(figsize=(16,9), dpi=300)

# Create linkage
Z = sch.linkage(X, method='ward', optimal_ordering=True) # note, we use method='ward'

# Specify cluster colors
sch.set_link_color_palette(['red', '#34eb34', 'blue', '#ae34eb'])

# Draw a dendrogram
sch.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=list(df_loc['Loc']), 
               color_threshold=25, above_threshold_color='black')
 
# Add horizontal line
plt.axhline(y=25, c='grey', lw=1, linestyle='dashed')

# Show the plot
plt.show()

# Set the model and its parameters
# note, options for linkage: {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
modela4 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
modelw4 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

# Fit HAC on our data
clusta4 = modela4.fit(X)
clustw4 = modelw4.fit(X)

# Attach cluster labels back to the location dataset
df_loc['Clusta4']=clusta4.labels_
df_loc['Clustw4']=clustw4.labels_

# Print data
df_loc

# Create a figure
fig = go.Figure(data=go.Scattergeo(
        lat=df_loc['Latitude'],
        lon=df_loc['Longitude'],
        hovertext=df_loc[['Loc', 'Clusta4']], 
        mode = 'markers',
        marker=dict(colorscale=['#34eb34', 'blue', '#ae34eb', 'red']),
        marker_color = df_loc['Clusta4'],
        ))

# Update layout so we can zoom in on Australia
fig.update_layout(
        showlegend=False,
        width=980,
        height=720,
        margin={"r":0,"t":10,"l":0,"b":10},
        geo = dict(
            scope='world',
            projection_type='miller',
            landcolor = "rgb(250, 250, 250)",
            center=dict(lat=-25.69839, lon=139.8813), # focus point
            projection_scale=6 # zoom in on
        ),
    )
fig.show()