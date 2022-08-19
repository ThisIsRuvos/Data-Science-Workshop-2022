# -*- coding: utf-8 -*-
"""
Title:
Author: Jake Hightower
Date: Thu Aug 11 13:13:23 2022
Version: 0.1
Last Updated:


Other Notes:
    -Spatial specificity - Country, State, County
    -Discuss Normalization
    -Scatterplot diagram
    -Discuss randomness in variables, correlation, autocorrelation
    -Voronoi Polygons
    -URL links / resources for other data sets
    -Visualization


1) What format problems do we have?
2) Where data do we trust?
    What don't we trust?
    WHY??
3) Given constraints, what can we do?
"""

####-------------------------------------------------------------------------------------------------------------------#
## Import libraries
import os
import pandas as pd
import numpy as np
import opendatasets as od
import matplotlib.pyplot as plt
#conda install -c plotly plotly-geo
import plotly.figure_factory as ff
import seaborn as sns


####-------------------------------------------------------------------------------------------------------------------#
## set wd
os.chdir('C:/Users/Jake/Desktop/Projects/Coding_Playground/PythonCode/datacleaning')
print("Current Working Directory " , os.getcwd())


####-------------------------------------------------------------------------------------------------------------------#
## Download Covid19 Data Set
#note you'll need username & to download a key (can open JSON with text file/ screengrab)
#dataset_url = 'https://www.kaggle.com/sudalairajkumar/covid19-in-usa'
#od.download(dataset_url)


####-------------------------------------------------------------------------------------------------------------------#
## Download Covid19 Data Set
c19 = pd.read_csv("covid19_in_usa/us_counties_covid19_daily.csv")

####-------------------------------------------------------------------------------------------------------------------#
## Data Cleaning

## Do you know what the data SHOULD look like? Compare that to quick dive here:
c19.dtypes   # Notice that cases and deaths are not stored as the same data type
c19.describe() #Notice the maxes of Cases and Deaths?! Will need to look into this
c19.shape
print('No. of rows: %s |  No. of cols: %s' % (c19.shape[0], c19.shape[1]))
## 800k rows, but how many are distinct days?
c19['date'].nunique() #320, 319 days though b/w these two dates
c19['date'].min()
c19['date'].max()
## SUPER IMPORTANT : Note that betweeen jan 21st and Dec 5th of 2020, that is 319 days. BUT the total unique count is 320 days. What does that mean?
## It means there are days in this dataset that are MISSING. Probably no data reporting, so the index is MISSING that date.
## We can add these days back in and defualt to zero, but this is a good moment to reach back out to date pipeline and CONFIRM those days should be missing.

##  How many distinct counties/ states? #################################################################
c19['county'].nunique() #1929, but there are 3,006 in the US. What's missing?
c19['state'].nunique() #55 states? List
c19_states = list(c19['state'].unique())
len(c19_states)
print(c19_states) #DC, Guam, N Mariana Islands, Puerto Rico, Virgin Islands (so States & Territories) --> should we update col name to reflect?)


## how much data is missing/ na #########################################################################
for col in c19.columns:
    pct_missing = np.mean(c19[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


c19_nan = c19.loc[c19.isnull().any(axis=1)]
c19_nan.head()

## Where are there missings- is there autocorrelation? (Note: visualizing not viable for most datasets, too large)
colours = ['#66CD00', '#CD3333']
sns.heatmap(c19.isnull(), cmap=sns.color_palette(colours))

## Check for duplicates ###################################################################################
## Pseudocode to check all below
#dupes = []
#for s in states:
#    for c in counties:
#        create new df
#        ise_dupe = check for dupes
#        dupes.append(is_dupe)

# Focus on one state/ county below as example:
FL_orange = c19[(c19.state == 'Florida') & (c19.county == 'Orange')]
for col in FL_orange.columns:
    pct_dupe = np.mean(FL_orange[col].duplicated())
    print('{} - {}%'.format(col, round(pct_dupe*100)))

####-------------------------------------------------------------------------------------------------------------------#
## Data ReFormate/ New View

#Format version 2
c19_daily = c19.groupby(['date']).sum()
c19_daily.drop('fips', axis=1, inplace=True)
c19_daily.dtypes #notice cases are int and deaths are float- fyi, as a datascientist my intution is there are probably NAs present in deaths which change the format
c19_daily.head()

plt.plot(c19_daily)
plt.plot(c19_daily['cases'])
plt.plot(c19_daily['deaths'])


## Axis is illegible because index is not stored as datetime
c19_daily.index = pd.to_datetime(c19_daily.index)

## Plot Cumsum Cases/ Death
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(c19_daily['cases'],
        color="blue")
# set x-axis label
ax.set_xlabel("Date", fontsize = 14)
# set y-axis label
ax.set_ylabel("Cases",
              color="blue",
              fontsize=14)
ax.tick_params(axis='x',rotation=90)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(c19_daily['deaths'],
         color="red")
ax2.set_ylabel("Deaths",
               color="red",
               fontsize=14)
ax2.tick_params(axis='x',rotation=90)
fig.suptitle('Cumsum Covid-19 Case & Death Count', fontsize=16)
plt.tight_layout()
plt.show()

####-------------------------------------------------------------------------------------------------------------------#
## Convert Cumsum to daily
## Plot Dialy Cases Death
c19_daily.rename(columns={'cases': 'cases_cumsum', 'deaths': 'deaths_cumsum'}, inplace=True)
c19_daily['cases_daily'] = c19_daily['cases_cumsum'].diff().fillna(c19_daily['cases_cumsum'])
c19_daily['deaths_daily'] = c19_daily['deaths_cumsum'].diff().fillna(c19_daily['deaths_cumsum'])
c19_daily.head()

## Plot Daily Cases/ Death
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(c19_daily['cases_daily'],
        color="blue")
# set x-axis label
ax.set_xlabel("Date", fontsize = 14)
# set y-axis label
ax.set_ylabel("Cases",
              color="blue",
              fontsize=14)
ax.tick_params(axis='x',rotation=90)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(c19_daily['deaths_daily'],
         color="red")
ax2.set_ylabel("Deaths",
               color="red",
               fontsize=14)
ax2.tick_params(axis='x',rotation=90)
fig.suptitle('Daily Covid-19 Case & Death Count', fontsize=16)
plt.tight_layout()
plt.show()





####-------------------------------------------------------------------------------------------------------------------#
## Check for missing dates (not empty, MISSING)
min_date = c19_daily.index.min()
max_date = c19_daily.index.max()

## Set timestamp as index

idx = pd.date_range(str(min_date), str(max_date))
print("Dates in dataframe: %s  vs Expected Length: %s " % (len(c19_daily), len(idx)))

##Uncomment below to fix missing dates in index
#c19_daily.set_index('new_dates', inplace=True)
#c19_daily = c19_daily.reindex(idx, fill_value=0)
#c19_daily.head()

## Simulate example without
c19_missings = c19_daily.copy()
c19_missings.drop(c19_missings.index[50:80], inplace=True)


## Plot
fig,ax = plt.subplots()
ax.plot(c19_missings['cases_daily'],
        color="blue")
ax.set_xlabel("Date", fontsize = 14)
ax.set_ylabel("Cases",
              color="blue",
              fontsize=14)
ax.tick_params(axis='x',rotation=90)
ax2=ax.twinx()
ax2.plot(c19_missings['deaths_daily'],
         color="red")
ax2.set_ylabel("Deaths",
               color="red",
               fontsize=14)
ax2.tick_params(axis='x',rotation=90)
fig.suptitle('Daily Covid-19 Case & Death Count', fontsize=16)
plt.tight_layout()
plt.show()


## Notice how small and nuanced the difference b/w the two graphs are


####-------------------------------------------------------------------------------------------------------------------#
## Bias Assessment

## Day Average
fig, axs = plt.subplots(figsize=(12, 4))
c19_daily.groupby(c19_daily.index.day)["cases_daily"].mean().plot(
    kind='bar', rot=0, ax=axs)
## Weekly Average
fig, axs = plt.subplots(figsize=(12, 4))
c19_daily.groupby(c19_daily.index.week)["cases_daily"].mean().plot(
    kind='bar', rot=0, ax=axs)
## Month Average
fig, axs = plt.subplots(figsize=(12, 4))
c19_daily.groupby(c19_daily.index.month)["cases_daily"].mean().plot(
    kind='bar', rot=0, ax=axs)

#------------------------------
# Daily cases with boxplot
fig, axs = plt.subplots(figsize=(12, 4))
sns.boxplot(x=c19_daily.index.day, y=c19_daily['cases_daily'])
axs.set_xlabel('Day of month')
axs.set_title('Boxplot of Daily Cases')
axs.set_ylabel('Cases')
# Weekly cases with boxplot
fig, axs = plt.subplots(figsize=(12, 4))
sns.boxplot(x=c19_daily.index.week, y=c19_daily['cases_daily'])
axs.set_xlabel('Week of year')
axs.set_title('Boxplot of Weekly Cases')
axs.set_ylabel('Cases')
# Monthly cases with boxplot
fig, axs = plt.subplots(figsize=(12, 4))
sns.boxplot(x=c19_daily.index.month, y=c19_daily['cases_daily'])
axs.set_xlabel('Month of year')
axs.set_title('Boxplot of Monthly Cases')
axs.set_ylabel('Cases')


####-------------------------------------------------------------------------------------------------------------------#
## Spatial Mapping
## Plot State and Counties by Fips
fips = list(c19['fips'].unique().astype(int)) #Select Unique, Convert Array of float to array of int, Convert array to list
values = range(len(fips))

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()

##https://www.python-graph-gallery.com/292-choropleth-map-with-folium
# import the folium library
import folium

# initialize the map and store it in a m object
m = folium.Map(location=[40, -95], zoom_start=4)

# show the map
m

## What to do with missing data

## Discuss randomness in variables, correlation, autocorrelation
## Variable
