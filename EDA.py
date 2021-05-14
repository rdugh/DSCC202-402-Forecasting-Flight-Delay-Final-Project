# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md ##Read in Air Traffic bronze data

# COMMAND ----------

databaseName = 'dscc202_group05_db'
airDF = spark.sql("""
  SELECT *
  FROM {}.air_traffic_delta_bronze
  """.format(databaseName))

# COMMAND ----------

display(airDF)

# COMMAND ----------

# MAGIC %md ###Extract the hour and create a new feature which is the day + hour of departure.

# COMMAND ----------

#extract time (join FL_DATE & DEP_TIME)

import pyspark.sql.functions as F
from pyspark.sql.functions import *
airDF2 = airDF.withColumn("DEP_TIME", col("DEP_TIME").cast('string')).withColumn("DEP_TIME", lpad("DEP_TIME", 4, "0")).withColumn("DEP_TIME",regexp_replace(col("DEP_TIME"),"(\\d{2})(\\d{2})" , "$1:$2" )).withColumn('time',concat(col("FL_DATE"), lit(" "), col("DEP_TIME"))) 

#covert the concatenation to TIMESTAMP
airDF2 = airDF2.withColumn('time', to_timestamp(col('time'), 'yyyy-MM-dd HH:mm'))
display(airDF2)
#display(airDF2)

# COMMAND ----------

airDF2.agg({"DEP_DELAY": "max"}).collect()[0] #max delay

# COMMAND ----------

# MAGIC %md ###Pandas profiling

# COMMAND ----------

from pathlib import Path

import numpy as np
import pandas as pd
import requests

import pandas_profiling
from pandas_profiling.utils.cache import cache_file

displayHTML(pandas_profiling.ProfileReport(airDF2.toPandas()).html)

#REPORT: 

# COMMAND ----------

# MAGIC %md Day of Week vs average Delay

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AVG(ARR_DELAY) AS avg_arr_delay,AVG(DEP_DELAY) AS avg_dep_delay, DAY_OF_WEEK
# MAGIC FROM dscc202_group05_db.air_traffic_delta_bronze
# MAGIC GROUP BY DAY_OF_WEEK
# MAGIC SORT BY DAY_OF_WEEK

# COMMAND ----------

# MAGIC %md Avg Delay for each origin

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AVG(ARR_DELAY) AS avg_arr_delay,AVG(DEP_DELAY) AS avg_dep_delay, ORIGIN
# MAGIC FROM dscc202_group05_db.air_traffic_delta_bronze
# MAGIC GROUP BY ORIGIN
# MAGIC SORT BY ORIGIN

# COMMAND ----------

# MAGIC %md Average Delay for each destination airport

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AVG(ARR_DELAY) AS avg_arr_delay,AVG(DEP_DELAY) AS avg_dep_delay, DEST
# MAGIC FROM dscc202_group05_db.air_traffic_delta_bronze
# MAGIC GROUP BY DEST
# MAGIC SORT BY DEST

# COMMAND ----------

display(airDF2.groupBy('OP_UNIQUE_CARRIER').agg(approx_count_distinct("ORIGIN_AIRPORT_ID").alias("Origin_Airports")).sort("Origin_Airports"))

# COMMAND ----------

# MAGIC %md Count Unique Airports Destinations for different carriers

# COMMAND ----------

display(airDF2.groupBy('OP_UNIQUE_CARRIER').agg(approx_count_distinct("DEST_AIRPORT_ID").alias("Dest_Airports")).sort("Dest_Airports"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Bronze Weather  data

# COMMAND ----------

databaseName = 'dscc202_group05_db'
weatherDF = spark.sql("""
  SELECT *
  FROM {}.weather_delta_bronze
  """.format(databaseName))

# COMMAND ----------

# MAGIC %md Summary statistics of continuous variables

# COMMAND ----------

display(weatherDF['avg_temp_f',
 'tot_precip_mm',
 'avg_wnd_mps',
 'avg_vis_m',
 'avg_slp_hpa',
 'avg_dewpt_f'].summary())


# COMMAND ----------

# MAGIC %md visualize statistics  using Tensorflow data validation lib

# COMMAND ----------


import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


stats = tfdv.generate_statistics_from_dataframe(dataframe=weatherDF.toPandas())
tfdv.visualize_statistics(stats)
displayHTML(get_statistics_html(stats)) #tot_precip_mm has about 90% zeros!, avg_wnd_mps has about 16% #No missing data


# COMMAND ----------

# MAGIC %md infer schema

# COMMAND ----------


weather_data_schema = tfdv.infer_schema(statistics=stats)
tfdv.display_schema(schema=weather_data_schema)

# COMMAND ----------

# MAGIC %md check for anomalies

# COMMAND ----------

weather_anomalies = tfdv.validate_statistics(statistics=stats, schema=weather_data_schema)
tfdv.display_anomalies(weather_anomalies)

# COMMAND ----------

# MAGIC %md average monthly temperature

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT x.month,AVG(x.avg_temp_f) as avg_temp_f FROM (
# MAGIC SELECT MONTH(time) as month,YEAR(time) as year,SUM(avg_temp_f) as avg_temp_f
# MAGIC FROM dscc202_group05_db.weather_delta_bronze
# MAGIC GROUP BY MONTH(time), YEAR(time)
# MAGIC   ) x
# MAGIC GROUP BY x.month
# MAGIC ORDER BY x.month

# COMMAND ----------

# MAGIC %md average monthly precipitation

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT x.month,AVG(x.precip_mm) as avg_precip_mm FROM (
# MAGIC SELECT MONTH(time) as month,YEAR(time) as year,SUM(tot_precip_mm) as precip_mm
# MAGIC FROM dscc202_group05_db.weather_delta_bronze
# MAGIC GROUP BY MONTH(time), YEAR(time) ) x
# MAGIC GROUP BY x.month
# MAGIC ORDER BY x.month

# COMMAND ----------

# MAGIC % md average monthly wind speed

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT x.month,AVG(x.wnd_mps) as avg_wnd_mps FROM (
# MAGIC SELECT MONTH(time) as month,YEAR(time) as year,SUM(avg_wnd_mps) as wnd_mps 
# MAGIC FROM dscc202_group05_db.weather_delta_bronze
# MAGIC GROUP BY MONTH(time), YEAR(time)) x
# MAGIC GROUP BY x.month
# MAGIC ORDER BY x.month

# COMMAND ----------

#weekofyear  
#'avg_vis_m'      FLOAT  required          -    
#'avg_slp_hpa'    FLOAT  required          -    
#'avg_dewpt_f'

# COMMAND ----------

# MAGIC %md ##EDA ->Silver tables (silverdep_delta & silverarr_delta)
# MAGIC 
# MAGIC  

# COMMAND ----------

databaseName = 'dscc202_group05_db'
Dep_SilverDF = spark.sql("""
  SELECT *
  FROM {}.silverdep_delta
  """.format(databaseName))

# COMMAND ----------

databaseName = 'dscc202_group05_db'
Arr_SilverDF = spark.sql("""
  SELECT *
  FROM {}.silverarr_delta
  """.format(databaseName))

# COMMAND ----------

# MAGIC %md #Tensorflow Data Validation of both silver tables

# COMMAND ----------

# MAGIC %md ##Arrivals one

# COMMAND ----------

# MAGIC %md Visualize stats

# COMMAND ----------

import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

train, test = train_test_split(Arr_SilverDF.toPandas(), test_size=0.3, random_state=42)
_eval, serve = train_test_split(test, test_size=0.5, random_state=42)

train_stats = tfdv.generate_statistics_from_dataframe(dataframe=train)
tfdv.visualize_statistics(train_stats)


# COMMAND ----------

displayHTML(get_statistics_html(train_stats)) #about 94% tot_precip_mm feature values is zeros!!!, arr_delay, dep_delay, and hour have missing values


# COMMAND ----------

# MAGIC %md infer schema

# COMMAND ----------

schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)

# COMMAND ----------

# MAGIC %md Check evaluation data for errors

# COMMAND ----------

eval_stats = tfdv.generate_statistics_from_dataframe(dataframe=_eval)
displayHTML(get_statistics_html(lhs_statistics=eval_stats, rhs_statistics=train_stats,lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET'))#same issue--total_precip_mm feature being mostly zeros

# COMMAND ----------

# MAGIC %md Check for evaluation & serving data anomalies

# COMMAND ----------

anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.display_anomalies(anomalies)

# COMMAND ----------

serving_stats = tfdv.generate_statistics_from_dataframe(dataframe=serve)
serving_anomalies = tfdv.validate_statistics(serving_stats, schema)

tfdv.display_anomalies(serving_anomalies)

# COMMAND ----------

# MAGIC %md ##Departures one

# COMMAND ----------

# MAGIC %md Compute and visualize statistics

# COMMAND ----------

import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

train2, test2 = train_test_split(Dep_SilverDF.toPandas(), test_size=0.3, random_state=42)
_eval2, serve2 = train_test_split(test2, test_size=0.5, random_state=42)

train_stats2 = tfdv.generate_statistics_from_dataframe(dataframe=train2)
tfdv.visualize_statistics(train_stats2)


# COMMAND ----------

displayHTML(get_statistics_html(train_stats2)) #about 94% tot_precip_mm features is zeros!

# COMMAND ----------

# MAGIC %md infer schema

# COMMAND ----------

schema2 = tfdv.infer_schema(statistics=train_stats2)
tfdv.display_schema(schema=schema2)

# COMMAND ----------

# MAGIC %md Check evaluation data for errors

# COMMAND ----------

eval_stats2 = tfdv.generate_statistics_from_dataframe(dataframe=_eval2)
displayHTML(get_statistics_html(lhs_statistics=eval_stats2, rhs_statistics=train_stats2,lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET'))#same issue--total_precip_mm!!

# COMMAND ----------

# MAGIC %md Check for evaluation & serving data anomalies

# COMMAND ----------

anomalies2 = tfdv.validate_statistics(statistics=eval_stats2, schema=schema2)
tfdv.display_anomalies(anomalies2)

# COMMAND ----------

serving_stats2 = tfdv.generate_statistics_from_dataframe(dataframe=serve2)
serving_anomalies2 = tfdv.validate_statistics(serving_stats2, schema2)

tfdv.display_anomalies(serving_anomalies2)

# COMMAND ----------

#%md Check for drift and skew

# COMMAND ----------

#replace airport ids with names: 

from pyspark.sql.functions import *
from pyspark.sql.types import *

replace_ids = {"10397": "ATL",
"10721": "BOS",
"11057": "CLT",
"13930": "ORD",
"11193": "CVG",
"11298": "DFW",
"11292": "DEN",
"12266": "IAH",
"12892": "LAX",
"12478": "JFK",
"14771": "SFO",
"14747": "SEA",
"11278": "DCA" }

Dep_SilverDF = Dep_SilverDF.withColumn('ORIGIN_AIRPORT_ID', col('ORIGIN_AIRPORT_ID').cast(StringType())).withColumn('DEST_AIRPORT_ID', col('DEST_AIRPORT_ID').cast(StringType()))

Arr_SilverDF = Arr_SilverDF.withColumn('ORIGIN_AIRPORT_ID', col('ORIGIN_AIRPORT_ID').cast(StringType())).withColumn('DEST_AIRPORT_ID', col('DEST_AIRPORT_ID').cast(StringType()))


Dep_SilverDF = Dep_SilverDF.na.replace(list(replace_ids.keys()),list(replace_ids.values()),'ORIGIN_AIRPORT_ID')
Dep_SilverDF = Dep_SilverDF.na.replace(list(replace_ids.keys()),list(replace_ids.values()),'DEST_AIRPORT_ID')
Arr_SilverDF = Dep_SilverDF.na.replace(list(replace_ids.keys()),list(replace_ids.values()),'ORIGIN_AIRPORT_ID')
Arr_SilverDF = Arr_SilverDF.na.replace(list(replace_ids.keys()),list(replace_ids.values()),'DEST_AIRPORT_ID')
display(Arr_SilverDF)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from scipy.stats import norm

arr_pd = Arr_SilverDF.toPandas()
dep_pd = Dep_SilverDF.toPandas()

# COMMAND ----------

# MAGIC %md #more EDA on the 2 silver tables

# COMMAND ----------

# MAGIC %md ### 1. Arrivals

# COMMAND ----------

#correlations: dep delay & arrival delay , avg_temp_f & avg_dewpt_f have significant correlation!! 
#Some slight correlations: hour & temp, month & temp, month & dew, temp & vis, precip & delays

corrmat = arr_pd.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# COMMAND ----------

# MAGIC %md Further exploration of vars with some slight correlation based on above plot: ( month & temp, month & dew, temp & vis, precip & delays)

# COMMAND ----------

#1. delays vs precipitation 
#scatterplot
sns.set()
cols = ['ARR_DELAY', 'DEP_DELAY','tot_precip_mm']
sns.pairplot(arr_pd[cols])
plt.show()

# COMMAND ----------

#2. temp & vis: very small correlation
sns.lmplot(x='avg_temp_f', y='avg_vis_m', data=arr_pd)

# COMMAND ----------

#3. month & temp: 

f,ax=plt.subplots(1,3,figsize=(20,8))
arr_pd[['MONTH','avg_temp_f' ]].groupby(['MONTH']).mean().plot(ax=ax[0])
ax[0].set_title('Average (avg_temp_f ) by month')

#4. month & dew
arr_pd[['MONTH','avg_dewpt_f']].groupby(['MONTH']).mean().plot(ax=ax[1])
ax[1].set_title('Average (avg_dewpt_f) by month')

#5. hour & temp
arr_pd[['hour','avg_temp_f']].groupby(['hour']).mean().plot(ax=ax[2])
ax[2].set_title('Average (avg_temp_f) by hour')




# COMMAND ----------

#average departure delay at origin
display(Arr_SilverDF.groupBy('ORIGIN_AIRPORT_ID').avg('DEP_DELAY')) #ORL hs highest departure delay on average followed by SFO if flight is orginating there


# COMMAND ----------

#average departure delay at destination
display(Arr_SilverDF.groupBy('DEST_AIRPORT_ID').agg(avg('DEP_DELAY'))) ##JFK hs highest departure delay on average followed by SFO if a destination airport

# COMMAND ----------

plt.figure(figsize=(15,6))
sns.boxplot('ORIGIN_AIRPORT_ID','DEP_DELAY', data=arr_pd, palette="Set3")#departure delay distribution by origin airport: 

# COMMAND ----------

plt.figure(figsize=(15,6))
sns.boxplot('DEST_AIRPORT_ID','ARR_DELAY', data=arr_pd, palette="Set3")#arrival delay distribution by destination airport: 

# COMMAND ----------

#histogram of departure delays: most departure delays are within 0 mins
fig = plt.gcf()
fig.set_size_inches( 20, 6)
plt.grid(True)
sns.distplot(arr_pd['DEP_DELAY'],bins=50)
plt.show()

# COMMAND ----------

#histogram of arrival delays: similarly, most arrival delays are within 0 mins, although arrival has more delay time than departure in comparison
fig = plt.gcf()
fig.set_size_inches( 20, 6)
plt.grid(True)
sns.distplot(arr_pd['ARR_DELAY'])
plt.show()

# COMMAND ----------

#1. depature delays per week: Sunday and mid-week have highest departure delays
import seaborn as sns

plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_WEEK", y="DEP_DELAY", data=arr_pd[['DAY_OF_WEEK', 'DEP_DELAY']].groupby(by='DAY_OF_WEEK').sum().reset_index(drop=False), palette="Set3")


#2. #arrival delays per week: similarly, Sunday and mid-week have most highest arrival delays
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_WEEK", y="ARR_DELAY", data=arr_pd[['DAY_OF_WEEK', 'ARR_DELAY']].groupby(by='DAY_OF_WEEK').sum().reset_index(drop=False))

# COMMAND ----------

#departure delays per month: most departure delays are around mid-month
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_MONTH", y="DEP_DELAY", data=arr_pd[['DAY_OF_MONTH', 'DEP_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False),palette="Set3")
plt.show()

# COMMAND ----------

#arrival delays per month: most arrival delays are around mid-month too, no wonder the correlation we saw :-
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_MONTH", y="ARR_DELAY", data=arr_pd[['DAY_OF_MONTH', 'ARR_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False),color='black')
plt.show()
arr_pd[['DAY_OF_MONTH', 'ARR_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False)

# COMMAND ----------

# MAGIC %md distribution of the 13 airports

# COMMAND ----------

import seaborn as sns
plt.figure(figsize=(15,6))
sns.countplot(x='ORIGIN_AIRPORT_ID', data=arr_pd, color = 'steelblue') # countplot: ORD is the most frequent airport among the origin ones, then ATL

plt.figure(figsize=(15,6))
sns.countplot(x='DEST_AIRPORT_ID', data=arr_pd, color = 'steelblue') # countplot: LAX appears most frequent among destination airports in our data, followed by ORD

# COMMAND ----------

# MAGIC %md Distribution of flight status: derived from arrival delay feature & departure delay: very similar distribution!

# COMMAND ----------

#flight status: arrival
for flight in arr_pd:
    arr_pd.loc[arr_pd['ARR_DELAY'] <= 15, 'Status'] = 0 #flight was on-time
    arr_pd.loc[arr_pd['ARR_DELAY'] >= 15, 'Status'] = 1 #sligtly delayed
    arr_pd.loc[arr_pd['ARR_DELAY'] >= 60, 'Status'] = 2 #highly delayed


#f,ax=plt.subplots(1,2,figsize=(20,8))

#flight status per airport
fl_status = pd.DataFrame(arr_pd['Status'].value_counts(dropna=True))#distribution of flight status: most flights arrive on time!
fl_status['Arrival_x'] = pd.Series(['On-time','Slight delay','High delay'])
plt.figure(figsize=(10,6))
sns.barplot(x='Arrival_x', y='Status',data=fl_status)


#flight status: departure
for flight in arr_pd:
    arr_pd.loc[arr_pd['DEP_DELAY'] <= 15, 'Status2'] = 0 #flight was on-time
    arr_pd.loc[arr_pd['DEP_DELAY'] >= 15, 'Status2'] = 1 #sligtly delayed
    arr_pd.loc[arr_pd['DEP_DELAY'] >= 60, 'Status2'] = 2 #highly delayed



#flight status per airport
fl_status2 = pd.DataFrame(arr_pd['Status2'].value_counts(dropna=True))#distribution of flight status: most flights depart on time too!
fl_status2['Departure_x'] = pd.Series(['On-time','Slight delay','High delay'])
plt.figure(figsize=(10,6))
sns.barplot(x='Departure_x', y='Status2',data=fl_status2)






# COMMAND ----------

# MAGIC %md do some months have worse arrival delays than others

# COMMAND ----------

plt.figure(figsize=(10,6))
plt.scatter(arr_pd['MONTH'],arr_pd['ARR_DELAY'],c='blue', edgecolors='none',alpha=0.5)
plt.xlabel("MONTH")
plt.ylabel("ARRIVAL DELAY")
#very little difference between months & arrival delay. Some months (Feb, June, Dec) have outlier arrival delays though

# COMMAND ----------

# MAGIC %md how about with depature delays?

# COMMAND ----------

plt.figure(figsize=(10,6))
plt.scatter(arr_pd['MONTH'],arr_pd['DEP_DELAY'],c='blue', edgecolors='none',alpha=0.5)
plt.xlabel("MONTH")
plt.ylabel("DEPARTURE DELAY")
#very little difference between months & arrival delay. some months have outlier delays too (Feb, June, Sept, Dec)

# COMMAND ----------

# MAGIC %md ### 2. Departures

# COMMAND ----------

#correlations: dep delay & arrival delay , avg_temp_f & avg_dewpt_f have significant correlation just like with arrivals table
#Some slight correlations: month & temp, hour & temp, month & dew, temp & vis, precip & delays
#Hourly correlation with temp is also seen here

corrmat = dep_pd.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# COMMAND ----------


#3. month & temp: same trend as with arrival table

f,ax=plt.subplots(1,3,figsize=(20,8))
dep_pd[['MONTH','avg_temp_f' ]].groupby(['MONTH']).mean().plot(ax=ax[0])
ax[0].set_title('Average (avg_temp_f ) by month')

#4. month & dew
dep_pd[['MONTH','avg_dewpt_f']].groupby(['MONTH']).mean().plot(ax=ax[1])
ax[1].set_title('Average (avg_dewpt_f) by month')

#5. hour & temp
dep_pd[['hour','avg_temp_f']].groupby(['hour']).mean().plot(ax=ax[2])
ax[2].set_title('Average (avg_temp_f) by hour')






# COMMAND ----------

#average departure delay at origin
display(Dep_SilverDF.groupBy('ORIGIN_AIRPORT_ID').avg('DEP_DELAY')) #ORL has highest departure delay on average followed by SFO if flight is orginating there, just as with arrival table


# COMMAND ----------

#average departure delay at destination
display(Dep_SilverDF.groupBy('DEST_AIRPORT_ID').agg(avg('DEP_DELAY'))) ##JFK has highest departure delay on average followed by SFO a destination airport (similar to arrivals)

# COMMAND ----------

plt.figure(figsize=(15,6))
sns.boxplot('ORIGIN_AIRPORT_ID','DEP_DELAY', data=dep_pd, palette="Set3")#departure delay distribution by origin airport: 

plt.figure(figsize=(15,6))
sns.boxplot('DEST_AIRPORT_ID','ARR_DELAY', data=dep_pd, palette="Set3")#arrival delay distribution by destination airport: 



# COMMAND ----------

#histogram of departure delays: most departure delays are within 0 mins
fig = plt.gcf()
fig.set_size_inches( 20, 6)
plt.grid(True)
sns.distplot(dep_pd['DEP_DELAY'],bins=50)
plt.show()

# COMMAND ----------

#histogram of arrival delays: similarly, most arrival delays are within 0 mins, although arrival has more delay time than departure in comparison
fig = plt.gcf()
fig.set_size_inches( 20, 6)
plt.grid(True)
sns.distplot(dep_pd['ARR_DELAY'])
plt.show()

# COMMAND ----------

#1. depature delays per week: Sunday and mid-week have highest departure delays here too
import seaborn as sns

plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_WEEK", y="DEP_DELAY", data=dep_pd[['DAY_OF_WEEK', 'DEP_DELAY']].groupby(by='DAY_OF_WEEK').sum().reset_index(drop=False), palette="Set3")


#2. #arrival delays per week: similarly, Sunday and mid-week have most highest arrival delays here too
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_WEEK", y="ARR_DELAY", data=dep_pd[['DAY_OF_WEEK', 'ARR_DELAY']].groupby(by='DAY_OF_WEEK').sum().reset_index(drop=False))

# COMMAND ----------

#departure delays per month: highest departure delays are around mid-month (same): 18th day of the month has highest delays(I wonder what's causes that)
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_MONTH", y="DEP_DELAY", data=dep_pd[['DAY_OF_MONTH', 'DEP_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False),color='black')
plt.show()



#arrival delays per month: most arrival delays are around mid-month too, no wonder the correlation we saw :-
plt.figure(figsize=(15,6))
sns.barplot(x="DAY_OF_MONTH", y="ARR_DELAY", data=dep_pd[['DAY_OF_MONTH', 'ARR_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False),color='steelblue')
plt.show()
#arr_pd[['DAY_OF_MONTH', 'ARR_DELAY']].groupby(by='DAY_OF_MONTH').sum().reset_index(drop=False)

# COMMAND ----------

#distribution of the 13 airports:
import seaborn as sns
plt.figure(figsize=(15,6))
sns.countplot(x='ORIGIN_AIRPORT_ID', data=dep_pd, color = 'black') # countplot: ORD is the most frequent airport among the origin ones, then ATL (same as arrivals df)

plt.figure(figsize=(15,6))
sns.countplot(x='DEST_AIRPORT_ID', data=dep_pd, color = 'orange') # countplot: LAX appears most frequent among destination airports in our data, followed by ORD (same too)

# COMMAND ----------

plt.figure(figsize=(10,6))
plt.scatter(arr_pd['MONTH'],dep_pd['ARR_DELAY'],c='blue', edgecolors='none',alpha=0.5)
plt.xlabel("MONTH")
plt.ylabel("ARRIVAL DELAY")
#very little difference between months & arrival delay. Some months (Feb, June, Dec) have outlier arrival delays

plt.figure(figsize=(10,6))
plt.scatter(arr_pd['MONTH'],dep_pd['DEP_DELAY'],c='blue', edgecolors='none',alpha=0.5)
plt.xlabel("MONTH")
plt.ylabel("DEPARTURE DELAY")
#very little difference between months & arrival delay. some months have outlier delays too (Feb, June, Sept, Dec)

# COMMAND ----------

import json

dbutils.notebook.exit("Success")
