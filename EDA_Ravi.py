# Databricks notebook source
print("IN EDA")

# COMMAND ----------



# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

import pandas as pd
import json
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import time

def untilStreamIsReady(namedStream: str, progressions: int = 3) -> bool:
    queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
    while len(queries) == 0 or len(queries[0].recentProgress) < progressions:
        time.sleep(5)
        queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
    print("The stream {} is active and ready.".format(namedStream))
    return True

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC --Example Air traffic data
# MAGIC select * from  dscc202_db.bronze_air_traffic limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC --Example weather data
# MAGIC select * from  dscc202_db.bronze_weather WHERE Date IS NOT NULL limit 10;

# COMMAND ----------

databaseName = 'dscc202_group05_db'

airDF = spark.sql("""
  SELECT *
  FROM {}.air_traffic_delta_bronze
  """.format(databaseName))



weatherDF = spark.sql("""
  SELECT *
  FROM {}.weather_delta_bronze
  """.format(databaseName))

# COMMAND ----------

display(airDF)

# COMMAND ----------

# Shape of data frame
print(f"Airline delta bronze has {airDF.count()} rows and {len(airDF.columns)} columns")
print(f"Weather delta bronze has {weatherDF.count()} rows and {len(weatherDF.columns)} columns")

# COMMAND ----------

raviairDF = (spark
               .readStream
               .table('dscc202_db.bronze_air_traffic'))



# COMMAND ----------

raviairDF.printSchema()

# COMMAND ----------

display(raviairDF)

# COMMAND ----------

databaseName = 'dscc202_group05_db'



raviairDF_1 = (raviairDF.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "DEP_TIME", "DEP_DELAY", "ARR_DELAY", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "ORIGIN", "DEST", "DAY_OF_WEEK", "MONTH", "YEAR", "QUARTER", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER","CANCELLED","CANCELLATION_CODE","AIR_TIME","DISTANCE","DISTANCE_GROUP","CARRIER_DELAY","WEATHER_DELAY")
.where("FL_DATE > '2014-01-01' AND FL_DATE < '2020-01-01'") #Date selection
.where("DEST == 'JFK' OR DEST == 'SEA' OR DEST == 'BOS' OR DEST == 'ATL' OR DEST == 'LAX' OR DEST == 'SFO' OR DEST == 'DEN' OR DEST == 'DFW' OR DEST == 'ORD' OR DEST == 'CVG' OR DEST == 'CLT' OR DEST == 'DCA' OR DEST == 'IAH'") #Only interested in flights going between these 13 airports, so filter dest and origin
.where("ORIGIN == 'JFK' OR ORIGIN == 'SEA' OR ORIGIN == 'BOS' OR ORIGIN == 'ATL' OR ORIGIN == 'LAX' OR ORIGIN == 'SFO' OR ORIGIN == 'DEN' OR ORIGIN == 'DFW' OR ORIGIN == 'ORD' OR ORIGIN == 'CVG' OR ORIGIN == 'CLT' OR ORIGIN == 'DCA' OR ORIGIN == 'IAH'")
         
        )

# COMMAND ----------

#raviairDF_1 is bronze delta table
AirPath=BASE_DELTA_PATH+"ravi_airlines_table"
AirCheckpoint=BASE_DELTA_PATH+"/_ravi_airlines_checkpoint"
raviairDF_1=(raviairDF_1
            .writeStream
            .option('checkpointLocation',AirCheckpoint)
            .format("delta")
            .queryName("ravi_air")
            .start(AirPath))

# COMMAND ----------

raviairDF_1.awaitTermination(1000000)
#AirlineSampleData=raviairDF_1.sample(0.2,100)
#AirlineSampleData=raviairDF_1.toPandas()
#raviairDF_1.describe()


# COMMAND ----------

raviairDF_1.stop()

# COMMAND ----------


       
display( dbutils.fs.ls(AirPath) )

# COMMAND ----------

#raviairDF_2 is dataframe based on the bronze delta table
raviairDF_2 = spark.read.format("delta").load(AirPath)

# COMMAND ----------

#RaviSample is a small sample of raviairDF_2. This will be used to view a small subset to select the columns/features to select/drop.
RaviSample=raviairDF_2.sample(0.1,100)
RaviSample=raviairDF_2.toPandas()
RaviSample.describe()

# COMMAND ----------

United=RaviSample.loc[RaviSample['OP_UNIQUE_CARRIER']=="DL"]

# COMMAND ----------

from pathlib import Path

import numpy as np
import pandas as pd
import requests

import pandas_profiling
from pandas_profiling.utils.cache import cache_file

displayHTML(pandas_profiling.ProfileReport(United).html)


# COMMAND ----------

# MAGIC %md ## EDA

# COMMAND ----------

display(raviairDF_2.groupBy('ORIGIN').agg(avg('DEP_DELAY')))

# COMMAND ----------

display(raviairDF_2.groupBy('DEST').agg(avg('DEP_DELAY')))

# COMMAND ----------

display(raviairDF_2)

# COMMAND ----------


#table1=RaviSample[['DAY_OF_WEEK','ORIGIN']].groupby('DAY_OF_WEEK').count()

# COMMAND ----------

#display(table1)
#display(table1.orderBy('DAY_OF_WEEK',ascending=True))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AVG(ARR_DELAY) AS avg_arr_delay,AVG(DEP_DELAY) AS avg_dep_delay, DAY_OF_WEEK
# MAGIC FROM delta.`/mnt/dscc202-group05-datasets/flightdelay/tables/ravi_airlines_table`
# MAGIC GROUP BY DAY_OF_WEEK
# MAGIC SORT BY DAY_OF_WEEK

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT AVG(ARR_DELAY) AS avg_arr_delay,AVG(DEP_DELAY) AS avg_dep_delay, ORIGIN
# MAGIC FROM delta.`/mnt/dscc202-group05-datasets/flightdelay/tables/ravi_airlines_table`
# MAGIC GROUP BY ORIGIN
# MAGIC SORT BY ORIGIN

# COMMAND ----------

from pyspark.sql.functions import *

 
UniqueAirportsOrigin = raviairDF_2.groupBy('OP_UNIQUE_CARRIER').agg(approx_count_distinct("ORIGIN_AIRPORT_ID").alias("Origin_Airports")).sort("Origin_Airports")
 
UniqueAirportsDest=raviairDF_2.groupBy('OP_UNIQUE_CARRIER').agg(approx_count_distinct("DEST_AIRPORT_ID").alias("Dest_Airports")).sort("Dest_Airports")

# COMMAND ----------

display(UniqueAirportsOrigin)

# COMMAND ----------

display(UniqueAirportsDest)

# COMMAND ----------

from pyspark.sql.functions import *

 
UniqueAirportsOriginDelay = raviairDF_2.groupBy("ORIGIN").agg(approx_count_distinct("WEATHER_DELAY").alias("Origin_Airports_Delays")).sort("Origin_Airports_Delays")
 
UniqueAirportsDestDelay=raviairDF_2.groupBy("DEST").agg(approx_count_distinct("WEATHER_DELAY").alias("Dest_Airports_Delays")).sort("Dest_Airports_Delays")


# COMMAND ----------

display(UniqueAirportsOriginDelay)

# COMMAND ----------

display(UniqueAirportsDestDelay)

# COMMAND ----------

print(raviairDF_2.count())

# COMMAND ----------

raviairDF_2.printSchema()

# COMMAND ----------

raviweatherDF = (spark
               .readStream
               .table('dscc202_db.bronze_weather'))

# COMMAND ----------

raviweatherDF.printSchema()

# COMMAND ----------



# COMMAND ----------

databaseName = 'dscc202_group05_db'

raviweatherDF_1 = (raviweatherDF.select("DATE", 'LATITUDE', 'LONGITUDE', 'TMP', 'VIS','ELEVATION', 'CIG', 'DEW', 'SLP', 'WND', 'AA1', 'CALL_SIGN')
        .where("DATE > '2014-01-01' AND DATE < '2020-01-01'")
        .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
        .withColumn('temp_qual', split(col('TMP'),",")[1])
        .withColumn('wnd_deg', split(col('WND'),",")[0])
        .withColumn('wnd_1', split(col('WND'),",")[1])
        .withColumn('wnd_2', split(col('WND'),",")[2])
        .withColumn('wnd_mps', split(col('WND'),",")[3]/10)
        .withColumn('wnd_4', split(col('WND'),",")[4])
        .withColumn('vis_m', split(col('VIS'),",")[0])
        .withColumn('vis_1', split(col('VIS'),",")[1])
        .withColumn('vis_2', split(col('VIS'),",")[2])
        .withColumn('vis_3', split(col('VIS'),",")[3])
        .withColumn('dew_pt_f', split(col('DEW'),",")[0]*9/50+32)
        .withColumn('dew_1', split(col('DEW'),",")[1])
        .withColumn('slp_hpa', split(col('SLP'),",")[0]/10)
        .withColumn('slp_1', split(col('SLP'),",")[1])
        .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
        .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
        .withColumn('precip_cond', split(col('AA1'),",")[2])
        .withColumn('precip_qual', split(col('AA1'),",")[3])
        .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
        .withColumn("time", date_trunc('hour', "DATE"))
        .withColumn("call_sign", substring("CALL_SIGN", 1, 3))
        .where("call_sign != '999' and (REPORT_TYPE='FM-15' or REPORT_TYPE='FM-13') and precip_qual='5' and temp_qual='5'")
        .groupby("time", "call_sign")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'), \
             mean('wnd_mps').alias('avg_wnd_mps'), \
             mean('vis_m').alias('avg_vis_m'),  \
             mean('slp_hpa').alias('avg_slp_hpa'),  \
             mean('dew_pt_f').alias('avg_dewpt_f'), \
            )                 )

# COMMAND ----------


raviweatherDF_2.printSchema()

# COMMAND ----------



# COMMAND ----------

#raviweatherDF_1 is bronze delta table for weather
WeatherPath=BASE_DELTA_PATH+"ravi_weather_table"
WeatherCheckpoint=BASE_DELTA_PATH+"/_ravi_weather_checkpoint"
raviweatherDF_1=(raviweatherDF_1
            .writeStream
            .option('checkpointLocation',WeatherCheckpoint)
            .format("delta")
            .outputMode("complete")
            .queryName("ravi_weather")
            .start(WeatherPath))

# COMMAND ----------

raviweatherDF_1.awaitTermination(1800000)

# COMMAND ----------

raviweatherDF_1.stop()

# COMMAND ----------

display( dbutils.fs.ls(WeatherPath) )

# COMMAND ----------

#raviairDF_2 is dataframe based on the bronze delta table
raviweatherDF_2 = spark.read.format("delta").load(WeatherPath)

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofweek, minute, second
 
raviweatherDF_3 = (raviweatherDF_2.withColumn("year", year(col("time")))
  .withColumn("month", month(col("time")))
  .withColumn("dayofmonth", dayofmonth(col("time")))
  .withColumn("hour", hour(col("time")))
  .withColumn("minute", minute(col("time")))
  .withColumn("second", second(col("time")))              
)
display(raviweatherDF_3)

# COMMAND ----------

raviweatherDF_3.printSchema()

# COMMAND ----------

raviairDF_2.printSchema()

# COMMAND ----------



# COMMAND ----------


 

# COMMAND ----------



# COMMAND ----------

#dbutils.notebook.exit("Success")