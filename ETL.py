# Databricks notebook source
# MAGIC %md
# MAGIC CLASS_DATA_PATH	/mnt/dscc202-datasets
# MAGIC 
# MAGIC GROUP_DATA_PATH	/mnt/dscc202-group05-datasets
# MAGIC 
# MAGIC BASE_DELTA_PATH	/mnt/dscc202-group05-datasets/flightdelay/tables/
# MAGIC 
# MAGIC GROUP_DBNAME	dscc202_group05_db

# COMMAND ----------

# MAGIC %run /Users/imanly@u.rochester.edu/flight_delay/includes/utilities

# COMMAND ----------

# MAGIC %run /Users/imanly@u.rochester.edu/flight_delay/includes/configuration

# COMMAND ----------

dbutils.fs.rm('/mnt/dscc202-group05-datasets/flightdelay/tables/', recurse = True) #CALL THIS TO UPDATE A DATA
#dbutils.fs.ls('/mnt/dscc202-group05-datasets/flightdelay/tables/')

# COMMAND ----------

import pandas as pd
import json
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import time
from pyspark.sql import functions as F


dbutils.fs.rm(BASE_DELTA_PATH, recurse = True)


# COMMAND ----------

# DBTITLE 1,Start of air traffic data
# MAGIC %sql
# MAGIC --Example Air traffic data
# MAGIC select * from  dscc202_db.bronze_air_traffic limit 10;

# COMMAND ----------

airTrafficDF = (spark
               .readStream
               .table('dscc202_db.bronze_air_traffic')
        )

# COMMAND ----------


trafficDF = (airTrafficDF.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "FL_DATE", "CRS_ARR_TIME", "CRS_DEP_TIME", "DEP_DELAY", "ARR_DELAY", "ORIGIN_CITY_NAME", "DEST_CITY_NAME", "ORIGIN", "DEST", "DAY_OF_WEEK", "MONTH", "YEAR", "QUARTER", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER", "DEP_TIME")
.where("FL_DATE > '2014-01-01' AND FL_DATE < '2020-01-01'") #Date selection
.where("DEST == 'JFK' OR DEST == 'SEA' OR DEST == 'BOS' OR DEST == 'ATL' OR DEST == 'LAX' OR DEST == 'SFO' OR DEST == 'DEN' OR DEST == 'DFW' OR DEST == 'ORD' OR DEST == 'CVG' OR DEST == 'CLT' OR DEST == 'DCA' OR DEST == 'IAH'") #Only interested in flights going between these 13 airports, so filter dest and origin
.where("ORIGIN == 'JFK' OR ORIGIN == 'SEA' OR ORIGIN == 'BOS' OR ORIGIN == 'ATL' OR ORIGIN == 'LAX' OR ORIGIN == 'SFO' OR ORIGIN == 'DEN' OR ORIGIN == 'DFW' OR ORIGIN == 'ORD' OR ORIGIN == 'CVG' OR ORIGIN == 'CLT' OR ORIGIN == 'DCA' OR ORIGIN == 'IAH'")

.withColumn("hour", round('DEP_TIME', 2) )
.withColumn("dep_time", F.expr("make_timestamp(YEAR, MONTH, DAY_OF_MONTH, substring(CRS_DEP_TIME, 1,2),0,0)")) #Create timestamps
.withColumn("arrival_time", F.expr("make_timestamp(YEAR, MONTH, DAY_OF_MONTH, substring(CRS_ARR_TIME, 1,2),0,0)"))
.where(col("dep_time").isNotNull())
.where(col("arrival_time").isNotNull())
       
        )

# COMMAND ----------

# DBTITLE 1,Write Air Traffic Stream


outputPathTraffic = BASE_DELTA_PATH + 'air_traffic/flights'
checkpointPathAir = BASE_DELTA_PATH + "flightCheckpoints"
                  

airTrafficStream = (trafficDF

  .writeStream
  .format("delta")
  .option("mode","append")
  .option('checkpointLocation', checkpointPathAir)
  .trigger(once = True)
  .queryName("stream_traffic")
  .start(outputPathTraffic)
)


# COMMAND ----------

airTrafficStream.awaitTermination()

# COMMAND ----------

spark.sql("""
   DROP TABLE IF EXISTS {}.air_traffic_delta_bronze;
   """.format(GROUP_DBNAME))

# COMMAND ----------

spark.sql("""
   CREATE TABLE IF NOT EXISTS {}.air_traffic_delta_bronze
   USING DELTA 
   LOCATION '{}'
  """.format(GROUP_DBNAME, outputPathTraffic))


# COMMAND ----------

# DBTITLE 1,Start of weather data
# MAGIC %sql
# MAGIC --Example weather data
# MAGIC select * from  dscc202_db.bronze_weather WHERE Date IS NOT NULL limit 10;

# COMMAND ----------

#dbutils.fs.rm(outputPathWeather, recurse = True)

# COMMAND ----------

# DBTITLE 1,Create weather stream
weatherDF = (spark
               .readStream
               .table('dscc202_db.bronze_weather'))

# COMMAND ----------

#dbutils.fs.ls(BASE_DELTA_PATH)

# COMMAND ----------


testDF=(spark
        .readStream
        .table('dscc202_db.bronze_weather')
        .select("DATE", 'LATITUDE', 'LONGITUDE', 'TMP', 'VIS','ELEVATION', 'CIG', 'DEW', 'SLP', 'WND', 'AA1', 'CALL_SIGN')
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
        .withColumn("call_sign", substring("CALL_SIGN", -4, 3))
        .where("call_sign != '999' and (REPORT_TYPE='FM-15' or REPORT_TYPE='FM-13') and precip_qual='5' and temp_qual='5'")
        .groupby("time", "call_sign")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'), \
             mean('wnd_mps').alias('avg_wnd_mps'), \
             mean('vis_m').alias('avg_vis_m'),  \
             mean('slp_hpa').alias('avg_slp_hpa'),  \
             mean('dew_pt_f').alias('avg_dewpt_f'), \
            )
        
        
   )

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Write weather stream
outputPathWeather = BASE_DELTA_PATH + 'weather/weather'

checkpointPath = BASE_DELTA_PATH + 'weatherCheckpoints'

dbutils.fs.rm(outputPathWeather, recurse = True)
dbutils.fs.rm(checkpointPath)

#dbutils.fs.mkdirs(outputPathWeather)

weatherStream = (testDF
  .writeStream
  .format("delta")
  .outputMode("complete")
  .trigger(once = True)
  .option('checkpointLocation', checkpointPath) 
  .queryName("stream_weather")
  .start(outputPathWeather)
)

# COMMAND ----------

weatherStream.awaitTermination()

# COMMAND ----------

spark.sql("""
   DROP TABLE IF EXISTS {}.weather_delta_bronze;
   """.format(GROUP_DBNAME))

# COMMAND ----------

#dbutils.fs.ls(outputPathWeather)
#display(spark.read.format("DELTA").load(outputPathWeather))

# COMMAND ----------

spark.sql("""
   CREATE TABLE IF NOT EXISTS {}.weather_delta_bronze
   USING DELTA 
   LOCATION '{}' 
  """.format(GROUP_DBNAME, outputPathWeather))

# COMMAND ----------

# DBTITLE 1,Silver View
#2 views, 1 airline 1 weather, join on time/airport ID, write out to a silver delta table

# COMMAND ----------

airBronzeDF = spark.sql("""
  SELECT *
  FROM {}.air_traffic_delta_bronze
  """.format(GROUP_DBNAME))



weatherBronzeDF = spark.sql("""
  SELECT *
  FROM {}.weather_delta_bronze
  """.format(GROUP_DBNAME))

# COMMAND ----------

airBronzeDF.createOrReplaceTempView("airTempView")
weatherBronzeDF.createOrReplaceTempView("weatherTempView")

#pull in arrival time
display(weatherBronzeDF)

# COMMAND ----------

cond1 = [(airBronzeDF.dep_time == weatherBronzeDF.time) & (weatherBronzeDF.call_sign == airBronzeDF.ORIGIN)]
cond2 = [(airBronzeDF.arrival_time == weatherBronzeDF.time) & (airBronzeDF.DEST == weatherBronzeDF.call_sign)]


silverDepDF = airBronzeDF.join(weatherBronzeDF, cond1)


silverDepDF = silverDepDF.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_DELAY", "ARR_DELAY", "FL_DATE", "DAY_OF_WEEK", "MONTH", "YEAR", "hour", "QUARTER", "DAY_OF_MONTH", "avg_temp_f", "tot_precip_mm", "avg_wnd_mps", "avg_vis_m", "avg_slp_hpa", "avg_dewpt_f")
silverDepDF.na.drop("all")

outputPathSilverDep = BASE_DELTA_PATH + 'silverDep'
checkpointPathSilver = BASE_DELTA_PATH + 'silverDepChecks'

display(silverDepDF)

# COMMAND ----------

(silverDepDF.write
 .format("delta")
 .mode("append")
 .save(outputPathSilverDep)
)

# COMMAND ----------

spark.sql("""
   DROP TABLE IF EXISTS {}.silverDep_delta;
   """.format(GROUP_DBNAME))

# COMMAND ----------

spark.sql("""
   CREATE TABLE IF NOT EXISTS {}.silverDep_delta
   USING DELTA 
   LOCATION '{}' 
  """.format(GROUP_DBNAME, outputPathSilverDep))

# COMMAND ----------

cond1 = [(airBronzeDF.dep_time == weatherBronzeDF.time) & (weatherBronzeDF.call_sign == airBronzeDF.ORIGIN)]
cond2 = [(airBronzeDF.arrival_time == weatherBronzeDF.time) & (airBronzeDF.DEST == weatherBronzeDF.call_sign)]


silverArrDF = airBronzeDF.join(weatherBronzeDF, cond2)

silverArrDF = silverArrDF.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "DEP_DELAY", "ARR_DELAY", "FL_DATE", "DAY_OF_WEEK", "MONTH", "YEAR", "hour", "QUARTER", "DAY_OF_MONTH", "avg_temp_f", "tot_precip_mm", "avg_wnd_mps", "avg_vis_m", "avg_slp_hpa", "avg_dewpt_f")
silverArrDF = silverArrDF.na.drop("all")

outputPathSilverArr = BASE_DELTA_PATH + 'silverArr'
checkpointPathSilver = BASE_DELTA_PATH + 'silverArrChecks'

display(silverArrDF)

# COMMAND ----------

(silverArrDF.write
 .format("delta")
 .mode("append")
 .save(outputPathSilverArr)
)

# COMMAND ----------

spark.sql("""
   DROP TABLE IF EXISTS {}.silverArr_delta;
   """.format(GROUP_DBNAME))

# COMMAND ----------

spark.sql("""
   CREATE TABLE IF NOT EXISTS {}.silverArr_delta
   USING DELTA 
   LOCATION '{}' 
  """.format(GROUP_DBNAME, outputPathSilverArr))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Quit Streams
for s in spark.streams.active:
  s.stop()

# COMMAND ----------

import json

dbutils.notebook.exit("Success")
