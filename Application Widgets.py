# Databricks notebook source
# MAGIC %md
# MAGIC ### Application Widgets

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

dbutils.widgets.removeAll()

dbutils.widgets.dropdown("00.Airport_Code", "JFK", ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"])
dbutils.widgets.text('01.training_start_date', "2018-01-01")
dbutils.widgets.text('02.training_end_date', "2019-03-15")
dbutils.widgets.text('03.inference_date', (dt.strptime(str(dbutils.widgets.get('02.training_end_date')), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))

airport_code = str(dbutils.widgets.get('00.Airport_Code'))
training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
  
print(airport_code,training_start_date,training_end_date,inference_date)

# COMMAND ----------

def abbrToID(data):
  """
  This function is designed to convert abbreviations to their airportID. For use with non-SparkDF datatypes.
  
  Input:  A  String, representing the airport Code
  Output: an Integer, representing the airportID
  """
  if data == "ATL":
    data = 10397
  elif data == "BOS":
    data = 10721
  elif data == "CLT":
    data =  11057
  elif data == "ORD":
    data =  13930
  elif data == "CVG":
    data =  11193
  elif data == "DFW":
    data =  11298
  elif data == "DEN":
    data =  11292
  elif data == "IAH":
    data =  12266
  elif data == "LAX":
    data =  12892
  elif data == "JFK":
    data =  12478
  elif data == "SFO":
    data =  14771
  elif data == "SEA":
    data =  14747
  elif data == "DCA":
    data =  11278
  else:
    data = 99999
  
  return(data)

# COMMAND ----------

import mlflow
from pprint import pprint
from mlflow.tracking import MlflowClient
import plotly.express as px
from datetime import timedelta, datetime
import numpy as np

client = MlflowClient()

# COMMAND ----------

# Select whether you would like to monitor arrival or depature models using "arr" or "dep":
model_type = "arr"

# Select model name:
  # group05_arr_sig_newfeatures
  # group05_dep_sig_newfeatures
  # g05_Arr_LinearModel
  # g05_Arr_LinearModel
model_name = "g05_Arr_LinearModel"

# COMMAND ----------

mlflow.set_experiment('/Users/rdugh@UR.Rochester.edu/flight_delay/dscc202_group05_experiment')

# COMMAND ----------

prod_version = None

# get the respective versions
for mv in client.search_model_versions(f"name='{model_name}'"):
  if dict(mv)['current_stage'] == 'Production':
    prod_version=dict(mv)['version']

if prod_version is not None:
  prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
  print("Production Model: ", prod_model)

# COMMAND ----------

# assemble dataset for forecasting

databaseName = GROUP_DBNAME
airport_id = abbrToID(airport_code)

fdf = spark.sql('''
  SELECT *
  FROM {0}.silver{1}_delta
  WHERE ORIGIN_AIRPORT_ID = {2} 
  AND
  FL_DATE BETWEEN '{3}' AND '{4}'
  '''.format(databaseName, model_type, airport_id, training_end_date, inference_date)
  )

# COMMAND ----------

# Forecast using the production model

df_forecast_production = fdf.toPandas().fillna(method='ffill').fillna(method='bfill')
df_forecast_production['model'] = 'Production'

if model_type == "arr":
  df_forecast_production['yhat'] = prod_model.predict(pd.DataFrame(df_forecast_production.drop(["model","FL_DATE","ARR_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))
elif model_type == "dep":
    df_forecast_production['yhat'] = prod_model.predict(pd.DataFrame(df_forecast_production.drop(["model","FL_DATE","DEP_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ARR_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))

# COMMAND ----------

df = df_forecast_production.reset_index()
df = df.sort_values(['hour'])

labels={
   "hour": "Forecast Time",
   "yhat": "Forecasted Delay",
   "model": "Model Stage"
   }

fig = px.line(df, x="hour", y="yhat", color='model', title=f"{airport_code} Delay Forecast", labels=labels)
fig.show()

# COMMAND ----------

import json

dbutils.notebook.exit("Success")
