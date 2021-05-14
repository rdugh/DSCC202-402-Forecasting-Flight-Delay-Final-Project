# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Monitoring

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
dbutils.widgets.text('04.promote_model', "No")

airport_code = str(dbutils.widgets.get('00.Airport_Code'))
training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
if dbutils.widgets.get("04.promote_model")=='Yes':
  promote_model = True
else:
  promote_model = False
  
print(airport_code,training_start_date,training_end_date,inference_date,promote_model)

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

# MAGIC %md
# MAGIC ## Forecast flight delay at selected airport

# COMMAND ----------

import mlflow
from pprint import pprint
from mlflow.tracking import MlflowClient
import plotly.express as px
from datetime import timedelta, datetime
import numpy as np
import pandas as pd

client = MlflowClient()

# COMMAND ----------

# Select whether you would like to monitor arrival or depature models using "arr" or "dep":
model_type = "arr"

# Select model names for staging and production:
  # group05_arr_sig_newfeatures
  # group05_dep_sig_newfeatures
  # g05_Arr_LinearModel
  # g05_Dep_LinearModel
model_name_staging = "group05_arr_sig_newfeatures"
model_name_production = "g05_Arr_LinearModel"

# COMMAND ----------

mlflow.set_experiment('/Users/rdugh@UR.Rochester.edu/flight_delay/dscc202_group05_experiment')

# COMMAND ----------

stage_version = None

# get the respective versions
for mv in client.search_model_versions(f"name='{model_name_staging}'"):
  if dict(mv)['current_stage'] == 'Staging':
    stage_version=dict(mv)['version']

if stage_version is not None:
  stage_model = mlflow.pyfunc.load_model(f"models:/{model_name_staging}/Staging")
  print("Staging Model: ", stage_model)

# COMMAND ----------

prod_version = None

# get the respective versions
for mv in client.search_model_versions(f"name='{model_name_production}'"):
  if dict(mv)['current_stage'] == 'Production':
    prod_version=dict(mv)['version']

if prod_version is not None:
  prod_model = mlflow.pyfunc.load_model(f"models:/{model_name_production}/Production")
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

# Forecast using the production and staging models

df_forecast_staging = fdf.toPandas().fillna(method='ffill').fillna(method='bfill')
df_forecast_staging['model'] = 'Staging'

if model_type == "arr":
  df_forecast_staging['yhat'] = stage_model.predict(pd.DataFrame(df_forecast_staging.drop(["model","FL_DATE","ARR_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))
if model_type == "dep":
  df_forecast_staging['yhat'] = stage_model.predict(pd.DataFrame(df_forecast_staging.drop(["model","FL_DATE","DEP_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ARR_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))

df_forecast_production = fdf.toPandas().fillna(method='ffill').fillna(method='bfill')
df_forecast_production['model'] = 'Production'

if model_type == "arr":
  df_forecast_production['yhat'] = prod_model.predict(pd.DataFrame(df_forecast_production.drop(["model","FL_DATE","ARR_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH','avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))
if model_type == "dep":
  df_forecast_production['yhat'] = prod_model.predict(pd.DataFrame(df_forecast_production.drop(["model","FL_DATE","DEP_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ARR_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH','avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))

# COMMAND ----------

df = pd.concat([df_forecast_staging,df_forecast_production]).reset_index()
df = df.sort_values(['hour'])

labels={
   "hour": "Forecast Time",
   "yhat": "Forecasted Delay",
   "model": "Model Stage"
   }

fig = px.line(df, x="hour", y="yhat", color='model', title=f"{airport_code} Delay Forecast by Model Stage", labels=labels)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring the model performance in training period

# COMMAND ----------

stage_version = None

# get the respective versions
for mv in client.search_model_versions(f"name='{model_name_staging}'"):
  if dict(mv)['current_stage'] == 'Staging':
    stage_version=dict(mv)['version']

if stage_version is not None:
  # load the training data assocaited with the staging model
  stage_model = mlflow.pyfunc.load_model(f"models:/{model_name_staging}/Staging")
  print("Production Model: ", stage_model)
  sdf = spark.sql(f"""SELECT * FROM {databaseName}.silver{model_type}_delta WHERE ORIGIN_AIRPORT_ID = {airport_id} AND
  FL_DATE BETWEEN '{training_start_date}' AND '{training_end_date}';""").toPandas()

# COMMAND ----------

prod_version = None

# get the respective versions
for mv in client.search_model_versions(f"name='{model_name_production}'"):
  if dict(mv)['current_stage'] == 'Production':
    prod_version=dict(mv)['version']

if prod_version is not None:
  # load the training data associated with the production model
  prod_model = mlflow.pyfunc.load_model(f"models:/{model_name_production}/Production")
  print("Production Model: ", prod_model)
  pdf = spark.sql(f"""SELECT * FROM {databaseName}.silver{model_type}_delta WHERE ORIGIN_AIRPORT_ID = {airport_id} AND
  FL_DATE BETWEEN '{training_start_date}' AND '{training_end_date}';""").toPandas()

# COMMAND ----------

# assemble dataset for training

airport_id = abbrToID(airport_code)

train_df = spark.sql('''
  SELECT *
  FROM {0}.silver{1}_delta
  WHERE ORIGIN_AIRPORT_ID = {2} 
  AND
  FL_DATE BETWEEN '{3}' AND '{4}'
  '''.format(databaseName, model_type, airport_id, training_start_date, training_end_date)
  )

# COMMAND ----------

# Train using the production and staging models

df_training_staging = train_df.toPandas().fillna(method='ffill').fillna(method='bfill')
df_training_staging['model'] = 'Staging'

if model_type == "arr":
  df_training_staging['yhat'] = stage_model.predict(pd.DataFrame(df_training_staging.drop(["model","FL_DATE","ARR_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))
if model_type == "dep":
  df_training_staging['yhat'] = stage_model.predict(pd.DataFrame(df_training_staging.drop(["model","FL_DATE","DEP_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ARR_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH', 'avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))

df_training_production = train_df.toPandas().fillna(method='ffill').fillna(method='bfill')
df_training_production['model'] = 'Production'

if model_type == "arr":
  df_training_production['yhat'] = prod_model.predict(pd.DataFrame(df_training_production.drop(["model","FL_DATE","ARR_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH','avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))
if model_type == "dep":
  df_training_production['yhat'] = prod_model.predict(pd.DataFrame(df_training_production.drop(["model","FL_DATE","DEP_DELAY"], axis=1).values, columns=['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ARR_DELAY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'hour', 'QUARTER', 'DAY_OF_MONTH','avg_temp_f', 'tot_precip_mm', 'avg_wnd_mps', 'avg_vis_m', 'avg_slp_hpa', 'avg_dewpt_f'], dtype=np.int32))

# COMMAND ----------

sdf['stage']="staging"
if model_type == "arr":
  sdf['residual']=sdf['ARR_DELAY']-df_training_staging['yhat']
if model_type == "dep":
  sdf['residual']=sdf['DEP_DELAY']-df_training_staging['yhat']
sdf['yhat']=df_training_staging['yhat']

pdf['stage']="prod"
if model_type == "arr":
  pdf['residual']=pdf['ARR_DELAY']-df_training_production['yhat']
if model_type == "dep":
  pdf['residual']=pdf['DEP_DELAY']-df_training_production['yhat']
pdf['yhat']=df_training_production['yhat']

df=pd.concat([sdf,pdf])

# COMMAND ----------

fig = px.scatter(
    df, x='yhat', y='residual',
    marginal_y='violin',
    color='stage', trendline='ols',
    title=f"{airport_code} Delay Forecast Model Performance Comparison for Training Period"
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Tensorflow Validation Library
# MAGIC - check schema between the training and serving periods of time
# MAGIC - check for data drift and skew between training and serving

# COMMAND ----------

from sklearn.model_selection import train_test_split
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

stats_train=tfdv.generate_statistics_from_dataframe(dataframe=train_df.toPandas())
stats_serve=tfdv.generate_statistics_from_dataframe(dataframe=fdf.toPandas())

schema = tfdv.infer_schema(statistics=stats_train)
tfdv.display_schema(schema=schema)

# COMMAND ----------

# Compare evaluation data with training data
displayHTML(get_statistics_html(lhs_statistics=stats_serve, rhs_statistics=stats_train,
                          lhs_name='SERVE_DATASET', rhs_name='TRAIN_DATASET'))

# COMMAND ----------

anomalies = tfdv.validate_statistics(statistics=stats_serve, schema=schema)
tfdv.display_anomalies(anomalies)

# COMMAND ----------

# Add skew and drift comparators
temp_f = tfdv.get_feature(schema, 'avg_temp_f')
temp_f.skew_comparator.jensen_shannon_divergence.threshold = 0
temp_f.drift_comparator.jensen_shannon_divergence.threshold = 0

precip_mm = tfdv.get_feature(schema, 'tot_precip_mm')
precip_mm.skew_comparator.jensen_shannon_divergence.threshold = 0
precip_mm.drift_comparator.jensen_shannon_divergence.threshold = 0

_anomalies = tfdv.validate_statistics(stats_train, schema, serving_statistics=stats_serve)

tfdv.display_anomalies(_anomalies)

# COMMAND ----------

hour = tfdv.get_feature(schema, 'hour')
hour.skew_comparator.jensen_shannon_divergence.threshold = 0
hour.drift_comparator.jensen_shannon_divergence.threshold = 0

dayofweek = tfdv.get_feature(schema, 'DAY_OF_WEEK')
dayofweek.skew_comparator.jensen_shannon_divergence.threshold = 0
dayofweek.drift_comparator.jensen_shannon_divergence.threshold = 0

_anomalies = tfdv.validate_statistics(stats_train, schema, serving_statistics=stats_serve)

tfdv.display_anomalies(_anomalies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote model if selected

# COMMAND ----------

# Criteria for retraining and promotion to production:
  # 1.) View the "Delay Forecast Model Performance Comparison for Training Period" graph
  # 2.) Analyze the graph to determine which model stage has a higher density of residuals around the horizontal line at residual=0. Note: The residuals are the difference between model predicted values and ground truth values.
  # 3a.) If the staging model observes a higher density of residuals around the horizontal line at residual=0 than the production model, then promote the current staging model to production and archive the old production model.
  # OR
  # 3b.) If the staging model does NOT observe a higher density of residuals around horizontal line at residual=0 than the production model, then retrain staging model and repeat the above process.

# COMMAND ----------

# promote staging to production
if promote_model and stage_version is not None and prod_version is not None:

  # Archive the production model
  client.transition_model_version_stage(
      name=model_name,
      version=prod_version,
      stage="Archived"
  )

  # Staging --> Production
  client.transition_model_version_stage(
      name=model_name,
      version=stage_version,
      stage="Production"
  )

# COMMAND ----------

import json

dbutils.notebook.exit("Success")