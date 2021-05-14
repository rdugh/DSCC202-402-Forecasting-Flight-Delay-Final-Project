# Databricks notebook source
# MAGIC %md #Gradient Boosted Trees Regression Model

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

databaseName = 'dscc202_group05_db'
silverDepDF = (spark
               .readStream
               .table('dscc202_group05_db.silverdep_delta'))


# COMMAND ----------

silverDepDF_1 = spark.sql("""
  SELECT *
  FROM {}.silverdep_delta
  """.format(GROUP_DBNAME))

silverDepDF_1.cache()

# COMMAND ----------

silverArrDF_1 = spark.sql("""
  SELECT *
  FROM {}.silverarr_delta
  """.format(GROUP_DBNAME))

silverArrDF_1.cache()
display(silverArrDF_1)



# COMMAND ----------

silverDepDF.printSchema()

# COMMAND ----------

print("The dep. dataset has %d rows." % silverDepDF_1.count())
print("The arr. dataset has %d rows." % silverArrDF_1.count())


# COMMAND ----------

display(silverDepDF_1)

# COMMAND ----------

from pyspark.sql.functions import col
silverDepDF_1 = silverDepDF_1.filter(col('DEP_DELAY').isNotNull())
silverDepDF_1 = silverDepDF_1.filter(col('ARR_DELAY').isNotNull())

# COMMAND ----------

silverArrDF_1 = silverArrDF_1.filter(col('DEP_DELAY').isNotNull())
silverArrDF_1 = silverArrDF_1.filter(col('ARR_DELAY').isNotNull())

# COMMAND ----------

from pyspark.sql.functions import col,sum
silverDepDF_1.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in silverDepDF_1.columns)).show()

# COMMAND ----------

from pyspark.sql.functions import col,sum
silverArrDF_1.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in silverArrDF_1.columns)).show()

# COMMAND ----------

#https://datascience.stackexchange.com/questions/72408/remove-all-columns-where-the-entire-column-is-null
def drop_null_columns(df):
  
    """
    This function drops columns containing all null values.
    :param df: A PySpark DataFrame
    """
    
    null_counts = df.select([sqlf.count(sqlf.when(sqlf.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    to_drop = [k for k, v in null_counts.items() if v >= df.count()]
    df = df.drop(*to_drop)
    
    return df

# COMMAND ----------

import pandas as pd
import numpy as np 
import pyspark.sql.functions as sqlf
silverDepDF_1 = drop_null_columns(silverDepDF_1)
silverArrDF_1 = drop_null_columns(silverArrDF_1)

# COMMAND ----------

silverDepDF_1.na.drop()
print("The Dep. dataset has %d rows." % silverDepDF_1.count())
silverArrDF_1.na.drop()
print("The Arr. dataset has %d rows." % silverArrDF_1.count())

# COMMAND ----------

silverDepDF_1=silverDepDF_1.drop('FL_DATE')
silverArrDF_1=silverArrDF_1.drop('FL_DATE')

# COMMAND ----------

print("The dep. dataset has %d rows." % silverDepDF_1.count())
print("The arr. dataset has %d rows." % silverArrDF_1.count())

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer

featuresCols = silverDepDF_1.columns
featuresCols.remove('DEP_DELAY')
# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
#vectorAssembler=vectorAssembler.transform(silverDepDF_1.na.drop).show(20)
# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# The next step is to define the model training stage of the pipeline. 
# The following command defines a GBTRegressor model that takes an input column "features" by default and learns to predict the labels in the "DEP_DELAY" column. 
gbt = GBTRegressor(labelCol="DEP_DELAY")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing. Passing a seed for deterministic behavior
train, test = silverDepDF_1.randomSplit([0.7, 0.3], seed = 0)
print("Departure Delay: There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(train.drop("DEP_DELAY"), train.select("DEP_DELAY"))

# COMMAND ----------

signature

# COMMAND ----------

display(silverDepDF_1)

# COMMAND ----------

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
# turn on autologging
mlflow.autolog(log_input_examples=True,log_model_signatures=True,log_models=True)

experiment_name = '/Users/rdugh@UR.Rochester.edu/flight_delay/dscc202_group05_experiment'
mlflow.set_experiment(experiment_name)
with mlflow.start_run(experiment_id=70892):
#with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
  pipelineModel = pipeline.fit(train)
  
  # Log the best model.
  
  mlflow.spark.log_model(spark_model=pipelineModel, artifact_path='best-model-dep-newfeatures',signature=signature, input_example=train.drop("DEP_DELAY").toPandas().head()) 
  
 


# COMMAND ----------

predictions_dep = pipelineModel.transform(test)

# COMMAND ----------

rmse = evaluator.evaluate(predictions_dep)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

display(predictions_dep.select("DEP_DELAY", "prediction", *featuresCols))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer

featuresCols = silverArrDF_1.columns
featuresCols.remove('ARR_DELAY')
# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
#vectorAssembler=vectorAssembler.transform(silverDepDF_1.na.drop).show(20)
# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)



# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# The next step is to define the model training stage of the pipeline. 
# The following command defines a GBTRegressor model that takes an input column "features" by default and learns to predict the labels in the "DEP_DELAY" column. 
gbt = GBTRegressor(labelCol="ARR_DELAY")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing. Passing a seed for deterministic behavior
train, test = silverArrDF_1.randomSplit([0.7, 0.3], seed = 0)
print("Arrival Delay: There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(train.drop("ARR_DELAY"), train.select("ARR_DELAY"))

# COMMAND ----------

signature

# COMMAND ----------

import mlflow
import mlflow.spark


# turn on autologging
mlflow.autolog(log_input_examples=True,log_model_signatures=True,log_models=True)
experiment_name = '/Users/rdugh@UR.Rochester.edu/flight_delay/dscc202_group05_experiment'
mlflow.set_experiment(experiment_name)
with mlflow.start_run(experiment_id=70892):
  pipelineModel = pipeline.fit(train)
  
  # Log the best model.
  #mlflow.set_experiment(experiment_name: '/databricks/mlflow-tracking/70892/')
  mlflow.spark.log_model(spark_model=pipelineModel, artifact_path='best-model-arr-newfeatures', signature=signature, input_example=train.drop("ARR_DELAY").toPandas().head()) 
  


# COMMAND ----------

predictions_arr = pipelineModel.transform(test)

# COMMAND ----------

rmse = evaluator.evaluate(predictions_arr)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

display(predictions_arr.select("ARR_DELAY", "prediction", *featuresCols))

# COMMAND ----------

# MAGIC %md
# MAGIC #Baseline Model

# COMMAND ----------

import pandas as pd 
from datetime import datetime
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
import pyspark.sql.functions as sqlf
from pyspark.sql.functions import col, to_date

# COMMAND ----------

# Load Departure Dataframe
silverArrDF_2 = spark.sql("""
  SELECT *
  FROM {}.silverarr_delta
  """.format(GROUP_DBNAME))

# Clean Arrival Data
silverArrDF_2 = silverArrDF_2.filter(col('ARR_DELAY').isNotNull())
silverArrDF_2 = silverArrDF_2.filter(col('DEP_DELAY').isNotNull())
silverArrDF_2 = silverArrDF_2.drop('FL_DATE')
silverArrDF_2 = drop_null_columns(silverArrDF_2)

#Transform into Pandas Dataframe
silverArrDF_2 = silverArrDF_2.toPandas()
display(silverArrDF_2.head())

# COMMAND ----------

# Load Departure Dataframe
silverDepDF_2 = spark.sql("""
  SELECT *
  FROM {}.silverdep_delta
  """.format(GROUP_DBNAME))

# Clean Departure Data
silverDepDF_2 = silverDepDF_2.filter(col('ARR_DELAY').isNotNull())
silverDepDF_2 = silverDepDF_2.filter(col('DEP_DELAY').isNotNull())
silverDepDF_2 = silverDepDF_2.drop('FL_DATE')
#silverDepDF_1 = idToAbbr(silverDepDF_1)
silverDepDF_2 = drop_null_columns(silverDepDF_2)

#Transform into Pandas Dataframe
silverDepDF_2 = silverDepDF_2.toPandas()
display(silverDepDF_2.head())

# COMMAND ----------

# Split datasets into features and targets
arrivalFeats  = silverArrDF_2.drop('ARR_DELAY', axis=1)
arrivalTarget = pd.Series.to_frame(silverArrDF_2["ARR_DELAY"])

departFeats  = silverDepDF_2.drop('DEP_DELAY', axis=1)
departTarget = pd.Series.to_frame(silverDepDF_2["DEP_DELAY"])

arrivalTrain_x, arrivalTest_x, arrivalTrain_y, arrivalTest_y = train_test_split(arrivalFeats, arrivalTarget, test_size=.25, random_state=0, shuffle=True)
departTrain_x,  departTest_x,  departTrain_y,  departTest_y  = train_test_split(departFeats, departTarget, test_size=.25, random_state=0, shuffle=True)

# COMMAND ----------

group05_arrBaselineModel = LinearRegression().fit(arrivalTrain_x, arrivalTrain_y)
group05_depBaselineModel = LinearRegression().fit(departTrain_x,  departTrain_y)

# COMMAND ----------

arrivalPredictions = group05_arrBaselineModel.predict(arrivalTest_x)
arrivalRMSE = mean_squared_error(arrivalTest_y, arrivalPredictions, squared=False)
arrivalEVS = explained_variance_score(arrivalTest_y, arrivalPredictions)
arrivalR2  = r2_score(arrivalTest_y, arrivalPredictions)
print("Arrival Dataset:\n   Root Mean Squared Error: {0}\n   Explained Variance: {1}\n   R^2 Score: {2}".format(arrivalRMSE, arrivalEVS, arrivalR2))

departPredictions = group05_arrBaselineModel.predict(departTest_x)
departRMSE = mean_squared_error(departTest_y, departPredictions, squared=False)
departEVS = explained_variance_score(departTest_y, departPredictions)
departR2  = r2_score(departTest_y, departPredictions)
print("Departure Dataset:\n   Root Mean Squared Error: {0}\n   Explained Variance: {1}\n   R^2 Score: {2}".format(departRMSE, departEVS, departR2))

# COMMAND ----------

import uuid

with mlflow.start_run(run_name="Group5 Arrival Linear Model") as run:
  mlflow.sklearn.log_model(group05_arrBaselineModel, "g05_Arr_LinearModel")
  mlflow.log_metric("rmse", arrivalRMSE)
  mlflow.log_metric("expVar", arrivalEVS)
  mlflow.log_metric("r2", arrivalR2)

  g05_Arr_LinearModel_runID = run.info.run_uuid

g05_Arr_LinearModel_name = f"g05_Arr_LinearModel"
g05_Arr_LinearModel_uri = "runs:/{run_id}/g05_Arr_LinearModel".format(run_id=g05_Arr_LinearModel_runID)
g05_Arr_LinearModel_details = mlflow.register_model(model_uri=g05_Arr_LinearModel_uri, name=g05_Arr_LinearModel_name)
#g05ArrLinearModel_name

# COMMAND ----------

with mlflow.start_run(run_name="Group5 Departure Linear Model") as run:
  mlflow.sklearn.log_model(group05_depBaselineModel, "g05_Dep_LinearModel")
  mlflow.log_metric("rmse", departRMSE)
  mlflow.log_metric("expVar", departEVS)
  mlflow.log_metric("r2", departR2)

  g05_Dep_LinearModel_runID = run.info.run_uuid

g05_Dep_LinearModel_name = f"g05_Dep_LinearModel"
g05_Dep_LinearModel_uri = "runs:/{run_id}/g05_Dep_LinearModel".format(run_id=g05_Dep_LinearModel_runID)
g05_Dep_LinearModel_details = mlflow.register_model(model_uri=g05_Dep_LinearModel_uri, name=g05_Dep_LinearModel_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Model Deletion Code

# COMMAND ----------

#from mlflow.tracking import MlflowClient

#client = MlflowClient()
#client.delete_registered_model(name="g05_Arr_LinearModel_edf53e361a")

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))
