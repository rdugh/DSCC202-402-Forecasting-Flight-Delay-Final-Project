# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

databaseName = 'dscc202_group05_db'
silverDepDF = (spark
               .readStream
               .table('dscc202_group05_db.silverdep_delta'))

# COMMAND ----------

#check number of null values
#silverDepDF.isnull().sum().sum()


# COMMAND ----------

#silverDepDF.isnull().values.any()

# COMMAND ----------

#display(silverDepDF)

# COMMAND ----------

display(silverDepDF_1)

# COMMAND ----------

silverDepDF_1 = spark.sql("""
  SELECT *
  FROM {}.silverdep_delta
  """.format(GROUP_DBNAME))

# COMMAND ----------



# COMMAND ----------

silverDepDF.printSchema()

# COMMAND ----------

print("The dataset has %d rows." % silverDepDF_1.count())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

row = silverDepDF_1.take(1)[0]
row

# COMMAND ----------

from pyspark.sql.functions import col,sum
silverDepDF_1.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in silverDepDF_1.columns)).show()

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

# COMMAND ----------

silverDepDF_1=silverDepDF_1.drop('FL_DATE','ORIGIN_CITY_NAME','DEST_CITY_NAME','ORIGIN','DEST','OP_UNIQUE_CARRIER','dep_time','arrival_time','time','call_sign')
#test=test.drop('FL_DATE')

# COMMAND ----------

(train_data, test_data) = silverDepDF_1.randomSplit([0.8, 0.2])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

display(train_data.select("DEP_DELAY", "DAY_OF_WEEK"))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer

featuresCols = silverDepDF_1.columns
featuresCols.remove('DEP_DELAY')
# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

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
print("There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

import mlflow
import mlflow.spark

# turn on autologging
mlflow.spark.autolog()

with mlflow.start_run():
  pipelineModel = pipeline.fit(train)
  
  # Log the best model.
  mlflow.spark.log_model(spark_model=pipelineModel.stages[2].bestModel, artifact_path='best-model') 

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

#features_data = vector_data.select(["features", "DEP_DELAY"])

# COMMAND ----------

#features_data.show()

# COMMAND ----------

#lr = LinearRegression(labelCol="DEP_DELAY", featuresCol="features")

# COMMAND ----------

#model = lr.fit(features_data)

# COMMAND ----------

#summary = model.summary

#print("R^2", summary.r2)