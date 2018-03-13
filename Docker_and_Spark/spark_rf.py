"""Training and storing Random Forest."""

import os

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load data
file_path = os.getcwd() + "data/iris_data.csv"

schema = StructType([
    StructField("sepal_length", FloatType()),
    StructField("sepal_width", FloatType()),
    StructField("petal_length", FloatType()),
    StructField("petal_width", FloatType()),
    StructField("class", StringType())
])

dataset = spark.read.csv(file_path, header=True, schema=schema)

# Setting up pipeline to convert features and labels
feature_cols = ['sepal_length',
                'sepal_width',
                'petal_length',
                'petal_width']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
label_stringIdx = StringIndexer(inputCol="class", outputCol="label")

pipeline = Pipeline(stages=[assembler, label_stringIdx])
pipelineModel = pipeline.fit(dataset)
dataset = pipelineModel.transform(dataset)

trainingData, testData = dataset.randomSplit([0.7, 0.3], seed=100)

# Train the model
rf = RandomForestClassifier(labelCol="label",
                            featuresCol="features",
                            numTrees=10)

rfModel = rf.fit(trainingData)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
predictions = rfModel.transform(testData)
acc = evaluator.evaluate(predictions)
print("Accuracy on testset is:", acc)

# Storing the model
rfModel.write().overwrite().save("spark_saves/rfModel")
pipelineModel.write().overwrite().save("spark_saves/pipelineModel")

print("Stored pipeline and model.")
