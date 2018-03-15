"""Training Random Forest and making predictions."""

import logging

from flask import Flask
from flask import request, jsonify

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sc.setLogLevel("ERROR")

app = Flask(__name__)

# Load data
file_path = "api/data/iris_data.csv"

schema = StructType([
    StructField("sepal_length", FloatType()),
    StructField("sepal_width", FloatType()),
    StructField("petal_length", FloatType()),
    StructField("petal_width", FloatType()),
    StructField("class", StringType())
])

spark = SparkSession.builder.getOrCreate()
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

logging.warning('Fitting RF.')
rfModel = rf.fit(trainingData)
logging.warning('Fitted RF.')

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
predictions = rfModel.transform(testData)
acc = evaluator.evaluate(predictions)
logging.warning("Accuracy on testset is:", acc)


# Setting up the API.
predict_schema = StructType(schema.fields[:-1])


@app.route('/get_prediction', methods=['POST'])
def calc_prob():
    """Calculate probability for species."""
    input_features = [[float(request.json["sepal_length"]),
                       float(request.json["sepal_width"]),
                       float(request.json["petal_length"]),
                       float(request.json["petal_width"])]]

    predict_df = spark.createDataFrame(data=input_features,
                                       schema=predict_schema)
    transformed_pred_df = pipelineModel.transform(predict_df)
    predictions = rfModel.transform(transformed_pred_df)
    probs = predictions.select('probability').take(1)[0][0]

    n_predictions = len(probs)
    labels = pipelineModel.stages[-1].labels
    result_dict = {labels[i]: probs[i] for i in range(n_predictions)}
    return jsonify(result_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
