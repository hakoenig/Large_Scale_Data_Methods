"""Making predictions."""

from flask import Flask
from flask import request, jsonify

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

sc = SparkContext()
sc.setLogLevel("ERROR")

app = Flask(__name__)

schema = StructType([
    StructField("sepal_length", FloatType()),
    StructField("sepal_width", FloatType()),
    StructField("petal_length", FloatType()),
    StructField("petal_width", FloatType()),
    StructField("class", StringType())
])

predict_schema = StructType(schema.fields[:-1])

pipelineModel = PipelineModel.load("api/sparksaves/pipelineModel")
rfModel = RandomForestClassificationModel.load("api/sparksaves/rfModel")

spark = SparkSession.builder.getOrCreate()


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
