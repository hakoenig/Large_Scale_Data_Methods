FROM jupyter/pyspark-notebook

RUN pip install flask
RUN pip install --upgrade pyspark

EXPOSE 5001

COPY flask_scoring_spark.py api/flask_scoring_spark.py
ADD sparksaves api/sparksaves

CMD ["python", "api/flask_scoring_spark.py"]
