import numpy as np
from flask import Flask
from flask import abort, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

clf = joblib.load('sklearn_saves/random_forest.pkl')

@app.route('/get_prediction', methods=['POST'])
def score():

    prediction = clf.predict_proba(np.array([request.json["sepal_length"],
                                       request.json["sepal_width"],
                                       request.json["petal_length"],
                                       request.json["petal_width"]])
                             .reshape(1, -1))

    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
