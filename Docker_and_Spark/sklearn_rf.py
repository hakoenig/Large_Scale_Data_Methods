"""Training and storing Random Forest."""

import numpy as np
from sklearn import datasets

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

clf = RandomForestClassifier(n_estimators=50, max_depth=3)
clf = clf.fit(iris_X, iris_y)
joblib.dump(clf, 'sklearn_saves/random_forest.pkl')

# clf.predict(np.array([1, 2, 3, 4]).reshape(1, -1))
