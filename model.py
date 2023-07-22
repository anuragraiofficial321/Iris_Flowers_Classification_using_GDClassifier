import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from joblib import dump

# Load the Iris dataset
data = pd.read_csv('iris.csv')

# Split the data into features and labels
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Create a stochastic gradient descent classifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Fit the classifier to the data
clf.fit(X, y)

# Save the classifier to a file
dump(clf, 'model.pkl')
