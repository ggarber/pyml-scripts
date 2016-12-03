import pandas as pd
import numpy as np
from perceptron import Perceptron

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

p = Perceptron(0.1, 10)
p.fit(X, y)

print p.errors