import numpy as np

class Perceptron(object):

  def __init__(self, eta=0.01, iterations=10):
    """ eta is a constant controlling how fast it adapts"""
    self.eta = eta
    self.iterations = iterations
    self.w = None
    self.errors = []

  def fit(self, X, y):
    """ X [samples, features]  y [labels]"""
    self.w = np.zeros(1 + X.shape[1])
    self.errors = []

    for _ in range(self.iterations):
      errors = 0

      for x, label in zip(X, y):
        update = self.eta * (label - self.predict(x))
        self.w[1:] += update * x
        self.w[0] += update
        errors += 1 if update != 0 else 0
      self.errors.append(errors)

  def predict(self, x):
    val = np.dot(x, self.w[1:]) + self.w[0]
    return 1 if val > 0 else -1
