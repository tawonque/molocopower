# imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

########## linear regression estimator (from scratch)

### instructions say we cannot use ML packages so we will 
### build the linear regression from scratch and 
### we will only use sklearn to take advantage of
### the ESTIMATOR framework, but nothing else!!!


### What are the steps???

### considering...
# Y = target vector
# X = predictor matrix with intercept dummy
# B = coefficients to find out

### then we need to...
# 1) define the problem
# Y = X B (remember X includes a dummy column for intercept!!!)
# 2) multiply both sides by the transpose of X, noted X'
# X' Y = (X' X) B
# 3) As (X' X) is invertable, then we can multiply both side by the inverse
# inv(X'X) X' Y = B

### the above is the equation we want to build within our estimator...

class linear_regression_estimator(BaseEstimator):
    '''
    let's do a multiple linear regression 
    - it's interpretable and relatively quick to code
    '''

    def __init__(self):
        self.betas = []

    def _add_intercept(self, X):
        dummies_shape = X.shape[0]
        dummies = np.ones(shape=dummies_shape).reshape(-1,1)
        print(dummies.shape)
        return np.concatenate((dummies, X), 1)
    
    def fit(self, X, y):
        '''
        Remember the steps 1) 2) 3)?
        let's apply them
        '''
        X = self._add_intercept(X)
        print(X.shape)
        print(X.head())
        # inv(X'X) X' Y = B
        self.betas = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    def predict(self, _X):
        b0 = self.betas[0]
        coeffs = self.betas[1:]
        prediction = b0
        # and now let's use all the coefficients of our equation
        for x_i, b_i in zip(_X, coeffs):
            prediction += x_i * b_i
        return prediction