from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import KNNImputer 
import pandas as pd,numpy as np
class AgeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,imputer="knn"):
        self.imputer=imputer
    def fit(self,X,y=None):
        if self.imputer == 'knn':
            self._imputer = KNNImputer(weights='distance')
        elif self.imputer == 'mean':
            self._imputer = SimpleImputer(strategy='mean')
        elif self.imputer == 'median':
            self._imputer = SimpleImputer(strategy='median')
        else:
            raise ValueError("imputer must be one of ['knn', 'mean', 'median']")
        self._imputer.fit(X[['Age']])
        return self
    def transform(self,X):
        X = X.copy()
        X[['Age']]=self._imputer.transform(X[['Age']])
        return X

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X = X.copy()
        bins = [0,4,9,14,19,24,29,34,39,44,49,54,59,64,70,80]
        groups = ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-70','71-80']
        X['Age_group'] = pd.cut(X['Age'],bins=bins,labels=groups)
        X['Has_Cabin'] = X['Cabin'].notna().astype(int)
        X['Familysize'] = X['SibSp'] + X['Parch'] +1 
        return X      