from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer
import pandas as pd, numpy as np
class PreFEKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=12, weights='distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.knn_cols = [
            'bedrooms', 'bathrooms', 'floors',
            'yr_built', 'sqft_living', 'sqft_above'
        ]

        self.zero_fill_cols = [
            'sqft_basement', 'sqft_lot',
            'waterfront', 'view', 'yr_renovated'
        ]

    def fit(self, X, y=None):
        X = X.copy()
        
        self.knn_cols_ = [c for c in self.knn_cols if c in X.columns]

        self._knn_imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )

        self._knn_imputer.fit(X[self.knn_cols_])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        X[self.knn_cols_] = self._knn_imputer.transform(X[self.knn_cols_])
        X[self.zero_fill_cols] = X[self.zero_fill_cols].fillna(0)

        return X




class featureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    # --------------------------------------------------
    # FIT: learn dataset-level statistics ONLY
    # --------------------------------------------------
    def fit(self, X, y=None):
        X = X.copy()

        # reference year for age calculation
        if 'yr_built' in X.columns:
            self.current_yr_ = int(X['yr_built'].max())
        else:
            self.current_yr_ = None

        self.is_fitted_ = True
        return self

    # --------------------------------------------------
    # TRANSFORM: pure feature construction
    # --------------------------------------------------
    def transform(self, X):
        check_is_fitted(self, attributes=['is_fitted_'])
        X = X.copy()

        # ---------- house age ----------
        if 'yr_built' in X.columns and self.current_yr_ is not None:
            X['houseage'] = (self.current_yr_ - X['yr_built']).astype(int)
        # ---------- total rooms ----------
        # if {'bedrooms', 'bathrooms'}.issubset(X.columns):
        #     X['totalrooms'] = (X['bedrooms'] + X['bathrooms']).astype(int)
            
        # ---------- basement flag ----------
        if 'sqft_basement' in X.columns:
            X['has_basement'] = (X['sqft_basement'] > 0).astype(int)

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # sklearn fitted flag
        self.is_fitted_ = True

        # internal FunctionTransformer
        self._log_fn_ = FunctionTransformer(
            self._log_transform,
            validate=False
        )
        return self

    def _log_transform(self, X):
        X = X.copy()

        for col in self.cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X

    def transform(self, X):
        return self._log_fn_.transform(X)

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.is_fitted_ =True
        self.cols_ = [c for c in self.cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.cols_, errors="ignore")


class feature_ordinal_encoding(BaseEstimator,TransformerMixin):
    def __init__(self):
        self._encoder=OrdinalEncoder(categories=[['Poor','Fair','Average','Good','Very Good'],['N','Y']])
    def fit(self,X,y=None):
        X=X.copy()
        self._encoder.fit(X[['condition','waterfront']])
        self.is_fitted_ = True
        return self
    def transform(self,X):
        X = X.copy()
        X[['condition','waterfront']] = self._encoder.transform(X[['condition','waterfront']]).astype(int)
        return X