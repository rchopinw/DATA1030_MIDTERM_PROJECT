import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class DetermineSparse:
    def __init__(
            self
    ):
        self.sparse_components = None

    def determine_sparse(
            self, x
    ) -> list:
        if self.sparse_components:
            return self.sparse_components
        self.sparse_components = [
            set(
                np.unique(x[:, i]).astype(int)
            ) == {0, 1}
            for i in range(x.shape[1])
        ]
        return self.sparse_components


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, self.columns]


class CategoricalFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns].astype(str)