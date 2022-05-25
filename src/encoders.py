# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:59:19 2022

@author: federicom
"""

import functools
import numpy as np
import pandas as pd

from category_encoders import (
    BackwardDifferenceEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    PolynomialEncoder,
    SumEncoder,
    WOEEncoder,
)

from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, default=-1, **kwargs):
        self.default = default
        self.encoding = defaultdict(lambda: defaultdict(lambda: self.default))
        self.inverse_encoding = defaultdict(
            lambda: defaultdict(lambda: self.default))
        self.cols = None

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        return X

    def fit_transform(self, X: pd.DataFrame, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, y, **kwargs)


class TargetEncoder(Encoder):
    """
    Maps categorical values into the average target associated to them
    """

    def __init__(self, default=-1, **kwargs):
        super().__init__(default=default, **kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        target_name = y.name
        X = X.join(y.squeeze())
        for col in self.cols:
            temp = X.groupby(col)[target_name].mean().to_dict()
            self.encoding[col].update(temp)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X


class CollapseEncoder(Encoder):
    """
    Evey categorical value is mapped to 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        self.cols = X.columns
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=['cat'])


class CVTargetEncoder(Encoder):
    """
    Has a column for every fold
    """

    def __init__(self, n_splits=5, default=-1, random_state=1444, **kwargs):
        super().__init__(default=default, **kwargs)
        self.n_splits = n_splits
        self.cv = StratifiedKFold(**{
            "n_splits" : n_splits, 
            "random_state" : random_state, 
            "shuffle" : True    
        })
        self.TEs = [TargetEncoder() for _ in range(n_splits)]
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns

        # Fit a different targetEncoder on each training fold
        for TE, (tr, te) in zip(self.TEs, self.cv.split(X, y)):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            TE.fit(Xtr, ytr)

        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()

        # Transform with each encoder the whole dataset
        XEs = []
        for fold, TE in enumerate(self.TEs):
            XEs.append(TE.transform(X).add_prefix(f'f{fold}_'))
        XE = functools.reduce(lambda x, y: x.join(y), XEs)

        return XE

    def __str__(self):
        return f'CVTargetEncoder{self.n_splits}'


class SmoothedTargetEncoder(Encoder):
    """
    TargetEncoder with smoothing parameter. 
    From https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_3_TargetEncoding.ipynb

    """

    def __init__(self, default=-1, w=20, **kwargs):
        super().__init__(default=default, **kwargs)
        self.w = w

    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        target_name = y.name
        X = X.join(y.squeeze())
        global_mean = y.mean()
        for col in self.cols:
            temp = X.groupby(col).agg(['sum', 'count'])
            temp['STE'] = (temp.target['sum'] + self.w *
                           global_mean) / (temp.target['count'] + self.w)
            # temp['TE'] = temp.target['sum'] / temp.target['count']
            self.encoding[col].update(temp['STE'].to_dict())
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].apply(lambda cat: self.encoding[col][cat])
        return X

