# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:59:19 2022

@author: federicom
"""

import functools
import math
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
from inspect import signature
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

    def __init__(self, default=-1, w=10, **kwargs):
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

class EncoderWrapper(Encoder):
    
    def __init__(self, encoder_class):
        self.encoder_class = encoder_class
        # Can the encoder handle a single column?
        self.singlecol = False
        if "col" in signature(self.encoder_class).parameters:
            self.singlecol = True
        self.encoders = {}
        
    def fit(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        X = X.copy().reset_index(drop=True)
        
        if self.singlecol:
            for col in self.cols:
                self.encoders[col] = self.encoder_class(col, **kwargs).fit(X, y)
        else:
            X = self.encoder_class(**kwargs).fit_transform(X, y)
        return self
        
    def fit_transform(self, X: pd.DataFrame, y, **kwargs):
        self.cols = X.columns
        X = X.copy().reset_index(drop=True)
        
        if self.singlecol:
            for col in self.cols:
                self.encoders[col] = self.encoder_class(col, **kwargs).fit(X, y)
                X = self.encoders[col].transform(X).drop(columns=col)
        else:
            X = self.encoder_class(**kwargs).fit_transform(X, y)
        return X
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.singlecol: 
            for col in self.cols:
                X = self.encoders[col].transform(X).drop(columns=col)
        else:
            raise Exception('invalid')
        return X

    def __str__(self):
        return f"{self.encoder_class.__name__}()"

class OOFTE(BaseEstimator, TransformerMixin):
    def __init__(self, col, default=-1, random_state=1444):
        self.col = col
        self.colname = f"{self.col}_TE"
        self._d = {}
        self.random_state = random_state
        self.kfold = StratifiedKFold(n_splits=10, random_state=self.random_state, shuffle=True)
        self.default = -1
    
    def fit(self, X, y):
        X = X.reset_index(drop=True)
        new_x = X[[self.col]].copy().reset_index(drop=True)
        X.loc[:, self.colname] = 0
        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold.split(new_x, y)):
            trn_x = new_x.iloc[trn_idx].copy()
            trn_x.loc[:, 'target'] = y.iloc[trn_idx]
            val_x = new_x.iloc[val_idx].copy()
            val_x.loc[:, 'target'] = y.iloc[val_idx]
            val = trn_x.groupby(self.col)['target'].mean().to_dict()
            # with default and other error handling
            val = defaultdict(lambda: -1, {
                k : v if not math.isnan(v) else -1 for k, v in val.items()  
            })
            self._d[n_fold] = val
        return self

    def fit_transform(self, X, y):
        X = X.reset_index(drop=True)
        new_x = X[[self.col]].copy().reset_index(drop=True)
        X.loc[:, self.colname] = 0
        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold.split(new_x, y)):
            trn_x = new_x.iloc[trn_idx].copy()
            trn_x.loc[:, 'target'] = y.iloc[trn_idx]
            val_x = new_x.iloc[val_idx].copy()
            val_x.loc[:, 'target'] = y.iloc[val_idx]
            val = trn_x.groupby(self.col)['target'].mean().to_dict()
            # with default and other error handling
            val = defaultdict(lambda: -1, {
                k : v if not math.isnan(v) else -1 for k, v in val.items()  
            })
            self._d[n_fold] = val
            X.loc[val_idx, self.colname] = X.loc[val_idx, self.col].map(val)
        return X

    def transform(self, X):
        X.loc[:, self.colname] = 0
        for key, val in self._d.items():
            X.loc[:, self.colname] += X[self.col].map(val)

        X.loc[:, self.colname] /= key + 1
        return X


# Already implemented in category_encoders.LeaveOneOutEncoder
class LOOTE:
    def __init__(self, col):
        self.col = col
        self.colname = f"{self.col}_TE"

    def fit_transform(self, X, y):
        new_x = X[[self.col]].copy()
        new_x.loc[:, 'target'] = y
        a = (new_x.groupby(self.col)['target'].transform(np.sum) - y)\
            / new_x.groupby(self.col)['target'].transform(len)
        X.loc[:, self.colname] = a
        self._d = X.groupby(self.col)[self.colname].mean()
        return X

    def transform(self, X):
        X.loc[:, self.colname] = X[self.col].map(self._d)
        return X

