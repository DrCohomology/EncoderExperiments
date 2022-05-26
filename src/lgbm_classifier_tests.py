# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:28:53 2022

@author: federicom

Test for logistic regression for binary classification with multiple encoders.
The expected result is for CatBoostEncoder and TargetEncoder to perform better (at least on average). 
 
Datasets used in the analysis are 
- adult
    https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- credit
    https://www.kaggle.com/c/home-credit-default-risk/data
- kaggle_cat_dat_1
    https://www.kaggle.com/competitions/cat-in-the-dat
- kaggle_cat_dat_2
    https://www.kaggle.com/c/cat-in-the-dat-ii
- kick
    https://www.kaggle.com/c/DontGetKicked/data
- promotion
    https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/
- telecom
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

All datasets were preprocessed following https://github.com/DenisVorotyntsev/CategoricalEncodingBenchmark
This consists of renaming the columns adn in particular renaming the target column
'target'.

The datasets were resampled down to at most 10k entries to ensure fast execution.
The model was not tuned, also to save runtime.  
"""

import numpy as np
import os
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import src.encoders as enc

# ---- Utility functions


def cohen_kappa(y_true, y_pred):
    """
    https://en.wikipedia.org/wiki/Cohen%27s_kappa
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))


def pre2process(df):
    """
    Get rid of nans, split dependent (y) and independent (X) variables
    """
    df2 = df[~df.target.isna()].dropna(axis=1, how="any")

    if len(df2.columns) <= 1:
        df2 = df.dropna(axis=0, how="any").reset_index(drop=True)
    X = df2.drop(columns="target")
    y = df2.target.astype(int)

    return X, y

# ---- Execution

# Go to correct directory
if Path.cwd().name == "src":
    os.chdir(Path.cwd().parent)
elif Path.cwd().name == "EncoderExperiments":
    pass
else:
    raise Exception(
        """The correct path could not be found. Ensure that the current working directory is EncoderExperiments""")

dataset = "telecom"
# df = pd.read_csv(f"./data/{dataset}.csv")
df = pd.read_csv(f"C:/Data/{dataset}.csv")
df = df.sample(min(10000, len(df)), random_state=1442)

X, y = pre2process(df)
X.reset_index(drop=True, inplace=True)

# encoders considered in the analysis
encoders = [
    # enc.BackwardDifferenceEncoder(),
    enc.BinaryEncoder(), # instead of OHE
    enc.CatBoostEncoder(),
    # enc.CountEncoder(), # does not produce a result
    # enc.GLMMEncoder(), # orders of magnitude too slow
    # enc.HashingEncoder(), # LogisticRegression does not converge
    # enc.HelmertEncoder(),
    # enc.JamesSteinEncoder(),
    enc.LeaveOneOutEncoder(),
    # enc.MEstimateEncoder(),
    # enc.OneHotEncoder(), # too memory inefficient
    # enc.PolynomialEncoder(),
    # enc.SumEncoder(),
    # enc.WOEEncoder(),
    enc.SmoothedTargetEncoder(), # default weight 10
    enc.TargetEncoder(),
    enc.EncoderWrapper(enc.OOFTE), 
]

# quality metrics considered - for binary classification
scores_list = [
    balanced_accuracy_score,
    cohen_kappa,
    accuracy_score,
]
scores = {
    s.__name__: make_scorer(s)
    for s in (scores_list)
}

# model of interest
model = LGBMClassifier()

# ---- Main loop

results = {}
exceptions = []
for encoder in tqdm(encoders):
    
    # Encode cazegorical variables, rescale numerical vars
    CT = ColumnTransformer([
        (
            "encoder",
            encoder,
            [col for col in X.columns if X[col].dtype not in ("float64", "int64")]
        ),
        (
            "scaler",
            RobustScaler(),
            [col for col in X.columns if X[col].dtype in ("float64", "int64")]
        ),

    ])
    
    PP = Pipeline([
        ("preproc", CT),
        ("model", model)
    ])

    try:
        out = cross_validate(PP, X, y,
                             scoring=scores,
                             n_jobs=-2,
                             verbose=0,
                             cv=10
                             )

    except Exception as e:
        out = None
        exceptions.append(e)
    finally:
        try:
            results[encoder.__name__] = out
        except AttributeError:
            results[str(encoder)] = out

# ---- Reorganize results into a df

new = True
end = False
while not end:
    try:
        for k, v in results.items():
            if v is None:
                if new:
                    print("Failed encoders:")
                    new = False
                print(k)
                del results[k]
        end = True
    except:
        pass

metrics = results[list(results.keys())[0]]
means = {
    metric: {
        enc_name: np.mean(enc_res[metric])
        for enc_name, enc_res in results.items()
    }
    for metric in metrics
}
stds = {
    metric: {
        enc_name: np.std(enc_res[metric])
        for enc_name, enc_res in results.items()
    }
    for metric in metrics
}

ms = pd.DataFrame().from_dict(means)
ss = pd.DataFrame().from_dict(stds)
ress = ms.join(ss, rsuffix='_std')
ress.columns = [
    "fit_time", "score_time", "Bacc", "ck", "acc",
    "fit_time_std", "score_time_std", "Bacc_std", "ck_std", "acc_std",
]
ress = ress[[
    "Bacc", "Bacc_std",
    "ck", "ck_std",
    "acc", "acc_std",
    "fit_time",
]]

# ----  Dump results

ress.to_csv(f"./results/LGBM/{dataset}_test.csv")
