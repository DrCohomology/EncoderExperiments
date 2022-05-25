# EncoderExperiments
 
Test for logistic regression and LGBM for binary classification with multiple encoders.
The expected result is for WOEEncoder to perform better (at least on average). 
 
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

Data can be accessed from Test for logistic regression for binary classification with multiple encoders.
The expected result is for WOEEncoder to perform better (at least on average). 
 
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

All datasets were pre-preprocessed following https://github.com/DenisVorotyntsev/CategoricalEncodingBenchmark
This consists of renaming the columns adn in particular renaming the target column
'target'.

The datasets were resampled down to at most 10k entries to ensure fast execution.
The model was not tuned, also to save runtime.  

## For reproducibility
The datasets should be in a directory called "data" inside of EncoderExperiments. 
Datasets can be downloaded from https://1drv.ms/u/s!Al3eqyMJEbpKgQ2kOiVDNsO03uRu?e=aJaoTs
