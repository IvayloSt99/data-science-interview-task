import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder

def modify_eliminate_cols(df):
    
    df = df.drop(columns=['EBS_JOB_NUMBER', 'ID', 'SNAPSHOT_NUMBER', 'RM_INVENTORY', 'PLANT', 'Job Coverage', 'LMS'])
    df[['ARCHIVE_DATE', 'SCHEDULED_DATE', 'JOB_CREATION_DATE', 'INVENTORY_DAY_BUCKET']] = df[['ARCHIVE_DATE', 'SCHEDULED_DATE', 'JOB_CREATION_DATE', 'INVENTORY_DAY_BUCKET']].apply(pd.to_datetime)
    df['ARCHIVE_DATE'] = df['ARCHIVE_DATE'].map(lambda x: x.month*100 + x.day)
    df['SCHEDULED_DATE_HOUR'] = df['SCHEDULED_DATE'].map(lambda x: x.hour)
    df = df.drop(columns='SCHEDULED_DATE')
    df['JOB_CREATION_DATE_HOUR'] = df['JOB_CREATION_DATE'].map(lambda x: x.hour)
    df['JOB_CREATION_DATE'] = df['JOB_CREATION_DATE'].map(lambda x: x.month*100 + x.day)
    df['INVENTORY_DAY_BUCKET'] = df['INVENTORY_DAY_BUCKET'].map(lambda x: x.month*100 + x.day)
    df = pd.get_dummies(df, columns=['JOB_STATUS', 'JOB_CREATION_FLAG', 'RM Item Coverage'], dtype='int64')
    df[['FG_CODE', 'RM_ITEM_CODE', 'Plant', 'LINE_CODE']] = df[['FG_CODE', 'RM_ITEM_CODE', 'Plant', 'LINE_CODE']].astype('str')

    return train_test_split(df.drop(columns='TARGET'), df['TARGET'], train_size=0.8, random_state=42)

def impute(x_train, x_test):
    proportions = {}
    for col in x_train[['Low', 'Medium', 'High']]:
        proportion = x_train[col].value_counts(normalize=True)
        proportions[col] = proportion
        x_train[col] = x_train[col].fillna(pd.Series(np.random.choice([0.0, 1.0], p=proportion, size=len(x_train[col]))))

    for col in x_test[['Low', 'Medium', 'High']]:
        proportion = proportions[col]
        x_test[col] = x_test[col].fillna(pd.Series(np.random.choice([0.0, 1.0], p=proportion, size=len(x_test[col]))))

    iterative_impute_cols = ['EPQ', 'SS', 'Scheduled-EPQ', 'Scheduled-EPQ_%', 'SEPQ']

    imputer = IterativeImputer(min_value=x_train[iterative_impute_cols].min(), tol=1e-5, sample_posterior=True)
    imputer.fit(x_train[iterative_impute_cols])

    x_train[iterative_impute_cols] = imputer.transform(x_train[iterative_impute_cols])
    x_test[iterative_impute_cols] = imputer.transform(x_test[iterative_impute_cols])

    return x_train, x_test

def encode(x_train, x_test, y_train):
    categorical_cols = ['FG_CODE', 'RM_ITEM_CODE', 'Plant', 'LINE_CODE']
    encoder = TargetEncoder(target_type='binary', cv=10)
    target_enc = encoder.fit(x_train[categorical_cols], y_train)
    x_train[categorical_cols] = target_enc.transform(x_train[categorical_cols])
    x_test[categorical_cols] = target_enc.transform(x_test[categorical_cols])
    return x_train, x_test
# class ProportionalBinaryImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns=None, random_state=None):
#         self.columns = columns
#         self.random_state = random_state
#         self.proportions_ = {}

#     def fit(self, X, y=None):
#         X = X.copy()
#         if self.columns is None:
#             self.columns = ['Low', 'Medium', 'High']
#         for col in self.columns:
#             value_counts = X[col].value_counts(normalize=True)
#             self.proportions_[col] = value_counts.reindex([0.0, 1.0], fill_value=0.0).values
#         return self

#     def transform(self, X):
#         X = X.copy()
#         rng = np.random.default_rng(self.random_state)
#         for col in self.columns:
#             mask = X[col].isna()
#             n_missing = mask.sum()
#             if n_missing > 0:
#                 X.loc[mask, col] = rng.choice([0.0, 1.0], size=n_missing, p=self.proportions_[col])
#         return X
