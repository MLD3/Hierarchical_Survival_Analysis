from config import *
import pandas as pd
import numpy as np
import sparse
from collections import defaultdict

from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from sklearn.feature_selection import VarianceThreshold
import sklearn
from collections import defaultdict

def print_header(*content, char='='):
    print()
    print(char * 80)
    print(*content)
    print(char * 80, flush=True)

from run_fiddle import *

######
# Post-filter: feature selection classes
######

# # Remove rare features
# class OccurrenceThreshold(
#     sklearn.base.BaseEstimator,
#     sklearn.feature_selection.base.SelectorMixin
# ):
#     def __init__(self, threshold=0., direction='<'):
#         self.threshold = threshold
#         self.direction = direction
    
#     def fit(self, X, y=None):
#         if hasattr(X, "toarray"):   # sparse matrix
#             X = X.toarray()
#         self.freqs_ = np.mean(X, axis=0)
#         return self

#     def _get_support_mask(self):
#         if self.direction == '>':
#             return self.freqs_ > self.threshold
#         elif self.direction == '<':
#             return self.freqs_ < 1 - self.threshold

# 
class FrequencyThreshold_temporal(
    sklearn.base.BaseEstimator,
    sklearn.feature_selection.SelectorMixin#sklearn.feature_selection.base.SelectorMixin
):
    def __init__(self, threshold=0., L=None):
        assert L is not None
        self.threshold = threshold
        self.L = L
    
    def fit(self, X, y=None):
        if hasattr(X, "toarray"):   # sparse matrix
            X = X.toarray()
        if hasattr(X, "todense"):   # sparse matrix
            X = X.todense()
        
        # Reshape to be 3-dimensional array
        NL, D = X.shape
        X = X.reshape(int(NL/self.L), self.L, D)
        
        # Collapse time dimension, generating NxD matrix
        X_notalways0 = X.any(axis=1)
        X_notalways1 = (1-X).any(axis=1)
        self.freqs_notalways0 = np.mean(X_notalways0, axis=0)
        self.freqs_notalways1 = np.mean(X_notalways1, axis=0)
        return self

    def _get_support_mask(self):
        return np.logical_and(
            self.freqs_notalways0 > self.threshold,
            self.freqs_notalways1 > self.threshold,
        )

# Keep only first feature in a pairwise perfectly correlated feature group
class CorrelationSelector(
    sklearn.base.BaseEstimator,
    sklearn.feature_selection.SelectorMixin,#sklearn.feature_selection.base.SelectorMixin,
):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        if hasattr(X, "toarray"):   # sparse matrix
            X = X.toarray()
        if hasattr(X, "todense"):   # sparse matrix
            X = X.todense()
        
        # Calculate correlation matrix
        # Keep only lower triangular matrix
        self.corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(self.corr_matrix, 0)
        self.corr_matrix *= np.tri(*self.corr_matrix.shape)
        
        # get absolute value
        corr = abs(self.corr_matrix)
        
        # coefficient close to 1 means perfectly correlated
        # Compare each feature to previous feature (smaller index) to see if they have correlation of 1
        to_drop = np.isclose(corr, 1.0).sum(axis=1).astype(bool)
        self.to_keep = ~to_drop
        
        return self

    def _get_support_mask(self):
        return self.to_keep
    
    def get_feature_aliases(self, feature_names):
        feature_names = [str(n) for n in feature_names]
        corr_matrix = self.corr_matrix
        flags = np.isclose(abs(corr_matrix), 1.0)
        alias_map = defaultdict(list)
        for i in range(1, corr_matrix.shape[0]):
            for j in range(i):
                if flags[i,j]:
                    if np.isclose(corr_matrix[i,j], 1.0):
                        alias_map[feature_names[j]].append(feature_names[i])
                    elif np.isclose(corr_matrix[i,j], -1.0):
                        alias_map[feature_names[j]].append('~{' + feature_names[i] + '}')
                    else:
                        assert False

                    # Only save alias for first in the list
                    break
        return dict(alias_map)


######
######
from __main__ import *


def get_unique_variables(df):
    return sorted(df[var_col].unique())

def get_frequent_numeric_variables(df_time_series, variables, threshold):
    df_types = pd.read_csv(data_path + 'value_types.csv').set_index(var_col)['value_type']
    numeric_vars = [col for col in variables if df_types[col] == 'Numeric']
    df_num_counts = calculate_variable_counts(df_time_series)[numeric_vars] #gets the count of each variable for each patient. 
    variables_num_freq = df_num_counts.columns[df_num_counts.mean() >= threshold * np.floor(T/dt)]
    return variables_num_freq

def calculate_variable_counts(df_data):
    """
    df_data in raw format with four columns
    """
    df = df_data.copy()
    df['count'] = 1
    df_count = df.groupby([ID_col, var_col]).count()[['count']].unstack(1, fill_value=0)
    df_count.columns = df_count.columns.droplevel()
    df_count = df_count.reindex(df_population.index, fill_value=0)
    ## Slower version
    # df_count = df[['ID', 'variable_name', 'count']].pivot_table(index='ID', columns='variable_name', aggfunc='count', fill_value=0)
    return df_count

def select_dtype(df, dtype, dtypes=None):
    if dtypes is None:
        ## Need to assert dtypes are not all objects
        assert not all(df.dtypes == 'object')
        if dtype == 'mask':
            return df.select_dtypes('bool')
        elif dtype == '~mask':
            return df.select_dtypes(exclude='bool')
    else:
        ## Need to assert df.columns and dtypes.index are the same
        if dtype == 'mask':
            return df.loc[:, (dtypes == 'bool')].astype(bool)
        elif dtype == '~mask':
            return df.loc[:, (dtypes != 'bool')]
        else:
            assert False
    return

def smart_qcut_dummify(x, q):
    z = smart_qcut(x, q)
    return pd.get_dummies(z, prefix=z.name)

def smart_qcut(x, q):
    # ignore strings when performing qcut
    x = x.copy()
    x = x.apply(make_float)
    m = x.apply(np.isreal)
    if x.loc[m].dropna().nunique() > 1:
        x.loc[m] = pd.qcut(x.loc[m].to_numpy(), q=q, duplicates='drop')
#         bins = np.percentile(x.loc[m].to_numpy(), [0, 20, 40, 60, 80, 100])
#         x.loc[m] = pd.cut(x, bins)
    return x

def make_float(v):
    if v is None:
        v = np.nan
    try:
        return float(v)
    except ValueError:
        return v
    assert False

def is_numeric(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
    assert False


######
# Time-series internals
######

def _get_time_bins(T, dt):
    return np.arange(0, T+dt, dt)

def _get_time_bins_index(T, dt):
    return pd.Index(pd.interval_range(start=0, end=T, freq=dt, closed='left'))

def pivot_event_table(df):
    df = df.copy()
    
    # Handle cases where the same variable is recorded multiple times with the same timestamp
    # Adjust the timestamps by epsilon so that all timestamps are unique
    eps = 1e-6
    m_dups = df.duplicated([ID_col, t_col, var_col], keep=False)
    df_dups = df[m_dups].copy()
    for v, df_v in df_dups.groupby(var_col):
        df_dups.loc[df_v.index, t_col] += eps * np.arange(len(df_v))
    
    df = pd.concat([df[~m_dups], df_dups])
    assert not df.duplicated([ID_col, t_col, var_col], keep=False).any()
    
    return pd.pivot_table(df, val_col, t_col, var_col, 'first')

def presence_mask(df_i, variables, T, dt):
    # for each itemid
    # for each time bin, whether there is real measurement
    if len(df_i) == 0:
        mask_i = pd.DataFrame().reindex(index=_get_time_bins_index(T, dt), columns=list(variables), fill_value=False)
    else:
        mask_i = df_i.groupby(
            pd.cut(df_i.index, _get_time_bins(T, dt), right=False)
        ).apply(lambda x: x.notnull().any())
        mask_i = mask_i.reindex(columns=variables, fill_value=False)
    
    mask_i.columns = [str(col) + '_mask' for col in mask_i.columns]
    return mask_i

def get_delta_time(mask_i):
    a = 1 - mask_i
    b = a.cumsum()
    c = mask_i.cumsum()
    dt_i = b - b.where(~a.astype(bool)).ffill().fillna(0).astype(int)
    
    # the delta time for itemid's for which there are no measurements must be 0
    # or where there's no previous measurement and no imputation
    dt_i[c == 0] = 0
    
    dt_i.columns = [str(col).replace('_mask', '_delta_time') for col in dt_i.columns]
    return dt_i

def impute_ffill(df, columns, T, dt, mask=None):
    if len(df) == 0:
        return pd.DataFrame().reindex(columns=columns, fill_value=np.nan)
    
    if mask is None:
        mask = presence_mask(df, columns)

    # Calculate time bins, sorted by time
    df_bin = df.copy()
    df_bin.index = pd.cut(df_bin.index, _get_time_bins(T, dt), right=False)
    
    # Compute the values used for imputation
    ## Collapse duplicate time bins, keeping latest values for each time bin
    df_imp = df_bin.ffill()
    df_imp = df_imp[~df_imp.index.duplicated(keep='last')]
    ## Reindex to make sure every time bin exists
    df_imp = df_imp.reindex(_get_time_bins_index(T, dt))
    ## Forward fill the missing time bins
    df_imp = df_imp.ffill()

    df_ff = df_imp
    df_ff[mask.to_numpy()] = np.nan
    df_ff.index = df_ff.index.mid ## Imputed values lie at the middle of a time bin
    df_ff = pd.concat([df, df_ff]).dropna(how='all')
    df_ff.sort_index(inplace=True)
    return df_ff

def most_recent_values(df_i, columns, T, dt):
    df_bin = df_i.copy()
    df_bin.index = pd.cut(df_bin.index, _get_time_bins(T, dt), right=False)
    df_v = df_bin.groupby(level=0).last()
    df_v.columns = [str(col) + '_value' for col in df_v.columns]
    df_v = df_v.reindex(_get_time_bins_index(T, dt))
    return df_v

def summary_statistics(df_i, columns, stats_functions, T, dt):
    # e.g. stats_functions=['mean', 'min', 'max']
    if len(columns) == 0:
        return pd.DataFrame().reindex(_get_time_bins_index(T, dt))
    else:
        # Encode statistics for numeric, frequent variables
        df_numeric = df_i[columns]
        df = df_numeric.copy().astype(float)
        df.index = pd.cut(df.index, _get_time_bins(T, dt), right=False)
        df_v = df.reset_index().groupby('index').agg(stats_functions)
        df_v.columns = list(map('_'.join, df_v.columns.values))
        df_v = df_v.reindex(_get_time_bins_index(T, dt))
        return df_v

def check_imputed_output(df_v):
    # Check imputation is successful
    ## If column is all null -> OK
    ## If column is all non-null -> OK
    ## If column has some null -> should only occur at the beginning
    not_null = df_v.notnull().all()
    all_null = df_v.isnull().all()
    cols_to_check = list(df_v.columns[~(not_null | all_null)])

    for col in cols_to_check:
        x = df_v[col].to_numpy()
        last_null_idx = np.argmax(np.where(pd.isnull(x))) # Find index of last nan
        assert pd.isnull(x[:(last_null_idx+1)]).all() # all values up to here are nan
        assert (~pd.isnull(x[(last_null_idx+1):])).all() # all values after here are not nan
    return
