"""
generic imputer based on scikit learn.
current version issues a convergence warning always, but this seems a bug not a real convergence problem
"""

from sklearn.experimental import enable_iterative_imputer   # DO NOT DROP
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from capacity_planning.utilities import sys_utils as s_ut
from functools import reduce
import time
import numpy as np


def set_nans(fcol, na_df, sep):
    c = fcol.name.split(sep)[0]
    b = na_df[c].astype(int)
    v = fcol * (1 - b) + (-1) * b
    return v.replace(-1, np.nan)


def impute(a_df_, i_vals=None, ex_cols=None):
    """
    generic imputer for a DF
    :param a_df_: imput DF
    :param ex_cols: columns to exclude from the imputation (eg ds, agent names, ...)
    :param i_vals: list of values to impute
    :return: a DF with the same columns and no NaNs or None values in the non-excluded columns
    """
    # ex_cols: columns to exclude from imputation
    if ex_cols is None:
        ex_cols = list()

    if i_vals is None:
        i_vals = [np.nan, None]
    else:
        i_vals = list(set(i_vals + [np.nan, None]))

    # prepare: all values to impute set to NaN
    df_ = a_df_.copy()
    df_.reset_index(inplace=True, drop=True)
    for c in df_.columns:
        if c not in ex_cols:
            df_[c].replace(i_vals, np.nan, inplace=True)

    # is there any work to do?
    imp_cols = list(set(a_df_.columns) - set(ex_cols))
    if df_[imp_cols].isnull().sum().sum() == 0:   # nothing to impute
        s_ut.my_print('No missing values for ' + str(a_df_.columns))
        return a_df_

    cat_cols = list(df_[imp_cols].select_dtypes(include='object').columns)
    ncat_cols = list(df_[imp_cols].select_dtypes(exclude='object').columns)
    
    cat_dict = {c: [x for x in df_[c].unique() if pd.notna(x)] for c in cat_cols}
    na_df = df_[cat_cols].isnull()                 # DF of missing values
    cat_list = list(cat_dict.values())
    df_[cat_cols] = df_[cat_cols].fillna('_nan_')   # replace NaN to be able to encode

    # encode cats
    if len(cat_list) > 0:
        start = time.time()
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False, categories=cat_list)
        sep = '<::>'
        enc_cols = [c + sep + str(vx) for c, v in cat_dict.items() for vx in v]
        df_enc = pd.DataFrame(enc.fit_transform(df_[cat_cols].copy()), columns=enc_cols)
        df_enc = df_enc.apply(set_nans, na_df=na_df, sep=sep, axis=0)       # add NaNs back for imputation
        end = time.time()
        s_ut.my_print('imputer: encoded categories: ' + str(list(cat_cols)) + ' encoding secs:  ' + str(end - start))
    else:
        enc = None
        df_enc = pd.DataFrame()

    # scale non-cats: scale numerical and bool
    if len(ncat_cols) > 0:
        scaler = MinMaxScaler()
        df_scl = pd.DataFrame(scaler.fit_transform(df_[ncat_cols].values), columns=ncat_cols)
    else:
        scaler = None
        df_scl = pd.DataFrame()

    # impute
    start = time.time()
    if len(ncat_cols) > 0:
        if len(cat_cols) > 0:
            imp_df = df_enc.merge(df_scl[ncat_cols], left_index=True, right_index=True, how='inner')   # DF to impute
        else:
            imp_df = df_scl[ncat_cols].copy()
    else:
        if len(cat_cols) > 0:
            imp_df = df_enc
        else:
            s_ut.my_print('imputer: nothing to impute')
            return a_df_.copy()

    my_imputer = IterativeImputer(max_iter=10, random_state=0, min_value=0.0, max_value=1.0,  n_nearest_features=min(3, len(imp_df.columns)), verbose=2)
    zf = pd.DataFrame(my_imputer.fit_transform(imp_df), columns=imp_df.columns)
    end = time.time()
    s_ut.my_print('imputer: imputation secs: ' + str(end - start))

    # decode
    if len(cat_list) > 0:
        start = time.time()
        df_dec = pd.DataFrame(enc.inverse_transform(zf[df_enc.columns]), columns=cat_cols)
        end = time.time()
        s_ut.my_print('imputer:  decoding secs: ' + str(end-start))
    else:
        df_dec = pd.DataFrame()

    # de-scale
    df_dsc = pd.DataFrame(scaler.inverse_transform(zf[ncat_cols]), columns=ncat_cols) if len(ncat_cols) > 0 else pd.DataFrame()

    # reduce
    start = time.time()
    if len(ncat_cols) > 0:
        if len(cat_cols) > 0:
            df_list = [df_[ex_cols], df_dec, df_dsc]
        else:
            df_list = [df_[ex_cols], df_dsc]
    else:
        if len(cat_cols) > 0:
            df_list = [df_[ex_cols], df_dec]
        else:
            s_ut.my_print('imputer: nothing to impute')
            return a_df_.copy()

    z_all = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True, how='inner'), df_list)
    end = time.time()
    s_ut.my_print('imputer:  reduce secs: ' + str(end-start))
    return z_all


if __name__ == '__main__':
    # df = pd.DataFrame({
    #     'a': [1,2,np.nan, 4, 5, 6, 7],
    #     'b': ['x', 'y', None, None, 'z', 'x', 'z'],
    #     'c': [1.5, 2.3, 5.2, 3, None, np.nan, 5.2],
    #     'ds': ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'],
    #     'd': [None, None, 'a', 'bb', 'a', 'a', 'bb']}
    # )
    # ex_cols_ = ['ds']
    df = pd.read_parquet('~/my_tmp/phone-aht_2019-11-23.par')
    from capacity_planning.utilities import pandas_utils as p_ut

    new_vals = p_ut.clean_cols(df, ["service_region", "language", "sector", "interaction_type"],
                               '~/my_repos/capacity_planning/data/config/col_values.json', check_new=True, do_nan=True)
    ex_cols_ = ['ds', 'agent_id']
    df['tenure_days'] = df['tenure_days'].apply(lambda x: x if x > 0 else np.nan)
    # df = df[df['ds'] >= '2019-06-01'].copy()
    print(df.head(10))
    zz = impute(df, ex_cols=ex_cols_)
    print(zz.head(10))
    print('DONE')
