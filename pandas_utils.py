import pandas as pd
import os
import sys
import numpy as np
import gzip
import flatdict
import json
import operator
from itertools import product
from fbprophet import Prophet
from functools import reduce

from capacity_planning.utilities import sys_utils as su
from capacity_planning.utilities import time_utils as tu
# from capacity_planning.utilities import TempUtils


def _get_bin_vals(df, t_col, bin_cnt, fbin, qfunc):
    """
    returns a DF with bin values based on the t_col bins, i.e t_col defines the aggregation bins for the other columns
    supports duplicated values
    :param df: input DF
    :param t_col: column to build the initial bin breakdown from
    :param bin_cnt:  number of bins to generate
    :param fbin:  binning method: count(qcut) or size(cut)
    :param qfunc:  function to use to compute the bin value: np.median, np.mean, ... in numerical columns. In discrete columns, the mean is always used.
    :return:  DF with same numerical cols as df and bin_cnt rows. The non-numerical cols are split into dummy cols.
    """
    def nbin_agg(zf, n_func, cols):
        fo = zf[cols].apply(n_func, axis=0)
        fo['_bin_prob'] = len(zf)     # normalized later
        return fo

    if len(df) == 0 or df is None:
        return None

    qidx = '_' + t_col + '_idx'                                             # column with the t_col bin number
    q_df = None
    s_df = df.copy()
    dropped_df = list()
    while q_df is None and bin_cnt >= 2:
        try:
            q_df = pd.concat([s_df, fbin(s_df[t_col], int(bin_cnt), labels=range(int(bin_cnt)))], axis=1, sort=True)    # add bin number column
        except ValueError:                                                    # repeat values: fix and try again
            vc = s_df[t_col].value_counts()
            cm_val, cm_cnt = vc.idxmax(), vc.max()                            # most common repeat value and repeat count
            if cm_cnt <= 1:                                                   # this should not happen!
                su.my_su.my_print('invalid max count for ' + t_col + '. Should be > 1')
                su.my_su.my_print(vc.head())
                return None
            mask = s_df[t_col] == cm_val
            if cm_cnt >= len(s_df) / bin_cnt:                                  # repeat value spans more than one bin
                bin_cnt *= float(len(s_df) - cm_cnt) / len(s_df)               # adjust qtile to keep proportions
                dropped_df.append(s_df[mask].copy())                          # save the portion dropped from s_df to bin later
                s_df = s_df[~mask].copy()                                     # new DF for fbin
            else:                                                             # add noise to the repeat value to drop repeats
                idx = np.argmin(np.abs(s_df[t_col][~mask].values - cm_val))   # closest value to cm_val, not equal to cm_val
                z = s_df[t_col][~mask].values[idx]
                std = min(np.abs(cm_val - z), np.abs(cm_val)) / 1000.0
                v_noise = np.where(mask, np.random.normal(0, std, len(s_df)), np.zeros(len(s_df)))
                s_df[t_col] += v_noise                                        # t_col with noise on most common value to remove repeats
        if q_df is not None and len(s_df) >= bin_cnt:
            break

    if q_df is not None and len(s_df) >= bin_cnt:
        q_df.columns = list(s_df.columns) + [qidx]
        if len(dropped_df) > 0:                    # bin repeated values
            su.my_print('_get_bin_vals:: ' + t_col + ' has duplicated values')
            bin_val = q_df[qidx].max()             # highest bin created
            for idx in range(len(dropped_df)):     # add bin number value to the dropped repeat values
                dropped_df[idx][qidx] = bin_val + 1 + idx
            dropped_df.append(q_df)                # add DF with bins
            q_df = pd.concat(dropped_df, axis=0, sort=True)   # DF with all bins, including repeated values

        # compute the value for each bin
        q_df[qidx] = q_df[qidx].astype(np.int64)                                                       # not a category
        n_cols = [c for c in q_df.columns if not(str(q_df.dtypes[c]) in ['object', 'category'])]       # only numerical cols
        n_df = q_df.groupby(qidx).apply(nbin_agg, n_func=qfunc, cols=n_cols).reset_index(drop=True)    # values in each bin for numerical cols
        n_df['_bin_prob'] /= n_df['_bin_prob'].sum()                                                   # fraction of samples (rows) in each bin
        d_cols = list(set(q_df.columns) - set(n_cols))                                                 # discrete columns
        d_df_list = [pd.get_dummies(q_df[d], prefix=d, prefix_sep='::') for d in d_cols]               # list of dummy DFs
        if len(d_df_list) > 0:                                                                         # compute value for discrete column: always use mean
            d_df = pd.concat(d_df_list, axis=1, sort=True)
            d_df[qidx] = q_df[qidx]
            c_df = d_df.groupby(qidx).apply(lambda z_df: z_df.apply(np.mean, axis=0)).reset_index(drop=True)  # value in each bin for non-numerical cols (prob)
            m_df = n_df.merge(c_df, on=qidx, how='outer')
        else:
            m_df = n_df.drop(qidx, axis=1)
        return m_df
    else:
        su.my_print('_get_bin_vals failed')
        return None


def _thres_bins(bin_df, thres):
    """
    non-numerical cols contain the prob that the cat happens in a bin
    this function allows to remove non-numerical cols from a bin DF with very low prob to get rid of categories with low probabilities
    :param bin_df: output of _get_bin_vals
    :param thres: removal threshold: cols with max less than thres get dropped
    :return: DF with bin values  for numerical and (thresholded) non-numerical cols
    """
    if bin_df is None:
        return None
    c_cols = [c for c in bin_df.columns if '::' in c]       # non-numerical columns
    n_cols = [c for c in bin_df.columns if not('::' in c)]  # numerical columns
    c_w = bin_df[c_cols].copy()
    n_w = bin_df[n_cols].copy()
    y = c_w.apply(lambda x: x if x.max() > thres else [np.nan] * len(x), axis=0)  # remove cat cols with very low values
    y.dropna(inplace=True, axis=1)
    col_dict = dict()
    for col in y.columns:
        n = col.split('::')[0]
        if n not in col_dict.keys():
            col_dict[n] = list()
        col_dict[n].append(col)
    for c, c_list in col_dict.items():
        y[c + '::other'] = 1.0 - c_w[c_list].sum(axis=1)      # lump all low prob cats into the 'other' cat
    return pd.concat([n_w, y], axis=1, sort=True)


def get_sz_bins(df, t_col, bin_cnt, qfunc=np.nanmedian, thres=None):
    """
    returns a DF with bin_cnt bins of equal size based on t_col, i.e t_col defines the aggregation bins for the other columns
    supports with duplicated values
    all bins are the same size in t_col
    for discrete columns, return the avg number of 1's in the bin
    :param df: input DF
    :param t_col: column to build the initial bin breakdown from
    :param bin_cnt:  number of hist bins to generate
    :param qfunc:  function to use to compute the bin value: np.median, np.mean, ...
    :param thres:  thres to drop non-numerical cols, None or between 0 and 1.
    :return:  DF with same numerical cols as df and bin_cnt rows
    """
    b_df = _get_bin_vals(df, t_col, bin_cnt, pd.cut, qfunc)
    if thres is None:
        return b_df
    else:
        return _thres_bins(b_df, thres)


def get_cnt_bins(df, t_col, bin_cnt, qfunc=np.nanmedian, thres=None):
    """
    returns a DF with bins based on the t_col quantiles, i.e t_col defines the aggregation bins for the other columns
    supports with duplicated values
    all bins have the same number of points
    for discrete columns, return the avg number of 1's in the bin
    :param df: input DF
    :param t_col: column to build the initial bin breakdown from
    :param bin_cnt:  number of bins to generate
    :param qfunc:  function to use to compute the bin value: np.median, np.mean, ...
    :param thres:  thres to drop non-numerical cols, None or between 0 and 1.
    :return:  DF with same numerical cols as df and bin_cnt rows
    """
    b_df = _get_bin_vals(df, t_col, bin_cnt, pd.qcut, qfunc)
    if thres is None:
        return b_df
    else:
        return _thres_bins(b_df, thres)


def df2json(df, f_out):
    """
    NOTE: only works if columns contain json-like elements: numbers, strings, lists, dict. Does not work for sets.
    write a df into a by-line json file
    df: input DF
    f_out: output file
    """
    f = gzip.open(f_out, 'wb') if f_out[-3:] == '.gz' else open(f_out, 'wb')
    f.write('\n'.join(r[1].to_json(orient='index') for r in df.iterrows()))
    f.close()


def df2json_gen(df, f_out):
    """
    NOTE: only works if columns contain json-like elements: numbers, strings, lists, dict. Does not work for sets.
    Generator version: slower but scales to larger DF's
    write a df into a by-line json file
    df: input DF
    f_out: output file
    """
    f = gzip.open(f_out, 'wb') if f_out[-3:] == '.gz' else open(f_out, 'wb')
    f.writelines((r[1].to_json(orient='index') + '\n' for r in df.iterrows()))
    f.close()


def df_to_json_lines(df, f_out):
    return df2json(df, f_out)


def json_line_to_df(fname):
    """
    reads json line file and returns DF
    :param fname: file in json line format
    :return: df
    """
    f = gzip.open(fname, 'r') if fname[-3:] == '.gz' else open(fname, 'r')
    lines = f.read().splitlines()
    f.close()
    data_list = [json.loads(line) for line in lines]
    return pd.DataFrame(data_list)


def df_inspect(df, df_name, rows=5, pid=False):
    """
    Inspect a dataframe
    pid su.my_prints the pid running
    """
    rows = max(rows, len(df.columns))
    pd.set_option('display.max_rows', rows)
    pd.set_option('display.max_columns', 2 + len(df.columns))
    w = max(20 * len(df.columns), 250)
    pd.set_option('display.width', w)
    su.my_print('------------ ' + df_name + ' ------------')
    if pid == True:
        su.my_print('pid: ' + str(os.getpid()))
    su.my_print('length: ' + str(len(df)))
    su.my_print('size: ' + str(df_sz_MBs(df)) + 'MB')
    su.my_print(df.head(rows))
    su.my_print(df.tail(rows))
    t_df = df.dtypes.reset_index()
    t_df.columns = ['cols', 'type']
    v = {}
    for i in t_df.index:
        c = t_df.iloc[i, 'cols']
        t = t_df.iloc[i, 'type']
        v[c] = [len(df[c].unique())] if t == np.object else [np.nan]
    v_df = pd.DataFrame(v).transpose()
    desc = df.describe().transpose()
    smry = pd.concat([df.isnull().sum(), df.dtypes, v_df, desc], axis=1, sort=True)
    smry.columns = ['nulls', 'dtypes', 'uniques'] + list(desc.columns)
    pd.set_option('display.max_columns', 2 + len(smry.columns))
    su.my_print(smry)
    su.my_print('------------------------------------')


def df_sz_MBs(df):
    return int((df.values.nbytes + df.index.nbytes + df.columns.nbytes) / 1048576.0)


def drop_cols(df):
    """
    drop single valued columns in a df or cols with NAs and a single value
    :param df: pd data frame
    :return: df with single value cols dropped
    unique() includes NA's
    value_counts().sum() excludes NAs
    len() - value_counts().sum() = number of NAs
    """
    z = df.dropna(axis=1, how='all')
    drop_list = []
    z_len = len(z)
    for c in z.columns:
        if len(z[c].unique()) == 1:
            drop_list.append(c)
        elif len(z[c].unique()) == 2 and z_len - z[c].value_counts().sum() > 0:  # NAs and one on-NA value
            drop_list.append(c)
        else:
            pass
    if len(drop_list) > 0:
        return z.drop(drop_list, axis=1)
    else:
        return z


def list_gb(df, gb_cols, from_col):
    """
    Take df and groupby gb_cols
    then list all the elements in column from_col in the column new_col as a list
    :return: Series indexed by gb_cols with lists
    """
    return df.groupby(gb_cols)[from_col].apply(lambda x: x.tolist())


def col_list_gb(df, gb_cols, col_list):
    """
    a wrapper for list_gb
    col_list: a list of columns
    :return: a DF indexed by gb_cols with the values in each col in col list turned into a list
    """
    df_list = [list_gb(df, gb_cols, c) for c in col_list]
    return pd.concat(df_list, axis=1, sort=True)


def nested_dict_to_df(a_dict, col_names, key_depth=None):
    """
    builds a df from a nested dict. Requires that all key paths be equally deep
    use pandas to_dict when 3 columns (keys levels) are left. otherwise, go recursive
    :param a_dict: dict to turn to a DF
    :param col_names: col names for the output DF
    :param key_depth: key depth entries to return. If None take the most common. Other entries are su.my_printed and dropped.
    :return: a DF with columns given by col_names and rows given by either the keys or the values in the input dict
    Example
    big_dict = {'a': {
                        'p1': {'d1': {'00': 1, '01': 2}, 'd2': {'00': 2, '02': 3}},
                        'p2':  {'d2': {'04': 4, '01': 2}, 'd3': {'00': 2, '01': 3}}
                    },
                'b': {
                    'p1': {'d1': {'00': 1, '01': 2}, 'd2': {'00': 2, '02': 3}},
                    'p3':  {'d4': {'00': 1, '01': 2}, 'd3': {'00': 2, '01': 3}},
                    'p2':  {'d1': {'00': 2, '01': 2}},
                    'p3':  {'d1': {'05': None}}
                },
                'c': {
                    'p1': {'d3': 2, 'd2': {'01': 5}},
                    'p2': 4,
                    'p4': 'd1'
                }
            }
    Example:
    nested_dict_to_df(big_dict, ['col', 'pod', 'date', 'hr', 'val'])
    su.my_prints:
    the following entries are incomplete and will be dropped from the output
        c:p4:d1
        c:p1:d3:2
        c:p2:4
    and returns:
       col date  hr pod  val
    0    b   d2  00  p1  2.0
    1    a   d3  00  p2  2.0
    2    a   d3  01  p2  3.0
    3    b   d1  00  p1  1.0
    4    b   d1  01  p2  2.0
    5    b   d2  02  p1  3.0
    6    a   d2  00  p1  2.0
    7    a   d2  02  p1  3.0
    8    c   d2  01  p1  5.0
    9    b   d1  05  p3  NaN
    10   a   d2  04  p2  4.0
    11   a   d2  01  p2  2.0
    12   b   d1  00  p2  2.0
    13   a   d1  01  p1  2.0
    14   a   d1  00  p1  1.0
    15   b   d1  01  p1  2.0
    """
    delim = '>>@:::#-->'  # use an odd delimiter to avoid split problems
    flat_dict = flatdict.FlatDict(a_dict, delimiter=delim)

    if key_depth is None:  # find the most common key depth
        k_lists = [k.split(delim) for k in flat_dict.keys()]
        d_lens = dict()    # {key_len: count}
        for l in k_lists:
            k_len = len(l)
            if k_len not in d_lens:
                d_lens[k_len] = 0
            d_lens[k_len] += 1
        max_k_len = max(d_lens.iteritems(), key=operator.itemgetter(1))[0]
    else:
        max_k_len = key_depth   # key depth to output

    # check that col_names length match key depth, otherwise error because we cannot name all the DF columns we need or have too many columns
    if len(col_names) != 1 + max_k_len:
        su.my_print('invalid col_names: ' + str(col_names) + '. There should be ' + str(max_k_len) + ' columns')
        raise RuntimeError('failure')

    # keep only the path depth we want and drop the others
    # su.my_print what is dropped
    flat_drop = {k: v for k, v in flat_dict.iteritems() if len(k.split(delim)) != max_k_len}
    if len(flat_drop) != 0:
        su.my_print('the following entries are incomplete and will be dropped from the output')
        for k, v in flat_drop.iteritems():
            k_col = k.replace(delim, ':')
            su.my_print('\t' + str(k_col) + ':' + str(v))

    # recover the dict to make into a DF
    flat_keep = {k: v for k, v in flat_dict.iteritems() if len(k.split(delim)) == max_k_len}
    t_list = [tuple(k.split(delim) + [v]) for k, v in flat_keep.iteritems()]
    dict_list = [dict(zip(col_names, row)) for row in t_list]
    return pd.DataFrame.from_dict(dict_list)


def nested_update(a_dict, old_val, new_val):
    """
    update all instances of old_val by new_val at any nesting level
    :param a_dict: 
    :param old_val: 
    :param new_val: 
    :return: 
    """
    def is_mutable(val):  # true if val is mutable
        if val is None:
            return False

        immutables = [str, int, bool, float, tuple]
        for t in immutables:
            if isinstance(val, t):
                return False
        return True

    if is_mutable(old_val):
        su.my_print('nested update: cannot update mutable value: ' + str(old_val))
        return

    for k, v in a_dict.iteritems():
        if isinstance(v, dict):
            nested_update(v, old_val, new_val)
        else:
            if old_val is None:
                if a_dict[k] is None:
                    a_dict[k] = new_val
            else:
                if a_dict[k] == old_val:
                    a_dict[k] = new_val


def w_mean(df, m_col, w_col):
    """
    weighted average of m_col using w_col as weights
    :param df:
    :param m_col:
    :param w_col:
    :return: the weighted average
    """
    return np.average(df[m_col].values, weights=df[w_col].values)


def trim_df(a_df, rge=1.5, qtiles=[0.25, 0.75], cols=None, upr=True, lwr=True, msg=None):
    """
    basic outlier IQR trim of a DF.
    Drop any row with any of its col in cols outside the interval [q1 - rge * iqr, q3 + rge * iqr]
    :param a_df: DF to trim
    :param cols: list of cols or str (single col) to use in the trimming. If None use all columns
    :param rge: iqr factor
    :param upr: if True drop upper (right) outliers
    :param lwr: if True drop lower (left) outliers
    :param qtiles: quantiles for the iqr range
    :param msg: report on data dropped with msg (eg cols, df, context) if not None
    :return: returns trimmed DF
    """
    if len(a_df) > 0:
        if isinstance(a_df, type(pd.DataFrame())):
            if isinstance(cols, str):
                cols = [cols]
            a_df_cols = list(a_df.columns)
        else:  # series
            cols, a_df_cols = None, None
        t_df = a_df.copy() if cols is None else a_df[cols].copy()
        q_df = t_df.quantile(qtiles)
        q_lwr, q_upr = qtiles
        if q_lwr > q_upr:
            q_upr, q_lwr = q_lwr, q_upr
        iqr = q_df.diff(1).dropna().reset_index(drop=True)
        if len(q_df) == 2 and len(iqr) == 1:
            lwr_thres = q_df.loc[q_lwr] - rge * iqr.iloc[0]
            upr_thres = q_df.loc[q_upr] + rge * iqr.iloc[0]

            # if bool_df == True, drop the row
            if upr == True and lwr == True:
                bool_df = (t_df < lwr_thres) | (t_df > upr_thres)
            elif upr == True and lwr == False:
                bool_df = (t_df > upr_thres)
            elif upr == False and lwr == True:
                bool_df = (t_df < lwr_thres)
            else:  # ignore the trim
                bool_df = pd.DataFrame([False] * len(a_df))

            if isinstance(a_df, type(pd.DataFrame())):
                z_bool = bool_df.apply(lambda x: any(x), axis=1)  # true if any row value are outside the interval [q1-rge * iqr, q3 + rge * iqr]
                z_out = a_df[~z_bool][a_df_cols].copy()
            else:  # series
                z_out = a_df[~bool_df].copy()

            if msg is not None:
                diff = len(a_df) - len(z_out)
                if diff != 0:
                    su.my_print(str(msg) + ':: dropped ' + str(diff) + ' outliers out of ' + str(len(a_df)) + ' points')
            return z_out
        else:
            return None
    else:
        return None


def iqr_filter(r_df, coef=3.0, q_lwr=0.25, q_upr=0.75):
    q_upr = r_df.quantile(q_upr)
    q_lwr = r_df.quantile(q_lwr)
    iqr = q_upr - q_lwr
    return q_upr + coef * iqr, q_lwr - coef * iqr   #


def median_filter(r_df, coef=3.0):
    m, stdev = r_df.median(), r_df.std()
    return m + coef * stdev, m - coef * stdev  #


def ts_outliers(y_df, t_col, y_col, coef=3.0, verbose=False, replace=False, ignore_dates=None, lbl_dict=None, r_val=1.0):      # set outliers to NaN
    """
    Find outliers in y_col which is a time series using IQR method or median filter.
    Assumes y_col >= 0
    :param df: DF with y_col (data) and t_col
    :param t_col: time column name.
    :param y_col: data column
    :param coef: IQR coefficient
    :param verbose: verbose
    :param lbl_dict: into dict (context)
    :param r_val: r_val = 1 replaces by the yhat_upr/yhat_lwr value, r_val=0 replaces by yhat. In between, a weighted avg
    :param replace: if True replace the outlier value(s) by the Prophet in-sample forecast. If false, set outlier to nan
    :param ignore_dates: do not replace outliers for dates in this list
    :return: DF with either nan in outliers or fit outliers
    """
    if len(y_df) <= 10:
        su.my_print(str(os.getpid()) + ' WARNING: not enough points for outlier detection: ' + str(len(y_df)))
        return y_df, np.nan, None

    # look for outliers
    _y_df = y_df.copy()
    _y_df.rename(columns={t_col: 'ds', y_col: 'y'}, inplace=True)
    _y_df.reset_index(inplace=True, drop=True)
    try:
        if verbose:
            m = Prophet(changepoint_range=0.9)
            m.fit(_y_df[['ds', 'y']])
        else:
            with su.suppress_stdout_stderr():
                m = Prophet(changepoint_range=0.9)
                m.fit(_y_df[['ds', 'y']])
    except ValueError:
        su.my_print(str(os.getpid()) + ' ERROR: prophet err: returning original DF. Data len: ' + str(len(_y_df)) + ' Saving to ' + '~/my_tmp/_prophet_df.par')
        _y_df.rename(columns={'ds': t_col, 'y': y_col}, inplace=True)
        save_df(_y_df, '~/my_tmp/_y_df')
        return None, np.nan, None

    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    y_vals = _y_df['y'].copy()  # they will be filtered later
    _y_df['yhat'] = forecast['yhat']
    _y_df['resi'] = _y_df['y'] - _y_df['yhat']

    # use iqr or median filter
    # using Prophet's interval_width does not work as it is a quantile,
    # and about the same number of outliers is always found on avg ~ len * (1 - interval_width)
    upr, lwr = iqr_filter(_y_df['resi'], coef=coef, q_lwr=0.25, q_upr=0.75)  # iqr
    # upr, lwr = median_filter(_y_df['resi'], coef=coef)                     # median filter

    _y_df['yhat_upr'] = forecast['yhat'] + upr
    _y_df['yhat_lwr'] = forecast['yhat'] + lwr
    _y_df.rename(columns={'ds': t_col, 'y': y_col}, inplace=True)

    # no outlier if yhat_lwr <= y <= yhat_upr
    _y_df['is_outlier'] = (y_vals > _y_df['yhat_upr']) | (y_vals < _y_df['yhat_lwr'])
    n_outliers = _y_df['is_outlier'].sum()
    err = np.round(100 * n_outliers / len(_y_df), 0)
    if ignore_dates is None:
        ignore_dates = list()

    off = None
    if n_outliers > 0:
        if verbose is True:
            save_df(_y_df, '~/my_tmp/outliers_DF_' + str(y_col) + '_' + str(lbl_dict))   # no outlier processing yet
        su.my_print(str(os.getpid()) + ' WARNING::column ' + y_col + ' has ' + str(len(_y_df)) +
                    ' rows and ' + str(n_outliers) + ' outliers (' + str(err) + '%) for context ' + str(lbl_dict))
        b_dates = ~_y_df[t_col].isin(ignore_dates)                          # boolean dates adjuster: when true, an outlier on that date can be adjusted
        b_adj = _y_df['is_outlier'] & b_dates                               # boolean outlier adjuster: if true it is an outlier we can adjust
        if replace is False:
            _y_df[y_col] = y_vals * (1 - b_adj) + np.nan * b_adj
        else:
            _y_df[y_col] = y_vals * (1 - b_adj) + \
                           (r_val * _y_df['yhat_upr'] + (1.0 - r_val) * _y_df['yhat']) * ((y_vals > _y_df['yhat_upr']) & b_dates) + \
                           (r_val * _y_df['yhat_lwr'] + (1.0 - r_val) * _y_df['yhat']) * ((y_vals < _y_df['yhat_lwr']) & b_dates)

        if verbose is True:    # print outlier info: note that actuals are already filtered wheras the original value is in the outlier column
            off = _y_df[b_adj].copy()
            su.my_print('*************** outlier detail ************')
            print(off)
    _y_df.drop(['resi', 'yhat', 'yhat_upr', 'yhat_lwr', 'is_outlier'], axis=1, inplace=True)
    return _y_df, err, off


def df_hist(in_df, v_col, w_col, g_col=None, v_col_vals=None):
    """
    get the v_col histogram of the DF with weights in w_col, ie p(v_col=v) = sum(x[r_col]: x[vcol]=v)/sum(x[r_col])
    assumes w_col is numeric and v_col has a finite number of values
    :param in_df: input DF
    :param w_col: col of float values (weights)
    :param v_col: col with the values to get the histogram of
    :param g_col: pre-pivot grouping columns wrt v_col to group duplicates in v_vol
    :param v_col_vals: subset of the v_col values
    :return: a DF with cols in the values of v_col and a totals columns for the total weight
    """
    b_df = in_df.copy() if v_col_vals is None else in_df[in_df[v_col].isin(v_col_vals)].copy()
    p_df = b_df if g_col is None else b_df.groupby([g_col, v_col]).agg({w_col: np.sum}).reset_index()      # aggregate duplicates before pivot
    cnts_df = p_df.pivot(index=g_col, columns=v_col, values=w_col)
    cnts_df.fillna(0, inplace=True)
    cnts_df.columns = [v_col + ':' + c for c in cnts_df.columns]
    s = cnts_df.sum(axis=1)
    hist_df = cnts_df.div(s, axis=0)
    hist_df[w_col] = s
    hist_df.reset_index(inplace=True)
    return hist_df


# def multi_merge(df_list, on, how):
#     """
#     merge the list of DFs according to m_cols (
#     :param df_list: list of DFs
#     :param on: list of columns. Must be in all DFs
#     :param how: inner, outer. Note with left and right, the order of df1 and df2 makes a difference in the lambda below
#     :return:
#     """
#     df_list_ = [f for f in df_list if f is not None]
#     return ft.reduce(lambda df1, df2: df1.merge(df2, on=on, how=how), df_list_) if len(df_list_) > 0 else None


def get_data(fpath, usecols=None, date_format=None, date_cols=list(), str_cols=list(), num_cols=list(), b_cols=list(), cat_cols=list(), other_cols=list()):
    # reads from csv files
    # manages NaN (NA vs. North America)
    # formats cols (time, str, float)
    def to_bool(pd_series):
        # only handles NaNs and values 0/1 or True/False
        is_null = pd_series.isnull()
        yn = pd_series[is_null]
        nn = pd_series[~is_null]
        d = {'False': False, 'True': True, '0': False, '1': True, 0: False, 1: False, True: True, False: False}
        for v in nn.unique():
            if v not in list(d.keys()):
                su.my_print(str(v) + ' :::::::WARNING:::::: to_bool: invalid values for ' + str(pd_series.name) + ': ' + str(nn.unique()))
                return pd_series
        s = pd.concat([yn, nn.map(d)], axis=0, sort=True)
        return s.sort_index()

    df = pd.read_csv(fpath,
                     usecols=usecols,
                     dtype=str,
                     keep_default_na=False,
                     na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                                '#1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'undefined', 'unknown'])
    # date cols
    if date_format is not None:
        date_list = [pd.DataFrame(pd.to_datetime(df[c].values, format=date_format), columns=[c]) for c in date_cols]
    else:
        try:
            date_list = [pd.DataFrame(pd.to_datetime(df[c].values, unit='s'), columns=[c]) for c in date_cols]
        except ValueError:
            date_list = [pd.DataFrame(pd.to_datetime(df[c].values, infer_datetime_format=True, errors='coerce'), columns=[c]) for c in date_cols]
    date_df = pd.concat(date_list, axis=1, sort=True) if len(date_list) > 0 else pd.DataFrame()
    for c in date_cols:
        n_len = len(date_df[date_df[c].isnull()])
        if n_len > 0:
            su.my_print('WARNING: invalid dates in ' + str(c) + ': ' + str(n_len))

    # str cols
    sdf = df[str_cols].fillna('nan')
    str_df = sdf[str_cols].astype(str) if len(str_cols) > 0 else pd.DataFrame()

    # cat cols
    cdf = df[cat_cols].fillna('nan')
    cat_df = cdf[cat_cols].astype(str) if len(cat_cols) > 0 else pd.DataFrame()

    # bool cols
    b_list = [pd.DataFrame(to_bool(df[c])) for c in b_cols]
    bool_df = pd.concat(b_list, axis=1, sort=True) if len(b_list) > 0 else pd.DataFrame()

    # num cols
    n_list = [pd.DataFrame(pd.to_numeric(df[c].values, errors='coerce'), columns=[c]) for c in num_cols]
    num_df = pd.concat(n_list, axis=1, sort=True) if len(n_list) > 0 else pd.DataFrame()
    if other_cols is None:  # keep all the other cols
        other_cols = list(set(df.columns) - set(date_cols) - set(str_cols) - set(cat_cols) - set(num_cols) - set(b_cols))
    return pd.concat([df[other_cols], date_df, str_df, cat_df,  num_df, bool_df], axis=1, sort=True)


def clean_cols(df, col_list, c_vals_path, check_new=True, do_nan=False, rename=False):
    # clean categorical cols
    # col_list: cols to fix
    # cat_cols: cols to check for new
    # c_vals contains translation dicts
    # from capacity_planning.utilities import col_values as c_vals

    def flip_dict(d, cols, c_map):
        # d[col] = {'good_val': [...bad_vals...], ...}
        new_dict = dict()
        for c in cols:
            nc = c_map.get(c, c)  # mapped col name
            new_dict[c] = {kv: kk for kk, vv in d[nc].items() for kv in vv}
        return new_dict

    # find new invalid values and replace bad values with good ones
    # with open(os.path.expanduser(c_vals_path), 'r') as fp:
    su.my_print('pid: ' + str(os.getpid()) + ' p_ut:clean_cols: values check for cols ' + str(col_list))
    with open(os.path.expanduser(c_vals_path), 'r') as fp:
        c_vals = json.load(fp)

    v_dict = c_vals['col_values']
    m_dict = c_vals['col_maps']

    if rename:
        r_dict = dict()
        c_cols = dict()
        for c in col_list:
            c_cols[c] = [k for k, v in m_dict.items() if v == c and k in df.columns] if c in m_dict.values() else list()
            if len(c_cols[c]) == 0:
                su.my_print('ERROR: column ' + str(c) + ' has no mapping in m_dict: ' + str(m_dict) + ' and DF has columns ' + str(df.columns))
                sys.exit()
            elif len(c_cols[c]) > 1:
                su.my_print('ERROR: column ' + str(c) + ' has too many matches in m_dict: ' + str(m_dict) + ' and DF has columns ' + str(df.columns))
                sys.exit()
            else:
                r_dict[c_cols[c][0]] = c
        df.rename(columns=r_dict, inplace=True)

    my_cols = list()
    for c in col_list:
        if c in m_dict.keys():
            my_cols.append(c)
        else:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: clean_cols:: no mapping for column ' + str(c))
    f_dict = flip_dict(v_dict, my_cols, m_dict)
    _ = [df[c].update(df[c].map(f_dict[c])) for c in my_cols]    # in place replacement
    if do_nan is True:
        df.replace(['_drop', 'nan', 'other'], np.nan, inplace=True)

    # find new invalid values in cat cols
    new_ones = dict()
    if check_new is True:
        su.my_print('pid: ' + str(os.getpid()) + ' p_ut:clean_cols: starting new invalid values check for cols ' + str(my_cols))
        for c in my_cols:
            uniques = df[c].unique()
            new_vals = list(set(uniques) - set(v_dict[m_dict.get(c, c)]))
            if len(new_vals) > 0:
                new_ones[c] = new_vals
                # su.my_print('pid: ' + str(os.getpid()) + ' p_ut:clean_cols::col: ' + str(c) + ' has new invalid values: ' + str(new_vals))
    return new_ones


def plot_by_year(df_in, cols, tcol, base_year=None, time_scale='W'):
    # cols: columns to plot
    # tcol: time column
    # returns a DF based on df_in ready to plot cols aligned by weekday and week for each year
    # the values are aligned by week and day of the week using the iso calendar
    # base year is used to index the resulting data. If None is given, take the largest or the one with a whole year of data.
    # If there is no complete year, sets values to 0
    # base year is only for plotting purposes. The dates and values in base year may not match the exact date of the year.
    # since 365 and 366 are not multiples of 7, there will be some NAs ate the beginning or end of some years.

    df = df_in[[tcol] + cols].copy()
    df.set_index(tcol, inplace=True)
    i_name = df.index.name
    s = pd.Series(df.index)
    iso = s.apply(lambda x: x.isocalendar())
    df['iso_nbr'] = iso.apply(lambda x: x[1]).values if time_scale == 'W' else iso.apply(lambda x: x[2]).values

    if len(df) != len(df.index.unique()):
        su.my_print('plot by year has duplicated dates')
        return None

    # set base year and check that data is daily (at least during the base year)
    years = list(df.index.year.unique())
    if base_year is None:
        base_year = min(years)

    # collect cols to plot by year
    g_list = list()
    for y in years:
        g = df[df.index.year == y].copy()
        g.index.name = y
        g.columns = [str(y) + '_' + c if c in cols else c for c in df.columns]
        g.reset_index(inplace=True)
        g.drop(y, axis=1, inplace=True)
        g_list.append(g)
    mf = reduce(lambda x, y: x.merge(y, on='iso_nbr', how='outer'), g_list) if len(g_list) > 0 else None

    # build index DF
    base_idx = df.index[df.index.year == base_year]
    i_df = pd.DataFrame(index=base_idx)
    s = pd.Series(base_idx)
    iso = s.apply(lambda x: x.isocalendar())
    i_df['iso_nbr'] = iso.apply(lambda x: x[1]).values if time_scale == 'W' else iso.apply(lambda x: x[2]).values
    i_df.reset_index(inplace=True)

    a_df = i_df.merge(mf, on='iso_nbr', how='left')
    a_df.drop('iso_nbr', axis=1, inplace=True)
    a_df.set_index(i_name, inplace=True)
    return a_df    # will have NaN entries


def plot_regression(df, xcol, ycol, xreg_col, yreg_col, title='Title', legend=None, pad=3.5, orientation=90):
    """
    plots scatter for original data and the regression line
    :param df: data DF
    :param xcol: x axis values for train/test data or time
    :param ycol: y axis values for train/test data
    :param xreg_col: x axis values for regression data. If None use xcol values
    :param yreg_col: y values from the regression model
    :param title: plot title
    :param pad: padding to avoid cutting values
    :param orientation: xaxis labels/values orientation
    :return: None
    """
    import matplotlib.pyplot as plt

    if xreg_col is None:
        xreg_col = xcol

    if xcol in df.select_dtypes(include=[np.datetime64]).columns:  # xcol is datetime
        xvals = np.arange(len(df))
        xreg_vals = np.arange(len(df))
    else:
        xvals = df[xcol].values
        xreg_vals = df[xreg_col].values

    plt.scatter(xvals, df[ycol].values, color='black', label=ycol)

    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout(pad=pad)
    plt.grid(True)
    plt.plot(xreg_vals, df[yreg_col], color='red', linewidth=3, label=yreg_col)
    plt.legend(loc='upper left')
    if xcol in df.select_dtypes(include=[np.datetime64]).columns:                 # xcol is datetime
        plt.xticks(xvals, df[xcol].astype(str).values, rotation=orientation)      # 90 = vertical


def to_parquet(df, f_path, str_cols=None, reset_and_drop=True):
    # save DF to a parquet file
    # json dumps to str the cols in str_cols. This way we deal with dict, nested lists, ... that json can handle
    def serial(x):
        # this function is to convert objects to json-serializable objects
        if isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return x

    if str_cols is None:
        str_cols = list()

    # look for numeric columns with np.nan. They do not load so well to hive tables
    df_c = df.copy()
    for c in df_c.columns:
        if df_c[c].dtype == float or df_c[c].dtype[c] == int:
            if df_c[c].isnull().sum() > 0:
                df_c[c].fillna(np.nan, inplace=True)
                df_c[c] = df_c[c].apply(json.dumps)
                su.my_print('WARNING:: numeric column ' + c + ' has nulls. Converted to json string')

    # _path = os.path.expanduser(f_path)
    if os.path.exists(os.path.dirname(f_path)):
        for c in str_cols:
            df_c[c] = df_c[c].apply(lambda x: json.dumps(serial(x)))
        if reset_and_drop is True:
            df_c.reset_index(inplace=True, drop=True)
        if '.gz' in f_path:
            df_c.to_parquet(f_path, compression='gzip')
        else:
            df_c.to_parquet(f_path)
    else:
        su.my_print('to_parquet:: directory ' + str(os.path.dirname(f_path)) + ' does not exit for file ' + str(f_path))
        raise RuntimeError('failure')


def from_parquet(f_path, str_cols=None):
    # read the DF from a parquet file
    # converts str cols into dict or other json dumped structures
    if str_cols is None:
        str_cols = list()
    # _path = os.path.expanduser(f_path)
    if os.path.isfile(f_path):
        df_out = pd.read_parquet(f_path)
        for c in str_cols:
            if c in df_out.columns:
               df_out[c] = df_out[c].apply(json.loads)
            else:
                su.my_print('from_parquet:: ' + c + ' is not a column')
        return df_out
    else:
        su.my_print('from_parquet:: file ' + f_path + ' does not exit')
        raise RuntimeError('failure')


def set_date_range(init_date, final_date, freq, w_start='MON'):
    # generate a pairs of start end end dates spaced by freq with start dates centered on Mondays by default
    wday = '-' + w_start if 'W' in freq else ''
    if isinstance(init_date, str):
        init_date = pd.to_datetime(init_date)
    if isinstance(final_date, str):
        final_date = pd.to_datetime(final_date)
    dr = list(pd.date_range(init_date, end=final_date, freq=freq + wday))
    if len(dr) > 0:
        if dr[0] > init_date:
            dr.insert(0, init_date)
    else:
        dr = [init_date, final_date]
    dr.sort()
    pairs = [(dr[i], dr[i + 1]) for i in range(len(dr) - 1)]  # start_d, end_d window
    if dr[-1] < final_date:
        pairs.append((dr[-1], final_date))
    return pairs


def save_df(a_df, os_path, sep=',', msg_hdr=None, msg_tail=None, verbose=True, index=False):
    hdr = '' if msg_hdr is None else msg_hdr
    tail = '' if msg_tail is None else msg_tail
    if os.path.dirname(os_path):  # dir exists
        try:
            a_df.to_parquet(os_path + '.par')
            if verbose:
                su.my_print(hdr + 'pid: ' + str(os.getpid()) + ' save_df::Saving to ' + os_path + '.par' + tail)
            return os_path + '.par'
        except:
            su.my_print(hdr + 'pid: ' + str(os.getpid()) + ' save_df::Failed for ' + os_path + '.par' + tail)
            try:
                a_df.to_csv(os_path + '.csv.gz', sep=sep, compression='gzip', index=index)
                if verbose:
                    su.my_print(hdr + 'pid: ' + str(os.getpid()) + ' save_df::Saving to ' + os_path + '.csv.gz' + tail)
                return os_path + '.csv.gz'
            except FileNotFoundError:
                su.my_print('ERROR: could not save: ' + os_path + ' not found')
                return None
    else:
        return None


def read_df(f_name, sep=None):
    fn = os.path.expanduser(f_name)
    root, ext = os.path.splitext(fn)
    if ext == '' or ext == '.':
        for ext in ['.par', '.csv', '.tsv', '.csv.gz', '.tsv.gz']:
            sep = '\t' if 'tsv' in ext else ','
            f = read_df(fn + ext, sep=sep)
            if f is not None:
                return f
        return None
    else:
        if os.path.isfile(fn):
            su.my_print('read_df file: ' + str(fn))
            root, ext = os.path.splitext(fn)
            if 'gz' in ext or 'csv' in ext:
                try:
                    return pd.read_csv(fn, sep=sep)
                except:
                    su.my_print('read_csv failed for file ' + str(fn))
                    return None
            elif 'par' in ext:
                try:
                    return pd.read_parquet(fn)
                except:
                    su.my_print('read_parquet failed for file ' + str(fn))
                    return None
        else:
            su.my_print(fn + ' does not exist')
            return None

#  ############# DF dedup by close values ###########
def num_dedup(df, idx_cols, ycol, tol):
    # finds if there are entries with ycol values different by more than tol in the data when grouping by idx_cols
    # returns a dups DF with dup vals (same idx_col values and y values off by more than tol) and a dedups DF with unique idx_col and averaged ycol (all values were within tol)
    def y_avg(xf, ycol_):
        xf[ycol_] = xf[ycol_].mean()
        return xf

    def num_unique(x, ycol, tol):
        def _err(x1, x2):
            return np.abs(x1 - x2) / (2 * np.abs(x1 + x2))
        vals = [list(x[ycol].values), list(x[ycol].values)]
        arr = np.array([_err(*p) for p in product(*vals) if p[0] > p[1]])
        return len(arr[arr > tol])  # number of different values

    df_cols = df.columns
    g = df.groupby(idx_cols)
    gu = pd.DataFrame(g.apply(lambda x: num_unique(x, ycol, tol)))
    gu.columns = ['_uy']
    gu.reset_index(inplace=True)
    zf = df.merge(gu, on=idx_cols, how='left')
    dups = zf[zf['_uy'] >= 1].copy()         # diff y's and same idx cols
    dups.sort_values(by=idx_cols, inplace=True)
    dedups = zf[zf['_uy'] == 0].groupby(idx_cols).apply(y_avg, ycol_=ycol).reset_index()   # values within tol or no idx_col repeats
    dedups.drop_duplicates(subset=idx_cols + [ycol], inplace=True)
    return dups[df_cols], dedups[df_cols]


def to_df(data_list, a_lbl, sep='\t', cols=None, debug=False):
    q_data = data_list[0] if len(data_list) == 1 else data_list
    if q_data is None or len(q_data) <= 1:
        return None

    # each row in q_data is a tab separated string
    if cols is None:  # infer?
        cols = ['.'.join(c.split('.')[1:]) for c in q_data[0].split(sep)] if '.' in q_data[0] else q_data[0].split(sep)  # col names of the form <tb_name>.<col_name>: drop table name
        rows = [r.split(sep) for r in q_data[1:]]
    else:
        rows = [r.split(sep) for r in q_data]
    su.my_print('pid: ' + str(os.getpid()) + ' to_df: collected for ' + a_lbl + ':: rows: ' + str(len(rows)) + ' columns: ' + str(cols))
    ncols = len(cols)
    rcols = [r for r in rows if len(r) == ncols]
    if len(rows) - len(rcols) > 0:
        su.my_print('pid: ' + str(os.getpid()) + ' WARNING: ' + str(len(rows) - len(rcols)) + ' dropped for ' + a_lbl)
    if len(rcols) > 0:
        _df = pd.DataFrame(rcols, columns=cols)
        if list(_df.columns) == list(_df.loc[0,].values):
            _df.drop(0, axis=0, inplace=True)
        if debug:
            save_df(_df, TempUtils.tmpfile(a_lbl + '_get_data_df'))
        return _df
    else:
        su.my_print('pid: ' + str(os.getpid()) + ' WARNING: no valid data returned for ' + a_lbl)
        return None


def set_week_start(adf, tcol='ds'):
   tu.set_week_start(adf, tcol=tcol)
