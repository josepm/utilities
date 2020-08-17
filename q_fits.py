"""
data fitting
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize

from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut

DO_MP = False
DEBUG = False


# ###########################################################
# ###########################################################
# ###########################################################
# ############## Quantile Fitting Functions #################
# ###########################################################
# ###########################################################
# ###########################################################


def set_fit(a):
    # generic function that sets stuff for fitting
    # returns the quantile dict {prob: x_prob, ...} and the data avg
    df, _y_col, tk = a
    _ts_name, _ts_key = tk

    avg = df.loc[df.index[0], _y_col]
    std = None
    if _y_col == 'yhat':
        qd = {(float(c[1:-3]) / 100.0): df.loc[df.index[0], c] for c in ['q10hat', 'q25hat', 'q50hat', 'q75hat', 'q90hat']}
    else:
        qd = {(float(c[1:]) / 100.0): df.loc[df.index[0], c] for c in ['q10', 'q25', 'q50', 'q75', 'q90']}
    return qd, avg, std


# ###############################################################
# #################### Exponential Fit ##########################
# ###############################################################


def expon_qfit(a):
    # fits distro using directly avg from capacity_planning.data using exponential (order = 1)
    # ignores order in cfg
    qd, avg, std = set_fit(a)      # get quantile dict and data avg
    df, _y_col, tk = a
    _ts_name, _ts_key = tk
    dict_out = {
        'prob': 1.0,
        'sh': np.nan,
        'sc': 1.0 / avg,
        'err': 0.0,
        'reg': None,
        'qs': None,
        'mdl_std': 1 / avg,
        'avg': avg,
        'std': std
    }
    dict_out['ts_name'] = _ts_name
    dict_out['ts_key'] = _ts_key
    dict_out['dist_name'] = 'expon'
    dict_out['probs'] = [[1.0]]
    dict_out['pars'] = [[dict_out['sc']]]
    dict_out['order'] = 1
    _ = [dict_out.pop(x, None) for x in ['sc', 'sh', 'prob']]
    return pd.DataFrame(dict_out)

# ###############################################################
# #################### Gamma Fit (no mixture) ###############
# ###############################################################


def _gamma_qfit(qdict, m, std):
    def _obj(x, *args):
        avg, qd, reg = args
        sh = x * x
        sc = avg / sh
        dobj = sps.gamma(sh, scale=sc)
        verr2 = np.array([(1 - dobj.ppf(q) / xq) ** 2 for q, xq in qd.items() if xq > 0.0])
        if len(verr2) == 0:
            s_ut.my_print('WARNING: gamma_qfit has invalid quantiles: ' + str(qd) + 'avg: ' + str(avg))
            err2 = 100000.0
        else:
            err2 = np.mean(verr2)
        return err2 + reg * sh

    dopt, e_min, v_max = None, np.inf, 0.0
    for reg in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        shx = np.array([1.0])
        res = minimize(_obj, shx, args=(m, qdict, reg), method='BFGS')      # Powell, BFGS, CG, Nelder-Mead
        if res.success is True:
            shx = res.x
            sh = shx * shx
            err = res.fun - reg * sh
            sc = m / sh
            mdl_mean = sh * sc
            mdl_var = mdl_mean * sc
            if err < e_min:
                e_min = err
                dopt = {'prob': 1.0, 'sh': sh, 'sc': sc,  'err': e_min, 'reg': reg, 'qs': [list(qdict.keys())], 'mdl_std': np.sqrt(mdl_var), 'std': std, 'avg': m}
    return dopt


def gamma_qfit(a):
    # _ts_name, _ts_key = a[3]
    _ts_name, _ts_key = a[2]
    qd, avg, std = set_fit(a)      # get quantile dict and data avg
    dict_out = _gamma_qfit(qd, avg, std) if (len(qd) > 0 and min(qd.values()) > 0.0 and avg > 0.0) else None

    if dict_out is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: gamma fit failed for ts_name: ' + str(_ts_name) + ' ts_key: ' + str(_ts_key))
        _ = p_ut.save_df(a[0], TempUtils.tmpfile('fit_df_' + _ts_name + '_' + _ts_key))
        return None
    else:    # fill output DF
        dict_out['ts_name'] = _ts_name
        dict_out['ts_key'] = _ts_key
        dict_out['dist_name'] = 'gamma'
        dict_out['probs'] = [[1.0]]
        dict_out['pars'] = [[dict_out['sh'][0], dict_out['sc'][0]]]
    _ = [dict_out.pop(x, None) for x in ['sc', 'sh', 'prob']]
    return pd.DataFrame(dict_out)

# #################################################################
# #################################################################
# #################################################################

# ###############################################################
# #################### LogNormal Fit (no mixture) ###############
# ###############################################################


def _lognorm_qfit(qdict, m, std):
    def _obj(x, *args):
        avg, qd, reg = args
        sh = x * x
        sc = avg * np.exp(- (sh * sh / 2))
        dobj = sps.lognorm(sh, scale=sc)
        verr2 = np.array([(1 - dobj.ppf(q) / xq) ** 2 for q, xq in qd.items() if xq > 0.0])
        if len(verr2) == 0:
            s_ut.my_print('WARNING: lognorm_qfit has invalid quantiles: ' + str(qd) + ' avg: ' + str(avg))
            err2 = 100000.0
        else:
            err2 = np.mean(verr2)
        return err2 + reg * sh

    dopt, e_min, v_max = None, np.inf, 0.0
    for reg in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0]:
        shx = np.array([1.0])
        res = minimize(_obj, shx, args=(m, qdict, reg), method='BFGS')      # Powell, BFGS, CG, Nelder-Mead
        if res.success is True:
            shx = res.x
            sh = shx * shx
            err = res.fun - reg * sh
            sc = m * np.exp(- (sh * sh / 2))
            mdl_mean = sc * np.exp((sh * sh / 2))
            mdl_var = mdl_mean ** 2 * (np.exp(sh * sh) - 1)
            if err < e_min:
                e_min = err
                dopt = {'prob': 1.0, 'sh': sh, 'sc': sc,  'err': e_min, 'reg': reg, 'qs': [list(qdict.keys())], 'mdl_std': np.sqrt(mdl_var),  'std': std, 'avg': m}
    return dopt


def lognorm_qfit(a):
    # _ts_name, _ts_key = a[3]
    _ts_name, _ts_key = a[2]
    qd, avg, std = set_fit(a)      # get quantile dict and data avg
    dict_out = _lognorm_qfit(qd, avg, std) if (len(qd) > 0 and min(qd.values()) > 0.0 and avg > 0.0) else None

    if dict_out is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: lognorm fit failed for ts_name: ' + str(_ts_name) + ' and ts_key: ' + str(_ts_key))
        _ = p_ut.save_df(a[0], TempUtils.tmpfile('fit_df_' + _ts_name + '_' + _ts_key))
        return None
    else:         # fill output DF
        dict_out['ts_name'] = _ts_name
        dict_out['ts_key'] = _ts_key
        dict_out['dist_name'] = 'lognorm'
        dict_out['probs'] = [[1.0]]
        dict_out['pars'] = [[dict_out['sh'][0], dict_out['sc'][0]]]
    _ = [dict_out.pop(x, None) for x in ['sc', 'sh', 'prob']]
    return pd.DataFrame(dict_out)

# #################################################################
# #################################################################
# #################################################################

