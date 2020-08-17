"""
u test for back transforms

"""
import scipy
import numpy as np
import pandas as pd

from capacity_planning.utilities import ts_transformations as tx
import statsmodels.tsa.arima_process as ts


def u_test(y, cls, name):
    dates = pd.date_range(start='2018-01-01', periods=len(y))
    obj = cls(pd.DataFrame({'t': dates, 'y': y}), name)
    tx_val = obj.tx_val
    r_val = obj.tx_inv(tx_val, '2018-01-01', None, 'D')
    # if obj.DD is False:
    #     y0 = obj.get_init('2018-01-01', None, 'D')
    #     r_val = obj.tx_inv(tx_val, y0)
    #     return np.max(np.abs(y - r_val)), obj
    # else:
    #     y0 = obj.bcD.get_init('2018-01-01', None, 'D')  # initial values
    #     rD = obj.tx_inv(tx_val, y0)
    #     rD = obj.bcD.tx_inv(tx_val, y0)
        # y0 = obj.get_init('2018-01-01', None, 'D')  # initial values
        # r_val = obj.Dbc.tx_inv(rD, y0)
        # r_val = obj.Dbc.tx_inv(rD, y0)
    return np.max(np.abs(y - r_val)), obj


tx_list = [[0, None, 0], [2, None, 0], [0, None, 3], [2, 0, 1], [None, None, None]]

# simple rvs fit
for dname in ['norm', 'expon']:
    dist = getattr(scipy.stats, dname)
    for name in tx_list:
        if name[0] == 0 and name[2] == 0:
            cls = tx.GenBoxCox
            cname = dname + '-GenBoxCox'
        elif name[0] == 0 and name[2] != 0:
            cls = tx.GenBoxCoxD
            cname = dname + '-GenBoxCoxD'
        elif name[0] != 0 and name[2] == 0:
            cls = tx.DGenBoxCox
            cname = dname + '-DGenBoxCox'
        else:
            cls = tx.DGenBoxCoxD
            cname = dname + '-DGenBoxCoxD'
        diff, obj = u_test(dist.rvs(size=1000), cls, name)
        print('label: ' + cname + ' tx name: ' + str([obj.pre_lag, np.round(obj.lmbda, 4), obj.pst_lag]) + ' diff: ' + str(diff))


# ARMA process fit
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams]      # add zero-lag and negate
ma = np.r_[1, maparams]       # add zero-lag
for name in tx_list:
    if name[0] == 0 and name[2] == 0:
        cls = tx.GenBoxCox
        cname = 'ARMA-GenBoxCox'
    elif name[0] == 0 and name[2] != 0:
        cls = tx.GenBoxCoxD
        cname = 'ARMA-GenBoxCoxD'
    elif name[0] != 0 and name[2] == 0:
        cls = tx.DGenBoxCox
        cname = 'ARMA-DGenBoxCox'
    else:
        cls = tx.DGenBoxCoxD
        cname = 'ARMA-DGenBoxCoxD'
    data = ts.arma_generate_sample(ar, ma, 1000)
    diff, obj = u_test(data, cls, name)
    print('label: ' + cname + ' tx name: ' + str([obj.pre_lag, np.round(obj.lmbda, 4), obj.pst_lag]) + ' diff: ' + str(diff))


