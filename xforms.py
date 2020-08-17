"""
##############################################################
################## boxcox transformations ####################
##############################################################
"""

import numpy as np
import pandas as pd
import os
import copy
from scipy.stats import norm

from capacity_planning.utilities import sys_utils as su
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from capacity_planning.utilities import pandas_utils as p_ut


class NoTransform(object):      # fake transform class used when xform = None to maintain the API
    def __init__(self):
        self.lmbda = None
        self.lambdas_ = [None]
        self.name = 'no_transform'

    @staticmethod
    def transform(y):
        return np.copy(y)

    def fit(self, y):
        return self

    @staticmethod
    def fit_transform(y):
        return np.copy(y)

    @staticmethod
    def inverse_transform(y, y_var, lbl=None):
        return np.copy(y)


class Anscombe(object):
    # For Anscombe transform (https://en.wikipedia.org/wiki/Anscombe_transform)
    def __init__(self):
        self.is_error = False
        self.lmbda = None
        self.xf_done = False

    def fit(self, y):
        return self

    @staticmethod
    def transform(y):
        return 2.0 * np.sqrt(y + 3/8)

    def fit_transform(self, y):
        return self.transform(y)

    @staticmethod
    def inverse_transform(yx, y_var, lbl=None):
        if np.min(yx) < 0:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: range: ' + str(np.min(yx)))
            return None
        else:
            return (1 / 4) * np.power(yx, 2) - (1/8) \
                   + (1/4) * np.sqrt(3/2) * np.power(yx, -1) \
                   - (11 / 8) * np.power(yx, -2) +\
                   (5 / 8) * np.sqrt(3 / 2) * np.power(yx, -3)


class Transform(object):
    #     Yeo-Johnson transform, which is an extension of Box-Cox transformation but can handle both positive and negative values.
    #     References:
    #     Weisberg, Yeo-Johnson Power Transformations. https://www.stat.umn.edu/arc/yjpower.pdf
    #     Yeo + Johnson, A new family of power transformations to improve normality or symmetry, http://www.stat.wisc.edu/sites/default/files/tr1002.pdf
    #     For Anscombe transform (https://en.wikipedia.org/wiki/Anscombe_transform),
    #
    #     Bias removal
    #     for bias removal, ciw and alpha must be not None, otherwise no bias removal is done and ew return the median rather than the mean
    #     for bias removal, see https://robjhyndman.com/hyndsight/backtransforming/
    #     see http://davegiles.blogspot.co.uk/2013/08/forecasting-from-log-linear-regressions.html
    #     see http://data.princeton.edu/wws509/notes/c2s10.html
    #     and https://robjhyndman.com/hyndsight/backtransforming/

    def __init__(self, method, nqs, ceiling=None, floor=None, unbias=False):
        self.method = method
        self.ceiling = ceiling
        self.floor = floor
        self.lmbda = None
        self.name = method
        self.xf_done = False
        self.unbias = unbias  # not implemented
        self.lbl = None
        if method == 'yeo-johnson' or method == 'box-cox':
            self.xobj = PowerTransformer(method=method, standardize=False, copy=False)  # MUST have standardize = False
        elif method == 'quantile':
            self.xobj = QuantileTransformer(n_quantiles=int(nqs), output_distribution='normal', copy=False)
        elif method == 'logistic':
            self.xobj = Linearizer(ceiling, floor, self.unbias)
        elif method == 'log':
            self.xobj = LogTransform(self.unbias)
        elif method == 'anscombe':
            self.xobj = Anscombe()
        elif method is None:
            self.method = None
            self.xobj = NoTransform()
        else:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: set_xform: invalid method: ' + str(method))
            self.method = None
            self.xobj = NoTransform()

    def check_input(self, y):
        if isinstance(y, (float, int, np.float, np.int)):
            y = np.array([y])

        if isinstance(y, np.ndarray) is False:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid type: ' + str(type(y)))
            return None

        yc = np.copy(y)
        if len(np.shape(yc)) > 1:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid input shape: ' + str(np.shape(yc)))
            return None
        yx = np.reshape(yc, (1, -1))[0]

        if np.max(yx) == np.min(yx):
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: constant series: ' + str(np.min(yc)))
            return None
        if np.min(yx) < 0.0 and self.method == 'box-cox' or self.method == 'log' or self.method == 'anscombe':
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid range for method: ' + self.method + ' min: ' + str(np.min(yx)) + ' lbda: ' + str(self.lmbda))
            return None
        return np.reshape(yx, (-1, 1))

    def transform(self, y):
        if self.xf_done is False:
            su.my_print('pid: ' + str(os.getpid()) + ' ' + self.method + ' : must fit before transform')
            return None
        if self.method == 'logistic':
            return self.xobj.transform(y)
        elif self.method == 'log':
            return self.xobj.transform(y)
        else:
            return self._transform(y)

    def _transform(self, y):
        if self.method is None:
            return y
        else:
            ys = self.check_input(y)
            if ys is None:
                return None
            yt = self.xobj.transform(ys)
            ya = np.reshape(yt, (1, -1))[0]
            return self.check_output(ya, y)

    def fit(self, y):
        if self.xf_done is True:
            su.my_print('pid: ' + str(os.getpid()) + ' ' + self.method + ' : already fit. Create new object')
            return None
        else:
            self.xf_done = True
            if self.method == 'logistic':
                self.xobj.fit(y)
                return self
            elif self.method == 'log':
                self.xobj.fit(y)
                return self
            else:
                return self._fit(y)

    def _fit(self, y):
        ys = self.check_input(y)
        if ys is None:
            return None
        else:
            try:
                self.xobj.fit(ys)
            except ValueError:
                return None
            try:
                self.lmbda = self.xobj.lambdas_[0]
            except AttributeError:
                self.lmbda = None
            return self

    def fit_transform(self, y):
        if self.xf_done is True:
            su.my_print('pid: ' + str(os.getpid()) + ' ' + self.method + ' : already fit. Create new Transform instance')
            return None
        else:
            r = self.fit(y)
            return None if r is None else self.transform(y)

    def check_output(self, ya, y):
        if len(np.unique(ya)) == 1:  # xform failed: over/underflow?
            su.my_print(' WARNING: transform ' + self.method + ' failed with lambda ' + str(self.lmbda)
                        + ' and label: ' + str(self.lbl) + ' Trying Quantile')
            return self.reset_xform(y)
        else:
            return ya

    def reset_xform(self, y):   # in case of failure, try quantile
        if self.method == 'yeo-johnson' or self.method == 'box-cox':
            self.method = 'quantile'
            del self.xobj   # drops lmbda also
            self.xobj = QuantileTransformer(output_distribution='normal', copy=False)
            self.xf_done = False
            return self.fit_transform(y)
        else:
            return y

    def fcast_var(self, rdf, w):  # used to unbias inverse transforms
        # see https://robjhyndman.com/hyndsight/backtransforming/
        # rdf contains yhat_upr and yhat_lwr
        # w is is the quantile used for the yhat bounds
        # y_upr/lwr = yhat +/- q * sig
        q = 1.0 - (1.0 - w) / 2.0
        qval = norm.ppf(q) / 2.0
        try:
            diff = np.abs(rdf.diff(axis=1).dropna(axis=1)) / 2.0
        except TypeError:
            return 0.0
        if len(diff) > 0 and len(diff.columns) == 1:
            diff.columns = ['var']
            return (diff / qval) ** 2   # y_var
        else:
            return 0.0

    def _u_inverse_transform(self, y_mean, y_var):
        # return f(y_mean) + (y_var /2) * f''(y_mean) where f in the inverse transform function
        if self.unbias is False:
            return y_mean
        else:
            if y_mean is None:
                return None
            else:
                y_mean_ = np.reshape(y_mean, (1, -1))[0]
                y_var_ = 0.0 if y_var is None else np.reshape(y_var, (1, -1))[0]
                if self.method == 'box-cox':
                    fy, d2fy = self._box_cox_unbias(y_mean_)
                elif self.method == 'yeo-johnson':
                    fy, d2fy = self._yeo_johnson_unbias(y_mean_)
                elif self.method == 'quantile':
                    fy, d2fy = self._quantile_unbias(y_mean_)
                else:
                    fy, d2fy = y_mean_, 0.0
                return None if fy is None else (fy if d2fy is None else fy + (y_var_ / 2.0) * d2fy)

    def _box_cox_unbias(self, y_mean):
        if np.abs(self.lmbda) > 1.0e-02:
            z = 1 + self.lmbda * y_mean
            fy = np.power(z, 1.0 / self.lmbda)
            fy = self.interpolate_(fy, y_mean, nan_pct=0.2)
            d2fy = (1 - self.lmbda) * np.power(z, (1.0 / self.lmbda) - 2.0)
            d2fy = self.interpolate_(d2fy, y_mean, nan_pct=0.2)
        else:
            fy = np.exp(y_mean)
            d2fy = fy
        return fy, d2fy

    def _yeo_johnson_unbias(self, y_mean):
        fy = np.zeros_like(y_mean)
        d2fy = np.zeros_like(y_mean)
        pos = y_mean >= 0  # binary mask
        if np.abs(self.lmbda) < 1.0e-02:
            fy[pos] = np.exp(y_mean[pos]) - 1
            d2fy[pos] = np.exp(y_mean[pos])
        else:
            z = 1 + self.lmbda * y_mean[pos]
            fy[pos] = np.power(z, (1.0 / self.lmbda)) - 1.0
            d2fy[pos] = (self.lmbda - 1) * np.power(z, (1.0 / self.lmbda) - 2.0)

        if np.abs(2 - self.lmbda) < 10.e-02:
            fy[~pos] = 1.0 - np.exp(-y_mean[~pos])
            d2fy[~pos] = -np.exp(-y_mean[~pos])
        else:
            theta = 2 - self.lmbda
            z = 1 - theta * y_mean[~pos]
            fy[~pos] = 1 - np.power(z, 1.0 / theta)
            d2fy[~pos] = (theta - 1) * np.power(z, (1.0 / theta) - 2.0)
        return fy, d2fy

    def _quantile_unbias(self, y_mean):
        return y_mean, 0.0

    def inverse_transform(self, y, y_var, lbl=None):
        self.lbl = lbl
        if y is not None:
            if isinstance(y, (pd.core.series.Series, pd.core.frame.DataFrame)):
                y = y.values
        else:
            return None

        if y_var is not None:
            if isinstance(y, (pd.core.series.Series, pd.core.frame.DataFrame)):
                y_var = y_var.values

        if isinstance(y, np.ndarray) is False:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid type: ' + str(type(y)))
            return None

        if self.xf_done is False:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: cannot inverse_transform before fit is done')
            return None

        yc = copy.deepcopy(y)
        if self.method == 'logistic':
            yt = self.xobj.inverse_transform(y, y_var, lbl=lbl)
            yt = self.interpolate_(yt, yc, nan_pct=0.2)
            if yt is None:
                su.my_print('pid: ' + str(os.getpid()) + ' WARNING: inverse transform failed for label: ' +
                            str(self.lbl) + ' (method: ' + str(self.method))
                return None
            else:
                return yt
        elif self.method == 'log':
            yt = self.xobj.inverse_transform(y, y_var, lbl=lbl)
            yt = self.interpolate_(yt, yc, nan_pct=0.2)
            if yt is None:
                su.my_print('pid: ' + str(os.getpid()) + ' WARNING: inverse transform failed for label: ' +
                            str(self.lbl) + ' (method: ' + str(self.method))
                return None
            else:
                return yt
        elif self.method is None:
            return y
        else:      # box-cox, yj
            yt = self._inverse_transform(y, yc, y_var)
            if yt is None:
                su.my_print('pid: ' + str(os.getpid()) + ' WARNING: inverse transform failed for label: ' +
                            str(self.lbl) + ' (method: ' + str(self.method) + ' and lambda: ' + str(self.lmbda) + ')')
                return None
            else:
                yout = np.reshape(yt, (1, -1))[0] if self.method is not None else y
                return yout

    def _inverse_transform(self, y, yc, y_var):
        if self.unbias is False:
            ys = np.reshape(y, (-1, 1))
            yt = self.xobj.inverse_transform(ys)  # box-cox returns NaN on failure
        else:
            yt = self._u_inverse_transform(copy.deepcopy(y), y_var)  # unbiased inverse transform
        yt = self.interpolate_(yt, yc, nan_pct=0.2)
        return yt

    def interpolate_(self, y, yt, nan_pct=0.2):
        # y: inverse-transformed values (values in natural scale)
        # yt: pre-inverse transform (values in transformed scale)
        if y is None:
            return None
        else:
            yx = np.reshape(y, (1, -1))[0] if self.method is not None else y
        nulls = pd.Series(yx).isnull().sum()
        pct = 100.0 * np.round(nulls / len(yx), 2)
        if nulls > nan_pct * np.ceil(len(yx)):
            su.my_print('WARNING: Too many NaN to interpolate for label ' + str(self.lbl) + ': ' + str(nulls) +
                        ' out of ' + str(len(yx)) + ' (' + str(pct) + '%) data points and lambda ' + str(self.lmbda))
            f = pd.DataFrame({'yt': list(yt), 'yx': list(yx)})
            f['lmbda'] = self.lmbda
            p_ut.save_df(f, '~/my_tmp/interpolDF')
            return None
        elif 0 < nulls <= nan_pct * np.ceil(len(yx)):      # interpolate yhat if some NaNs
            su.my_print('WARNING: interpolating for label ' + str(self.lbl) + ': ' + str(nulls) + ' NaNs out of ' +
                        str(len(yx)) + ' data points (' + str(pct) + '%)')
            st = pd.Series(yx)
            sint = st.interpolate(limit_direction='both')
            yhat = sint.values
            ys = np.reshape(yhat, (1, -1))
            return ys[0]
        else:  # all OK
            return y


class Linearizer(object):
    # transforms logistic like data between floor and ceiling (floor < y < ceiling, no equality) into unbound (linear)
    # See Fisher-Prys transform
    # See http://newmaeweb.ucsd.edu/courses/MAE119/WI_2018/ewExternalFiles/Simple%20Substitution%20Model%20of%20Technological%20Change%20-%20Fischer%20Pry%201971.pdf
    # y = 1/(1+e^g(t))
    # y/C = z   -> z/(1-z) = e^(-g(t))
    # y = F + (C-F)/(1+e^(g(t)))
    # z = (y-F) / (C-F) and z = 1/(1+e^g(t))
    # inverse: log(t/(1-t)) = y -> t = (1-t) e^y -> t = e^y/(1+e^y)
    def __init__(self, ceiling, floor, unbias):
        # super().__init__('logistic', 100, ceiling=None, floor=None, unbias=False)
        self.lmbda = None
        self.ceiling = ceiling
        self.floor = floor
        self.xf_done = False
        self.unbias = unbias
        if isinstance(self.ceiling, type(None)) or isinstance(self.floor, type(None)):
            su.my_print('ERROR: must have ceiling and floor for logistic transform')
            self.is_error = True
        elif self.ceiling < self.floor:
            su.my_print('ERROR: must have ceiling and floor for logistic transform')
            self.is_error = True
        else:
            self.is_error = False

    def check_input(self, y):
        if isinstance(y, np.ndarray) is False:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid type: ' + str(type(y)))
            return True

        if self.xf_done is True:
            su.my_print('pid: ' + str(os.getpid()) + ' Linearizer: already fit. Create new object')
            return self.xf_done

        yc = np.copy(y)
        if len(np.shape(yc)) > 1:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid input shape: ' + str(np.shape(yc)))
            return True

        yx = np.reshape(yc, (1, -1))[0]
        if np.max(yx) == np.min(yx):
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: constant series: ' + str(np.min(yc)))
            return True
        return False

    def transform(self, y):
        ret = self.check_input(y)
        if ret is False:
            if np.max(y) >= self.ceiling or np.min(y) <= self.floor:
                su.my_print('pid: ' + str(os.getpid()) + ' WARNING: data outside (floor, ceiling)')
                return None
            else:
                self.xf_done = True
                yr = (y - self.floor) / (self.ceiling - self.floor)  # 0 < yf < 1
                return np.log(yr / (1.0 - yr))
        else:
            su.my_print('pid: ' + str(os.getpid()) + ' Linearizer: already fit. Create new object')
            return None

    def fit(self, y):  # this is to preserve the API
        return self

    def fit_transform(self, y):
        return self.transform(y)

    def _u_inverse_transform(self, y_mean):
        fy = self._b_inverse_transform(y_mean)
        d2fy = (self.ceiling - self.floor) * np.exp(y_mean) * (1.0 - np.exp(y_mean)) / np.power(1.0 + np.exp(y_mean), 3.0)
        return fy, d2fy

    def inverse_transform(self, y, y_var, lbl=None):
        if y is None:
            return None
        elif self.xf_done is False:
            su.my_print('pid: ' + str(os.getpid()) + ' WARNING: cannot inverse_transform before transform was done')
            return None
        else:
            if self.unbias is False:
                return self._b_inverse_transform(y)
            else:
                fy, d2fy = self._u_inverse_transform(y)
                return None if fy is None else (fy if d2fy is None else fy + (y_var / 2.0) * d2fy)

    def _b_inverse_transform(self, y):
        z = pd.DataFrame({'y': list(y)})
        z['yf'] = z['y'].apply(lambda x: 1.0 if np.isinf(np.exp(x)) else np.exp(x) / (1.0 + np.exp(x)))
        yout = self.floor + (self.ceiling - self.floor) * z['yf'].values
        return list(yout)


class LogTransform(object):      # fake transform class used when xform = None to maintain the API
    def __init__(self, unbias):
        self.lmbda = None
        self.lambdas_ = [None]
        self.name = 'log_transform'
        self.unbias = unbias

    @staticmethod
    def transform(y):
        if np.min(y) < 0.0:
            return None
        else:
            return np.log1p(y)

    def fit(self, y):
        return self

    def fit_transform(self, y):
        return self.transform(y)

    def inverse_transform(self, y, y_var, lbl=None):
        if y is None:
            return None
        else:
            if self.unbias is False:
                z = pd.DataFrame({'y': list(y)})
                z['yf'] = z['y'].apply(lambda x: x if pd.isna(x) else np.inf if np.isinf(np.exp(x)) else np.exp(x) - 1)
                return list(z['yf'].values)
            else:
                fy, d2fy = self._u_inverse_transform(y)
                z = pd.DataFrame({'fy': list(fy), 'd2fy': list(d2fy)})
                yt = z.apply(lambda x: x if pd.isna(x) else np.inf if np.isinf(x['d2fy']) else x['fy'] + (y_var / 2) * x['d2fy'], axis=1)
                return list(yt)

    @staticmethod
    def _u_inverse_transform(y):
        return np.exp(y) - 1, np.exp(y)




