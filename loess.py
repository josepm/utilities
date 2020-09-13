"""
from https://gist.github.com/agramfort/850437
also http://ml.stat.purdue.edu/hafen/preprints/Hafen_thesis.pdf
This module implements the Loess function for nonparametric regression.
Functions:
loess fits a smooth nonparametric regression curve to a scatterplot.
For more information, see
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.

Usage:
y_loess = lowess(x, y, f=2./3., iter_=3)
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar
import pwlf


class Loess(object):
    """
     Lowess smoother class: Robust locally weighted regression.
     Fits a nonparametric regression curve to a scatterplot.
     The arrays x and y contain an equal number of elements; each pair
     (x[i], y[i]) defines a data point in the scatterplot.
     The function returns the estimated (smooth) values of y.
     The smoothing span (bandwidth) is given by f.
     A larger value for f will result in a smoother curve.
     The number of robustifying iterations is given by iter_.
     The function will run faster with a smaller number of iterations.
     x: np.array
     y: np.array
     f: relative bandwidth (0 < f < 1)
     returns the smoothed values of y
     """
    def __init__(self, x, y, f, iter_=3, degree=1):
        self.x = np.array(x)
        self.y = np.array(y)
        self.f = f
        self.n = len(x)
        self.iter_ = iter_
        self.degree = degree
        self.L, self.Lambda = self.operator_matrix()

    def weights(self):
        r = int(np.ceil(self.f * self.n))
        h = [np.sort(np.abs(self.x - self.x[i]))[r] for i in range(self.n)]
        w = np.clip(np.abs((self.x[:, None] - self.x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        return w  # the j column of w contains the weights of each point on the jth point

    def pars(self):
        return np.trace(np.dot(np.transpose(self.L), self.L))

    @staticmethod
    def set_delta(residuals):
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        return (1 - delta ** 2) ** 2

    def operator_matrix(self):
        # get the L-matrix and the Lambda-matrix, Lambda = (I-L)^T (I-L)
        # equivalent to the hat matrix in a LR
        w = self.weights()
        delta = np.ones(self.n)
        L = None
        for iteration in range(self.iter_):
            L, delta = self._operator_matrix(w, delta)
        IL = np.identity(self.n) - L
        Lambda = np.dot(np.transpose(IL), IL)
        return L, Lambda

    def _operator_matrix(self, w, delta):
        l_rows = list()
        for i in range(self.n):  # ith row of L
            wi = delta * w[:, i]
            nzi = np.nonzero(wi)[0]
            x_cols_ = [(self.x - self.x[i]) ** n for n in range(self.degree+1)]
            x_cols = [c[nzi] for c in x_cols_]
            X = np.column_stack(x_cols)
            W = np.diag(wi[wi != 0.0])
            Xt = np.transpose(X)
            H = np.dot(np.dot(Xt, W), X)
            Hinv = linalg.inv(H)
            lr = np.dot(np.dot(Hinv, Xt), W)
            lx = np.zeros(self.n)
            lx[nzi] = lr[0]
            l_rows.append(lx)
        L = np.array(l_rows)
        IL = np.identity(self.n) - L
        residuals = np.dot(IL, self.y)
        delta = self.set_delta(residuals)
        return L, delta

    def ssr(self):
        resid = self.resid()
        return np.sum(resid ** 2)

    def resid(self):
        return self.y - np.dot(self.L, self.y)

    def var(self):
        # loess model variance
        resid = self.resid()
        return np.sum(resid ** 2) / np.trace(self.Lambda)

    def select_criterion(self, lbda=1.0):
        # lbda: regularizing coef for the numper of parameters
        # returns aic's, gcv and the number of pars
        # https://support.sas.com/documentation/onlinedoc/stat/131/loess.pdf
        # https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/1467-9868.00125?casa_token=bd_sxNZP46gAAAAA:or6-R32Ztc47ydFHRA4o4KV9YGy2r7vsepi19E5stXLi0JeRjB0W4aEeSizNHPyPOysUQ2xgeSWtwoQk
        iLbda = np.identity(self.n) - self.Lambda
        m = np.dot(np.transpose(iLbda), iLbda)
        delta1 = np.trace(m)
        delta2 = np.trace(np.dot(m, m))
        r = delta1 / delta2
        sig2 = self.var()
        npars = lbda * self.pars()
        tL = np.trace(self.L)
        aicc = 1 + np.log(sig2) + 2.0 * (1.0 + npars) / (self.n - npars - 2.0)
        aicc1 = self.n * np.log(sig2) + self.n * r * (self.n + npars) / (delta1 * r - 2.0)
        gcv = self.n * sig2 / (self.n - tL) ** 2
        return aicc1, aicc, gcv, npars


def opt_func(f, *args):
    # cost function as a function of f and args
    # args: (x, y, iter_, degree, lbda, criterion, cost_arr, f_arr)
    # cost_arr collects all the costs seen
    # f_arr collects all the spans seen
    # f_opt = min(argmin(aicc(f), argmax(Delta(aicc(f)), argmin(abs(Delta(aicc(f))))). The last one is to get the solution of Delta(aicc(f))=0
    x, y, iter_, degree, lbda, criterion, cost_arr, f_arr = args
    loess_obj = Loess(x, y, f, iter_=iter_, degree=degree)
    cost = loess_obj.select_criterion(lbda=lbda)
    print('f: ' + str(f) + ' cost: ' + str(cost[:3]) + ' npars: ' + str(cost[3]))
    cost_arr.append(cost[int(criterion)])        # save the cost
    f_arr.append(f)                              # save the span
    return cost_arr[-1]


def loess_performance(x, y, iter_=3, degree=1, criterion=1, lbda=1.0, fmin=0.1, fmax=0.9):
    # select f_opt as f_opt = min(argmin_f(aicc(f), argmax_f(Delta(aicc(f)), argmin_f(abs(Delta(aicc(f))))). Note: The last one is to get the solution of Delta(aicc(f))=0
    # see https://pubmed.ncbi.nlm.nih.gov/24905059/
    cost_arr, f_arr = list(), list()
    args = (x, y, iter_, degree, lbda, criterion, cost_arr, f_arr)  # x, y, iter, degree, lbda, criterion
    bounds = (fmin, fmax)
    res = minimize_scalar(opt_func, bounds=bounds, args=args, method='bounded')
    if criterion != 2:
        aic_arr = np.array(cost_arr)
        res = [min(f_arr)]
        Daic_arr = np.diff(aic_arr)
        res.append(f_arr[np.argmax(Daic_arr)])
        res.append(f_arr[np.argmin(Daic_arr)])
        print(res)
        return min(res)
    else:
        return res.x


def opt_loess(x, y, iter_=3, lbda=1.0, degree=1, criterion=1, fmin=0.1, fmax=0.9):
    # one stop optimizer: return the smoothed y by automatically selecting f
    # x, y scatter points
    # iter_: outlier iterations
    # lbda: npars regularization parameter
    # degree: 1 (linear), 2 (quadratic), ...
    # criterion: index of [aicc1, aicc, gcv]
    # fmin, fmax: bounds for the span parameter f
    # return the smoothed y's and the min cost
    f_opt = loess_performance(x, y, iter_=iter_, lbda=lbda, criterion=criterion, degree=degree, fmin=fmin, fmax=fmax)
    loess_obj = Loess(x, y, np.round(f_opt, 3), iter_=iter_, degree=degree)
    return np.dot(loess_obj.L, y), f_opt


def loess(x, y, f=0.66, iter_=3):
    """
    Direct loess function. Should be faster than using the class (no operator matrix used)
    Only degree = 1
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot.
    The function returns the estimated (smooth) values of y.
    The smoothing span (bandwidth) is given by f.
    A larger value for f will result in a smoother curve.
    The number of robustifying iterations is given by iter_.
    The function will run faster with a smaller number of iterations.
    x: np.array
    y: np.array
    f: relative bandwidth (0 < f < 1)
    returns the smoothed values of y
    """
    n = len(x)
    yhat = np.zeros(n)
    delta = np.ones(n)
    loess_obj = Loess(x, y, f, iter_=iter_, degree=1)
    w = loess_obj.weights(x, f)
    for iteration in range(iter_):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yhat[i] = beta[0] + beta[1] * x[i]
        delta = loess_obj.set_delta(y - yhat)
    return yhat


