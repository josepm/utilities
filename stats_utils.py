"""
statistical stuff
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from functools import partial, lru_cache
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from statsmodels.tools.eval_measures import aic, bic
import scipy.special as ssp
import scipy.stats as sps
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from lifelines import KaplanMeierFitter
from itertools import combinations

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import xforms as xf

DO_MP = False
if DO_MP is True:
    pass


# #########################################################################
# #########################################################################
# ######################### KDE functions #################################
# #########################################################################
# #########################################################################

dist_names = [
    'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine',
    'dgamma', 'dweibull', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk',
    'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto', 'gennorm', 'genexpon',
    'genextremJe', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l',
    'hyperexp',
    'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss',
    'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable',
    'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf',
    'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal',
    'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda',
    'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'
]


def to_probs(q):
    # q is an array that gets converted into a prob array (between 0 and 1 and sum to 1)
    # deals with overflows and underflows in prob vectors
    q_max = np.max(np.abs(q))
    if q_max == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR to_probs: ' + str(q))
        return None
    else:
        qv = q / q_max
        q_sq = qv ** 2
        return q_sq / np.sum(q_sq)


# ###########################################################################################
# ###########################################################################################
# ################################## Simulations ############################################
# ###########################################################################################
# ###########################################################################################
def sim_ecdf(e_df, q_vals=None, sz=1, min_prob=None, max_prob=None):
    """
    simulate sz points of a continuous rv using an empirical CDF.
    to generate values outside the x-range in the ecdf, one must provide an ecdf with cdf.max < 1 and cdf_min > 0 and set min_prob such that min_prob <= cdf.min
    and max_prob such that max_prob >= cdf.max
    :param e_df: DF with a col labelled 'x' and a col labelled 'cdf' that contains the CFD values, cdf = Prob(<= x).
    :param sz: # pts to simulate
    :param min_prob: If None, set to e_df['cdf'].min(), otherwise a nbr between 0 and 1.
    :param max_prob: If None, set to e_df['cdf'].max(), otherwise a nbr between 0 and 1.
    :param q_vals: If None, simulate sz x-values.
                   If not None, q_vals is an np.array of quantiles. In this case, we return the x-values associated to these quantiles.
                   Quantile values outside the (min_prob, max_prob) range will generate x-values outside the e_df['x'] range.
    :return: sz simulated x data points
    """
    x_df = e_df.copy()

    # check valid ecdf
    if x_df['cdf'].min() < -1.0e-6 or x_df['cdf'].max() > 1.0 + 1.0e-6:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: sim_ecdf: invalid prob range: min prob: ' + str(x_df['cdf'].min()) + ' max prob: ' + str(x_df['cdf'].max()))
        return None
    if x_df['x'].nunique() < len(x_df):   # cannot have repeated x_values
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: sim_ecdf: duplicated x values')
        return None

    if max_prob is None: max_prob = min(x_df['cdf'].max(), 1.0)
    if min_prob is None: min_prob = max(x_df['cdf'].min(), 0.0)
    if max_prob > 1 or min_prob < 0 or max_prob < min_prob:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' sim_ecdf error: prob bounds error')
        return None

    # simulate
    x_df.sort_values(by='cdf', inplace=True, ascending=True)
    x_df.reset_index(inplace=True, drop=True)
    u_sim = sps.uniform(min_prob, max_prob - min_prob).rvs(size=sz) if q_vals is None else np.array(q_vals)        # rnd unif vals between min_prob and max_prob
    return np.percentile(x_df['x'].values, 100.0 * u_sim, interpolation='linear')


def bootstrap(data, sple_fraction=0.5, n_iters=1000, ci=0.95, idx_arr=None):
    """
    boostrap stats (mean, std and CI bounds) for the cols in the data DF.
    The sampling happens at the row level of the DF, not individually for each column.
    :param data: DF with the results from simulation variables in the rows. One column per variable from the simulation.
    :param sple_fraction: sets the number of samples to collect to samples = sple_fraction * len(data)
    :param n_iters: number of times we sample
    :param idx_arr: if not None and the right size, it is a pre-computed sampling array
    :return: dict with key the cols in the DF and mean, std, low CI, high CI
    """
    n_samples = int(len(data) * sple_fraction)
    data.reset_index(inplace=True, drop=True)
    if idx_arr is None or len(idx_arr[0]) != n_samples or len(idx_arr) != n_iters:
        # idx_arr: list with <n_iters> arrays of indices to sample
        idx_arr = [np.random.choice(range(len(data)), size=n_samples, replace=True) for _ in range(n_iters)]
    spl_df = [data[data.index.isin(idx_list)] for idx_list in idx_arr]                            # sampled data
    means_df = pd.concat([pd.DataFrame(f.mean(axis=0)).transpose() for f in spl_df], sort=True)              # cols = cols from data, len(mean_df) = n_iters
    srt_df = means_df.apply(lambda x: x.sort_values().values, axis=0).reset_index(drop=True)      # sort each col individually
    eps = (1 - ci) / 2
    l_idx, h_idx = int(n_iters * eps), int(n_iters * 1 - eps)
    means = pd.DataFrame(srt_df.mean(axis=0)).transpose(); means.index = ['mean']   # mean from each col
    stds = pd.DataFrame(srt_df.std(axis=0)).transpose(); stds.index = ['std']       # std from each col
    l_ci = pd.DataFrame(srt_df.loc[l_idx, ]).transpose(); l_ci.index = ['l_ci']     # std from each col
    h_ci = pd.DataFrame(srt_df.loc[h_idx, ]).transpose(); h_ci.index = ['h_ci']     # std from each col
    return pd.concat([means, stds, l_ci, h_ci], axis=0, sort=True)


# ###########################################################################################
# ###########################################################################################
# ################################## Densities ##############################################
# ###########################################################################################
# ###########################################################################################
def kde(x, x_grid=None, max_cv=5, max_len=None, **kwargs):
    """
    Kernel Density Estimation with Scikit-learn. Automatically sets best bandwidth
    x: input data. np array or list of shape nrows x 2
    x_grid: density support.
    return DF with x_grid, epdf and ecdf. Columns: 'x', 'pdf', 'cdf'
    """
    if np.shape(x)[1] != 2:
        s_ut.my_print('ERROR: invalid shape for kde: ' + str(np.shape(x)))
        return None

    x_spl = np.array(x) if max_len is None else (np.random.choice(x, size=max_len, replace=True, p=None) if len(x) > max_len else np.array(x))
    dfx = pd.DataFrame(x_spl, columns=['x', 'y'])
    dfx.sort_values(by='x', inplace=True)
    X = dfx.values
    # x_s = dfx['y]'.values
    # _x_grid = dfx['x'].values

    # _x_grid = x_s if x_grid is None else np.sort(x_grid)
    # y_vals = x_s[:, np.newaxis]
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=min(max_cv, max(2, int(len(x)/4))),
                        return_train_score=False,
                        n_jobs=-1)
    try:
        # g = grid.fit(y_vals)
        g = grid.fit(X)
        bw = g.best_params_['bandwidth']
        kdens = KernelDensity(bandwidth=bw, **kwargs)
    except ValueError:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' stats_utils:kde() WARNING: not enough data for test/train: ' + str(len(_x_grid)))
        return None   # pd.DataFrame({'x': _x_grid, 'pdf': [0] * len(_x_grid), 'cdf': [0] * len(_x_grid)})

    # kdens.fit(y_vals)
    kdens.fit(X)
    epdf_ = np.exp(kdens.score_samples(_x_grid[:, np.newaxis]))        # score_samples() returns the log-likelihood of the samples
    epdf_df = pd.DataFrame({'x': _x_grid, 'pdf': epdf_})
    epdf_df = epdf_df.groupby('x').agg({'pdf': np.sum}).reset_index()
    xdiff = epdf_df['x'].diff()
    xdiff.fillna(xdiff[xdiff.index <= min([len(xdiff), 5])].mean(), inplace=True)
    xq = xdiff * epdf_df['pdf']
    epdf_df['cdf'] = xq.cumsum() / xq.sum()
    return epdf_df


def c_dens(in_df, qtile, x_col, y_col, g_size, x_vals):
    """
    compute y_1, ... y_{g_size} such that qtile = Prob(Y <= y_i| X = x_i) for x_i = [x_1, ...x_{g_size}]
    Assumes qtile function increase with the value of y
    :param in_df: input df
    :param qtile: quantile to estimate (between 0 and 1)
    :param x_col: df column with conditioning variable
    :param y_col: df column with observed variable
    :param g_size: number of points to condition on
    :param x_vals: how to get the conditioning x values. Possible options are:
                  - by_count: use the g_size-th quantiles of the x_col.
                  - by_length: divide x values into g_size segments of equal length between the min and max value in the x_col
                  - using the provided numerical list of np.array
    :return: dataframe with x_col and the quantile quantities observed
    """
    def func(y, x, qtile, cdens):
        return (cdens.cdf(endog_predict=[y], exog_predict=[x])[0] - qtile)**2

    df = in_df.dropna()
    cond_dens_obj = sm.nonparametric.KDEMultivariateConditional(endog=[df[y_col].values], exog=[df[x_col].values], dep_type='c', indep_type='c', bw='normal_reference')
    if isinstance(x_vals, list) or isinstance(x_vals, np.ndarray):
        X_grid = np.array(list(x_vals))
    elif x_vals == 'by_count':
        X_grid = df[x_col].quantile(np.arange(1, g_size) / float(g_size))
    elif x_vals == 'by_length':
        X_grid = np.linspace(df[x_col].min(), df[x_col].max(), num=g_size+2)[1:-1]
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' invalid xaxis: ' + str(x_vals))
        return None

    result = []
    y_name = y_col
    bracket = (df[y_col].min(), df[y_col].max())
    for x_val in X_grid:
        fx = partial(func, x=x_val, qtile=qtile, cdens=cond_dens_obj)
        res = minimize_scalar(fx, method='Brent', bracket=bracket)
        if res.get('success', False):
            y = res.x
        else:
            if res.get('fun', np.nan):
                y = res.get('x', np.nan) if res.fun < 1E-06 else np.nan
        result.append({x_col: x_val, y_name: y})
    z = pd.DataFrame(result)
    return z


def dist_df(epdf, inc, p0=0.0):
    """
    return DF with columns x and cdf with cdf prob values incremented by inc between 0 and 1, i.e. in row n, P(X <= x_n) = p_n = n / len(df)
    Only for non-negative support.
    epdf normally is the output of kde()
    :param epdf: DF with cols x and pdf
    :param inc:  prob inc
    :param p0:  mass at 0, only used when epdf['x'].min() > 0.
    :return: ecdf DF with cols x and prob(X<=x) with prob increments equal to inc
    """
    _epdf = epdf.copy()
    _epdf.sort_values(by='x', inplace=True)
    _epdf.reset_index(inplace=True, drop=True)

    if np.abs(1.0 - _epdf['cdf'].max()) > 1e-3:
        s_ut.my_print('ERROR_: q_df invalid inputs: p_max: ' + str(_epdf['cdf'].max()) + ' should be 1')
        return None

    r0 = _epdf.loc[0, ]
    if r0['x'] > 0.0:    # add 0 mass.
        if r0['cdf'] < p0:
            s_ut.my_print('ERROR_: q_df invalid inputs: p0: ' + str(p0) + ' and r0: ' + str(r0))
            return None
        else:
            _epdf = _epdf.append(pd.DataFrame({'x': [0.0], 'cdf': [p0], 'pdf': [p0]}), sort=True)
            _epdf.sort_values(by='x', inplace=True)
            _epdf.reset_index(inplace=True, drop=True)
    else:  # the 0 mass is already included in the epdf and p0 is ignored
        pass

    probs = np.linspace(0.0, 1.0, num=1 + int(1.0 / inc), endpoint=True)
    pct = np.percentile(epdf['cdf'].values, probs)
    return pd.DataFrame({'x': pct, 'cdf': probs})

# ###############################################
# ###############################################
# ################ shape and scale ##############
# ###############################################
# ###############################################


def fobj(sh, _args):
    v1, v2, d_obj = _args
    _dobj = d_obj(sh ** 2, loc=0, scale=1)
    return v2[1] * _dobj.ppf(v1[0]) - v1[1] * _dobj.ppf(v2[0])


def fobj2(sh, _args):
    return fobj(sh, _args) ** 2


def get_bnds(func, fargs, x_min=1.0e-4, x_max=100.0, lin_sz=100, ctr=0):
    """
    get bounds for the smallest solution in sc_vals
    :param func: func to find bounds for
    :param fargs: args tofunc
    :param x_min: initial min
    :param x_max: initial max
    :param lin_sz: grid size
    :param ctr: ctr
    :return: final x_min and x_max
    """
    lsx = np.linspace(x_min, x_max, num=lin_sz)
    fvals = np.array([func(x, fargs) for x in lsx])
    nz_fvals = [x for x in fvals if np.abs(x) > 1.0e-8]
    if min(nz_fvals) * max(nz_fvals) < 0.0:
        idx_min = np.argmin(fvals)
        idx_max = np.argmax(fvals)
        if idx_min > 0:
            if func(lsx[0], fargs) > 0:
                x_min, x_max = lsx[0], lsx[idx_min]
            else:
                x_min, x_max = lsx[idx_min], lsx[idx_max]
        else:  # idx_min = 0
            if func(lsx[0], fargs) > 0:  # this cannot happen
                s_ut.my_print('s_ut.get_bnds:: ERROR. This cannot happen!')
            else:
                x_min, x_max = lsx[0], lsx[idx_min]
        return x_min, x_max
    else:
        if ctr < 6:
            return get_bnds(func, fargs, x_min=1.0e-4, x_max=1.5 * x_max, lin_sz=2 * lin_sz, ctr=ctr+1)
        else:
            s_ut.my_print('ERROR_: get_bnds:: could not find bounds')  # func is such that it must have a 0 (see https://www.johndcook.com/quantiles_parameters.pdf)
            return None


def sc_vals(dist_name, qdict, avg):
    """
    find shape and scale from quantiles
    https://www.johndcook.com/quantiles_parameters.pdf
    only for distributions with 0 (expon) or 1 shape parameters
    :param dist_name: name of the distro
    :param qdict: dict with quantile values, {p: x_p, ...} where prob(X <= x_p) = p
    :param avg: avg of the distribution
    :return:
    """
    if len(qdict) != 2:
        s_ut.my_print('sc_vals:: invalid data: ' + str(qdict))
        return None
    try:
        dist_obj = getattr(sps, dist_name)
    except AttributeError:
        s_ut.my_print('ERROR_: sc_vals::  ' + dist_name + ': invalid distribution name')
        return None

    if dist_name == 'gamma' or dist_name == 'lognorm':
        if avg > 1.0e6:
            s_ut.my_print('WARNING: sc_vals::' + dist_name + ' needs rescaling: input avg is too large: ' + str(avg) + 'for stability')
            return None
        fargs = tuple(list(qdict.items()) + [dist_obj])
        bnds = get_bnds(fobj, fargs, dist_obj, x_min=1.0e-4, x_max=1000, lin_sz=1000)  # need to look for bnds so that we do not convrger to huge values
        if bnds is not None:
            res = minimize_scalar(fobj2, bounds=bnds, args=list(qdict.items()) + [dist_obj], method='Bounded')
            if res.status == 0:
                shape = res.x ** 2
                dobj = dist_obj(shape, loc=0, scale=1)
                scale = np.mean(np.array([qdict[k] / dobj.ppf(k) for k in qdict.keys()]))
                return shape, scale
            else:
                s_ut.my_print('ERROR_: sc_vals: no solution found::')
                return None
        else:
            return None
    elif dist_name == 'expon':
        dobj = dist_obj(loc=0, scale=1)
        scale = np.mean(np.array([qdict[k] / dobj.ppf(k) for k in qdict.keys()]))
        return None, scale
    else:
        s_ut.my_print('sc_vals:: ' + dist_name + ' is not implemented')
        return None


# ###########################################################################################
# ###########################################################################################
# ################################## EM Algo ################################################
# ###########################################################################################
# ###########################################################################################
class EM(object):
    def __init__(self, data, w, dist_name, max_iter=1500, tol=1.0e-3, floc=0.0, reg_info=None, verbose=False):
        """
        This is the basic EM algo for mixtures and general distros.
        We impose that the distro shape parameters are > 0 and that location is 0.
        Stop when MLE change is smaller than tol or max_iter is reached
        See http://cs229.stanford.edu/notes/cs229-notes8.pdf
        :param data: np array with input data
        :param w: number of elements in mixture
        :param dist_name: scipy distro name, ie dist_name is such that dist_obj = getattr(scipy.stats, str(dist_name))
        :param max_iter: number of steps in main loop
        :param tol: tolerance in the MLE. Stop when MLE change is smaller than tol
        :param floc: force the location parameter value to the value of floc. If None, treat location as an unknown.
        :param reg_info: regularization dict of the form {'loc': frozen_dist, 'scale': frozen_dist, 'shapes': [frozen_dist1, ...]}
        :param verbose:
        """
        # break data into chunks to set initial values
        self.dist_name = str(dist_name)
        try:
            self.dist_obj = getattr(sps, self.dist_name)
        except AttributeError:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' EM ERROR dist_name: ' + self.dist_name)
            self.dist_obj = None

        self.data = np.sort(data)
        self.order = w
        self.max_iter = max_iter
        self.tol = tol
        self.floc = floc
        self.reg_lambda = None
        self.len = len(self.data)
        if reg_info is not None:
            self.loc_prior = reg_info['loc'] if self.floc is None else None
            self.scale_prior = reg_info['scale']
            self.shape_priors = reg_info['shapes']  # a list
        else:
            self.reg_lambda = 0.0
            self.scale_prior, self.loc_prior, self.shape_priors = None, None, None
        self.verbose = verbose

        # initialize values
        inc = int(np.ceil((self.len / self.order)))
        data_parts = [self.data[k * inc: min(self.len, (k + 1) * inc)] for k in range(self.order)]
        self.phi = [1.0] * self.order      # prob of each phase
        if self.floc is None:             # location is a mdl parameter to set
            params = [self.dist_obj.fit(data_parts[i]) for i in range(self.order)]  # with loc included params[i] = [...(shape_pars_i, loc_i, scale_i)...]
        else:                             # fixed location
            params = [self.dist_obj.fit(data_parts[i], floc=self.floc) for i in range(self.order)]  # with loc = included,  params[i] = [...(pars[i], loc)...]
        self.params = [list(params[i]) for i in range(self.order)]              # params[i] = [.. shape_pars ..., loc, scale]

    def fit(self, lbda=0.0):
        """
        fits to data using EM
        :return: dict with dist_name
        """
        self.reg_lambda = lbda
        ctr = 0
        fval, nval, llf = [0.0] * self.order, [0.0] * self.order, None
        while ctr < self.max_iter:

            # E step:
            # joint likelihoods (w_ij = prob(order = j| data = i, dist_pars)  len(data) X order
            # w_ij is denoted Q_i(z = j) = p(z = j | x_i) where z is the mixture label
            # phi_j = Prob(z = j) unknown
            rvs = np.array([self.dist_obj(*self.params[i]) for i in range(self.order)])            # frozen dists
            dist_pdf = np.transpose(np.array([rvs[i].pdf(self.data) for i in range(self.order)]))  # shape: len(data_) X w array p_ij = f(x_i|theta_j)
            mix_pdf = self.phi * dist_pdf                   # prob(order = j | data = i, pars) has shape: len(data_) X w with row i: p_i1 * phi_1, p_i2 * phi_2, ..., p_iw * phi_w
            q_sum = np.sum(mix_pdf, axis=1)                # len(data) X 1 with row i: x_i1 * pi1 + ... + x_iw * pi_w
            q_sum = np.reshape(q_sum, (len(self.data), 1))
            w_ij = mix_pdf / q_sum                         # len(data) X w with row i: p_ij * phi_j / sum_k(p_ik * phi_k) = Prob(z = j |x_i)

            # M step
            self.phi = (np.sum(w_ij, axis=0) + self.len * self.reg_lambda) / (self.len + self.len * self.reg_lambda * self.order)  # New Prob(z = j) = phi_j, 1 <=j <= K, sum over rows. w X 1 array
            err, llf = list(), 0.0
            for j in range(self.order):
                pars = self.params[j]           # shapes, loc, scale
                for m in ['Powell', 'Nelder-Mead', 'BFGS']:
                    opt_res = self.em_opt(pars, w_ij[:, j], m)
                    if opt_res is not None:
                        break
                if opt_res is None:
                    return None
                nval[j], p_out = opt_res
                if self.floc is not None:
                    scale = p_out[-1]
                    self.params[j] = list(p_out[:-1]) + [self.floc, scale]
                else:
                    self.params[j] = list(p_out)
                err.append(np.abs(2.0 * (fval[j] - nval[j]) / (fval[j] + nval[j])))   # use relative error
                fval[j] = nval[j]
                llf += nval[j]
            ctr += 1
            if self.verbose:
                s_ut.my_print(str(ctr) + ' ' + str(np.round(self.phi, 3)) + ' ' + str(list(np.round(self.params, 4))) + ' ' + str(np.round(np.array(err), 7)) + ' ' + str(nval))
            if max(err) < self.tol and ctr > 100:
                break

        # create the output arrays
        shapes = np.array([self.params[i][:-2] for i in range(self.order) if len(self.params[i][:-2]) > 0])
        scales = np.array([self.params[i][-1] for i in range(self.order)])
        locations = np.array([self.params[i][-2] for i in range(self.order)]) if self.floc is None else np.array([self.floc] * self.order)
        bic_ = bic(llf, self.len, len(shapes) + len(scales) + len(locations))
        aic_ = aic(llf, self.len, len(shapes) + len(scales) + len(locations))
        return {'dist_name': self.dist_name,
                'probs': self.phi, 'scales': scales,
                'shapes': shapes, 'locations': locations,
                'bic': bic_, 'aic': aic_,
                'reg_lambda': self.reg_lambda, 'llf': llf,
                'data_len': self.len,
                'iterations': ctr} if ctr < self.max_iter else None

    def priors_logpdf(self, dpars):
        sc = 0.0 if self.scale_prior is None else self.scale_prior.logpdf(dpars[-1])
        lc = 0.0 if self.loc_prior is None else self.loc_prior.logpdf(dpars[-2])
        if self.shape_priors is None:
            sh = 0.0
        else:
            sh = sum([self.shape_priors[i].logpdf(dpars[i]) for i in range(len(dpars) - 2)]) if len(dpars) > 2 else 0
        return sc + lc + sh

    def em_obj(self, x_pars, *args):
        w_i_,  = args                    # N x 1 array
        if self.floc is None:
            shapes = list(x_pars[:-2] ** 2)
            ls = [x_pars[-2], x_pars[-1] ** 2]
        else:
            shapes = list(x_pars[:-1] ** 2)
            ls = [self.floc, x_pars[-1] ** 2]
        qars = np.array(list(shapes + ls))
        f_rv = self.dist_obj(*qars)
        l_pdf = f_rv.logpdf(self.data)                    # N x 1 array
        vals = w_i_ * l_pdf                               # N x 1 array
        l_prior = self.priors_logpdf(qars) if self.reg_lambda > 0.0 else 0.0
        llf_ = np.sum(vals, axis=0)
        return -(llf_ + self.reg_lambda * l_prior)

    def em_opt(self, pars, w_i, method):
        # square and sqrt trick to ensure pars > 0 and not have to deal with constraints but loc may be < 0
        shapes = list(np.sqrt(np.array(np.abs(pars[:-2]))))
        ls = [pars[-2], np.sqrt(pars[-1])] if self.floc is None else [np.sqrt(pars[-1])]
        x0 = np.array(shapes + ls)
        args = (w_i, )
        res = minimize(self.em_obj, x0, method=method, args=args)              # BFGS, Powell, Nelder-Mead
        if res.status == 0 and res.success == True:
            if len(np.shape(res.x)) == 0:                                      # minimization returns a scalar: turn to array
                res.x = np.array([res.x])
            opt_pars = np.array(list(res.x[:-1]) + [res.x[-1]])                # add location back
            return -res.fun, opt_pars ** 2                                     # llf and pars
        else:
            if self.verbose:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' mix order: ' + str(self.order) +
                      ' minimization failure at component  with initial value: ' + str(x0) + ' and method ' + method)
            return None


class prior_scale(object):
    """
    lambda (exp rate) has prior Prior_cdf(prior_pars), i.e Prob(lambda <= x) = Prior_cdf(x; prior_pars)
    X|lambda has dist Exp(lambda). The scale of X|lambda ~ Exp is 1/lambda = scale_X
    Prob(scale_X <= l) = Prob(lambda > 1 / l) = 1 - Prior_cdf(1/l; prior_pars)
    pdf(scale_X = a_scale) = prior_pdf(1/a_scale; prior_pars) / a_scale^2
    """
    def __init__(self, a, b, prior):
        if prior == 'gamma':
            self.shape = a
            self.scale = b
            self.dist = sps.gamma(self.shape, loc=0, scale=self.scale)
            scale_max = (a - 1) ** (a - 1) * np.exp(-(a - 1)) / (ssp.gamma(a) * b)
        elif prior == 'laplace':
            self.loc = a
            self.scale = b
            scale_max = 1.0 / (2 * b)
            self.dist = sps.laplace(loc=self.loc, scale=self.scale)
        else:
            scale_max = 1.0
            s_ut.my_print('NOT HERE!!!')
            pass
        self.max_logpdf = np.sum(self.dist.logpdf(1.0 / scale_max)) - 2 * np.sum(np.log(scale_max))

    def logpdf(self, scale_val):
        # data here is a scale value (inverse of the exponential rate)
        if len(np.shape(scale_val)) == 0:  # minimization returns a scalar: turn to array
            scale_val = np.array([scale_val])
        return np.sum(self.dist.logpdf(1.0 / scale_val)) - 2 * np.sum(np.log(scale_val)) - len(scale_val) * self.max_logpdf   # ensure always < 0: penalty


class HyperExp(object):
    """
    Fit to HyperExponential using divide and conquer
    if data CV < 1, we fit to a plain exponential
    """
    def __init__(self, data_in, em_phases=4, max_splits=5, max_cv=1.25, min_size=100, floc=0.0, max_iter=1500, tol=1.0e-3):
        """
        http://www.paper.edu.cn/scholar/showpdf/MUj2AN4INTz0gxeQh
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.122.9616&rep=rep1&type=pdf
        https://static.aminer.org/pdf/PDF/000/012/061/the_regularized_em_algorithm.pdf
        http://alumni.cs.ucr.edu/~hli/paper/hli05rem.pdf
        Note that w and min_size are dependent. In an exponential EM, there are 2*w - 1 parameters to find, so min_size > 10 * (2 * w -1) at least.
        If w = 4, min_size >= 70
        Conversely, set min_size = 100, then w <= 4
        The len(data_in) is also relevant here.
        If we want at most 5 splits, min_size <= len(data_in) / max_split and therefore w <= floor((min_size / 20) + 0.5)
        :param data_in: np array with input data
        :param em_phases: phases of each EM approx in a data chunk. Can be an int or a range of the form [w_min, ..., w_max]
        :param max_cv: max CV in each sub data set
        :param floc: mixture location. If None or missing it is computed otherwise it is the value set
        :param max_splits: max number of dividing splits
        :param min_size: min size of a split
        :param max_iter: number of steps in main loop
        :param tol: tolerance in the MLE. Stop when MLE change is smaller than tol
        """
        if len(data_in) < min_size:
            s_ut.my_print('WARNING: not enough data for EM')
        self.weights = None
        self.fit_res = list()
        self.dist_obj = None
        self.data_in = np.sort(np.array(data_in))
        self.cv_in = sps.variation(self.data_in)
        self.max_splits = max_splits
        self.min_size = min(min_size, int(len(data_in) / self.max_splits))
        if isinstance(em_phases, int):
            self.em_phases = range(em_phases, em_phases + 1)
        elif isinstance(em_phases, (list, tuple)):  # w = [w_min, w_max]
            self.em_phases = range(em_phases[0], em_phases[1] + 1)
        else:
            self.em_phases = range(1, 2)
        self.max_cv = max_cv
        self.floc = floc
        self.max_iter = max_iter
        self.tol = tol
        self.data_splits = self.div_n_conq()
        self.probs, self.scales, self.locations, self.shapes = None, None, None, None
        self.em_mean, self.em_std, self.rnd, self.prob_df, self.pval, self.m_err, self.s_err, self.em_cv = None, None, None, None, None, None, None, None

    def div_n_conq(self):
        idx, cv = 0, 0.0
        data_splits = list()
        inc = max(1, int(self.min_size / 10))
        while idx < len(self.data_in):                           # split in arrays with CV <= cv_max
            split_chunk = list()
            while cv < self.max_cv and idx < len(self.data_in):  # build a chunk and add it
                idx_upr = min(len(self.data_in), idx + inc)
                if idx_upr >= len(self.data_in):                 # last piece: add all the way to then end (more than a chunk)
                    n_split = list(self.data_in[idx:])
                    if len(n_split) < self.min_size and len(data_splits) > 0:  # last chunk would be less than min_size: put it with the last created chunk
                        l_split = data_splits.pop(-1)
                        n_split += l_split
                else:
                    n_split = list(self.data_in[idx:idx_upr])
                split_chunk.extend(n_split)             # current chunk
                cv = sps.variation(split_chunk)         # CV of the current chunk split added
                if len(split_chunk) < self.min_size:    # make sure each split chunk is large enough
                    cv = 0
                idx += inc
            cv = 0.0
            data_splits.append(split_chunk)
            if len(data_splits) == self.max_splits:
                n_split = list(self.data_in[idx:])
                l_split = data_splits.pop(-1)
                n_split += l_split
                data_splits.append(n_split)
                break

        _ = [s_ut.my_print('pid: ' + str(os.getpid()) + ' splits: cv: ' + str(sps.variation(d)) + ' len: ' + str(len(d))) for d in data_splits]
        len_arr = np.array([len(d) for d in data_splits])
        if np.sum(len_arr) == len(self.data_in):
            self.weights = len_arr / len(self.data_in)
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: divide and conquer failed')
        return data_splits

    @staticmethod
    def laplace_prior_pars(data):
        """
        Set Laplace (mu, b) prior for regularization
        No closed forms
        Initialize:
        E[lambda] = mu and E[X|lambda]=1/lambda => mu = 1/E[data]
        var(lambda) = 2b^2 and Var(X|lambda) = 1/lambda^2 => 2b^2 = 1/Var(data) - mu^2
        :param data:
        :return:
        """
        def laplace_prior_obj(x_pars, *args_):
            x_i,  = args_
            n = len(x_i)
            mu, b = x_pars ** 2
            arr = 2 * mu + b * (1 + b * x_i) ** 2 * np.exp(mu * (x_i - 1/b)) - 2 * b ** 2 * x_i * (2 + mu * x_i)  # array of length n
            if np.min(arr) < 0:   # arr should never be < 0, but in case
                arr = np.array([0.0] * n)
            llf = -mu * np.sum(x_i) \
                  + np.sum(np.log(arr)) \
                  - n * np.log(2) \
                  - 2.0 * np.sum(np.log(np.abs(1 - b ** 2 * x_i ** 2)))
            return -llf

        mu = 1.0 / np.mean(data)
        if mu * np.std(data) < 1.0:
            b = (1.0 / np.sqrt(2)) * np.sqrt(-mu ** 2 + 1.0 / np.var(data))
        else:
            b = 0.5 * np.std(data)   # 0.75 or less
        x0 = np.sqrt(np.array([mu, b]))                                           # shape, scale
        args = (data, )
        for m in ['Powell', 'Nelder-Mead', 'BFGS']:
            res = minimize(laplace_prior_obj, x0, method=m, args=args)              # BFGS, Powell, Nelder-Mead
            if res is not None:
                if res.status == 0 and res.success == True:
                    return -res.fun, np.array(res.x) ** 2                         # llf and pars
        return None

    @staticmethod
    def gamma_prior_pars(data):
        """
        Set Gamma prior for regularization
        Exp/Gamma system
        Gamma(a, b): a = shape, b = scale
        If X | lambda ~ Exp(lambda) and lambda ~ Gamma(a, b), then lambda | x_1, ..., x_N ~ Gamma(a + N, b + sum_1^N x_i)
        Conversely, if lambda ~ Gamma(a, b)
        pdf_X(x) = integrate_0^inf z exp(-z x) z^(a-1) exp(-z/b) / (Gamma(a) * b^a) dz = a b /(1 + bx )^(a+1)
        EX = 1 / (b * (a - 1)), a > 1
        VX = a / (b^2 (a-1)^2 (a - 2)), a > 2
        LLF = N log(a) + N log(b) - (a+1) sum_i log(1 + b x_i)
        Moments:
        If CV(X) > 1 can use moments
        a = 2 CV^2 / (CV^2 - 1)
        b = (CV^2 + 1) / (CV^2 - 1)

        :param data:
        :return:
        """
        def gamma_prior_obj(x_pars, *args_):
            x_i,  = args_
            n = len(x_i)
            a, b = x_pars ** 2
            llf = n * np.log(a) + n * np.log(b) - (a+1) * np.sum(np.log(1.0 + b * x_i))
            return -llf

        cv = sps.variation(data)
        if cv > 1.0:
            cv_sq = cv ** 2
            a = 2.0 * cv_sq / (cv_sq - 1)
            b = 1.0 / (np.mean(data) * (a - 1.0))
        else:
            a, b = 5, 10
        x0 = np.sqrt(np.array([a, b]))                                           # shape, scale
        args = (data, )
        for m in ['Powell', 'Nelder-Mead', 'BFGS']:
            res = minimize(gamma_prior_obj, x0, method=m, args=args)              # BFGS, Powell, Nelder-Mead
            if res is not None:
                if res.status == 0 and res.success == True:
                    return -res.fun, np.array(res.x) ** 2                         # llf and pars
        return None

    def set_reg_info(self, data, prior):
        """
        :param data:
        :param prior:
        :return:
        """
        if prior == 'gamma':
            res = self.gamma_prior_pars(data)
            s_ut.my_print('pid: ' + str(os.getpid()) + ' prior gamma fit failed. Trying laplace')
            if res is None:  # try laplace
                prior = 'laplace'
                res = self.laplace_prior_pars(data)
        elif prior == 'laplace':
            res = self.laplace_prior_pars(data)
            if res is None:  # try gamma
                s_ut.my_print('pid: ' + str(os.getpid()) + ' prior laplace fir failed. Trying gamma')
                prior = 'gamma'
                res = self.gamma_prior_pars(data)
        else:
            res = None

        if res is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' all prior fits failed. No regularization used')
            return None
        else:
            llf, pars = res
            if prior == 'gamma':
                if pars[0] < 1.0:
                    pars[0] = 1.0
            elif prior == 'laplace':
                pass
            else:
                pass  # should never get here
            pars = list(pars) + [prior]
            reg_info = dict()
            reg_info['scale'] = prior_scale(*pars)  # 1 / scale
            reg_info['loc'] = None
            reg_info['shapes'] = None
            return reg_info

    def fit_default(self, data, idx):
        em_fit = dict()
        em_fit['dist_name'] = 'expon'
        em_fit['probs'] = np.array([1.0])
        em_fit['scales'] = np.array([np.mean(data)])
        em_fit['shapes'] = list()
        em_fit['locations'] = np.array([self.floc])
        em_fit['iterations'] = np.array([0])
        em_fit['probs'] *= self.weights[idx]
        em_fit['split_id'] = idx
        em_fit['em_phases'] = 1
        em_fit['bic'] = None
        em_fit['aic'] = None
        em_fit['reg_lambda'] = 0.0
        em_fit['llf'] = None
        return em_fit

    def fit(self):
        for idx in range(len(self.data_splits)):  # get a mixture distro for each data split
            data = np.array(self.data_splits[idx])
            if sps.variation(data) < 1.0:
                self.em_phases = range(1, 2)
            # select best w
            for w in list(self.em_phases):  # range(min_order, max_order + 1):
                d_out = list()
                em_fit = EM(data, w, 'expon', max_iter=self.max_iter, tol=self.tol, floc=self.floc, reg_info=None).fit(lbda=0)
                if em_fit is not None:
                    em_fit['split_id'] = idx
                    em_fit['em_phases'] = w
                    em_fit['probs'] *= self.weights[idx]
                    d_out.append(em_fit)

                if len(d_out) == 0:  # EM failed: fall back to a default
                    d_out.append(self.fit_default(data, idx))

                u_fits = self.reduce(d_out)
                if len(u_fits) < w or w == max(self.em_phases):  # found the best w for data: works better than aic and bic
                    w_opt = max(1, w - 1) if len(u_fits) < w else w
                    reg_info = self.set_reg_info(data, 'laplace')  # set the prior for the exponential rates
                    llf = np.sum(sps.expon(0, np.mean(data)).logpdf(data))
                    lreg = reg_info['scale'].logpdf(np.mean(data))
                    # s_ut.my_print('llf: ' + str(llf) + ' lreg: ' + str(lreg) + ' llf/lreg: ' + str(llf/(len(data) * lreg)))
                    lbda = 0.0 if reg_info is None else 0.01 * len(data) * np.abs(llf / (len(data) * lreg))                      # regularization
                    reg_fit = EM(data, w_opt, 'expon', max_iter=self.max_iter, tol=self.tol, floc=self.floc, reg_info=reg_info).fit(lbda=lbda)  # find regularized pars
                    if reg_fit is None:
                        reg_fit = EM(data, w_opt, 'expon', max_iter=self.max_iter, tol=self.tol, floc=self.floc, reg_info=reg_info).fit(lbda=0.0)  # find regularized pars
                    if reg_fit is not None:
                        reg_fit['split_id'] = idx
                        reg_fit['em_phases'] = w_opt
                        reg_fit['probs'] *= self.weights[idx]
                    else:                              # regularized fit failed: go to default
                        reg_fit = self.fit_default(data, idx)
                    s_ut.my_print('reg_fit: ' + str(reg_fit))  # regularized fit for this data split
                    u_fits = self.reduce([reg_fit])
                    self.fit_res.extend(u_fits)
                    break

        self._set_rv()
        self.hx_stats()
        self.check_fit()

    @staticmethod
    def reduce(d_out):
        d_list = list()
        for d in d_out:
            if len(d['shapes']) == 0:
                for i in range(len(d['probs'])):
                    d_list.append({'dist_name': 'expon', 'prob': d['probs'][i], 'params': [d['locations'][i], d['scales'][i]],
                                   'bic': d['bic'], 'aic': d['aic'],
                                   'reg_lambda': d['reg_lambda'],
                                   'em_phases': d['em_phases'], 'split_id': d['split_id'], 'llf': d['llf']})
            else:
                for i in range(len(d['probs'])):
                    d_list.append({'dist_name': 'expon', 'prob': d['probs'][i], 'params': list(d['shapes'][i]) + [d['locations'][i], d['scales'][i]],
                                   'bic': d['bic'], 'aic': d['aic'],
                                   'reg_lambda': d['reg_lambda'], 'em_phases': d['em_phases'], 'split_id': d['split_id'], 'llf': d['llf']})
        fit_res = get_udists(d_list, min_pval=0.1)  # merge if pval >= min_pval
        # s_ut.my_print('Original mixtures: ' + str(len(d_list)) + ' final mixtures: ' + str(len(fit_res)))
        return fit_res

    def _set_rv(self):
        self.dist_obj = getattr(sps, 'expon')
        self.probs = np.array([d['prob'] for d in self.fit_res])
        self.scales = np.array([d['params'][-1] for d in self.fit_res])
        self.shapes = np.array([d['params'][:-2] for d in self.fit_res if len(d['params']) > 2])
        self.locations = np.array([d['params'][-2] for d in self.fit_res])

    def rvs(self, size=1):
        cum_probs = np.cumsum(np.array([0] + list(self.probs)))
        rnd_ = sps.uniform().rvs(size=size)
        m_pts = [len(rnd_[(rnd_ >= cum_probs[i]) & (rnd_ < cum_probs[i+1])]) for i in range(len(self.probs))]  # nbr of points from each mix
        v_arr = list()
        for i in range(len(self.probs)):  # generate the right fraction of rnd's for each component
            if len(self.shapes) == 0:
                v_arr += list(self.dist_obj(loc=self.locations[i], scale=self.scales[i]).rvs(size=m_pts[i]))
            else:
                v_arr += list(self.dist_obj(self.shapes[i], loc=self.locations[i], scale=self.scales[i]).rvs(size=m_pts[i]))
        np.random.shuffle(v_arr)   # inplace shuffle
        return v_arr

    def hx_stats(self, size=1000):
        self.rnd = np.sort(self.rvs(size)) if self.rnd is None or len(self.rnd) < size else self.rnd
        self.em_mean = np.mean(self.rnd)
        self.em_std = np.std(self.rnd)
        self.em_cv = sps.variation(self.rnd)
        # self.prob_df = pd.DataFrame({'x': self.rnd})
        # self.prob_df['pdf'] = 1.0 / len(self.prob_df)
        # self.prob_df['cdf'] = self.prob_df['pdf'].cumsum()

    def ppf(self, q):    # Get x such that prob(X <= x) = q
        if self.prob_df is None:
            self.hx_stats()
        df = self.prob_df.copy()
        diff = np.abs(df['cdf'] - q)
        r = df[diff == diff.min()]
        return r.loc[r.index[0], 'x']

    def cdf(self, x):    # prob(X <= x)
        if self.prob_df is None:
            self.hx_stats()
        df = self.prob_df.copy()
        diff = np.abs(df['x'] - x)
        r = df[diff == diff.min()]
        return r.loc[r.index[0], 'cdf']

    def check_fit(self):
        # size = 1000
        # self.rnd = np.sort(self.rvs(size)) if self.rnd is None or len(self.rnd) < size else self.rnd
        if self.rnd is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR????')
        else:
            _, self.pval = sps.ks_2samp(self.data_in, self.rnd)
            self.m_err = np.abs(1 - self.em_mean / np.mean(self.data_in))
            self.s_err = np.abs(1 - self.em_std / np.std(self.data_in))
            m_err = np.round(100.0 * self.m_err, 2)
            s_err = np.round(100.0 * np.abs(1 - self.em_std / np.std(self.data_in)), 2)
            # s_ut.my_print('pid: ' + str(os.getpid()) + ' fit test: ' + str(self.pval) +
            #       ' mean err: ' + str(m_err) + '% std_err: ' + str(s_err) + '%')  # pval: prob rejecting Null when True => should be high (eg > 0.1) for good fit


# ###########################################################################################
# ###########################################################################################
# ################################## Prony  #################################################
# ###########################################################################################
# ###########################################################################################
def prony(t, y, order):
    """
    https://en.wikipedia.org/wiki/Prony%27s_method
    http://sachinashanbhag.blogspot.com/2017/08/exam-question-on-fitting-sums-of.html
    http://sachinashanbhag.blogspot.com/2017/09/prony-method.html
    Input  : y(t) function to fit np.array
             t: equidistant points, np.array
    Output : arrays a and b such that y(t) ~ sum_i ai exp(bi*t)
    """
    import numpy.polynomial.polynomial as poly
    N = len(t)
    Amat = np.zeros((N - order, order))
    bmat = y[order:N]

    for jcol in range(order):
        Amat[:, jcol] = y[(order - jcol - 1):(N - 1 - jcol)]

    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    d = sol[0]

    # Solve the roots of the polynomial in step 2
    # first, form the polynomial coefficients
    c = np.zeros(order + 1)
    c[order] = 1.
    for i in range(1, order + 1):
        c[order - i] = -d[i - 1]

    u = poly.polyroots(c)
    b_est = np.log(u) / (t[1] - t[0])

    # Set up LLS problem to find the "a"s in step 3
    Amat = np.zeros((N, order))
    bmat = y

    for irow in range(N):
        Amat[irow, :] = u ** irow

    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    a_est = sol[0]

    return a_est, b_est


# ###########################################################################################
# ###########################################################################################
# ############################### Censored Data  ############################################
# ###########################################################################################
# ###########################################################################################
class SurvivalML(object):
    """
    MLE estimate for censored data
    Usage: res = SurvivalML(T, E, **kwds).fit()
        # T: time durations
        # E: event observed indicator: 1 if observed (not censored), 0 if censored
        # a_dist: scipy dist obj
    """
    def __init__(self, T, E, a_dist=None, is_pos=False):
        # T: time durations
        # E: Event: 1 if observed (ie not censored), 0 if not observed (censored)
        # is_pos = True: dist support must be > 0: we separate the data with observed value = 0 from the rest (mass_0)
        self.durations = T
        self.observed = E
        self.d_obj = a_dist
        self.is_pos = is_pos

    def _sv_sim(self, fitter, increasing):
        fitter.fit(self.durations, event_observed=self.observed)
        e_df = fitter.survival_function_.reset_index()
        e_df.columns = ['x', 'cdf']
        e_df['cdf'] = 1 - e_df['cdf']

        # separate 0-mass (P(X = 0) = 0) from the continuous component (otherwise the vectorized sim fails)
        if self.is_pos == True:
            mass_0 = e_df[e_df['x'] == 0]['cdf'].values[0] if e_df['x'].min() == 0 else 0.0
        else:
            mass_0 = 0.0  # the case P(X = 0) is treated together with the rest

        e_df['cdf'] -= mass_0
        e_df['cdf'] /= (1 - mass_0)
        if increasing is True:   # force cdf to be increasing
            e_df = e_df.groupby('cdf').agg({'x': np.mean}).reset_index()
        return e_df, mass_0

    def rvs(self, size=1):
        vals = self.km_sim()
        if vals is None:
            return None
        e_df, mass_0 = vals
        cum_probs = np.array([0, mass_0, 1.0])
        rnd_ = sps.uniform.rvs(size=size)
        m_pts = [len(rnd_[(rnd_ >= cum_probs[i]) & (rnd_ < cum_probs[i+1])]) for i in range(2)]  # points from each mix
        try:
            rnd = np.array([0.0] * m_pts[0] + list(sim_ecdf(e_df, sz=m_pts[1])))
            return rnd, mass_0
        except TypeError:
            s_ut.my_print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            s_ut.my_print('pid: ' + str(os.getpid()) + 'ERROR_ rvs TypeError')
            s_ut.my_print('mpts: ' + str(m_pts))
            s_ut.my_print('edf: ' + str(e_df.head()))
            s_ut.my_print('mass: ' + str(mass_0))
            s_ut.my_print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            return -1

    def km_sim(self, increasing=True):
        # Kaplan-Meier SF -> simulate uncensored data from KM SF -> MLE fit on simulated data
        if len(self.durations) == 0 or len(self.observed) == 0:
            return None
        e_df, mass_0 = self._sv_sim(KaplanMeierFitter(), increasing)
        return e_df, mass_0

    def km_fit(self):
        e_df, mass_0 = self.km_sim()
        x_sim = sim_ecdf(e_df, sz=1000)                     # simulate data
        params = self.d_obj.fit(x_sim)                      # fit simulated data
        return params, mass_0


# ###########################################################################################
# ###########################################################################################
# ############################### Data  Models  #############################################
# ###########################################################################################
# ###########################################################################################
class DataObj(object):
    """
    Model data by finding best MLE fit distribution to data
    From https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html
    We force data > 0 because we are only dealing with positive values. This means that loc = 0.
    A positive loc would not allow values less than loc and a negative loc would allow negative values.
    I other words, we always call DataObj with loc_val = 0.
    :param data_df: DF with data and censoring info if applicable
    :param data_col: name of col with data. Censored info is in col_name 'cens'.
    :param censored: if None or False use regular fit.
                     If True, the data has a col called cens with censoring values
    :param loc_val: loc value. SHould be 0
    :param samples: if None, do not sample. If numeric use a sample of size <samples> from teh actual data for parameter estimation (if data < samples, we use the whole data anyway)
    :return: DF sorted by decreasing pval with distribution name, and the loc, scale and shape parameters
    """

    def __init__(self, data_df, data_col, loc_val=0.0, samples=None, censored=False, min_mass=5.0e-2, verbose=False):
        self.dist_list = list()          # list of distribution models for the data
        self.udist_list = list()          # list of distribution models for the data after dropping identical distros
        self.loc_val = loc_val
        self.samples = samples           # nbr of samples to use if not None
        self.data_avg = data_df[data_col].mean()
        self.data_min = data_df[data_col].min()
        self.data_name = data_col
        self.verbose = verbose
        self.min_mass = min_mass
        s_df = data_df.sample(frac=samples / len(data_df)) if samples is not None and len(data_df) > 2 * samples else data_df.copy()
        self.mass_dict = self.get_mass(s_df[data_col])
        if self.mass_dict is not None:
            c_data = s_df[~s_df[data_col].isin(list(self.mass_dict.keys()))].copy()  # also drop 0
        else:
            c_data = s_df.copy()

        c_data.sort_values(by=data_col, ascending=True)
        self.data = c_data[data_col].values
        self.censored = censored
        self.cens_data = None if self.censored is False else c_data['cens'].values
        if self.censored is True:
            s_obj = SurvivalML(self.data, self.cens_data)
            vals = s_obj.rvs(size=len(self.data))
            if vals is not None:
                self.data, mass_0 = vals
                self.update_mass(0.0, mass_0)
            else:
                self.data = None
        self.data = None if self.data is None else self.data[self.data > 0]

    def get_mass(self, s):
        cnt_df = pd.DataFrame(s.value_counts(normalize=True)).reset_index()
        cnt_df.columns = ['data', 'pmass']
        mass_pnts = cnt_df[cnt_df.pmass > self.min_mass].copy()
        if len(mass_pnts) > 0:
            mass_pnts.sort_values(by='pmass', ascending=False, inplace=True)
            mass_pnts.set_index('data', inplace=True)
            mass_dict = mass_pnts.to_dict()
            return mass_dict['pmass']   # {'value1': prob1, 'value2': prob2,...}
        else:
            return None

    def update_mass(self, pt, mass):
        if mass > 0.0:
            if self.mass_dict is not None:
                dist = [(k, np.abs(k - pt)) for k in self.mass_dict.keys()]
                dist.sort(key=lambda x: x[1])
                closest = dist[0]
                if closest[1] <= 1.0e-4:
                    self.mass_dict[closest[0]] += mass
                else:
                    self.mass_dict[pt] = mass
            else:
                self.mass_dict = {pt: mass}

    def get_udists(self, dist_list=None):   # drop statistically identical distros from a dist_list
        if dist_list is None:
            self.udist_list = get_udists(self.dist_list)
            return self.udist_list
        else:
            if len(dist_list) > 1:
                return get_udists(dist_list)
            else:
                return dist_list


# ###########################################################################################
# ###########################################################################################
# ########################## Fit Dist Mix to Quantiles  #####################################
# ###########################################################################################
# ###########################################################################################
# Quantiles CI
# see http://www.math.mcgill.ca/~dstephens/OldCourses/556-2006/Math556-Median.pdf
# see https://stats.stackexchange.com/questions/99829/how-to-obtain-a-confidence-interval-for-a-percentile
# If F(x) = Prob(X<=x) = q,
# the sdt error on the quantile estimate is se(q)^2 = q (1 - q) / (n * f(x)^2) where f() is the pdf so that CI = q +/ 1.96 * se(q)

# only support for distributions with at most one shape parameter

class QuantileFit(object):
    # initial default pars: [shape. loc, scale]
    pars_default = {'expon': [None, 0, 1], 'gamma': [1, 0, 1], 'lognorm': [1, 0, 1], 'weibull_min': [1, 0, 1], 'fatiguelife': [1, 0, 1],
                    'pareto': [1, 0, 1], 'genpareto': [1, 0, 1], 'chi': [1, 0, 1], 'chi2': [1, 0, 1], 'invgamma': [1, 0, 1], 'invgauss': [1, 0, 1],
                    'invweibull': [1, 0, 1], 'loglapace': [1, 0, 1]}

    def __init__(self, name, dist_name, order, qdict, uq=True, pmin=1.0e-4):
        """
        :param qdict: dict { ... quantile: x_value ...}
        :param order: number of components in the mix
        :param dist_name: name of the dist family to use
        :param uq: if True, remove identical or low prob distros in the fit, ese leave as is
        :param pmin: if uq True, drop components with prob below pmin
        """
        self.name = name
        self.dist_name = dist_name
        self.qdict = qdict    # {..., q: x_q, ...} dict with quantiles to match
        self.order = order    # nbr of dist instances in the mixture
        self.dobj = getattr(sps, str(dist_name), None)
        if self.dobj is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR mix_cdf invalid dist name: ' + str(dist_name))
        min_order = np.floor((1 + len(self.qdict)) / 2)
        if self.order > min_order:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: qfit::input order (' + str(self.order) + ') should at most be ' + str(min_order))
        self.probs, self.scales, self.shapes, self.pars = None, None, None, None
        self.b_size = 2   # should be at least 1 + #shape pars. Could be 1 for expon, but must be at least 2 for all distros with 1 shape parameter
        self.uq = uq
        self.pmin = pmin
        dpars = self.pars_default.get(self.dist_name, None)
        if dpars is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: ' + self.dist_name + ' not supported')
            raise RuntimeError('failure')
        else:
            self.has_shape = False if dpars[0] is None else True

    def take_ends(self, a_list, cnt=1):
        # finds #order sets of non-overlapping indices of size b_size as far apart from each other in an array
        if len(a_list) < self.b_size:
            return []
        elif len(a_list) < 2 * self.b_size:  # only one block to return:pick the middle
            trim = int(np.floor(len(a_list) - self.b_size) / 2)
            return list(a_list[trim: - trim])[:self.b_size]
        else:   # return 2 extreme blocs of indices and continue if needed
            cnt += 1
            if self.order == cnt:
                return [a_list[:self.b_size]] + [a_list[-self.b_size:]]
            else:
                return [a_list[:self.b_size]] + [a_list[-self.b_size:]] + self.take_ends(a_list[self.b_size:-self.b_size], cnt=cnt)

    def mix_cdf(self, xvals_, probs_=None, scales_=None, shapes_=None):
        """
        return array with the CDF for each x in xvals for a mix dist in the famility dist_name with pars in scale and shapes
        :param probs_:  array of mixing probs
        :param scales_: array of scale pars for each RV
        :param xvals_: x values to compute the CDF
        :param shapes_: shape pars of the dist. ONLY 1 shape parameter supported
        :return: array with P(X<=x_i) fpr x_i in xvals_
        """
        if probs_ is None:
            probs_ = self.probs
        if scales_ is None:
            scales_ = self.scales
        if shapes_ is None:
            shapes_ = self.shapes
        if shapes_ is not None:
            if len(shapes_) == len(scales_):
                par_arr = [(scales_[i], shapes_[i]) for i in range(len(scales_))]
                d_arr = np.array([self.dobj(a[1:], loc=0.0, scale=a[0]) for a in par_arr])  # must have the word scale, otherwise scale gets incremented
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR mix_cdf invalid shapes: ' + str(shapes_))
                return None
        else:
            d_arr = np.array([self.dobj(loc=0.0, scale=a) for a in scales_])  # must have the word scale, otherwise scale gets incremented
        ex = np.array([d.cdf(xvals_) for d in d_arr])
        return np.sum(np.transpose(ex) * probs_, axis=1)

    def _qfit_obj(self, params, *args):             # quantile fit objective function for distribution mixtures
        d_name_, order_, qdict_, = args[0][0], args[0][1], args[0][2:]
        p, r = params[0: order_], params[order_:]
        r = np.array_split(r, order_)
        if self.has_shape:
            sh = [x[0] for x in r]   # shapes
            sc = [x[1] for x in r]   # scale
        else:
            sh = None
            sc = [x[0] for x in r]   # scale
        probs = to_probs(np.array(p))
        scales = np.power(np.array(sc), 2)
        shapes = np.power(np.array(sh), 2) if sh is not None else None
        xvals = np.array([x[1] for x in self.qdict.items()])
        qvals = np.array([x[0] for x in self.qdict.items()])
        qhat = self.mix_cdf(xvals, probs_=probs, scales_=scales, shapes_=shapes)
        err = np.mean((qvals - qhat) ** 2)                       # MSE
        return err

    def set_init(self):
        # set initial values for fit()
        d_items = np.array(sorted([[k, v] for k, v in self.qdict.items()], key=lambda x: x[0]))   # sorted by increasing quantile
        idx_list = self.take_ends(list(range(len(d_items)))) if self.order > 1 else [0]                           # d_items indices to group together to initialize minimization
        if len(idx_list) < self.order and self.order > 1:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: mixture order too large (' + str(self.order) + ') for qdict size (' + str(len(self.qdict)) + ')')
            return None
        rs = list()
        s_ut.my_print(idx_list)
        for i in range(self.order):            # take the sqrt of init vals to avoid constraints: we will square later in obj func
            jvals = idx_list[i]
            dpars = self.pars_default[self.dist_name]
            if dpars is None:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: ' + self.dist_name + ' not supported')
                return None
            else:
                if dpars[0] is not None:   # add shape
                    rs.append(dpars[0])

                # initialize with exponential scale and location and shape to default. Seems to work well enough
                # rs.append(0)                                      # DO NOT ADD location: not an unknown
                ql = d_items[:, 1] * np.log(1.0 - d_items[:, 0])
                q2 = d_items[:, 1] ** 2
                rs.append(np.sqrt(-np.sum(ql[jvals]) / np.sum(q2[jvals])))  # add scale
        p0 = [1.0] * self.order         # gets normalized later
        return np.array(p0 + rs)

    def fit(self):
        """
        Fit from quantiles a mixture if dist from the dist_name family
        If p(X<=x) = q  then q = P(X<=x) = sum_i p_i P(X_i<=x)
        dist params: [shape_par1, ... shape_par_n, loc, scale]
        :return: params: probs, shape and scales
        """
        x0 = self.set_init()
        if x0 is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: could find initial values')
            return None
        res = minimize(self._qfit_obj, x0=x0, args=[self.dist_name] + [self.order] + list(self.qdict.items()), method='Powell')
        # s_ut.my_print('name: ' + str(self.name) + ' fit:::init: ' + str(x0) + ' order: ' + str(self.order) + ' sol: ' + str(res.x))
        if res['success'] is True or res['status'] == 0:
            v = list(res.x)
            probs, self.pars = np.array(v[:self.order]), np.array(v[self.order:])  # pars: [shape_par11, ... shape_par_1n, scale1, shape_par21, ... shape_par_2n, scale2, ... ]
            if self.dist_name == 'expon':
                scales, shapes = self.pars, None
            else:
                shapes = np.array([self.pars[2 * i] for i in range(self.order)])
                scales = np.array([self.pars[1 + 2 * i] for i in range(self.order)])

            sh = None if shapes is None else shapes ** 2
            self.mdl_adjust(to_probs(probs), scales ** 2, sh)
            return self.probs, self.scales, self.shapes
        else:
            if self.order > 1:
                self.order -= 1
                return self.fit()
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: fit could not converge for dist name ' + self.dist_name + ' and order: ' + str(self.order) + ' results: ' + str(res))
                return None

    def mdl_adjust(self, probs, scales, shapes):
        # adjust for pmin
        # we should also drop statistically identical distros (close pars) --> see udists below
        if self.uq is True:
            idx = [i for i in range(len(probs)) if probs[i] > self.pmin]   # indices to keep
            probs = np.array([probs[i] for i in idx])
            probs /= np.sum(probs)
            scales = np.array([scales[i] for i in idx]) if scales is not None else None
            shapes = np.array([shapes[i] for i in idx]) if shapes is not None else None

        self.probs = list(probs)
        self.scales = None if scales is None else list(scales)
        self.shapes = None if shapes is None else list(shapes)
        self.order = len(self.probs)

# ###########################################################################################
# ###########################################################################################
# ########################## Dist Mix Reduction ##################################################
# ###########################################################################################
# ###########################################################################################


def get_udists(dist_list, min_pval=0.1):   # drop statistically identical distros from a dist_list. Assumes loc=0
    # dist_list = [..., {'dist_name': name, 'params': params, 'prob': prob}, ...]
    keep_list = list(range(len(dist_list)))
    drop_list = list()
    _ = [dist_list[i].update({'g_idx__': i}) for i in range(len(dist_list))]  # add a unique id to each distro
    for d1, d2 in combinations(dist_list, 2):
        rvs_arr = list()
        for md in [d1, d2]:
            try:
                dobj = getattr(sps, md['dist_name'])
                if isinstance(md['params'], (float, int)):
                    md['params'] = [md['params']]
                rvs_arr.append(dobj(*md['params']).rvs(size=1000))

            except AttributeError:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' get_udists::invalid distribution: ' + str(md['dist_name']))
                return None

        D, pval = sps.ks_2samp(*rvs_arr)
        if pval > min_pval:                                                # two "statistically similar" distros have "high" pval: drop one
            drop_list.append(d1['g_idx__'])
            d2['prob'] += d1['prob']

    idx_list = list(set(keep_list) - set(drop_list))
    out_list = [d for d in dist_list if d['g_idx__'] in idx_list]
    psum = sum([vmdl['prob'] for vmdl in out_list])
    for vmdl in out_list:
        vmdl['prob'] /= psum
    l_out = [v for v in out_list]
    done = True if len(out_list) == len(dist_list) else False         # compare to out_list which is what we recurse on
    if done is False:
        return get_udists(l_out)
    else:
        return l_out


# ###########################################################################################
# ###########################################################################################
# ########################## Distribution Mixtures  #########################################
# ###########################################################################################
# ###########################################################################################
class DistributionMixer(object):
    def __init__(self, dist_name, params, data_obj=None, max_order=4):
        """
        """
        self.dist_name = dist_name
        self.has_shape = False if self.dist_name == 'expon' else True
        # p = mp.current_process()
        # p.daemon = False

        self.data_obj = data_obj
        self.disc_dist = None
        self.sim_vals = None
        self.dist_df = None  # a DF with cols x, pdf, cdf and avg
        self.cdf_int = None
        self.ppf_int = None

        try:
            dobj = getattr(sps, str(dist_name))
        except AttributeError:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' Distribution::invalid distribution: ' + str(dist_name))
            dobj = None

        # ##################################################
        # ##################################################
        # ONLY THE CASE OF PARAMS IS DONE
        # ##################################################
        # ##################################################
        # if self.data_obj is not None:       # run EM
        #     self.dist_obj = dobj
        #     self.order_list = range(1, 1 + max_order)
        #     self.order, self.probs, self.pars, self.params = None, None, None, None
        #     self.disc_dist = self.data_obj.mass_dict
        #     self.pval, self.cont_dist = None, None
        #     if DO_MP is True:
        #         with mp.Pool(processes=int(mp.cpu_count())) as pool:
        #             mvals = pool.map(self.mix, self.order_list)
        #         s_ut.reap_children()
        #     else:
        #         mvals = [self.mix(w) for w in self.order_list]
        #     self.set_order(mvals)                     # sets order, probs, pars and pval for this family of mixes
        #     self.params = [self.probs, self.pars]
        #     self.set_data()
        #     s_ut.my_print('pid: ' + str(os.getpid()) + ' SET_ORDER::: name: ' + self.data_obj.data_name + ' distr: ' + self.dist_name + ' order: ' +
        #           str(self.order) + ' probs: ' + str(self.probs) + ' pars: ' + str(self.pars) + ' pval: ' + str(self.pval))
        # else:
        self.dist_obj = dobj
        self.params = params
        if len(params) == 3:
            self.probs, self.pars, self.disc_dist = self.params
            self.order = len(self.probs)
        elif len(params) == 2:
            self.probs, self.pars = self.params
            self.order = len(self.probs)
            self.disc_dist = None
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + 'DistributionMixer ha invalid parameters: ' + str(params))
            raise RuntimeError('failure')
        self.dist_pars = self.set_pars()  # array of len= order with each component containing [shape, loc, scale] or [loc, scale]
        self.vobj = self.set_vobj()

    def set_pars(self):
        # s_ut.my_print('pid: ' + str(os.getpid()) + ' set_pars::: params: ' + str(self.params) + ' order: ' + str(self.order) +
        #       ' has_shape: ' + str(self.has_shape) + ' pars: ' + str(self.pars))
        locs = [0] * self.order
        if self.order == 1:
            if self.has_shape:
                scale = [self.pars[1]]
                shape = [self.pars[0]]
                return list(zip(shape, locs, scale))
            else:
                scale = self.pars
                return list(zip(locs, scale))
        else:
            if self.has_shape:
                scales = self.pars[1]
                shapes = self.pars[0]
                return list(zip(shapes, locs, scales))
            else:
                scales = self.pars
                return list(zip(locs, scales))

    def set_vobj(self):
        if self.has_shape:
            return [self.dist_obj(*v[:-2], loc=v[-2], scale=v[-1]) for v in self.dist_pars]
        else:
            return [self.dist_obj(loc=v[0], scale=v[1]) for v in self.dist_pars]

    def __str__(self):
        return 'dist name: ' + str(self.dist_name) + ' probs: ' + str(self.probs) + ' pars: ' + str(self.pars) + \
               ' disc_dict: ' + str(self.disc_dist) + ' avg: ' + str(self.mean(size=1000)) + ' std: ' + str(self.std(size=1000))

    # def set_order(self, vals):                 # find best order
    #     d_list = [{'dist_name': self.dist_name, 'params': [v[1], v[2]]} for v in vals if v is not None]
    #     u_list = self.data_obj.get_udists(dist_list=d_list)  # first drop identical dists
    #     res = [(dpars['params'], self.kstest() for dpars in u_list]
    #     res.sort(key=lambda x: x[1][1])
    #     self.probs, self.pars = res[0][0]
    #     self.order = len(self.probs)
    #     self.pval = res[0][1][1]

    def kstest(self):
        size = len(self.data_obj.data) if self.data_obj.mass_dict is None else len(self.data_obj.data) + len(self.data_obj.mass_dict)
        mdl_data = self.rvs(size=size)
        D, pval = sps.ks_2samp(self.data_obj.data, mdl_data)
        return D, pval

    def mix(self, w, max_iter=1500, tol=1.0e-3):
        # this is the EM algo
        # impose loc always 0 because loc = min(data) = 0 (dealing with time here)
        return EM(self.data_obj.data, w, self.dist_name, max_iter=max_iter, tol=tol, floc=0.0)

    def rvs(self, size=1, reuse=True):
        if self.sim_vals is None or size > len(self.sim_vals) or reuse == False:
            self.sim_vals = self._rvs(size=size)
            self.dist_df = kde(self.sim_vals, max_len=1000)  # cols: 'x', 'pdf', 'cdf', 'avg'
        return self.sim_vals[:size]

    def _rvs(self, size=1):
        if self.disc_dist is not None:   # there is a discrete part
            disc_probs = list(self.disc_dist.values())
            mass = list(self.disc_dist.keys())
            cum_disc_probs = np.cumsum(np.array([0] + list(disc_probs)))  # len = order + 1
            cum_disc_probs = list(cum_disc_probs) + [1]
            rnd_ = sps.uniform().rvs(size=size)
            m_pts = [len(rnd_[(rnd_ >= cum_disc_probs[i]) & (rnd_ < cum_disc_probs[i + 1])]) for i in range(len(cum_disc_probs) - 1)]
            w_disc = [[mass[i]] * m_pts[i] for i in range(len(mass)) if m_pts[i] > 0]  # simulation of discrete values
            w_disc = [x for yx in w_disc for x in yx]
            w_cont = self._cont_rvs(size=m_pts[-1])                                    # sim cont part
            w = np.array(w_disc + list(w_cont))
            np.random.shuffle(w)    # needed for reuse
            return w
        else:
            return self._cont_rvs(size=size)

    def _cont_rvs(self, size=1):
        cum_probs = np.cumsum(np.array([0] + list(self.probs)))
        rnd_ = sps.uniform().rvs(size=size)
        m_pts = [len(rnd_[(rnd_ >= cum_probs[i]) & (rnd_ < cum_probs[i+1])]) for i in range(self.order)]  # nbr of points from each mix
        rnd = list()
        for i in range(self.order):  # generate the right fraction of rnd's for each component
            if self.has_shape:
                rnd += list(self.dist_obj(*self.dist_pars[i][:-2], loc=self.dist_pars[i][-2], scale=self.dist_pars[i][-1]).rvs(size=m_pts[i]))
            else:
                rnd += list(self.dist_obj(loc=self.dist_pars[i][-2], scale=self.dist_pars[i][-1]).rvs(size=m_pts[i]))
        np.random.shuffle(rnd)   # inplace shuffle
        return rnd

    def set_data(self):
        d = {k: v for k, v in self.__dict__.items()}
        d['avg'] = self.data_obj.data_avg
        d['min'] = self.data_obj.data_min
        for k in ['probs', 'pars']:
            d[k] = list(d[k])
            arr = list()
            for v in d[k]:
                if isinstance(v, np.ndarray) or isinstance(v, tuple) or isinstance(v, list):
                    arr.append(list(v))
                else:  # numeric?
                    arr.append(v)
            d[k] = arr
        d['params'] = [d['probs'], d['pars']]
        for k in ['dist_obj', 'data_obj', 'cont_dist']:
            d.pop(k, None)
        self.data_obj.dist_list.append(d)  # d has keys: dist_name, disc_dist, probs, params, avg, min

    # @staticmethod
    # def set_row(row, arr, col):
    #     return np.array([np.abs(row[col] - x) for x in arr])
        # for x in arr:
        #     row[x] = np.abs(row[col] - x)
        # return row

    @lru_cache(maxsize=None)
    def ppf(self, q, size=1000):
        # 0 < q < 1
        if self.dist_df is None:      # generate dist_df
            _ = self.rvs(size=size)
            self.dist_df = kde(self.sim_vals, x_grid=None, max_cv=5, max_len=1000)

        if self.ppf_int is None:
            self.ppf_int = interp1d(self.dist_df['cdf'].values, self.dist_df['x'].values, kind='cubic', fill_value='extrapolate')

        if not(isinstance(q, (list, tuple, np.ndarray))):
            q = [q]

        if min(q) < 0.0 or max(q) > 1.0:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: DistributionMixer ppf: ' + str(q))

        vals = self.ppf_int(np.array(q))
        if self.is_pos is True:
            return np.array([x for x in vals if x > 0.0])
        else:
            return vals

    def mean(self, size=1000, reuse=True):
        return sum([self.probs[idx] * self.vobj[idx].mean() for idx in range(self.order)])
        # rv_arr = self.rvs(size=size, reuse=reuse)
        # return np.mean(rv_arr)

    def std(self, size=1000, reuse=True):
        return sum([self.probs[idx] * self.vobj[idx].std() for idx in range(self.order)])
        # rv_arr = self.rvs(size=size, reuse=reuse)
        # return np.std(rv_arr)

    @lru_cache(maxsize=None)
    def cdf(self, x, size=1000):
        if self.dist_df is None:      # generate dist_df
            _ = self.rvs(size=size)
            self.dist_df = kde(self.sim_vals, x_grid=None, max_cv=5, max_len=1000)

        if self.cdf_int is None:
            self.cdf_int = interp1d(self.dist_df['x'].values, self.dist_df['cdf'].values, kind='cubic', fill_value='extrapolate')

        if not(isinstance(x, (list, tuple, np.ndarray))):
            x = [x]

        pvals = self.cdf_int(np.array(x))
        return np.array([min(max(0.0, x), 1.0) for x in pvals])  # ensure between 0 and 1!

    def sf(self, x, size=1000):
        return 1.0 - self.cdf(x, size=size)


###############################################################################
###############################################################################
# ############################### Error Measures ##############################
###############################################################################
###############################################################################

def err_func(y_in, yhat_in, etype):
    # flatten to 1d by concatenating rows
    try:
        y = y_in.flatten()
        yhat = yhat_in.flatten()
    except TypeError:
        s_ut.my_print('WARNING: invalid type:: y: ' + str(type(y_in)) + ' yhat: ' + str(type(yhat_in)))
        return None

    if len(y) != len(yhat):
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: shapes:: y: ' + str(np.shape(y_in)) + ' yhat: ' + str(np.shape(yhat_in)))
        return None

    nz_y = y[y != 0.0]
    if len(nz_y) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: all y values are 0')
        return None
    nz_yhat = yhat[y != 0.0]
    if etype == 'sMAPE':
        p = y * yhat
        p_y = y[p != 0.0]
        p_yhat = yhat[p != 0.0]
        return 2.0 * np.mean(np.abs(p_y - p_yhat) / np.abs(p_y + p_yhat))
    elif etype == 'RMSE':
        return np.sqrt(np.mean((y - yhat) ** 2))
    elif etype == 'MAPE':
        return np.mean(np.abs(nz_y - nz_yhat) / np.abs(nz_y))
    elif etype == 'mMAPE':
        return np.median(np.abs(nz_y - nz_yhat) / np.abs(nz_y))
    elif etype == 'wMAPE':
        return np.mean(np.abs(y - yhat)) / np.mean(np.abs(y))
    elif etype == 'MASE':
        return np.mean(np.abs(nz_y - nz_yhat)) / np.mean(np.abs(nz_y[1:] - nz_y[:-1])) if len(nz_y) > 1 else None
    elif etype == 'LAR':  # close to 0 is good
        p = y * yhat
        p_y = y[p > 0]
        p_yhat = yhat[p > 0]
        if len(p_y) > 0:
            return np.mean(np.log(p_yhat / p_y))
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR::LAR: all values <= 0')
            return None
    elif etype == 'BC':
        xform_obj = xf.Transform('box-cox', None)
        if np.min(y) > 0.0 and np.min(yhat) > 0.0:
            ty = xform_obj.fit_transform(y)
            tyhat = xform_obj.transform(yhat)
            if ty is not None and tyhat is not None:
                return err_func(ty, tyhat, 'MAPE')
            else:
                return None
        else:
            return None
    elif etype == 'YJ':
        xform_obj = xf.Transform('yeo-johnson', None)
        ty = xform_obj.fit_transform(y)
        tyhat = xform_obj.transform(yhat)
        return err_func(ty, tyhat, 'MAPE')
    elif etype == 'YJd':
        xform_obj = xf.Transform('yeo-johnson', None)
        ty = xform_obj.fit_transform(y - yhat)
        return np.mean(ty) if ty is not None else None
    elif etype == 'SR':
        if np.min(y) > 0.0 and np.min(yhat) > 0.0:
            ty = np.sqrt(y)
            tyhat = np.sqrt(yhat)
            return err_func(ty, tyhat, 'MAPE')
        else:
            return None
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: invalid error type: ' + str(etype))
        return None


def rmse(arr):
    """
    return the rmse of np array
    :param arr: np array
    :return:
    """
    if isinstance(arr, (float, int)):
        arr = [arr]
    if isinstance(arr, list):
        arr = np.array(arr)
    if any(np.isnan(arr)):
        return np.inf
    else:
        return np.sqrt(np.mean(arr ** 2))


def mad(arr):
    """
    median absolute deviation
    :param arr:
    :return: median(abs(arr - median(arr)))
    """
    return np.median(np.abs(arr - np.median(arr)))


# def mape(y, yhat):
#     u = np.abs(y - yhat)
#     if len(y[y == 0]) == 0:
#         return np.mean(u / np.abs(y))
#     else:
#         return smape(y, yhat)


def smape(y, yhat):
    u = np.abs(y - yhat)
    v = np.abs(y + yhat) / 2.0
    if len(u[u == 0]) == 0:
        return np.mean(v / u)
    else:
        return np.nan


def mape(f, y_col='y', yhat_col='yhat'):
    # f is a DF
    # computed on all the DF
    # assumes col names: y and yhat
    if len(f[(f[y_col] == 0.0) & (f[yhat_col] == 0.0)]) > 0:
        s_ut.my_print('WARNNG: dropping zero/zero from mape computation')
        return mape(f[~((f[y_col] == 0.0) & (f[yhat_col] == 0.0))], y_col=y_col, yhat_col=yhat_col)
    elif len(f[(f[y_col] == 0.0) & (f[yhat_col] != 0.0)]) > 0:
        s_ut.my_print('WARNNG: dropping zero from mape computation')
        return mape(f[~((f[y_col] == 0.0) & (f[yhat_col] != 0.0))], y_col=y_col, yhat_col=yhat_col)
    else:
        return (1 - f[yhat_col] / f[y_col]).abs().mean() if len(f) > 0 else np.nan                     # MAPE


def wmape(f, y_col='y', yhat_col='yhat'):
    # f is a DF
    # computed on all the DF
    # assumes col names: y and yhat
    if len(f[f[y_col] == 0.0]) > 0:
        return wmape(f[f[y_col] != 0.0], y_col=y_col, yhat_col=yhat_col)
    else:
        return (f[yhat_col] - f[y_col]).abs().sum() / f[y_col].abs().sum() if len(f) > 0 else np.nan      # W-MAPE


def mase(f, shift_=1, y_col='y', yhat_col='yhat'):                  # len(f) < season
    # f is a DF with col names: y (actuals) and yhat (for the fancy forecast)
    # compare fcast error to the error of the naive forecaster yhat(t) = yhat(t|t-shift_) = y(t - shift_)
    # shift >=1
    if shift_ <= 0:
        s_ut.my_print('ERROR: invalid shift: ' + str(shift_))
        return np.nan
    f.dropna(inplace=True)
    if len(f) <= np.floor(shift_ / 2):
        s_ut.my_print('WARNING: not enough data for mase: len(f) = ' + str(len(f)) + ' and shift: ' + str(shift_))
        return np.nan
    fs = f.shift(-shift_)
    num = (f[yhat_col] - f[y_col]).abs().mean()        # MASE (for a window < season and factoring in upr and lwr)
    den = (fs[y_col] - f[y_col]).abs().mean()          # MASE (for a window < season and factoring in upr and lwr)
    if den == 0.0:
        return np.inf if num != 0 else np.nan
    else:
        return num / den          # MASE (for a window < season and factoring in upr and lwr)


def thiel_u(f, shift_=1, y_col='y', yhat_col='yhat'):
    # f is a DF
    # assumes col names: y and yhat
    # compare fcast RMSE to the error of the naive forecaster yhat(t) = yhat(t|t-shift_) = y(t - shift_)
    # shift >= 1
    if shift_ <= 0:
        s_ut.my_print('ERROR: invalid shift: ' + str(shift_))
        return np.nan
    f.dropna(inplace=True)
    if len(f) <= np.floor(shift_ / 2):
        s_ut.my_print('WARNING: not enough data for mase: len(f) = ' + str(len(f)) + ' and shift: ' + str(shift_))
        return np.nan
    fs = f.shift(-shift_)
    nu = (((fs[yhat_col] - fs[y_col]) / f[y_col]) ** 2).mean()
    du = (((fs[y_col] - f[y_col]) / f[y_col]) ** 2).mean()
    if du == 0.0:
        return np.inf if nu != 0.0 else np.nan
    else:
        return np.sqrt(nu / du)




###############################################################################
###############################################################################
# ############################### Tests ##############################
###############################################################################
###############################################################################

def white_noise_test(yres, p_thres=0.05, lags=None, verbose=True):
    """
    Test that yres is white noise (normal and no correl):
    Shapiro-Wilkes test for normality (null = normal) for yres and Ljung-Box test for serial correlation
    :param yres: series of residuals
    :param p_thres: p-value threshold to reject null hypothesis
    :param lags: correlation lags to check. See https://robjhyndman.com/hyndsight/ljung-box-test/ , for optimal lags
    :param verbose: print stuff
    :return: True (is white noise), False (not)
    """
    _, s_pval = sps.shapiro(yres)                         # null: it is normal
    _, n_pval = sps.normaltest(yres, nan_policy='omit')   # null: it is Normal
    if s_pval < p_thres or n_pval < p_thres:
        if verbose:
            s_ut.my_print('Residuals are not normal: ' + str(min(s_pval, n_pval)))
        return False   # reject null (that it is Normal)
    else:              # serial correlation. LB Null = No serial correl
        _, lb_pval = smd.acorr_ljungbox(yres, boxpierce=False, lags=lags)
        if not(isinstance(lb_pval, list)) and not(isinstance(lb_pval, np.ndarray)):
            lb_pval = [lb_pval]
        if min(lb_pval) < p_thres:
            if verbose:
                s_ut.my_print('There are serial correlations: ' + str(min(lb_pval)))
            return False
        else:
            if verbose:
                s_ut.my_print('There are no serial correlations: ' + str(min(lb_pval)))
            return True

