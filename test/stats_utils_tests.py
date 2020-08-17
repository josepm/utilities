"""
some tests for stats_utiles
"""
import numpy as np
import scipy.stats as sps
from capacity_planning.utilities import stats_utils as s_ut


def sim_data(d_name, probs_, scales_, shapes_, n_pts_):
    shape_pars_default = {'expon': None, 'gamma': [1], 'lognorm': [1], 'weibull_min': [1], 'fatiguelife': [1],
                          'pareto': [1], 'genpareto': [1], 'chi': [1], 'chi2': [1], 'invgamma': [1], 'invgauss': [1], 'invweibull': [1], 'loglapace': [1]}
    dobj = getattr(sps, d_name)
    vals = list()
    sz = [int(n_pts_ * p) for p in probs_]
    for i in range(len(scales_)):
        p = [scales_[i]] if shape_pars_default[d_name] is None else [shapes_[i]] + [scales_[i]]
        if len(p[:-1]) > 0:
            vals += list(dobj(*p[:-1], loc=0, scale=p[-1]).rvs(sz[i]))
        else:
            vals += list(dobj(loc=0, scale=p[-1]).rvs(sz[i]))
    return vals


# ########################################
# Tests for Quantile Fits
# ########################################
# select dist family, dist params and mixture
dist_name = 'gamma'              # family
probs = [0.6, 0.4]               # mixture probs
scales = [1.0, 10.0]             # scale pars for distros
shapes = [4, 1]                  # shape pars (if applicable)
n_pts = 1000                     # total pts to simulate
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]     # quantiles to use for fitting

# sim data and set quantiles dict
vals = sim_data(dist_name, probs, scales, shapes, n_pts)
v = np.sort(vals)
qdict = {q: v[int(n_pts * q) - 1] for q in quantiles}

# estimate and check
q_obj = s_ut.QuantileFit(dist_name, len(probs), qdict)
probs, scales, shapes = q_obj.fit()
fit_quantiles = q_obj.mix_cdf(list(qdict.values()))

# initial quantiles and fit quantiles should be pretty close
print('dist name: ' + str(dist_name) + ' initial quantiles: ' + str(quantiles) + ' fit quantiles: ' + str(fit_quantiles))

# ########################################
# Tests for EM
# ########################################
# select dist family, dist params and mixture
dist_name = 'gamma'              # family
probs = [0.2, 0.4, 0.2, 0.2]               # mixture probs
scales = [1.0, 10.0, 20.0, 100.0]          # scale pars for distros
shapes = [4, 1, 10, 20]                    # shape pars (if applicable)
n_pts = 100                               # total pts to simulate

vals = sim_data(dist_name, probs, scales, shapes, n_pts)
reg_info = set_reg_info(vals)
print('Orig params::probs: ' + str(probs) + ' scales: ' + str(scales) + ' shapes: ' + str(shapes))
em_obj = s_ut.EM(vals, len(probs), dist_name, floc=0.0, reg_info=reg_info)
em_obj.fit()


dist_name = 'lognorm'
probs = [1.0]
scales = [0.00015730416925571387]
shapes = [3.244304697913489]
n_pts = 1000
vals = sim_data(dist_name, probs, scales, shapes, n_pts)
h_obj = s_ut.HyperExp(vals, em_phases=3, max_splits=1, max_cv=1.25, min_size=100, floc=0.0, max_iter=1500, tol=1.0e-3)
h_obj.fit()
