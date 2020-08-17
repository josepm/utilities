"""
quick unit test for boxcox
"""
import scipy.stats as sp
import numpy as np
import sys
import os

import xforms as xf

# ##################### BC tests
# Fit model to some data
r = sp.norm.rvs(size=100)                      # input data: has negative values!
bc = xf.BoxCox(lmbda=None, verbose=False)      # BC obj
bc.fit(r)                                      # find lmbda

# Use the bc object to transform and back transform.
# Note that the data is not the same as the one used to fit.
v = sp.norm.rvs(size=100)
res = bc.inv_transform(bc.transform(v))
print('1: rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# memory test: BC does not remember where pos and neg vals are
bc = xf.BoxCox(lmbda=0.5, verbose=False)
v = sp.norm.rvs(size=100)
vx = bc.transform(v)
new_bc = xf.BoxCox(lmbda=0.5, verbose=False)  # back-transform with a new object
vv = new_bc.inv_transform(vx)
print('2: rmse: ' + str(np.sqrt(np.sum((v - vv) ** 2)) / len(v)))


# Preset lmbda: can still deal with negative values!
bc = xf.BoxCox(lmbda=0.5, verbose=False)
v = sp.norm.rvs(size=100)
res = bc.inv_transform(bc.transform(v))
print('3: rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# Log case (lmbda = 0): can still deal with negative values!
bc = xf.BoxCox(lmbda=0.0, verbose=False)
v = sp.norm.rvs(size=100)
res = bc.inv_transform(bc.transform(v))
print('4: rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# shifted data
r = sp.norm.rvs(size=100)                      # input data: has negative values!
bc = xf.BoxCox(lmbda=None, verbose=False)      # BC obj
bc.fit(r)                                      # find lmbda

# Use the bc object to transform and back transform
shift = -2
v = sp.norm.rvs(size=100)
res = bc.inv_transform(bc.transform(v + shift)) - shift
print('5: rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# negative lmbda
bc = xf.BoxCox(lmbda=-1.0, verbose=False)
v = sp.norm.rvs(size=100)
v[-1] = 0.0   # enforce 1/0
res = bc.inv_transform(bc.transform(v))
print('rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# lmbda = 1.0
bc = xf.BoxCox(lmbda=1.0, verbose=False)
v = sp.norm.rvs(size=100)
v[-1] = 0.0   # enforce 1/0
res = bc.inv_transform(bc.transform(v))
print('6: rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# ###################### Diff tests
# OC Diff
print('large RMSE may be because stability issues')
v = np.cumsum(sp.norm.rvs(size=1000))
co = xf.ARDiff('test1', v, lag=1)
res = co.inv_diff(co.diff(v), [v[0]])
print('1:slope: ' + str(co.slope) + ' rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# lag not set
v = np.cumsum(sp.norm.rvs(size=1000))
co = xf.ARDiff('test2', v, lag=None)           # sets best lag
res = co.inv_diff(co.diff(v), v[:co.lag])
print('2:lag: ' + str(co.lag) + ' slope: ' + str(co.slope) + ' rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# lag > 1
v0 = sp.norm.rvs(size=1000)
v = np.diff(v0)
lag = 7
co = xf.ARDiff('test3', v, lag=lag)
d_lag = co.diff(v)
res = co.inv_diff(d_lag, v[:lag])
print('3:lag: ' + str(co.lag) + ' slope: ' + str(co.slope) + ' rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))

# lag > 1
v0 = sp.norm.rvs(size=1000)
v = np.cumsum(np.cumsum(np.cumsum(np.cumsum(v0))))
lag = 7
co = xf.ARDiff('test4', v, lag=lag)
d_lag = co.diff(v)
res = co.inv_diff(d_lag, v[:lag])
print('4:lag: ' + str(co.lag) + ' slope: ' + str(co.slope) + ' rmse: ' + str(np.sqrt(np.sum((v - res) ** 2)) / len(v)))
