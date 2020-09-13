"""
functions to generate fake time series
usage:
ts_obj = TimeSeries(n_pts, mix=1.0,
                    seasonalities=None, events=None, t_size, bounded,
                    trend_cpt=-1, level_cpt=-1,
                    trend_noise_level=0.0, trend_slow_bleed=0.0,
                    level_noise_level=0.0, level_slow_bleed=0.0
                    )
ts = ts_obj.ts
creates:
    # ts: overall time series
    # TS components: seasons, events, trend, level
    # changepoint times for each component: tbreak_trend_, tbreak_level_
    # model parameters at each time: theta_trend_, theta_level_
additional parameters:
    # t_size: xform size. The number of YJ transforms on the initial additve TS (0 = purely additive, neg purely mult, pos a random mixture).
             The abs value of t_size tells the number of YJ transforms to apply.
    # bounded: if len(bounded) = 0, it is not bounded. Otherwise bounded = (lwri_bnd, upr_bnd)
    # trend_cpt: no trend if -1 otherwise have a trend with trend_cpt change points
    # level_cpt: no level if -1 otherwise have a level with level_cpt change points
    # seasonalities: list with season info (name, period, fourier order)
    # events: list with event info (period of each event)
    # noise level: relative noise level. 0.0 <= noise_level < 1.0
    # slow_bleed: relative geometric trend change over time. -1.0 << slow_bleed << 1.0. Should be very close to 0 othrwise things blow up

TODO: impute missing points
TODO: generate TS with outliers
TODO: test slow bleed and noise in general
"""
import sys
import numpy as np
import pandas as pd
import scipy.stats as sps


class TimeSeries(object):
    def __init__(self, n_pts, seasonalities=None, event=None, t_size=3, bounded=(),
                 trend_cpt=-1, level_cpt=-1, trend_noise_level=0.0, trend_slow_bleed=0.0, level_noise_level=0.0, level_slow_bleed=0.0):
        """
        n_pts: number of points in the time series
        trend_cpt: no trend if -1 otherwise have a trend with trend_cpt change points
        level_cpt: no level if -1 otherwise have a level with level_cpt change points
        seasonalities: list with season info (name, period, fourier order). Name is not used
        event: list with event info (period of each event)
        t_size: xform size. The number of YJ transforms on the initial additve TS (0 = purely additive, neg purely mult, pos a random mixture).
                The abs value of t_size tells the number of YJ transforms to apply.
        bounded: TS is bounded if bounded is a tuple (floor, ceiling)
        noise level: trend/level relative noise level. 0.0 <= noise_level < 1.0
        slow_bleed: trend/level relative geometric change over time. -1.0 << slow_bleed << 1.0. Should be very close to 0 otherwise things blow up
        """
        self.n_pts = n_pts
        self.seasonalities = seasonalities
        self.event = event
        self.t_size = t_size
        self.bounded = bounded
        self.trend_cpt = trend_cpt
        self.level_cpt = level_cpt
        self.level_noise_level = level_noise_level
        self.level_slow_bleed = level_slow_bleed
        self.trend_noise_level = trend_noise_level
        self.trend_slow_bleed = trend_slow_bleed
        self.gen_ts()
        if len(self.bounded) > 0:
            self.bounded_ts(floor=min(self.bounded), ceil=max(self.bounded))

    # noinspection PyAttributeOutsideInit
    def gen_ts(self):
        """
        generate data for trend changes
        returns the new ts, season_ts, trend_ts, level_ts, change times for each trend and level, model param ts for each trend and level
        """
        season_ = self.seasonal_data()
        event_ = self.event_data()
        trend_, self.trend_tbreak_, self.trend_thetat_ = self._gen_ts('trend', self.trend_cpt, self.trend_noise_level, self.trend_slow_bleed)
        level_, self.level_tbreak_, self.level_thetat_ = self._gen_ts('level', self.level_cpt, self.level_noise_level, self.level_slow_bleed)
        if self.t_size > 0:     # mixture
            lambdas = np.random.uniform(low=-2.0, high=2.0, size=self.t_size)
            print('TS mixture:: lambdas: ' + str(lambdas))
        elif self.t_size == 0:  # addtive
            lambdas = [1.0]
            print('TS is additive:: lambdas: ' + str(lambdas))
        else:                   # mutiplicative
            lambdas = np.random.uniform(low=1.5, high=3.0, size=int(np.abs(self.t_size)))
            print('TS is multiplicative:: lambdas: ' + str(lambdas))

        # additive components
        self.event = 0 if event_ is None else event_
        self.season = 0 if season_ is None else season_
        self.trend = 0 if trend_ is None else trend_
        self.level = 0 if level_ is None else level_
        self.ts = self.event + self.season + self.trend + self.level
        for lb in lambdas:
            self.ts = sps.yeojohnson(self.ts, lmbda=lb)

    def _gen_ts(self, type_, n_cpt, noise_level, slow_bleed):
        """
        generate data for trend/level changes
        returns: the TS, change times and a TS with the model param values
        type_: 'level' or 'trend'
        """
        if n_cpt <= -1:
            return None, None, None
        else:
            w_step = 10  # start point for changepoints
            tbreak_ = np.sort(np.random.randint(w_step, self.n_pts - w_step, n_cpt))  # changepoints

            if len(tbreak_) == 0:
                tbreak_ = np.array([0])
            sigma = 5
            theta = self.get_theta(type_, n_cpt, sigma)
            print('\nthetat_::::' + str(type_) + ' model values: ' + str(theta))
            thetat_ = self.set_breaks(self.n_pts, tbreak_, theta, slow_bleed)
            if noise_level > 0.0:  # add noise to the trend or the level model parameters
                noise = np.random.normal(0, sigma * noise_level, size=self.n_pts)
                if type_ == 'level':
                    noise = np.abs(noise)
                thetat_ += noise
            if type_ == 'trend':
                y_ = np.cumsum(thetat_)
            else:  # level
                y_ = np.random.poisson(thetat_).astype(float)
                y_ -= np.mean(y_)
            return y_, tbreak_, thetat_

    def bounded_ts(self, floor=0.0, ceil=1.0):
        # TS bounded between floor and ceil
        return floor + (ceil - floor) * np.exp(self.ts) / (1.0 + np.exp(self.ts))

    @staticmethod
    def get_theta(type_, n_cpt, sigma, div=10.0):
        """
        generates model parameters for trend and level
        type_: trend or level
        n_cpt: number of change points
        sigma: Poisson rate (level) or Normal std (trend)
        div: ensure the trend growth rates are smallish
        """
        if type_ == 'trend':
            return np.random.normal(0, sigma / div, size=n_cpt + 2)  # array with slopes
        elif type_ == 'level':
            return np.random.exponential(sigma, size=n_cpt + 2)     # array with Poisson rates
        else:
            print('ERROR: invalid type_')
            sys.exit(0)

    @staticmethod
    def fourier_series(n_pts, seasonality, do_beta=True):
        tm = np.arange(n_pts)
        p, n = seasonality[1:]
        x = 2 * np.pi * np.arange(1, n + 1) / p              # 2 pi n / p
        x = x * tm[:, None]                                  # (2 pi n / p) * t
        x = np.concatenate((np.cos(x), np.sin(x)), axis=1)   # n_pts X 2 * n
        if do_beta:                                          # return a random combination of the fourier components: this sets a random phase
            beta = np.random.normal(size=2 * n)
            return x * beta
        else:
            return x

    def seasonal_data(self):
        """
        generate seasonal data
        """
        if self.seasonalities is None:
            return None
        else:
            if self.n_pts < 2 * max([x[1] for x in self.seasonalities]):
                print('ERROR: no seasonalities set up because not enough data points')
                return None
            terms = {s[0]: self.fourier_series(self.n_pts, s).sum(axis=1) for s in self.seasonalities}
            df = pd.DataFrame(terms)
            df['total'] = df.sum(axis=1)   # add all the seasonal components
            df['total'] -= df['total'].mean()
            return df['total'].values

    def event_data(self, window=0):
        """
        generate event data
        window: pre and post event day change
        """
        if self.event is None:
            return None
        else:
            ev_arr = np.zeros(self.n_pts)
            for ev in self.event:
                if ev > self.n_pts:
                    continue
                evt = np.random.randint(low=0, high=ev, size=1)  # initial event time
                while evt < self.n_pts:
                    val = np.abs(np.random.normal())
                    ev_arr[evt] += val
                    for ix in range(1, window + 1):
                        ev_arr[evt + ix] += val
                        ev_arr[evt - ix] += val
                    evt += ev
            return np.array([max(x, 0) for x in ev_arr])

    @staticmethod
    def set_breaks(n_pts, tbreak_, theta, slow_bleed):
        """
        builds the time series of model parameters, i.e. what the model parameter is at each point in time
        n_pts: nbr of TS points
        tbreak_: times of changepoints
        theta: model parameter values after each change point
        slow_bleed: (small) geom rate change of model parameter between change points
        """
        n_cpt = len(tbreak_)
        thetat_ = np.zeros(n_pts)
        thetat_[:tbreak_[0]] = theta[0]
        print('\nindex: ' + str(0) + ' start: ' + str(0) + ' end: ' + str(tbreak_[0]) + ' value: ' + str(theta[0]))
        for i in range(1, n_cpt):
            print('index: ' + str(i) + ' start: ' + str(tbreak_[i-1]) + ' end: ' + str(tbreak_[i]) + ' value: ' + str(theta[i]))
            thetat_[tbreak_[i-1]:tbreak_[i]] = theta[i] * np.array([(1 + slow_bleed) ** n for n in range(tbreak_[i] - tbreak_[i - 1])])
        thetat_[tbreak_[n_cpt - 1]:] = theta[n_cpt] * np.array([(1 + slow_bleed) ** n for n in range(n_pts - tbreak_[n_cpt - 1])])
        print('index: ' + str(n_cpt) + ' start: ' + str(tbreak_[n_cpt - 1]) + ' end: ' + str(n_pts) + ' value: ' + str(theta[n_cpt]))
        return thetat_

