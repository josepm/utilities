__author__ = 'josep'
"""
Basic time conversion functions
"""

import sys
import os
import time
from datetime import datetime
from datetime import timedelta
from datetime import date
import pytz
import monthdelta
from isoweek import Week
import numpy as np
import pandas as pd
import calendar
from capacity_planning.utilities import sys_utils as su
from capacity_planning.utilities import pandas_utils as p_ut

# see https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior for date string format options
# The isoweek module provide the class Week. Instances represent specific weeks spanning Monday to Sunday.
# Week 1 is defined to be the first week with 4 or more days in January.

# common date format
datetime_format_list = ['%Y-%m-%d', '%Y-%m', '%Y-%m-%d-%H', '%Y-%m-%d %H:%M:%S',
                        '%a, %d %b %Y %H:%M:%S +0000', '%Y-%m-%d %H:%M', '%Y-%m-%d %H', '%Y-%m-%d-%H-%M', '%Y-%m-%d %H:%M',
                        '%Y-%m-%dT%H:%M:%SZ', '%m/%d/%y', '%m/%d/%Y', '%m-%d-%y', '%m-%d-%Y'
                        ]


def iso_weeks(yyyy):
    w = Week(yyyy, 53)
    return 53 if w.year == yyyy else 52


def iso_dates(yyyy):
    # start and end dates of an iso year in week-starting Sunday!
    weeks = iso_weeks(yyyy)
    start = pd.to_datetime(Week(yyyy, 1).monday()) - pd.to_timedelta(1, unit='D')
    end = pd.to_datetime(Week(yyyy, weeks).monday()) - pd.to_timedelta(1, unit='D')
    return start, end


def week_from_date(dd):
    dt = pd.to_datetime(dd)
    return Week.withdate(dt)[1]  # week number



def time_now():
    # current epoch in secs
    return time.time()


def get_date(days_back=0, date_format='%Y-%m-%d', tz_str='US/Pacific'):
    """
    get the date days_back from now
    :param days_back: 0, get today's date, 1 yesterday's
    :param date_format: output format
    :param tz_str: TZ of the output
    :return: a string in date_format
    """
    t = time.time() - days_back * 86400
    return to_date(t, date_format, tz_str)


def to_date(epoch, date_format='%a, %d %b %Y %H:%M:%S +0000', tz_str='US/Pacific'):
    """
    return the date in tz_str timezone for the epoch
    :param epoch: input unix timestamp in secs
    :param date_format: format of output
    :param tz_str: timezone of output
    :return: astring date in date_format at tz_str
    """
    if tz_str in pytz.common_timezones:
        t_gmt = time.gmtime(epoch)
        dt_gmt = datetime(t_gmt.tm_year, t_gmt.tm_mon, t_gmt.tm_mday, t_gmt.tm_hour, t_gmt.tm_min, t_gmt.tm_sec)
        dt_str = dt_gmt.strftime('%Y-%m-%d')
        hr_shift = get_shift(dt_str, tz_str)
        dt = dt_gmt + timedelta(hours=hr_shift)
        return dt.strftime(date_format)
    else:
        return ''


def to_timestamp(a_date, date_format='%Y-%m-%d %H:%M:%S', tz_str='US/Pacific'):
    """
    Date in tz_str to unix timestamp
    :param a_date: a string date
    :param date_format: date format
    :param tz_str: string to describe timezone of a_date
    :return: timestamp in secs
    """
    epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
    tz = pytz.timezone(tz_str)
    tt = tz.localize(datetime.strptime(a_date, date_format))
    return int((tt - epoch).total_seconds())


def days_diff(d1, d2, date_format=None):
    """
    days between d1 and d2 both included, so if d1 = d2, it returns 1.
    :param d1: date string
    :param d2: date string
    :param date_format:
    :return:
    """
    f1 = get_date_format(d1) if date_format is None else date_format
    f2 = get_date_format(d2) if date_format is None else date_format
    if f1 is None or f2 is None:
        return None
    else:
        date1 = datetime.strptime(d1, f1)
        date2 = datetime.strptime(d2, f2)
        return (date2 - date1).days


def months_diff(d1, d2, date_format=None):
    """
    days between d1 and d2 both included, so if d1 = d2.
    :param d1: date string
    :param d2: date string
    :param date_format:
    :return: d2 - d1 in months
    """
    f1 = get_date_format(d1) if date_format is None else date_format
    f2 = get_date_format(d2) if date_format is None else date_format
    if f1 is None or f2 is None:
        return None
    else:
        date1 = datetime.strptime(d1, f1)
        date2 = datetime.strptime(d2, f2)
        return monthdelta.monthmod(date1, date2)[0].months


# compute the hour shift between GMT and a timezone
def get_shift(str_date, tz_str='US/Pacific'):
    """
    tz: string of the time zone
    a_date: dat in yyyy-mm-dd format
    shift hours, positive if ahead of GMT, negative if behind GMT
    """
    if tz_str in pytz.common_timezones:
        tz = pytz.timezone(tz_str)
        delta = tz.utcoffset(datetime.strptime(str_date, '%Y-%m-%d'))
        return delta.days * 24 + delta.seconds / 3600  # negative if behind GMT, positive if ahead
    else:
        return None


def get_from_str_date(str_date, date_format, what):
    """
    get day, month, year, hour, minute, sec from str date
    :param str_date: date as a string
    :param date_format: date format
    :param what: year, month, day, hour, min, sec
    :return:
    """
    t = datetime.strptime(str_date, date_format)
    if what == 'year':
        return t.year
    elif what == 'month':
        return t.month
    elif what == 'day':
        return t.day
    elif what == 'hour':
        return t.hour
    elif what == 'minute':
        return t.minute
    elif what == 'second':
        return t.second
    elif what == 'weekday':
        return t.weekday()
    else:
        print('invalid what value: ' + str(what))


def get_yday(str_date, date_format=None):
    """
    get day number within a year
    :param str_date:
    :param date_format:
    :return:
    """
    fmt = get_date_format(str_date) if date_format is None else date_format
    if fmt is None:
        return None
    else:
        t = datetime.strptime(str_date, fmt).timetuple()
        return t.tm_yday


def get_yweek(str_date, date_format=None):
    """
    get week number within a year, from week 0 to week 52
    :param str_date:
    :param date_format:
    :return:
    """
    fmt = get_date_format(str_date) if date_format is None else date_format
    if fmt is None:
        return None
    else:
        date_str = change_format(str_date, in_format=fmt, out_format='%Y-%m-%d')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        w = Week.withdate(date(date_obj.year, date_obj.month, date_obj.day))
        return w[1]


def get_yw(str_date, date_format=None):
    """
    get week number within a year, from week 0 to week 52
    :param str_date:
    :param date_format:
    :return:
    """
    fmt = get_date_format(str_date) if date_format is None else date_format
    if fmt is None:
        return None
    else:
        date_str = change_format(str_date, in_format=fmt, out_format='%Y-%m-%d')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        w = Week.withdate(date(date_obj.year, date_obj.month, date_obj.day))
        return str(w[0]) + '-' + str(w[1] if w[1] > 9 else '0' + str(w[1]))    #  yyyy-ww


def check_date_format(a_date, date_format='%Y-%m-%d'):  # check yyy-mm-dd
    if isinstance(a_date, str):
        try:
            datetime.strptime(a_date, date_format)
            return True
        except ValueError:
            return False
    else:   # try UX time stamp
        if a_date is None:
            return False
        else:
            if np.isnan(a_date):
                return False
            elif a_date < 0.0:
                return False
            else:
                try:
                    f_date = float(a_date)
                    return isinstance(f_date, float)   # a time stamp could, in theory, be negative
                except ValueError:
                    return False


def get_date_format(a_date):
    for f in datetime_format_list:
        if check_date_format(a_date, date_format=f):
            return f
    print('Unsupported date format for ' + str(a_date))
    return None


def add_days(a_date, days, date_format=None):
    """
    add days to a date
    :param a_date: string date
    :param days: days to add
    :param date_format: input and output date format
    :return: new string date with days added
    """
    fmt = get_date_format(a_date) if date_format is None else date_format
    if fmt is not None:
        dt = datetime.strptime(a_date, fmt) + timedelta(days=days)
        return dt.strftime(fmt)
    else:
        return None


def add_months(a_date, months, date_format=None):
    """
    add days to a date
    :param a_date: string date
    :param months: months to add
    :param date_format: input and output date format
    :return: new string date with months added
    """
    fmt = get_date_format(a_date) if date_format is None else date_format
    if fmt is not None:
        dt = datetime.strptime(a_date, fmt) + monthdelta.monthdelta(months)
        return dt.strftime(fmt)
    else:
        return None


def change_format(date_str, in_format=None, out_format='%Y-%m'):
    in_format = get_date_format(date_str) if in_format is None else in_format
    return datetime.strftime(datetime.strptime(date_str, in_format), out_format)


def date_range(s_date, e_date, inc=1, out_format=None):
    # list of dates in date format from init_date <= date < end_date
    # if out_format contains %d, add days
    # if outformat contains %m, add months
    # if outformat contains %H:%M:%S ...  (needs extending)
    s_fmt = get_date_format(s_date)
    e_fmt = get_date_format(e_date)
    if e_fmt is None or s_fmt is None:  # cannot decide what the dates are!
        print('date_range: unsupported input date format: ' + str(s_date) + ' ' + str(e_date))
        return None
    else:                               # if out_format is None, use s_fmt
        if out_format in datetime_format_list:
            s_date = change_format(s_date, in_format=s_fmt, out_format=out_format)
            e_date = change_format(e_date, in_format=e_fmt, out_format=out_format)
        else:  # default to s_fmt
            # print('date_range: unsupported output date format: ' + str(out_format) + ' defaulting to ' + str(s_fmt))
            e_date = change_format(e_date, in_format=e_fmt, out_format=s_fmt)
    if '%d' in out_format:
        add_func = add_days
    elif '%m' in out_format:
        add_func = add_months
    else:
        print(out_format + ' add not implemented')
        return None
    ym = s_date
    dates_list = list()
    while ym < e_date:  # generate all lease story
        dates_list.append(str(ym))
        ym = add_func(ym, inc, date_format=out_format)
    return dates_list


def next_weekday(ddate, weekday):
    """
    Get the next date with give weekday
    :param ddate: a datetime date
    :param weekday: datetime weekday (0 Monday, 1 Tuesday, ...
    :return:
    """
    days_ahead = weekday - ddate.weekday()
    if days_ahead <= 0:            # Target day already happened this week
        days_ahead += 7
    return ddate + pd.to_timedelta(days_ahead, unit='D')


def last_saturday(a_date):      # last Saturday in a month before a_date (can be same month or prior month)
    day_, year_, month_ = a_date.day, a_date.year, a_date.month
    last_sat_ = max([week[calendar.SATURDAY] for week in calendar.monthcalendar(year_, month_)])
    if last_sat_ <= day_:  # in the same month
        return pd.to_datetime('-'.join([str(year_), str(month_), str(last_sat_)]))
    else:                # in the previous month
        nmonth = (month_ - 1) if month_ > 1 else 12
        nyear = year_ if nmonth < 12 else year_ - 1
        nlast_sat = max([week[calendar.SATURDAY] for week in calendar.monthcalendar(nyear, nmonth)])
        return pd.to_datetime('-'.join([str(nyear), str(nmonth), str(nlast_sat)]))


def last_saturday_month(a_date):      # last Saturday in a month before a_date (can be same month or prior month)
    day_, year_, month_ = a_date.day, a_date.year, a_date.month
    last_sat_ = max([week[calendar.SATURDAY] for week in calendar.monthcalendar(year_, month_)])
    if last_sat_ <= day_:  # in the same month
        return pd.to_datetime('-'.join([str(year_), str(month_), str(last_sat_)]))
    else:                # in the previous month
        nmonth = (month_ - 1) if month_ > 1 else 12
        nyear = year_ if nmonth < 12 else year_ - 1
        nlast_sat = max([week[calendar.SATURDAY] for week in calendar.monthcalendar(nyear, nmonth)])
        return pd.to_datetime('-'.join([str(nyear), str(nmonth), str(nlast_sat)]))


def last_saturday_my(yy, mm):
    # last saturday of month mm (mm number for 1 to 12) in year yy (yy in 4 digits, ie 2019 not 19)
    dd = max([week[calendar.SATURDAY] for week in calendar.monthcalendar(yy, mm)])
    mm_str = str(mm) if mm >= 10 else '0' + str(mm)
    dd_str = str(dd) if dd >= 10 else '0' + str(dd)
    return pd.to_datetime(str(yy) + '-' + mm_str + '-' + dd_str)


def get_last_sat(r_date):
    # get max(saturdays) such that Saturday <= r_date
    dt = pd.to_datetime(r_date)
    for i in range(7):
        ds = dt - pd.to_timedelta(i, unit='D')
        if ds.weekday() == 5:       # a Saturday
            return ds
    return None


def set_week_start(adf, tcol='ds'):
    if 'ds_week_starting' in adf.columns and 'ds_week_ending' in adf.columns:
        adf['ds_week_starting'] = pd.to_datetime(adf['ds_week_starting'])
    elif 'ds_week_starting' in adf.columns and 'ds_week_ending' not in adf.columns:
        adf['ds_week_starting'] = pd.to_datetime(adf['ds_week_starting'])
    elif 'ds_week_starting' not in adf.columns and 'ds_week_ending' in adf.columns:
        adf['ds_week_ending'] = pd.to_datetime(adf['ds_week_ending'])
        adf['ds_week_starting'] = adf['ds_week_ending'] - pd.to_timedelta(6, unit='D')
    else:
        if tcol not in adf.columns:
            su.my_print('ERROR: invalid tcol: ' + str(tcol))
            print(adf.head())
            sys.exit()
        adf_copy = adf.copy()
        adf_copy.reset_index(inplace=True, drop=True)
        ts = adf_copy[tcol].copy()
        ts.drop_duplicates(inplace=True)
        ts.sort_values(inplace=True)
        ts.dropna(inplace=True)
        ts = pd.Series(pd.to_datetime(ts.values))
        ts_freq = pd.infer_freq(ts)
        if ts_freq is None:        # make a guess
            su.my_print('WARNING: no time frequency found. making educated guesses')
            diffs = ts.diff().dt.days
            v = diffs.value_counts(normalize=True)
            freq_days = v.index[0]  # most common freq in days
            if v.max() > 0.9:  # 90% of the times the freq in days is freq_days
                if freq_days == 1:
                    ts_freq = 'D'
                elif freq_days == 7:
                    ts_freq = 'W'
                else:
                    su.my_print('ERROR: unknown time frequency. Frequency values observed (in days) and fraction: ')
                    print(v)
                    sys.exit()

        if 'W' in ts_freq:
            dw = adf_copy.loc[adf_copy.index[0], tcol].weekday()
            if dw == 5:  # set to week_ending Sat
                if tcol == 'ds':
                    adf[tcol] = pd.to_datetime(adf[tcol])
                    adf[tcol] -= pd.to_timedelta(6, unit='D')
                    return
                else:
                    adf[tcol] = pd.to_datetime(adf[tcol])
                    adf['ds_week_starting'] = adf[tcol] - pd.to_timedelta(6, unit='D')
                    adf[tcol] = adf[tcol] - pd.to_timedelta(6, unit='D')
                    return
        elif 'D' in ts_freq:
            pass
        else:
            su.my_print('ERROR: invalid frequency')
            adf = None
