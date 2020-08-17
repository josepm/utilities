"""
plotting function
"""

import matplotlib.pyplot as my_plt
import matplotlib.dates as mdates
import numpy as np
import time_utils as tm_ut
from datetime import datetime as dtime
from scipy.interpolate import spline

my_plt.rcParams.update({'figure.max_open_warning': 0})  # turn off max open figs warning


class PlotInfo(object):
    # collect plotting information for a single graph
    def __init__(self, xaxis, yaxis, zaxis=None, show_legend=False, grid=False, marker='o', size=50, zunit=None,
                 date_fmt=None, show_lines=False, xname=None, yname=None, ymin=None, ymax=None, yunit=None, yscale='linear'):
        self.xaxis = xaxis                                                                            # axis name: str
        self.yaxis = {yaxis: 0} if isinstance(yaxis, str) else yaxis                                  # a dict with key = name and value: 0 (left yaxis), 1 (right yaxis)
        self.yname = self.yaxis if yname is None else {yname: self.yaxis.values()[0]}                 # display name in yaxis. Either None or a dict
        self.ymin = ymin                                                    # manually set min value for y axis
        self.ymax = ymax                                                    # manually set max value for y axis
        self.yunit = yunit                                                  # yunit if any
        self.zunit = zunit                                                  # zunit if any

        if len(self.yaxis) != 1 or isinstance(self.yaxis, dict) == False:
            print('plotting::invalid yaxis PlotInfo: ' + str(yname))

        if isinstance(self.yname, dict) == False:
            print('plotting::invalid yname PlotInfo: ' + str(yname))

        self.yname = {v: k for k, v in self.yname.iteritems()}              # reverse so that yname = {axis_idx: label}
        self.xname = self.xaxis if xname is None else xname                 # display name in xaxis
        self.zaxis = zaxis                                                  # axis name for colors (str)
        self.marker = marker                                                # plot marker
        self.size = size                                                    # marker size
        self.date_fmt = date_fmt                                            # date display format if not None
        self.show_lines = show_lines                                        # show lines or not
        self.show_legend = show_legend                                      # shoiw legend or not
        self.grid = self.yaxis.values()[0] if grid == True else None          # set to yaxis number to know what grid to set
        self.yscale = yscale

        if self.zaxis is not None and self.zunit is None:
            print('plotting::must provide z unit for ' + str(self.yaxis))

    def validate(self, xaxis, zaxis, date_fmt, grid, yaxis, data_cols, xname, zunit, yname):     # checks consistency across graphs within the same plot
        if self.xaxis not in data_cols:
            print('plotting::invalid xaxis: ' + str(self.xaxis) + ' for ' + str(data_cols))

        # only single xaxis
        if xaxis is not None and self.xaxis != xaxis:
            print('plotting::invalid xaxis: ' + str(self.xaxis) + ' should be ' + str(xaxis))

        # only single color scale for everybody
        if zaxis is not None and self.zaxis != zaxis:
            print('plotting::invalid zaxis: ' + str(self.zaxis) + ' should be ' + str(zaxis))

        # only single display date format
        if date_fmt is not None and self.date_fmt != date_fmt:
            print('plotting::date format reset from ' + str(self.date_fmt) + ' to %Y-%m')
            if self.date_fmt == '%Y-%m-%d':
                self.date_fmt = '%Y-%m'

        grid = self.only_one(self.grid, grid, 'grid')         # only one grid
        xname = self.only_one(self.xname, xname, 'xname')     # only one xname
        zunit = self.only_one(self.zunit, zunit, 'zunit')     # only one zunit
        for yidx, yn in self.yname.iteritems():
            yname[yidx] = self.only_one(yn, yname[yidx], 'yname')     # only one yname per yaxis

        ykey = self.yaxis.keys()[0]
        if ykey not in yaxis:
            yaxis[ykey] = self.yaxis[ykey]
        else:
            print('plotting::dup yaxis name: ' + str(ykey))

        if len(set(yaxis.values())) > 2:
            print('plotting::too many yaxis: ' + str(yaxis))

        return self.xaxis, self.zaxis, self.date_fmt, grid, xname, zunit, yname

    @staticmethod
    def only_one(self_val, new_val, lbl):
        if new_val is None:
            return self_val
        else:
            if self_val is None:
                return new_val
            else:
                if self_val != new_val:
                    print('plotting::only one ' + lbl + ' allowed')
                    return None
                else:
                    return new_val


def scatter_gradient_plot(df, plot_info_list, title=None, cset='autumn', lloc='best', f_out=None):
    """
    Single figure scatter or line plot from a dataframe values
    option to have colors, plot lines, ... See PlotInfo
    x-axis and y-axis limits set to be between 95% and 105% of min and max unless min is < 0 in which case it is set to 0.
    Must have a single xaxis and a single figure
    :param df: dataframe with data
    :param plot_info_list: list of PlotInfo objects for each line/scatter to draw
    :param title: plot title
    :param lloc: legend location if any
    :param cset: color set one of: 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'autumn', 'viridis', ...
    :param f_out: output file
    :return: nothing
    """
    # prepare data
    in_df = df.copy()
    in_df.reset_index(inplace=True, drop=True)

    # load plot info
    yaxis = dict()
    xaxis, zaxis, date_fmt, grid, xname, zunit, yname = None, None, None, None, None, None, [None, None]
    for plot_info in plot_info_list:    # check consistency across graphs
        xaxis, zaxis, date_fmt, grid, xname, zunit, yname = plot_info.validate(xaxis, zaxis, date_fmt, grid, yaxis, in_df.columns, xname, zunit, yname)
    yscale = plot_info_list[0].yscale

    # set date to datetime if needed
    if date_fmt is not None:
        d_fmt = in_df[xaxis].apply(tm_ut.get_date_format).unique()
        if len(d_fmt) != 1 or isinstance(d_fmt[0], str) == False:
            print('plotting::invalid input date format')
            print('formats: ' + str(d_fmt))
            return None
        in_df['xaxis'] = in_df[xaxis].apply(lambda x: dtime.strptime(x, d_fmt[0]))
    else:
        in_df['xaxis'] = in_df[xaxis].copy()

    # ensure that 0 is the yaxis when only one yaxis
    yaxis_cnt = len(set(yaxis.values()))
    ymin = [0.0] * yaxis_cnt
    ymax = [None] * yaxis_cnt
    fig, ax0 = my_plt.subplots(1)
    my_plt.gcf().subplots_adjust(bottom=0.15)
    ax_list = [ax0]

    if yaxis_cnt == 1:
        yaxis = {k: 0 for k in yaxis.keys()}            # force left y axis when only one y axis is used
    elif yaxis_cnt == 2:
        ax1 = ax0.twinx()
        ax_list.append(ax1)
        yaxis = {yaxis.keys()[r]: r for r in range(2)}  # ensure 0 and 1 axis numbers
    else:
        print('plotting::invalid yaxis: ' + str(yaxis))
        return None

    x_leg, x_inc = 0.15, 1.0 / len(plot_info_list)
    if zaxis is not None:
        in_df_z = in_df.sort_values(by=zaxis)  # sort by zaxis increasing: drawing order and color sorting?
        nvals, lvals, cvals = get_ticks(in_df_z[zaxis], out_fmt=date_fmt)
        for plot_info in plot_info_list:
            y, ax_idx = plot_info.yaxis.iteritems().next()
            sz = plot_info.size
            marker = plot_info.marker
            show_lines = plot_info.show_lines
            show_legend = plot_info.show_legend
            ymin[ax_idx] = plot_info.ymin if plot_info.ymin is not None else min(ymin[ax_idx], in_df[y].min())
            ymax[ax_idx] = plot_info.ymax if plot_info.ymax is not None else max(ymax[ax_idx], in_df[y].max())
            yname = plot_info.yname[ax_idx] if plot_info.yunit is None else plot_info.yname[ax_idx] + '(' + plot_info.yunit + ')'
            im = ax_list[ax_idx].scatter(np.array(in_df_z['xaxis']), np.array(in_df_z[y]), s=sz, marker=marker, c=cvals, cmap=my_plt.get_cmap(cset), label=yname)
            ax_list[ax_idx].set_ylabel(yname)
            if show_lines == True:
                ax_list[ax_idx].plot(in_df['xaxis'].values, in_df[y].values, c='black')
            if show_legend == True:
                ax_list[ax_idx].legend(scatterpoints=1, loc=9, bbox_to_anchor=(x_leg, 1.1), fontsize='small')  # put legend on top of graph
                x_leg += x_inc
                # ax_list[ax_idx].legend(scatterpoints=1, loc=lloc, fontsize='small')

        # set color bar
        pad = 0.1 if yaxis_cnt == 1 else 0.15
        cbar = fig.colorbar(im, ax=ax_list, pad=pad)
        cbar.set_ticks(nvals)
        cbar.ax.set_yticklabels(lvals, fontsize=10)
        zlabel = zaxis if zunit is None else zaxis + '(' + zunit + ')'
        cbar.set_label(zlabel, labelpad=-2)
    else:
        cmap = my_plt.get_cmap(cset)
        colors = iter(cmap(np.linspace(0, 1, len(plot_info_list))))

        for plot_info in plot_info_list:
            y, ax_idx = plot_info.yaxis.iteritems().next()
            sz = plot_info.size
            marker = plot_info.marker
            show_lines = plot_info.show_lines
            show_legend = plot_info.show_legend
            yname = plot_info.yname[ax_idx] if plot_info.yunit is None else plot_info.yname[ax_idx] + '(' + plot_info.yunit + ')'
            ymin[ax_idx] = plot_info.ymin if plot_info.ymin is not None else min(ymin[ax_idx], in_df[y].min())
            ymax[ax_idx] = plot_info.ymax if plot_info.ymax is not None else max(ymax[ax_idx], in_df[y].max())
            this_color = colors.next()
            im = ax_list[ax_idx].scatter(x=in_df['xaxis'].values, y=in_df[y].values, s=sz, marker=marker, label=plot_info.yaxis.keys()[0], c=this_color)
            ax_list[ax_idx].set_ylabel(yname)
            if show_lines == True:
                ax_list[ax_idx].plot(in_df['xaxis'].values, in_df[y].values, c=this_color)
            if show_legend == True:
                ax_list[ax_idx].legend(scatterpoints=1, loc=9, bbox_to_anchor=(x_leg, 1.1), fontsize='small')  # put legend on top of graph
                x_leg += x_inc

    for i in range(len(ax_list)):
        ymini = 0.95 * ymin[i] if ymin[i] > 0 else 1.05 * ymin[i]
        ymaxi = 1.05 * ymax[i] if ymax[i] > 0 else 0.95 * ymax[i]
        ax_list[i].set_ylim([ymini, ymaxi])

    if date_fmt is not None:
        my_plt.xlim([in_df['xaxis'].min(), in_df['xaxis'].max()])  # ensures getting dates in get_dates
        ax0.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        labels = ax0.get_xticklabels()
        my_plt.setp(labels, rotation=45)
    else:
        my_plt.xlim([min(0, 0.95 * in_df[xaxis].min()), 1.05 * in_df[xaxis].max()])

    if grid is not None:
        ax_list[grid].grid(True)

    my_plt.xlabel(xname)
    if title is not None:
        my_plt.title(title)

    my_plt.yscale(yscale)

    if f_out is not None:
        fmt = 'png'
        fig = ax0.get_figure()
        fig.savefig(f_out + fmt, format=fmt, bbox_inches='tight')


def get_ticks(z_series, out_fmt=None):  # z_series pd.Series already sorted increasing
    zlen = len(z_series)
    if zlen > 5:
        q_arr = [0.0, 0.25, 0.50, 0.75, 1.0]
    elif zlen >= 3:
        q_arr = [0.0, 0.50, 1.0]
    elif zlen >= 2:
        q_arr = [0.0, 1.0]
    elif zlen >= 1:
        q_arr = [0.5]
    else:
        q_arr = None

    if q_arr is not None:
        if 'int' not in str(z_series.dtypes) and 'float' not in str(z_series.dtypes):   # a str: lbl and values are different arrays
            z_list = list(z_series)
            in_fmt = tm_ut.get_date_format(z_list[0])
            cvals = np.array(range(len(z_series)))
            nvals = [int(min(zlen * x, zlen - 1)) for x in q_arr]
            if in_fmt is not None and out_fmt is not None:             # if it is a date
                lvals = [tm_ut.change_format(z_list[idx], in_format=in_fmt, out_format=out_fmt) for idx in nvals]
            else:
                lvals = [z_list[idx] for idx in nvals]
        else:  # numerical z values
            cvals = np.array(z_series)
            nvals = z_series.quantile(q_arr)
            lvals = ['%.2f' % x for x in nvals]
        return nvals, lvals, cvals
    else:
        return None
