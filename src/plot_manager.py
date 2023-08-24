import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, LogLocator
from matplotlib.ticker import (AutoMinorLocator)

import src.global_constants as g
import src.utils as u
from src.exp_manager import SettingsManager


def randint(high):
    return g.rng.integers(0, high=high, endpoint=True)


def get_display_names(algo_names):
    algo_names = [algo_names] if isinstance(algo_names, str) else algo_names

    # Replace names with display names
    # CW -> Cw, CW-SV -> Prop., CRB_unconditional_asym -> Unc. bound asym.
    # CRB_unconditional -> Unc. bound, CRB_conditional -> Cond. bound

    display_names = [algo_name
                      .replace('CW-SV', 'Wideband (prop.)')
                      .replace('CW-EV-SV', 'Wideband (prop. 2)')
                      .replace('CW', 'Narrowband')
                      .replace('CRB_unconditional_asym', 'Bound unc.')
                      .replace('CRB_unconditional', 'Bound unc.')
                      .replace('CRB_conditional', 'Bound cond.') for algo_name in algo_names]

    return display_names


def get_y_label(metric_name):
    # replace matric name with y label, for readability and consistency:
    # 'MSE dB' -> 'MSE [dB]'
    metric_name = metric_name.replace('RMSE dB', 'RMSE [dB]')
    metric_name = metric_name.replace('MSE dB', 'MSE [dB]')

    return metric_name


def plot_errors_wrapper(x_values, errors_array, suptitle=None, title=None, metric_names='metric_name',
                        algo_names='algo_name', x_label='x_label', algo_visible=None, ylim=(None, None), fig=None,
                        ax=None,
                        xscale_log=False, dpi=None, colors=None, font_size=None, legend_font_size=None,
                        legend_num_cols=None, show_plot=True):
    """
    Plots errors_array as a function of x_values. Each subplot is a metric, each line is an algorithm.
    :param ax: axes to plot on
    :param x_values: list: x-axis values
    :param errors_array: (num_x_values, num_algorithms=num_labels, num_metrics=num_subplots, 2=(mean + std))
    :param metric_names: list: one plot per metric
    :param algo_names: list: labels inside each plot
    :param suptitle: plot title
    :param x_label: label for x-axis
    :param algo_visible: ex np.array([1,0,1]) shows first and third algorithms only
    :param ylim: tuple: (ymin, ymax)
    :param fig: figure to plot on
    :param xscale_log: bool: if True, x-axis is log scale
    :param dpi: int: figure dpi
    :param colors: list: colors for each algorithm
    :param font_size: str: font size for labels
    :param legend_font_size: str: font size for legend
    :param legend_num_cols: int: number of columns for legend

    :return: fig, ax
    """

    if font_size is None:
        font_size = 'x-large'

    if legend_font_size is None:
        legend_font_size = 'large'

    if colors is None:
        colors = g.colors

    while errors_array.ndim < 4:
        errors_array = errors_array[..., np.newaxis]

    if algo_visible is not None:
        errors_array = errors_array[:, np.array(algo_visible)]
        algo_names = np.array(algo_names)[algo_visible].tolist()

    algo_names = [algo_names] if isinstance(algo_names, str) else algo_names
    algo_names = get_display_names(algo_names)

    metric_names = [metric_names] if isinstance(metric_names, str) else metric_names
    if len(algo_names) != errors_array.shape[1]:
        warnings.warn(f"{len(algo_names)=} but {errors_array.shape[1]=}: names and actual algorithms results mismatch")

    # assert (len(metric_names) <= errors_array.shape[2])
    if len(metric_names) != errors_array.shape[2]:
        warnings.warn(f"{len(metric_names)=} but {errors_array.shape[2]=}: names and actual metrics results mismatch")
        metric_names = metric_names[:errors_array.shape[2]]

    # preprocess values
    variation_factor_name, variation_factor_values = x_label, x_values
    x_values = np.asarray([float(x) for x in variation_factor_values])
    sorted_indices = x_values.argsort()[:len(errors_array)]
    x_values = x_values[sorted_indices]
    errors_array = errors_array[sorted_indices]

    # set up plot
    num_plots = len(metric_names)
    if fig is None:
        if ax is None:
            plot_area_size = 3.5
            fig = plt.figure(figsize=(1 + plot_area_size, 1 + num_plots * plot_area_size), dpi=dpi)
            axes = fig.subplots(nrows=num_plots, ncols=1, squeeze=False)
        else:
            fig = ax.get_figure()
            axes = np.array([ax])
    else:
        axes = fig.subplots(nrows=num_plots, ncols=1, squeeze=False)

    # x axis
    num_max_x_ticks = 16
    num_x_ticks = min(num_max_x_ticks, len(x_values))

    for metric_idx, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
        min_y = -1e9
        max_y = 1e9

        # find max_y and min_y considering also the std and the possible negative values
        minus = errors_array[..., 1]
        plus = errors_array[..., 2]

        min_plus, max_plus = np.min(plus), np.max(plus)
        min_minus, max_minus = np.min(minus), np.max(minus)

        if metric_name == 'Hermitian angle':
            max_y = min(1.0, max(max_plus, max_minus))
            min_y = max(0.0, min(min_plus, min_minus))
        elif metric_name == 'MSE dB' or metric_name == 'MSE [dB]':
            max_y = min(g.mse_db_max_error, max(max_plus, max_minus))
            min_y = max(g.mse_db_min_error, min(min_plus, min_minus))

        border_len = np.abs(max_y - min_y) / 50
        # max_y += border_len
        # min_y -= border_len

        # if last dimension is not unitary, it contains (mean, standard deviation = std). Otherwise, only mean is present.
        # Plot the std as a shaded area around the mean
        if errors_array.shape[-1] > 1:
            for algo_idx, algo_name in enumerate(algo_names):
                if not (u.is_crb(algo_name) and metric_name == 'Hermitian angle'):
                    # mean = errors_array[:, algo_idx, metric_idx, 0]
                    # mean = np.maximum(np.minimum(mean, max_y - border_len), min_y + border_len)
                    # std = errors_array[:, algo_idx, metric_idx, 1]
                    # ax.fill_between(x_values, mean - std, mean + std, facecolor=colors[algo_idx], alpha=0.2)

                    ax.fill_between(x_values, errors_array[:, algo_idx, metric_idx, 1],
                                    errors_array[:, algo_idx, metric_idx, 2],
                                    facecolor=colors[algo_idx], alpha=0.2)

        # Plot the mean
        for algo_idx, algo_name in enumerate(algo_names):
            if not (u.is_crb(algo_name) and metric_name == 'Hermitian angle'):
                mean = errors_array[:, algo_idx, metric_idx, 0]
                mean = np.maximum(np.minimum(mean, max_y - border_len), min_y + border_len)
                col = colors[algo_idx]

                if u.is_crb(algo_name):
                    if 'cond.' in algo_name.lower():
                        col = "tab:green"
                        line_style = (randint(2), (5, 1, 1))
                    else:
                        col = "tab:purple"
                        line_style = (randint(4), (4, 1))  # Unconditional CRB lines are long-dashed
                elif u.is_cw(algo_name):
                    line_style = (randint(1), (1, 1))  # 'dotted'  # benchmark (CW) lines are dotted
                else:
                    line_style = 'solid'  # other lines are solid

                # ax.plot(x_values, mean, markerfacecolor='none', ms=15, markeredgecolor='red')

                ax.plot(x_values, mean, c=col,
                          label=algo_name, linestyle=line_style, linewidth=1.2,
                          marker=g.markers[algo_idx], markersize=4.5, markeredgecolor=col,
                          markerfacecolor='none', markeredgewidth=0.5)

        # improve legend
        if legend_num_cols is None:
            # num_cols = int(np.ceil(len(algo_names) / 3)) # legend has at most 3 entries per each column.
            legend_num_cols = 2 if len(algo_names) > 2 else 1
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=legend_font_size, ncol=legend_num_cols)
        variation_factor_name_spaces = variation_factor_name.replace('_', ' ')

        # subplot title
        if len(variation_factor_name_spaces) > 0 and variation_factor_name_spaces[-1] not in ['s', ']', ')']:
            variation_factor_name_spaces = variation_factor_name_spaces + "s"
        if title is None:
            ax.set_title(metric_names[metric_idx] + " for different " + variation_factor_name_spaces, fontsize=font_size)
        else:
            ax.set_title(title, fontsize=font_size)

        # x,y label
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_size)
        else:
            ax.set_xlabel(variation_factor_name_spaces, fontsize=font_size)

        ax.set_ylabel(get_y_label(metric_names[metric_idx]), fontsize=font_size)

        # y limits and scale
        if ylim is not None and ylim != (None, None):
            ax.set_ylim(ylim)
        elif metric_name == 'Hermitian angle' or metric_name == 'MSE dB':
            ax.set_ylim(bottom=min_y, top=max_y)

        if metric_name == 'MSE':
            ax.set_yscale("log")

        # x scale
        if xscale_log or \
                variation_factor_name == 'noise_estimate_perturbation_amount' \
                or variation_factor_name == 'duration_output_sec' \
                or variation_factor_name == 'nstft' \
                or variation_factor_name == 'duration_output_frames' \
                or variation_factor_name == 'Time frames':
            ax.set_xscale("log")

        # x, y ticks and grid
        if ax.get_xscale() == 'log':
            x_locator = LogLocator(base=10, numticks=num_x_ticks)
            x_minor_locator = None
            y_minor_locator = AutoMinorLocator(4)
            if any(np.log(x_values) % 1 == 0):
                x_locator = FixedLocator(x_values)
        else:
            x_locator = MaxNLocator(num_x_ticks, integer=True)
            x_minor_locator = AutoMinorLocator(4)
            y_minor_locator = AutoMinorLocator(2)

        ax.set_xticks(x_values, fontsize=font_size)
        if x_locator is not None:
            ax.xaxis.set_major_locator(x_locator)
        ax.tick_params(axis='both', labelsize=font_size)
        ax.grid(which='both')

        # Change minor ticks to show every 5. (20/4 = 5)
        if x_minor_locator is not None:
            ax.xaxis.set_minor_locator(x_minor_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
        ax.grid(which='major', color='#CCCCCC')
        ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
        ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=font_size)

    if show_plot:
        fig.tight_layout()
        fig.show()

    return fig


def plot_errors(settings, err_mean_std_array, title=None):
    x_label, x_values = SettingsManager.get_variation_key_values(settings)
    return plot_errors_wrapper(x_values, errors_array=err_mean_std_array,
                                            suptitle=settings['exp_name'] if title is None else title,
                                            metric_names=settings['metric_names'],
                                            algo_names=settings['algo_names'],
                                            x_label=x_label)


def set_transparent_background_figure(fig):
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.3)
    fig.set_edgecolor('none')
    for ax in fig.get_axes():
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.7)
    return fig

