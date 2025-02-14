import warnings

import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, LogLocator
from matplotlib.ticker import (AutoMinorLocator)
import matplotlib.ticker as tck

import src.global_constants as g
import src.utils as u
from src.settings_manager import SettingsManager


def randint(high):
    return g.rng.integers(0, high=high, endpoint=True)


def get_display_names(algo_names):
    algo_names = [algo_names] if isinstance(algo_names, str) else algo_names

    # Replace names with display names
    # CW -> Cw, CW-SV -> Prop., CRB_unconditional_asym -> Unc. bound asym.
    # CRB_unconditional -> Unc. bound, CRB_conditional -> Cond. bound

    display_names = [algo_name
                     .replace('CW-SV-orig-phase', 'Wideband (original phase)')
                     .replace('CW-EV-SV', 'Wideband (prop. 2)')
                     .replace('CW-SV', 'Wideband (proposed)')
                     .replace('CW', 'Narrowband')
                     .replace('Ideal', 'True RTF')
                     .replace('Unprocessed', 'Noisy')
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


def plot_errors_wrapper(x_values, errors_array, suptitle=None, title=None, rtf_metrics='metric_name',
                        algo_names='algo_name', x_label='x_label', algo_visible=None, ylim=(None, None), fig=None,
                        ax=None,
                        xscale_log=False, dpi=None, colors=None, font_size=None, legend_font_size=None,
                        legend_num_cols=None, show_plot=True):
    """
    Plots errors_array as a function of x_values. Each subplot is a metric, each line is an algorithm.
    :param ax: axes to plot on
    :param x_values: list: x-axis values
    :param errors_array: (num_x_values, num_algorithms=num_labels, num_metrics=num_subplots, 2=(mean + std))
    :param rtf_metrics: list: one plot per metric
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

    if errors_array.size == 0:
        return None

    while errors_array.ndim < 4:
        errors_array = errors_array[..., np.newaxis]

    if algo_visible is not None:
        errors_array = errors_array[:, np.array(algo_visible)]
        algo_names = np.array(algo_names)[algo_visible].tolist()

    algo_names = [algo_names] if isinstance(algo_names, str) else algo_names
    algo_names = get_display_names(algo_names)

    rtf_metrics = [rtf_metrics] if isinstance(rtf_metrics, str) else rtf_metrics
    if len(algo_names) != errors_array.shape[1]:
        warnings.warn(f"{len(algo_names)=} but {errors_array.shape[1]=}: names and actual algorithms results mismatch")

    # assert (len(rtf_metrics) <= errors_array.shape[2])
    if len(rtf_metrics) != errors_array.shape[2]:
        warnings.warn(f"{len(rtf_metrics)=} but {errors_array.shape[2]=}: names and actual metrics results mismatch")
        rtf_metrics = rtf_metrics[:errors_array.shape[2]]

    # preprocess values
    variation_factor_name, variation_factor_values = x_label, x_values
    # x_values = np.asarray([float(x) for x in variation_factor_values])
    x_values = np.array(x_values)
    sorted_indices = x_values.argsort()[:len(errors_array)]
    x_values = x_values[sorted_indices]
    errors_array = errors_array[sorted_indices]

    # set up plot
    num_plots = len(rtf_metrics)
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

    for metric_idx, (ax, metric_name) in enumerate(zip(axes.flat, rtf_metrics)):
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
                    line_style = (algo_idx % 2, (1, 1))  # 'dotted'  # benchmark (CW) lines are dotted
                elif 'phase' in algo_name.lower():
                    # long dashed
                    line_style = (randint(4), (4, 1))
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
            ax.set_title(rtf_metrics[metric_idx] + " for different " + variation_factor_name_spaces, fontsize=font_size)
        else:
            ax.set_title(title, fontsize=font_size)

        # x,y label
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_size)
        else:
            ax.set_xlabel(variation_factor_name_spaces, fontsize=font_size)

        ax.set_ylabel(get_y_label(rtf_metrics[metric_idx]), fontsize=font_size)

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
                or variation_factor_name == 'nstft' \
                or variation_factor_name == 'duration_output_frames' \
                or variation_factor_name == 'Time frames':
            ax.set_xscale("log")
                # or variation_factor_name == 'duration_output_sec' \

        # x, y ticks and grid
        if ax.get_xscale() == 'log':
            x_locator = tck.LogLocator(base=10, numticks=num_x_ticks)
            x_minor_locator = None
            y_minor_locator = tck.AutoMinorLocator(4)
            if any(np.log(x_values) % 1 == 0):
                x_locator = tck.FixedLocator(x_values)
        else:
            if num_x_ticks < 8:  # only a few x-ticks, so show all of them
                x_locator = tck.FixedLocator(x_values)
                x_minor_locator = tck.AutoMinorLocator(4)
                y_minor_locator = tck.AutoMinorLocator(2)
            else:
                x_locator = tck.MaxNLocator(num_x_ticks, integer=True)
                x_minor_locator = tck.AutoMinorLocator(4)
                y_minor_locator = tck.AutoMinorLocator(2)

        ax.set_xticks(x_values)
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
        fig.show()

    return fig


def compute_plot_text(sett, use_tex_labels_, xscale_log):
    """ Compute the title and x_label for the plot, depending on the experiment name and settings."""

    exp_name = sett['exp_name']
    sett = SettingsManager.assign_default_values(sett)
    var_key, x_values = SettingsManager.get_variation_key_values(sett)
    var_key_display = var_key + '_display' if isinstance(var_key, str) else var_key
    if var_key_display in sett:
        _, x_values = SettingsManager.get_variation_key_values(sett, [var_key_display])
    correlation_name = 'Noise' if 'correlation_noise' in sett['varying_factors'] else 'Target'
    correlation_name = correlation_name + f" {sett['correlation_noise_type']} corr."

    if exp_name == 'target_correlation':
        # x_values_transformed = [int(100 * x) for x in x_values]
        x_values_transformed = [round(x, 2) for x in x_values]
        if use_tex_labels_:
            title = f"$L={sett['duration_output_frames']}, \\upsilon_f={sett['correlation_noise']}$"
            # x_label = f"$\\rho_f\ [\%]$"
            x_label = rf"$\rho_f$"
        else:
            title = f"{sett['duration_output_frames']} frames, " \
                    f"{sett['correlation_noise'] * 100:.0f}% noise corr."
            x_label = f"{correlation_name} [%]"
        # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

    elif exp_name == 'noise_correlation':

        # x_values_transformed = [int(100 * x) for x in x_values]
        x_values_transformed = [round(x, 2) for x in x_values]
        if use_tex_labels_:
            title = f"$L={sett['duration_output_frames']}, \\rho_f={sett['correlation_target']}$"
            if sett['correlation_noise_type'] == 'frequency':
                subscript = 'f'
            elif sett['correlation_noise_type'] == 'space':
                subscript = 's'
            else:
                subscript = 'f+s'
            x_label = rf"$\upsilon_{subscript}$"
        else:
            title = f"{sett['duration_output_frames']} frames, " \
                    f"{sett['correlation_target'] * 100:.0f}% target corr."
            x_label = f"{correlation_name} [%]"
        # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

    elif exp_name == 'time_frames':
        if use_tex_labels_:
            title = f"$\\rho_f={sett['correlation_target']}, \\upsilon_f={sett['correlation_noise']}$"
            x_label = "Number of frames"
        else:
            title = f"{sett['correlation_target'] * 100:.0f}% target corr., " \
                    f"{sett['correlation_noise'] * 100:.0f}% noise corr."
            x_label = "Number of frames"
        xscale_log = True
        x_values_transformed = [int(x) for x in x_values]

        # suptitle = f"Varying number of frames, {sett['noises_info'][0]['snr'][0]} dB SNR"

    elif exp_name == 'snr':
        if use_tex_labels_:
            title = rf"$\rho_f={sett['correlation_target']}, \upsilon_f={sett['correlation_noise']}$"
            x_label = r"SNR [dB]"
        else:
            title = f"{sett['correlation_target'] * 100:.0f}% target corr., " \
                    f"{sett['correlation_noise'] * 100:.0f}% noise corr."
            x_label = f"SNR [dB]"
        x_values_transformed = [int(x) for x in x_values]

    elif exp_name == 'speech_snr':
        title = f""
        # title = f"Real speech"
        x_label = f"SNR [dB]"
        x_values_transformed = [int(x) for x in x_values]

    elif exp_name == 'speech_time_frames':
        title = f""
        # title = f"Real speech"
        x_label = f"Number of frames"
        x_values_transformed = [int(x) for x in x_values]
        xscale_log = True

    elif exp_name == 'speech_nstft':
        title = f""
        # title = f"Real speech"
        x_label = f"FFT size"
        if use_tex_labels_:
            x_label = f"FFT size $K_2$"
        x_values_transformed = [int(x) for x in x_values]
        xscale_log = False

    elif exp_name == 'speech_num_mics':
        title = f""
        x_label = f"Number of microphones"
        if use_tex_labels_:
            x_label = f"Number of microphones $M$"
        x_values_transformed = [int(x) for x in x_values]
        xscale_log = False

    elif exp_name == 'speech_noise_position':
        title = f""
        x_label = f"Target/interf. angular distance [deg]"
        x_values_transformed = [int(x) - sett['rir_settings']['target_angle'] for x in x_values]
        xscale_log = False

    elif exp_name == 'speech_time_seconds':
        title = f""
        if use_tex_labels_:
            x_label = f"Segment length $T$ [s]"
        else:
            x_label = f"Segment length [s]"
        x_values_transformed = [round(x, 1) for x in x_values]
        xscale_log = False

    else:
        title = 'debug'
        x_values_transformed = [int(x) if x > 1 else int(100 * x) for x in x_values]
        x_label = var_key
        # x_label = f"SNR [dB]"
        # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

    return title, x_label, x_values_transformed, xscale_log


def set_transparent_background_figure(fig):
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.3)
    fig.set_edgecolor('none')
    for ax in fig.get_axes():
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.7)
    return fig


def plot_spectrogram(y, suptitle=''):
    # y = beamforming_evaluators_dict[-1]['STOI'][-1].estimates_dict['CW-SV'][-1]
    # y = beamforming_evaluators_dict[-1]['STOI'][-1].estimates_dict['CW'][-1]
    # y = beamforming_evaluators_dict[-1]['STOI'][-1].estimates_dict['Ideal'][-1]
    # y = beamforming_evaluators_dict[-1]['STOI'][-1].ground_truth[-1]

    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]

    sr = 16000
    nstft = 1024
    hop_length = nstft // 2

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=nstft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax)
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    if suptitle != '':
        fig.suptitle(suptitle)

    fig.show()
    return fig


def plot_errors(errors_all_figures, settings_all_figures, is_beamforming_error=False, use_tex_labels_=False,
                show_plot=False):
    """ Plot errors for each figure. Each figure contains errors for different algorithms and metrics. """

    suptitle = ''
    u.set_plot_options(use_tex_labels_)
    y_min = np.inf
    y_max = -np.inf
    xscale_log = False
    plot_area_size = 2.75
    figure_size = (1 + plot_area_size, 1 + plot_area_size)
    figures = []

    for errors_fig, settings_fig in zip(errors_all_figures, settings_all_figures):

        if not isinstance(errors_fig, np.ndarray):
            raise ValueError(f"Expected errors_fig to be a numpy array, but got {type(errors_fig)} instead.")
        if not isinstance(settings_fig, dict):
            raise ValueError(f"Expected settings_fig to be a dict, but got {type(settings_fig)} instead.")

        fig_opt = dict(figsize=figure_size, constrained_layout=True, sharex=True, sharey=True)
        fig, axes = plt.subplots(**fig_opt, squeeze=True)

        title, x_label, x_values_transformed, xscale_log = \
            compute_plot_text(settings_fig, use_tex_labels_, xscale_log)

        metric_names = settings_fig['beamforming_metrics'] if is_beamforming_error else settings_fig['rtf_metrics']

        algo_names = settings_fig['algo_names_displayed'] if 'algo_names_displayed' in settings_fig else \
            settings_fig['algo_names']
        if is_beamforming_error:
            algo_names = SettingsManager.get_algo_names_beamforming(algo_names)
        algo_names = SettingsManager.convert_algo_names_correlation_type(algo_names, settings_fig)

        f = plot_errors_wrapper(x_values_transformed,
                                ax=axes,
                                errors_array=errors_fig,
                                suptitle=suptitle, title=title,
                                x_label=x_label,
                                rtf_metrics=metric_names,
                                algo_names=algo_names,
                                legend_num_cols=1,
                                show_plot=show_plot,
                                font_size='x-large',
                                legend_font_size='medium',
                                xscale_log=xscale_log,
                                )
        y_min = min(y_min, axes.get_ylim()[0])
        y_max = max(y_max, axes.get_ylim()[1])
        figures.append(f)

    # set ylims, remove legends, and label outer axes.
    for figure, settings_fig in zip(figures, settings_all_figures):
        # figure = plot_manager.set_transparent_background_figure(figure)
        for ax_idx, ax in enumerate(figure.get_axes()):
            # ax.set_ylim(0, 0.3)
            if 'y_lim' in settings_fig:
                ax.set_ylim(settings_fig['y_lim'])
            else:
                if abs(y_max - y_min) > 5:
                    ax.set_ylim(np.floor(y_min), np.ceil(y_max))
                else:  # round to 1 decimal
                    ax.set_ylim(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10)
            ax.label_outer()
            if ax_idx != 0:
                try:
                    ax.get_legend().remove()
                except AttributeError:
                    pass

    return figures


def plot_hermitian_angle_per_frequency_and_psd(stimuli_stft, loud_bins_mask, processed_freqs_mask, err, nstft):

    if 'Hermitian angle' not in err['CW']:
        return

    target_pow = np.mean(u.MSE_single(stimuli_stft['wet']), axis=(0, 2))[loud_bins_mask]
    noise_pow = np.mean(u.MSE_single(stimuli_stft['noise_mix']), axis=(0, 2))[loud_bins_mask]

    # err contains errors not averaged over frequency. plot them in a subplot. other subplot shows psd of target
    # and noise signals. only use matplotlib
    plt.subplot(2, 1, 1)
    x_axis = np.fft.rfftfreq(nstft, 1 / g.fs)[processed_freqs_mask]
    x_axis_loud = x_axis[loud_bins_mask]

    plt.plot(x_axis_loud, u.log_pow(target_pow), label='target', linestyle='-', color='green')
    plt.plot(x_axis_loud, u.log_pow(noise_pow), label='noise', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Power spectral density')
    plt.grid(True, which='both')
    plt.subplot(2, 1, 2)
    plt.plot(x_axis_loud, err['CW']['Hermitian angle'])
    plt.plot(x_axis_loud, err['CW-SV']['Hermitian angle'])
    plt.legend(err.keys())
    plt.grid(True, which='both')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Hermitian angle (rad)')
    plt.tight_layout()
    plt.show()
