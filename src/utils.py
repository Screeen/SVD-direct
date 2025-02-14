import copy
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import scipy
from matplotlib import pyplot as plt
import pydub
from functools import reduce
import operator
from itertools import zip_longest

from numba import njit

import src.global_constants as g
import sys
# noinspection PyUnresolvedReferences
from pprint import pprint as pprint


def normalize_volume(x_samples, max_value=0.9):
    if np.max(np.abs(x_samples)) < 1e-6:
        warnings.warn(f"Skipping normalization as it would amplify numerical noise.")
        return x_samples
    else:
        return max_value * x_samples / np.max(np.abs(x_samples))


def play(sound, max_length_seconds=5, normalize_flag=True, volume=0.75, fs=16000):
    import sounddevice  # needs to stay inside function because not available on cluster
    sound_normalized = volume * normalize_volume(sound) if normalize_flag else sound
    max_length_samples = int(max_length_seconds * fs)

    if sound.ndim > 1:
        if sound.shape[0] > sound.shape[1]:
            sound_normalized = sound_normalized.T

    if sound.ndim > 1 and sound_normalized.shape[0] >= 2:  # multichannel input was given! Play first and last channel
        sounddevice.play(sound_normalized[(0, -1), :max_length_samples].T, fs, blocking=True)
    else:
        sounddevice.play(np.squeeze(sound_normalized)[:max_length_samples], fs, blocking=True)

def secondsToSamples(s):
    if s == -1:
        return s
    return int(s * g.fs)


def signed16bitToFloat(x_: np.array):
    assert np.max(x_) > 1
    assert np.min(x_) < -1
    x = x_.copy()
    x = x / 32768.
    return x


def kth_diag_indices(a, k=0):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def hermitian_angle_single_frequency(ref, est, tol):

    num = np.linalg.norm(ref.conj().T @ est)
    den = np.linalg.norm(ref) * np.linalg.norm(est)
    ratio = np.clip(num / den, a_min=-1, a_max=2)

    # assign maximum error if ratio outside bounds
    if ratio > 1 + tol:
        ratio = -1
    elif ratio > 1:  # do not penalize if marginally higher than 1
        ratio = 1

    return np.arccos(ratio)


def hermitian_angle(reference, estimate, tol=1e-2, mean=True):
    num_mics, num_freqs = reference.shape
    assert num_freqs >= 1
    assert num_mics >= 2

    err = [hermitian_angle_single_frequency(reference[:, kk], estimate[:, kk], tol=tol) for kk in range(num_freqs)]
    err = np.array(err)
    if mean:
        err = np.mean(err)
    return err


def calculate_error_all_mics_all_freqs(error_function):
    def impl(*args):
        inputs = [*args]
        for idx, input_ in enumerate(inputs):
            if input_.ndim <= 1 or input_.ndim > 3:
                raise ValueError(f"reference_ should be 2D or 3D, but has shape {input_.shape}")
            elif input_.ndim == 2:
                inputs[idx] = input_[..., np.newaxis]

        reference_ = inputs[0]
        estimate_ = inputs[1]
        num_mics, num_freqs, num_frames = reference_.shape

        errors = np.zeros((num_mics, num_freqs, num_frames), dtype=float)
        for mm in range(num_mics):
            for kk in range(num_freqs):
                for ll in range(num_frames):
                    idx = (mm, kk, ll)
                    errors[idx] = error_function(reference_[idx], estimate_[idx])

        return errors

    return impl


@calculate_error_all_mics_all_freqs
def MSE_normalized_dB(x, x_hat):
    return MSE_normalized_dB_single_measurement(x, x_hat)


def MSE_normalized_dB_single_measurement(x, x_hat):
    return np.clip(10 * np.log10(g.log_pow_threshold + squared_euclidean_norm(x - x_hat) / squared_euclidean_norm(x)),
                   a_min=g.mse_db_min_error,
                   a_max=g.mse_db_max_error)
    # return np.clip(log_pow((np.linalg.norm(x - x_hat)) / (g.eps + np.linalg.norm(x))), a_min=g.mse_db_min_error,
    #                a_max=g.mse_db_max_error)


@calculate_error_all_mics_all_freqs
def MSE_not_normalized_dB(x, x_hat):
    return MSE_not_normalized_dB_single_measurement(x, x_hat)


def MSE_not_normalized_dB_single_measurement(x, x_hat):
    return np.clip(10 * np.log10(g.log_pow_threshold + squared_euclidean_norm(x - x_hat)),
                   a_min=g.mse_db_min_error,
                   a_max=g.mse_db_max_error)


# @calculate_error_all_mics_all_freqs
# def MAE_normalized_dB(x, x_hat):
#     return log_pow((np.linalg.norm(x - x_hat)) / (g.eps + np.linalg.norm(x)))


def MSE(x, x_hat, mean=True):
    err = MSE_single(x, x_hat)
    if mean:
        err = np.mean(err)
    return err


@calculate_error_all_mics_all_freqs
def return_real_part_second_argument(x, x_hat):
    return np.real(x_hat)


def MSE_single(x, x_hat=0):
    # return np.abs(x - x_hat) ** 2
    y = x - x_hat
    return np.real(y) ** 2 + np.imag(y) ** 2


def log_pow(x, thr=g.log_pow_threshold):
    return 2 * linear_to_db(np.abs(x))


def pad_or_trim_to_len(x_original, target_length):
    x_new = np.array(x_original, ndmin=2, copy=True)
    num_channels, num_samples = x_new.shape[-2:]

    if target_length > num_samples:  # pad
        x_new = np.zeros((num_channels, target_length), dtype=x_original.dtype)
        x_new[..., :num_samples] = x_original
    elif target_length < num_samples:
        x_new = x_new[..., :target_length]

    return x_new


def stft(x, nstft, noverlap=None, window=None):
    warnings.warn("Deprecated. Use stft_object from SignalGenerator instead", DeprecationWarning)

    # noverlap should be absolute, not percentage! E.g. 256 for nstft 512
    assert not 0 < noverlap <= 1
    if window is not None and isinstance(window, str) and \
            (g.window_function_data is None or len(g.window_function_data) != nstft):
        window = scipy.signal.windows.get_window(window, nstft, fftbins=True)
        window = window / np.linalg.norm(window)  # normalize to unit energy
        # should be checked before sqrt(), because window is applied twice
        if not scipy.signal.check_COLA(window, nstft, int(noverlap)):
            raise ValueError('COLA violated')
        window = np.sqrt(window)
        g.window_function_data = window
    else:
        window = g.window_function_data
    return scipy.signal.stft(x, fs=g.fs, window=window, nperseg=nstft, nfft=nstft,
                             noverlap=noverlap)
    # , scaling='psd', boundary='zeros')


def istft(x_stft, nstft, noverlap=None, window=None):
    warnings.warn("Deprecated. Use stft_object from SignalGenerator instead", DeprecationWarning)

    # noverlap should be absolute, not percentage! E.g. 256 for nstft 512
    assert not 0 < noverlap <= 1
    if window is not None:
        window = scipy.signal.windows.get_window(window, nstft)
        window = np.sqrt(window)
    # noinspection PyTypeChecker
    return scipy.signal.istft(x_stft, fs=g.fs, window=window, nperseg=nstft, nfft=nstft,
                              noverlap=noverlap)
    # , scaling='psd', boundary='zeros')


def resample(x: np.array, current_fs, desired_fs=g.fs):
    if current_fs == desired_fs:
        return x
    if x.ndim > 1 and x.shape[0] > x.shape[1]:
        raise ValueError("Dimensions should be (num_channels, num_samples)")
    secs = x.shape[-1] / current_fs  # Number of seconds in signal X
    samps = int(secs * desired_fs)  # Number of samples to downsample
    return scipy.signal.resample(x, samps, axis=-1)


def plot(x, ax=None, titles='', fs=16000, time_axis=True, show=True):
    """For one or multiple 1-D plots, i.e. for time-domain plots."""

    if isinstance(x, list) and len(x) == 1:
        x = x[0]
        if isinstance(titles, list):
            titles = titles[0]

    if isinstance(x, list):
        num_plots = len(x)

        fig_opt = dict(figsize=(5, 1.5 + num_plots * 1.5), constrained_layout=True)
        fig, axes = plt.subplots(num_plots, 1, sharey='all', sharex='all', **fig_opt)

        for ax, audio_sample, title in zip_longest(axes, x, titles):
            plot(audio_sample, ax, fs=fs, time_axis=time_axis)
            ax.set_ylabel("Amplitude")

            if time_axis:
                ax.set_xlabel("Time [s]")
                x_locs, _ = ax.get_xticks(), ax.get_xticklabels()
                labels = np.linspace(0, audio_sample.shape[-1] / fs, len(x_locs), dtype=float)
                labels_str = [f"{x:.1f}" for x in labels]
                ax.set_xticks(x_locs[1:-1], labels_str[1:-1])

            ax.set_title(title)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            ax.label_outer()
            ax.grid(True)

        if show:
            fig.show()
        return fig

    else:  # not a list but a single ndarray
        is_subplot = ax is not None
        if is_subplot:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(titles)

    if x.ndim == 2 and x.shape[1] > x.shape[0]:
        ax.plot(x.T)
    else:
        ax.plot(x)

    ax.grid(True)

    if not is_subplot and show:
        plt.show()
        plt.pause(0.05)

    return fig


def plot_matrix(x_input, ax=None, frame_index=13, title='', log=True, xy_label=None, amp_range=(None, None),
                stft_shape=None, show_colorbar=True, freq_range_hz=(None, None), font_size='x-large', **kwargs):
    """ Plot the magnitude of a matrix, e.g. a STFT matrix.
    amp_range: if -1, determine automatically
    """

    x = copy.deepcopy(x_input)
    x = np.squeeze(x)
    x_ax_label = None
    is_subplot = ax is not None
    if is_subplot:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    one_mic_and_frequency_labels = x.ndim > 1 and stft_shape is not None
    if one_mic_and_frequency_labels:
        num_mics, num_freqs, _ = stft_shape
        # assert False  # correct reshaping is probably (num_mics, num_freqs, num_retained_eigva)! Check
        x = x.reshape((num_freqs, num_mics, num_freqs, num_mics))
        if freq_range_hz != (None, None):
            # Convert range from Hz to bin and select only those bins for the plot
            freq_range_bin = [int(x * num_freqs // (g.fs / 2)) for x in freq_range_hz]
            x = x[freq_range_bin[0]:freq_range_bin[1], :, freq_range_bin[0]:freq_range_bin[1], :]
            num_freqs = freq_range_bin[1] - freq_range_bin[0]

        x = np.squeeze(x[:, 0, :, 0])
        x = x / np.max(abs(x))

        if freq_range_hz == (None, None):
            x_ax_label = np.linspace(0, g.fs // 2, num_freqs, endpoint=True, dtype=int)
        else:
            # noinspection PyTypeChecker
            x_ax_label = np.linspace(freq_range_hz[0], freq_range_hz[1], num_freqs, endpoint=True, dtype=int)

    transform = log_pow if log else np.abs
    if isinstance(amp_range, tuple) and amp_range != (None, None):
        options = dict(vmin=amp_range[0], vmax=amp_range[1])
    elif np.isscalar(amp_range):
        options = dict(vmin=transform(np.min(x)), vmax=transform(np.max(x)))
    else:
        options = dict(vmin=-120, vmax=0)

    if x.ndim == 3:
        tf = min(frame_index, x.shape[-1] - 1)
        z = transform(x[..., tf])
        pcm_mag = ax.matshow(z, **options)
    elif x.ndim == 2:
        tf = -1
        z = transform(x)
        if x_ax_label is not None:
            pcm_mag = ax.matshow(x_ax_label, x_ax_label, z, **options)
        else:
            pcm_mag = ax.matshow(z, **options)
    elif x.ndim == 1:
        tf = -1
        pcm_mag = ax.plot(transform(x))
    else:
        raise ValueError(f"Matrix has {x.ndim} dimensions and cannot be plotted")

    if title == '':
        title = f"Magnitude of bifrequency spectrum"
        if tf != -1:
            title += f" at time-frame {tf}"

    ax.set_title(title, fontsize=font_size)
    if show_colorbar and x.ndim != 1:
        cl = fig.colorbar(pcm_mag, ax=ax, shrink=kwargs.get('colorbar_shrink', 1.0))
        label = 'Magnitude [dBm]' if log else 'Magnitude'
        cl.set_label(label, size=kwargs.get('colorbar_font_size', font_size))
        cl.ax.tick_params(labelsize=kwargs.get('colorbar_font_size', font_size))

    if one_mic_and_frequency_labels:
        x_locs, _ = ax.get_xticks(), ax.get_xticklabels()
        # labels = np.linspace(0, g.fs // 2, len(x_locs), dtype=int)
        labels_str = [f"{x / 1000:.1f}" for x in x_locs]  # Show kHz
        # labels_str = [f"{x / 1000:.0f}" for x in labels]  # Show kHz
        # labels_str = [f"{x:.0f}" for x in labels]  # Show Hz
        ax.set_xticks(x_locs[1:-1], labels_str[1:-1])
        ax.set_yticks(x_locs[1:-1], labels_str[1:-1])
        # ax.set_yticks(x_locs, labels_str)
        # ax.set_xticks(x_locs, labels_str)

    # ax.invert_yaxis()
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    # ax.xaxis.set_ticks_position('both')
    # ax.xaxis.set_label_position('top')

    if xy_label is None:
        if one_mic_and_frequency_labels:
            xy_label = 'Frequency [kHz]'
        else:
            xy_label = 'Frequency bin * microphone index'
        ax.set_ylabel(xy_label, fontsize=font_size)
        ax.set_xlabel(xy_label, fontsize=font_size)

    elif not isinstance(xy_label, str):
        ax.set_ylabel(xy_label[0], fontsize=font_size)
        ax.set_xlabel(xy_label[1], fontsize=font_size)

    # Grid lines between pixels, not at pixel centers
    if kwargs.get('grid', False):
        ax.set_xticks(np.arange(-.5, x.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, x.shape[1], 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    if not is_subplot:
        fig.show()

    return fig, pcm_mag


def plot_dict(named_samples_dict):
    """ Plot a dictionary of named samples """
    samples = list(named_samples_dict.values())
    samples = np.array([np.squeeze(x) for x in samples])
    plot(samples, titles=named_samples_dict.keys())


def db_to_linear(self_noise_db_power):
    """ Convert dBm to linear power """
    return 10.0 ** (self_noise_db_power / 10.0)


def linear_to_db(power_linear):
    """" Convert linear power to dBm """
    return 10 * np.log10(power_linear + g.eps)


# numpy array to MP3
def write_mp3(x_numpy_float, target_path, fs_):
    return write_audio_file(x_numpy_float, target_path, fs_, 'mp3')


# numpy array to file
def write_audio_file(x_numpy_float, target_path, fs_, extension):
    assert x_numpy_float.ndim == 2
    assert (np.all(np.abs(x_numpy_float) <= 1))
    assert x_numpy_float.shape[0] > x_numpy_float.shape[1]  # likely, samples will be more than channels

    channels = x_numpy_float.shape[1]
    y = np.int16(x_numpy_float * 2 ** 15)  # u.normalized array - each item should be a float in [-1, 1)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=fs_, sample_width=2, channels=channels)

    if extension == 'mp3':
        out = song.export(target_path, format="mp3", bitrate="320k")
    else:
        out = song.export(target_path, format="wav")
    out.close()


def herm(A: np.array):
    return A.conj().T


def set_printoptions_numpy():
    """ Set numpy print options to make it easier to read. Also set pprint as default for dict() """
    desired_width = 220
    np.set_printoptions(precision=3, linewidth=desired_width, suppress=True)

    # use pprint by default for dict()
    sys.displayhook = lambda x: exec(['_=x; pprint(x)', 'pass'][x is None])

    # make warnings more readable
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line


def is_positiveDefinite(A):
    return np.all(0 < np.linalg.eigvals(A))


def is_positiveSemiDefinite(A, tol=0):
    return np.all(np.linalg.eigvals(A) >= -tol)


def is_hermitian(A):
    return np.allclose(A, A.conj().T)


def is_symmetric(A):
    return np.allclose(A, A.T)


@njit(cache=True)
def ForceToZeroOffBlockDiagonal(m_, block_size, max_distance_diagonal=-1):
    if max_distance_diagonal == -1:
        return m_

    m = m_.copy()
    w, h = m.shape[:2]
    for ii in np.arange(w, step=block_size):
        for jj in np.arange(h, step=block_size):
            element_close_enough_to_diagonal = abs(ii - jj) <= max_distance_diagonal * block_size
            if not element_close_enough_to_diagonal:
                m[ii:ii + block_size, jj:jj + block_size, ...] = 0

    return m


def get_by_path(dictionary, paths):
    """
    Get a value in a nested object in root by item sequence.
    https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    """
    return reduce(operator.getitem, paths, dictionary)


def set_by_path(dictionary, paths, new_value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(dictionary, paths[:-1])[paths[-1]] = new_value


# def show_graph_with_labels(adjacency_matrix, mylabels=None):
#     rows, cols = np.where(adjacency_matrix == 1)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
#     plt.show()
#     return gr


# a = np.abs(xx1) > 1e-6
# a = a[:,:,0]
# gr = u.show_graph_with_labels(a)
# aa = list(networkx.find_cliques(gr))
# c1 = networkx.shortest_path(gr, 0).keys()
#
# a = np.abs(xx1) > 1e-6
# a[np.diag_indices_from(a)] = 0
# gr = u.show_graph_with_labels(a)
# c = list(networkx.shortest_path(gr, 0).keys())
# c.sort()


def save_figure(figure, user_file_name='', dpi='figure', transparent=False, out_dir_name='', extension='pdf', **kwargs):
    # day_string, day_time_string = get_day_time_strings()
    file_name = make_complete_file_name(out_dir_name=out_dir_name, user_file_name=user_file_name, **kwargs)

    # add extension if not already there
    if not file_name.endswith(extension):
        file_name = file_name + '.' + extension

    figure.savefig(file_name, dpi=dpi, transparent=transparent, bbox_inches='tight',
                   facecolor=figure.get_facecolor(), edgecolor=figure.get_edgecolor())

    print(f"Figure saved as {file_name}")

    return out_dir_name


def save_array(x: np.array, user_file_name='', out_dir_name='', **kwargs):
    # out_dir_name = create_save_folders(day_string, save_to_parent_dir=save_to_parent_dir)
    day_string, day_time_string = get_day_time_strings()
    file_name_complete = make_complete_file_name(day_time_string, out_dir_name, user_file_name, **kwargs)

    # save with pickle
    with open(file_name_complete, 'wb') as f:
        pickle.dump(x, f)

    print(f"Numpy array saved as {file_name_complete}")

    # return out_dir_name


def make_complete_file_name(day_time_string='', out_dir_name='', user_file_name='', extension='', **kwargs):
    if kwargs.get('omit_time_from_file_name', False) and user_file_name != '':
        user_file_name = os.path.join(out_dir_name, f"{user_file_name}{extension}")
    elif kwargs.get('append_time', False) and user_file_name != '':
        user_file_name = os.path.join(out_dir_name, f"{user_file_name}_{day_time_string}{extension}")
    elif user_file_name != '':
        if day_time_string != '':
            user_file_name = os.path.join(out_dir_name, f"{day_time_string}_{user_file_name}{extension}")
        else:
            user_file_name = os.path.join(out_dir_name, f"{user_file_name}{extension}")
    else:
        user_file_name = os.path.join(out_dir_name, f"{day_time_string}{extension}")
    return user_file_name


def create_output_folder(parent_dir_name='', child_dir_name='', out_dir_name=None):
    if out_dir_name is None:
        day_string, day_time_string = get_day_time_strings()
        out_dir_name = os.path.join(g.out_dir_experiments, parent_dir_name, '-'.join([child_dir_name, day_time_string]))

    # create folder parent_dir_name if it doesn't exist (recursive)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    return out_dir_name


def make_folders_errors_figures_settings(out_dir_name, use_tex_figures=False):
    array_dir = check_create_folder(out_dir_name, 'arrays')
    figures_dir = check_create_folder(out_dir_name, 'figures_tex' if use_tex_figures else 'figures')
    settings_dir = check_create_folder(out_dir_name, 'settings')
    return array_dir, figures_dir, settings_dir


def check_create_folder(parent_folder, folder_name):
    child_folder = os.path.join(parent_folder, folder_name)
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)
    return child_folder


def get_day_time_strings():
    day_string = datetime.now().strftime("%Y-%m-%d")
    time_string = datetime.now().strftime('%H-%M-%S')
    day_time_string = '--'.join([day_string, time_string])
    return day_string, day_time_string


# Clips real and imag part of complex number independently
def clip_cpx(x, a_min, a_max):
    if np.iscomplexobj(x):
        x = np.clip(x.real, a_min, a_max) + 1j * np.clip(x.imag, a_min, a_max)
    else:
        x = np.clip(x, a_min, a_max)
    return x


def circular_gaussian(shape_):
    # call generator only once, then use half vector for real part and half for imaginary
    rnd = g.rng.standard_normal(((2,) + shape_))
    return (rnd[0] + rnd[1] * 1j) / np.sqrt(2)
    # return (g.rng.standard_normal(shape_) + g.rng.standard_normal(shape_) * 1j) / np.sqrt(2)


def squared_euclidean_norm(y):
    return np.real(y.flatten() @ y.flatten().conj())


def is_crb(estimator_name):
    return 'crlb' in estimator_name.lower() or 'crb' in estimator_name.lower() or 'bound' in estimator_name.lower()


def set_plot_options(use_tex=False):
    plt.style.use('seaborn-v0_8-paper')

    if not use_tex:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    else:
        """
        This might be interesting at some point
        https://github.com/Python4AstronomersAndParticlePhysicists/PythonWorkshop-ICE/tree/master/examples/use_system_latex
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams["axes.formatter.use_mathtext"] = True
        font = {'family': 'serif',
                'size': 10,
                'serif': 'cmr10'
                }
        plt.rc('font', **font)


def is_cw(algo_name):
    return ('narrowband' in algo_name.lower() or 'cw' in algo_name.lower()) and not is_cw_sv(algo_name)


def is_cw_sv(algo_name):
    return ('cw' in algo_name.lower() and 'sv' in algo_name.lower()) or \
        'prop' in algo_name.lower()


def save_settings(settings_figure, user_file_name, out_dir_name, **param):
    # very similar to save_array, but settings_figure is a list of dictionaries
    day_string, day_time_string = get_day_time_strings()
    file_name_complete = make_complete_file_name(day_time_string, out_dir_name, user_file_name, **param)

    # save with pickle
    with open(file_name_complete, 'wb') as f:
        pickle.dump(settings_figure, f)

    print(f"Settings saved as {file_name_complete}")

    return out_dir_name


def find_largest_sequence(signal):
    """ Find the largest sequence of True values in a NumPy array of boolean values"""
    max_sequence_length = 0
    current_sequence_length = 0
    max_sequence_start = 0

    for i, bit in enumerate(signal):
        if bit:
            current_sequence_length += 1
            if current_sequence_length > max_sequence_length:
                max_sequence_length = current_sequence_length
                max_sequence_start = i - current_sequence_length + 1
        else:
            current_sequence_length = 0

    mask = np.zeros(len(signal), dtype=bool)
    mask[max_sequence_start:max_sequence_start + max_sequence_length] = True

    return mask

    # Example signal as a NumPy array of boolean values
    # signal = np.array(
    #     [False, False, True, True, False, False, True, True, True, False, True, True, True, True, False, False, True, True,
    #      True, True, True, False], dtype=bool)
    # mask = find_largest_sequence(signal)
    # print("Signal:", signal)
    # print("Mask:  ", mask)


def get_frequency_bins_from_frequencies(freqs_hz, nstft=None, list_all_freqs_hz=None):

    freqs_hz = np.array(freqs_hz if not np.isscalar(freqs_hz) else [freqs_hz])

    if list_all_freqs_hz is None:
        if nstft is None:
            raise ValueError("Either nstft or list_all_freqs_hz should be given")
        else:
            list_all_freqs_hz = np.fft.rfftfreq(nstft, 1 / g.fs)

    if np.all(np.isfinite(freqs_hz)) and np.all(freqs_hz >= 0):
        freqs_bins = np.argmin(np.abs(list_all_freqs_hz - freqs_hz[:, np.newaxis]), axis=1)
        freqs_bins = np.unique(freqs_bins)
        return freqs_bins if len(freqs_bins) > 1 else freqs_bins[0]
    else:
        return -1


def compute_mask_frequency_range(freq_range=(0, 30000), nstft=None):
    """ Compute a mask for the frequency range of interest. Only the frequencies in the range are kept."""

    num_freqs_real = nstft // 2 + 1
    freq_mask = np.ones(num_freqs_real).astype(bool)
    bin_min = get_frequency_bins_from_frequencies(freq_range[0], nstft)
    bin_max = get_frequency_bins_from_frequencies(freq_range[1], nstft)
    bin_max = bin_max if bin_max > 1 else get_frequency_bins_from_frequencies(g.fs // 2, nstft) // 2 + 1
    assert bin_max > bin_min >= 0
    freq_mask[:bin_min] = False
    freq_mask[bin_max + 1:] = False

    return freq_mask


def cholesky_or_svd(r_true):
    try:
        # Introduce correlations between variables by multiplying the white data with the square root of the true noise covariance
        rv_true_sqrt = np.linalg.cholesky(r_true)
    except np.linalg.LinAlgError:
        # Alternative procedure to avoid Cholesky decomposition, which is not defined for singular matrices.
        # (https://stats.stackexchange.com/questions/159313/generating-samples-from-singular-gaussian-distribution)
        warnings.warn("Cholesky decomposition failed. Generating correlated signal using SVD instead.")
        U, d, _ = np.linalg.svd(r_true, hermitian=True)
        assert np.all(
            d >= 0), "Singular values of covariance matrix are negative. Thus this not a valid covariance matrix."
        rv_true_sqrt = U @ np.diag(np.sqrt(d))

    return rv_true_sqrt


def smoothen_corners(s, alpha=0.5):
    # Apply Tukey window to smoothen corners. Higher alpha in Tukey window leads to more smoothing.
    # 0 is rectangular window, 1 is Hann window.

    y = copy.deepcopy(s)
    win_len = min(500, s.shape[-1] // 4)
    win = scipy.signal.windows.tukey(win_len, alpha=alpha)
    y[:win_len // 2] = s[:win_len // 2] * win[:win_len // 2]
    y[-win_len // 2:] = s[-win_len // 2:] * win[-win_len // 2:]
    return y
