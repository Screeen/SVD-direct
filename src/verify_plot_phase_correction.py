import numpy as np
import matplotlib.pyplot as plt
import scipy

import src.utils as u
from icecream import ic

rng = np.random.default_rng(636253)
u.set_plot_options(use_tex=True)


def generate_harmonic_signal(f0_, fs_, L_, num_harmonics_=3, frequency_error_=0.0, rnd_amplitude=False, rnd_phase=True):
    # Generate a harmonic signal with random phase and amplitude.

    freq_error = rng.uniform(-frequency_error_, frequency_error_, num_harmonics_)
    discrete_frequencies = f0_ * (np.arange(num_harmonics_) + 1 + freq_error)
    phases = rng.uniform(0, 2 * np.pi, num_harmonics_) if rnd_phase else np.zeros(num_harmonics_)
    amplitudes = rng.uniform(0.5, 1.0, num_harmonics_) if rnd_amplitude else np.ones(num_harmonics_)
    discrete_times = np.arange(L_) / fs_

    y_ = np.sum(amplitudes[:, None] *
                np.cos(2 * np.pi * discrete_frequencies[:, None] * discrete_times[np.newaxis, :] + phases[:, None]),
                axis=0)

    y_ = y_ / (1e-10 + np.max(np.abs(y_)))
    y_ -= np.mean(y_)

    return y_


def fig_to_subplot(existing_fig, title, ax, xy_ticks=(None, None), xlabel='', ylabel=''):

    if existing_fig is None or not existing_fig:
        return None

    # Retrieve the image data from the existing figure
    img = existing_fig.axes[0].collections[0].get_array().data

    # Retrieve vmin and vmax from the existing figure
    vmin, vmax = existing_fig.axes[0].collections[0].get_clim()

    # Retrieve the x ticks, y ticks, color map, and labels from the existing figure
    cmap = existing_fig.axes[0].collections[0].get_cmap()

    # Display the image data in the new subplot
    if xy_ticks != (None, None):
        im = ax.pcolormesh(*xy_ticks, img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        im = ax.pcolormesh(img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)

    if xlabel == '':
        xlabel = 'Cyclic freq.~$\\alpha_p$ [kHz]'

    if ylabel == '':
        ylabel = 'Freq.~$\\omega_k$ [kHz]'

    # Set the title of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return im


ic.configureOutput(prefix='    ')
u.set_printoptions_numpy()
u.set_plot_options(use_tex=True)

markers = ['D', '^', '*', 'x', 'o', "^", ".", "H", '>', '<', "v", '+']

fs = 16000
Nfft = 512
save = True
L = 2 * Nfft
frequency_error = 0.0

Nfft_with_padding = 1 * Nfft
num_samples = int(3 * Nfft)
fund_freq = (fs / Nfft) * 3.
fund_freq_bin = round(fund_freq * Nfft / fs)
shift_samples = Nfft // 4  # Nfft - overlap
num_harmonics = 3

# Calculate num frames considering the shift
num_frames = int((num_samples - Nfft + shift_samples) / shift_samples)
s = generate_harmonic_signal(f0_=fund_freq, fs_=fs, L_=num_samples,
                             num_harmonics_=num_harmonics, frequency_error_=frequency_error,
                             rnd_phase=False)
noise = rng.normal(size=len(s)) * 0
y = s + noise

assert num_samples % Nfft == 0  # STFT assumes that num_samples is a multiple of Nfft (no padding)

# win = np.ones(Nfft)
win = scipy.signal.windows.hann(Nfft)

# Manual STFT: first window the signal, then take (real) FFT. Output: Y
y_win = np.zeros((Nfft, num_frames), complex)
for ii in range(num_frames):
    y_win[:, ii] = y[ii * shift_samples:ii * shift_samples + Nfft] * win
Y = np.fft.rfft(y_win, axis=0, n=Nfft_with_padding)

# Phase correction, output: Yc
shifts_samples = np.arange(num_frames) * shift_samples
frequencies_hz_manual = np.arange(Nfft_with_padding) * fs / Nfft_with_padding
frequencies_hz = np.fft.rfftfreq(n=Nfft_with_padding, d=1 / fs)
phase_correction = np.exp(-2j * np.pi * frequencies_hz[:, None] * shifts_samples[None, :] / fs)
Yc = Y * phase_correction

# Inverse STFT of Yc
yc_win = np.zeros((Nfft_with_padding, num_frames), complex)
for ii in range(num_frames):
    yc_win[:, ii] = np.fft.irfft(Yc[:, ii], n=Nfft_with_padding)

# fig = u.plot(y, titles='Waveform of harmonic signal')
# ax = fig.gca()
# for shift in shifts_samples:
#     ax.axvline(shift, color='r', linestyle='--')
# fig.show()

# u.plot(list(y_win.real.T), titles=['Original windowed'], time_axis=False)
# u.plot(list(yc_win.real.T), titles=['Phase corrected windowed'], time_axis=False)

freqs_hz = np.arange(1, num_harmonics + 1) * fund_freq
freq_bins = freqs_hz * Nfft / fs
freq_bins = np.round(freq_bins).astype(int)

# Plot phase of multiple harmonics across frames
# Make two subplots, one for Y and one for Yc. In each subplot, plot phase of multiple harmonics across frames.
# Each harmonic is a different line in the plot.
x_size = 3.2
y_size = 2.8
f, axes = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(x_size, y_size), squeeze=True,
                       constrained_layout=True)
Y_or_Yc = Yc
font_size = 12
for ax, Y_ in zip([axes], [Y_or_Yc]):
    for idx, (freq_bin, freq_hz, marker) in enumerate(zip(freq_bins, freqs_hz, markers)):
        ax.plot(np.angle(Y_)[freq_bin, :],
                # label=f'{freq_hz:.0f} Hz',
                label=f"$h = {idx + 1}$",
                marker=marker, linewidth=0.2)
    ax.legend(fontsize=font_size - 2)
    ax.set_title('Phase, conventional' if Y_ is Y else
                 'Phase, phase-corrected', fontsize=font_size)
    ax.grid(True)
    ax.set_ylim([-np.pi*1.1, np.pi*1.1])
    ax.set_xlabel('Frame index', fontsize=font_size)
    ax.set_ylabel('Phase (radians)', fontsize=font_size)
    name = 'phase_original' if Y_ is Y else 'phase_phase_corrected'

f.show()
if save:
    u.save_figure(f, user_file_name=name)

"""
I observe that compensating for the phase shift leads to discontinuities in the time domain.
In fact, the length of the IFFT is short, so that the signal is "wrapped around" in the time domain.
However, if we removed the wrapping, then the phase correction would work.
Probably giving higher correlation between frames.

When plotting the phase of signals, we see that for the corrected STFT, the phase is
continuous across frames (for multiple harmonics).
"""

# Compute the spectral correlation between different frequencies
mask = np.zeros(Nfft // 2 + 1, dtype=bool)
mask[freq_bins] = True
Ym = Y[mask, :]
Ycm = Yc[mask, :]

Sc_ym = Ym @ Ym.conj().T / num_frames
Sc_ycm = Ycm @ Ycm.conj().T / num_frames

global_max = max(np.max(np.abs(Sc_ym)), np.max(np.abs(Sc_ycm)))
Sc_ym = Sc_ym / global_max
Sc_ycm = Sc_ycm / global_max

plt_cfg = {'amp_range': (-30, 0), 'xy_label': ('Harmonic idx', 'Harmonic idx'), 'font_size': 12,
           'colorbar_shrink': 0.8, 'colorbar_font_size': 8, 'grid': True}

existing_figs = []
titles = ['Spectral correlation, conventional', 'Spectral correlation, phase-corrected']
matrices_to_plot = [Sc_ym, Sc_ycm]

for ii, matrix_ii in enumerate(matrices_to_plot):
    ff, _ = u.plot_matrix(matrix_ii, title=titles[ii], **plt_cfg)
    ax = ff.gca()
    xticks_labels = [f'{int(xt + 1):.0f}' for xt in ax.get_xticks()]
    yticks_labels = [f'{int(yt + 1):.0f}' for yt in ax.get_yticks()]
    ax.set_xticklabels(xticks_labels, minor=False)
    ax.set_yticklabels(yticks_labels, minor=False)

    ff.set_size_inches(x_size, y_size)
    existing_figs.append(ff)
    ff.show()

if save:
    u.save_figure(existing_figs[0], user_file_name='spectral_correlation_original')
    u.save_figure(existing_figs[1], user_file_name='spectral_correlation_phase_corrected')

