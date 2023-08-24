import scipy
from src import cov_manager, utils as u, global_constants as g
from pathlib import Path
from src.utils import plot as plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

u.set_printoptions_numpy()

# plt.style.use('seaborn-paper')
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['mathtext.fontset'] = 'stix'
u.set_plot_options()
plt.rcParams['text.usetex'] = True

out_dir_name = Path(__file__).parent.parent / "Datasets"
file_name = out_dir_name / "long_a.wav"
# file_name = out_dir_name / "Anechoic" / "SI Harvard Word Lists Male_16khz.wav"

fs, dry_samples = scipy.io.wavfile.read(file_name)
dry_samples = u.signed16bitToFloat(dry_samples).T
dry = u.resample(dry_samples, fs, g.fs)
dry = dry[0] if dry.ndim == 2 else dry
# dry = dry[33000:34500]
# dry = dry[15000:18000]
dry = dry[15000:25000]
dry = u.normalize_volume(dry)

# dry = g.rng.normal(size=dry.shape)


nstft = 1024
noverlap_percentage = 0.5
_, _, dry_stft = u.stft(dry, nstft=nstft, noverlap=noverlap_percentage * nstft, window='hann')
min_num_frames = dry_stft.shape[-1]

"""The trick here is to create a regular grid but not to have a plot in every cell. 
For example, in this case, I created a 2x4-cell grid. Each plot spans 2 cells. 
So on the first row, I have 2 plots (2x2 cells). 
On the second row, I have one empty cell, 1 plot (1x2 cells) and another empty cell.
"""

fig_opt = dict(figsize=(6., 2.5))
fig = plt.figure(**fig_opt)
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])
ax3.set_aspect('equal')
axes = np.array([ax1, ax2, ax3])
font_size = 'medium'
ticks_font_size = 'x-small'

# fig_opt = dict(figsize=(6., 5.2))
# fig = plt.figure(**fig_opt)
# gs = GridSpec(2, 4)
# ax1 = fig.add_subplot(gs[0, :2])  # First row, first column
# ax2 = fig.add_subplot(gs[0, 2:])
# ax3 = fig.add_subplot(gs[1, 1:3])
# axes = np.array([ax1, ax2, ax3])
# font_size = 'medium'

audio_sample = dry[:dry.shape[0] // 8]
plot(audio_sample, ax1)
# font_size = 'x-large'
ax1.set_xlabel("Time [s]", fontsize=font_size)
ax1.set_ylabel(r'$x(t)$', fontsize=font_size)
# ax1.set_ylabel("Amplitude", fontsize=font_size)
x_locs, _ = ax1.get_xticks(), ax1.get_xticklabels()
labels = np.linspace(0, audio_sample.shape[-1] / g.fs, len(x_locs), dtype=float)
labels_str = [f"{x:.2f}" for x in labels]
ax1.set_xticks(x_locs[1:-1], labels_str[1:-1])
ax1.grid(True)

# hide y ticks labels
ax1.set_yticklabels([])

# ax1.set_title("Time-domain", fontsize='x-large')
for line in ax1.lines:
    line.set_linewidth(0.75)

f, psd = scipy.signal.welch(dry, g.fs, nperseg=nstft)
f = f[f < 4000]
psd = psd[:len(f)]
ax2.semilogy(f, psd)
ax2.set_ylim([1e-12, 1e-3])
ax2.set_xlabel('Frequency [kHz]', fontsize=font_size)
ax2.set_ylabel(r'$|X(f)|^2$', fontsize=font_size)
# ax2.set_ylabel(r'PSD [$V^2/\mathrm{Hz}$]', fontsize=font_size)
ax2.grid(True)
x_locs, _ = ax2.get_xticks(), ax2.get_xticklabels()
labels_str = [f"{x / 1000:.1f}" for x in x_locs]  # Show kHz
ax2.set_xticks(x_locs[1:-1], labels_str[1:-1])
ax2.set_title("PSD", fontsize=font_size)
for line in ax2.lines:
    line.set_linewidth(0.75)

ax2.set_yticklabels([])

sh = (1, dry_stft.shape[0], min_num_frames)
correction_term = \
    cov_manager.CovarianceManager.compute_correction_term(nstft, noverlap_percentage * nstft, sh)

d = dry_stft
c = correction_term
b, _ = cov_manager.CovarianceManager.estimate_cpsd_bifreq(d,
                                                          warning_level=None,
                                                          correction_term=c)

_, im = u.plot_matrix(b, ax3, show_colorbar=True, log=True,
                      stft_shape=dry_stft[None].shape, freq_range_hz=(0, 4000), font_size=font_size)

for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=ticks_font_size)
    ax.set_title('')

# fig.get_axes()[0].set_title(r'$x(t)$', fontsize=font_size)
# fig.get_axes()[1].set_title(r'$|X(f)|^2$', fontsize=font_size)
fig.get_axes()[2].set_title(r'$|X(f_1)X(f_2)^*|^2$', fontsize=font_size)

fig.patch.set_facecolor('white')
fig.patch.set_alpha(0.3)
for ax in fig.get_axes():
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.7)

fig.tight_layout()
fig.show()
_, user_file_name = u.get_day_time_strings()
u.save_figure(fig, user_file_name=user_file_name, out_dir_name='out')
