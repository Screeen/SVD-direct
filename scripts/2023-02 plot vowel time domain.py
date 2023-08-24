import scipy
import src.utils as u, global_constants as g
from pathlib import Path
from src.utils import plot as plot
import numpy as np
import matplotlib.pyplot as plt

u.set_printoptions_numpy()

out_dir_name = Path(__file__).parent.parent / "Datasets"
file_name = out_dir_name / "long_a.wav"
# file_name = out_dir_name / "Anechoic" / "SI Harvard Word Lists Male_16khz.wav"

fs, dry_samples = scipy.io.wavfile.read(file_name)
dry_samples = u.signed16bitToFloat(dry_samples).T
dry_samples = u.resample(dry_samples, g.fs)
dry = u.normalize_volume(dry_samples)
# dry = dry[:, 33000:60000]  # first sentence
# dry = dry[:, 33000:40000]  # first word
# dry_mono = dry[0] if dry.ndim == 2 else dry
dry_mono = dry

fig, axes = plt.subplots(nrows=1, ncols=2, sharey='all')
for ax, audio_sample in zip(axes, [dry_mono[33000:40000], dry_mono[33000:36000]]):
    plot(audio_sample, ax)
    ax.set_xlabel("Time [s]", fontsize='x-large')
    ax.set_ylabel("Amplitude", fontsize='x-large')
    x_locs, _ = ax.get_xticks(), ax.get_xticklabels()
    labels = np.linspace(0, audio_sample.shape[-1] / g.fs, len(x_locs), dtype=float)
    labels_str = [f"{x:.2f}" for x in labels]
    ax.set_xticks(x_locs[1:-1], labels_str[1:-1])
    ax.grid(True)

fig.suptitle("Man reading the word 'Harvard'", fontsize='x-large')
axes[0].set_title("Harvard", fontsize='x-large')
axes[1].set_title("Ha-", fontsize='x-large')
fig.tight_layout()
fig.show()
u.save_figure(fig, dpi=300)
