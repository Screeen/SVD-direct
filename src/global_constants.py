from numpy.random import default_rng
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append('../src')

out_dir_experiments = '../out'
# if sys.platform == 'darwin':
#     out_dir_experiments = os.path.join("..", out_dir_experiments)

use_fixed_random_seed = True
fs = 16e3

colors = ["tab:red", "tab:brown", "tab:blue", "tab:green", "tab:orange", "tab:pink", "tab:purple", "tab:gray", "tab:cyan",
          "tab:orange", "tab:red", "tab:blue", "black", "tab:brown", "tab:pink", "tab:cyan"]
# colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:pink", "tab:purple", "tab:gray", "tab:cyan",
#           "tab:orange", "tab:red", "tab:blue", "black", "tab:brown", "tab:pink", "tab:cyan"]

markers = ['D', '*', '^', 'x', 'o', "^", ".", "H", '>', '<', "v", '+']
# markers = ['D', '^', 'x', 'o', "^", ".", "H", '>', '<', "v", '+']

# stft options
window_function_name = 'hann'
# window_function_name = 'cosine'
window_function_data = None
# window_function_name = 'rect'
idx_ref_mic = 0

# log_pow_threshold and mse_db_min_error should be modified together
log_pow_threshold = 1e-16
mse_db_min_error = -160
mse_db_max_error = 160

eps = np.finfo(float).eps

# high limits are better when evaluating CRBs, which otherwise don't correspond for low SNRs
# rtf_min = -200
# rtf_max = 200

# lower limits are more useful in practical scenarios
rtf_min = -20
rtf_max = 20

alpha_cov = 0.95  # rolling average smoothing factor

white_noise_floor_db = 40  # dB
max_relative_difference_loud_bins_default = 35  # db. If this number is higher, more bins are included in evaluation of MSE

noise_estimation_time = 2  # seconds. if -1, use the same length as the signal, but a different realization
# noise_estimation_time = -1  # seconds. if -1, use the same length as the signal, but a different realization

if use_fixed_random_seed:
    print("WARNING! DEFAULT RANDOM SEED, experiments will give same result over and over.")
    # rng = default_rng(4456)  # experiments oldenburg june 2023
    # rng = default_rng(4471)  # experiments tsp 2023
    # rng = default_rng(6361)  # experiments tsp 2023 (november 2023)
    rng = default_rng(10004)  # experiments tsp 2023 (november 2023)
else:
    rng = default_rng()

if sys.gettrace() is None:
    debug_mode = False
else:
    debug_mode = True

debug_show_plots = True
debug_save = False
release_save_plots = True
sleeping_time_figure_saving = 0

diagonal_loading = 1e-8  # amount of diagonal loading when adding identity matrix to covariance matrix

dataset_folder = Path(__file__).parent.parent / 'datasets'
