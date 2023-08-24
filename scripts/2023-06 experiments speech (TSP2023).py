import copy

import global_constants
import utils
from src.exp_manager import ExperimentManager
import src.plot_manager as plot_manager

atf = None
global_constants.rtf_max = 15
global_constants.rtf_min = -15

settings = {
    'exp_name': 'Experiments speech (TSP2023)',
    'algo_names': ['CW-SV', 'CW'],
    'num_mics_max':  2,
    'nstft': [1024],
    'varying_factors': ['noises_info', 0, 'snr'],
    # 'varying_factors': ['duration_output_frames'],
    'num_repeated_experiments': 10,
    'avg_time_frames': True,
    'metric_names': ['RMSE dB', 'Hermitian angle'],
    'noverlap_percentage': 0.5,
    'gen_signals_freq_domain': False,
    'flag_scree_method': False,
    'noises_info': [
        {'names': ['white'], 'snr': [-5, 0, 10, 20], 'isDirectional': False},
        {'names': ['white'], 'snr': [40]},
    ],

    'add_identity_noise_noisy': True,
    'use_true_covariance': False,
    'perc_active_target_freq': 1.0,

    # 'rtf_type': 'random-small-once',
    'rtf_type': 'real',
    # 'num_nonzero_samples_rir': 512,
    # 'desired': ['vowel'],
    'desired': ['female'],
    # 'desired': ['male'],
    'duration_output_frames': 500,
    # "duration_output_frames": [100, 1000, 5000],
    'max_relative_difference_loud_bins': 35,
}

error_mse, rtf_evaluators, _, _, atf, _, error_ha = ExperimentManager.run_experiment(settings, atf_target=atf)
settings_mse, settings_hermitian_angle = copy.deepcopy(settings), copy.deepcopy(settings)
settings_mse['metric_names'] = ['RMSE dB']
settings_hermitian_angle['metric_names'] = ['Hermitian angle']

f = []
for err, sett in zip([error_mse, error_ha], [settings_mse, settings_hermitian_angle]):
    f.append(plot_manager.plot_errors(settings=sett, err_mean_std_array=err, title='Real speech in WGN'))

_, day_time_strings = utils.get_day_time_strings()
utils.save_figure(f[0], user_file_name=f"Real speech MSE {settings['exp_name']}", day_time_string=day_time_strings)
out_dir_name = utils.save_figure(f[1], user_file_name=f"Real speech Hermitian angle {settings['exp_name']}", day_time_string=day_time_strings)

print(f"{out_dir_name=}")
print(f"End of experiment {settings['exp_name']}")
