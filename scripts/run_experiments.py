import argparse
import sys
import time
import numpy as np

sys.path.append('.')
sys.path.append('..')
sys.path.append('../src')

from src.kickstarter import Kickstarter
import src.utils as u
import src.global_constants as g

start_time = time.time()

# use_multiple_processes_default = True if not g.debug_mode else False
use_multiple_processes_default = False

exp_names_speech = ['speech_num_mics', 'speech_time_seconds', 'speech_snr', 'speech_noise_position']
exp_names_synthetic = ['target_correlation', 'noise_correlation', 'time_frames', 'snr']
exp_names_all = exp_names_speech + exp_names_synthetic
exp_names_all_with_debug = exp_names_all + ['debug']

parser = argparse.ArgumentParser(description='Run RTF estimation and beamforming experiments.')
parser.add_argument('--exp_name', type=str, default='debug', help=f'Experiment name, options: {exp_names_all}')
parser.add_argument('--repeated_experiments_constant', type=float, default=1,
                    help='Number of repeated experiments, e.g. 1e6, 1e8, 1e11')
parser.add_argument('--use_multiple_processes', type=bool, default=use_multiple_processes_default,
                    help='Use multiple processes')
parser.add_argument('--target_noise_equal_variances', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

exp_name_temp = args.exp_name
repeated_experiments_constant_ = args.repeated_experiments_constant
use_multiple_processes_ = args.use_multiple_processes

if exp_name_temp == 'all':
    exp_names = exp_names_all
elif exp_name_temp == 'speech_all':
    exp_names = exp_names_speech
elif exp_name_temp == 'synthetic_all':
    exp_names = exp_names_synthetic
else:
    exp_names = [exp_name_temp]

# top level of abstraction: we can run multiple experiments at once (different subsections of the paper)
results_dict_list = []
for exp_name_ in exp_names:
    print('*' * 80)
    print(f"Running experiment: {exp_name_}")
    results_dict = Kickstarter.run_script(exp_name_,
                                          num_experiments_constant=repeated_experiments_constant_,
                                          use_multiple_processes=use_multiple_processes_,
                                          use_tex_labels_=False,
                                          target_noise_equal_variances=args.target_noise_equal_variances)
    results_dict_list.append(results_dict)
    print('*' * 80)
    print()

# Open results folder when all experiments are done
s = results_dict_list[-1]['settings'][0]
if not g.debug_mode and not s['plot_correlation_histogram'] and not len(exp_names) > 1 and g.release_save_plots:
    Kickstarter.make_sound_open_folder(s['out_dir_name'], s['exp_name'], start_time=start_time)

bbs, bb, rrs, rr = None, None, None, None


def plot_rtfs():
    r1 = rrs[which_var_value]
    nstft = settings['nstft']
    if not np.isscalar(nstft):
        nstft = nstft[which_var_value]
    freq_range_mask = exp_data_dict[list(exp_data_dict.keys())[which_var_value]].selected_bins_mask[-1]
    freq_range_hz = np.fft.rfftfreq(int(nstft), 1 / g.fs)[freq_range_mask]
    trans = np.abs
    f = u.plot([trans(r1.estimates_dict['CW'].T),
                trans(r1.estimates_dict['CW-SV'].T), trans(r1.ground_truth.T), ],
               titles=['CW', 'CW-SV', 'Ground truth'], time_axis=False, show=False)
    f.suptitle('RTF magnitude')
    # use freq_range_hz as x-axis labels. Do not show more than 10 labels
    ticks = np.linspace(0, len(freq_range_hz) - 1, 10).astype(int)
    f.axes[-1].set_xticks(ticks)
    f.axes[-1].set_xticklabels(freq_range_hz[ticks].astype(int))
    f.axes[-1].set_xlabel('Frequency [Hz]')
    f.axes[-1].set_ylim([0, 10])
    f.show()

    trans = np.angle
    f1 = u.plot([trans(r1.estimates_dict['CW'].T),
                 trans(r1.estimates_dict['CW-SV'].T), trans(r1.ground_truth.T), ],
                titles=['CW', 'CW-SV', 'Ground truth'], time_axis=False, show=False)
    f1.suptitle('RTF phase')
    f1.axes[-1].set_xticks(ticks)
    f1.axes[-1].set_xticklabels(freq_range_hz[ticks].astype(int))
    f1.axes[-1].set_ylabel('Phase [rad]')
    f1.axes[-1].set_xlabel('Frequency [Hz]')
    f1.show()


if len(results_dict_list) == 1:
    result_dict = dict(results_dict_list[0])
    num_figures = len(result_dict['rtf_errors_dict'])

    # If there is only one fig, all variables above are lists of one element. Extract it
    # Example: rtf_errors_dict = result_dict['rtf_errors_dict'] if num_figures > 1 else result_dict['rtf_errors_dict'][0]
    # The variables names can be obtained as the keys of the result_dict
    # dict_keys(['rtf_errors_dict', 'rtf_evaluators_dict', 'beamforming_errors_dict',
    # 'beamforming_evaluators_dict', 'exp_data_dict', 'settings'])
    dict_keys = list(result_dict.keys())

    for key in dict_keys:  # Extract the last element of each list (the last figure)
        result_dict[key] = result_dict[key][-1]

    rtf_errors_dict = result_dict['rtf_errors_dict']
    rtf_evaluators_dict = result_dict['rtf_evaluators_dict']
    beamforming_errors_dict = result_dict['beamforming_errors_dict']
    beamforming_evaluators_dict = result_dict['beamforming_evaluators_dict']
    exp_data_dict = result_dict['exp_data_dict']
    settings = result_dict['settings']

    bf_metrics = list(beamforming_evaluators_dict.keys())
    if bf_metrics:
        bbs = beamforming_evaluators_dict[bf_metrics[0]]
        bb = bbs[-1]
    rtf_metrics = list(rtf_evaluators_dict.keys())
    if rtf_metrics:
        rrs = rtf_evaluators_dict[rtf_metrics[0]]
        rr = rrs[-1]

    max_length_seconds = 3
    which_var_value = 0
    extended = True
    show_rtfs = False
    if show_rtfs:
        plot_rtfs()

"""
plot_manager.plot_spectrogram(bbs[which_var_value].ground_truth, suptitle='Ground truth')
plot_manager.plot_spectrogram(bbs[which_var_value].estimates_dict['Unprocessed'], suptitle='Unprocessed')
plot_manager.plot_spectrogram(bbs[which_var_value].estimates_dict['Ideal'], suptitle='Ideal')
plot_manager.plot_spectrogram(bbs[which_var_value].estimates_dict['CW'], suptitle='CW')
plot_manager.plot_spectrogram(bbs[which_var_value].estimates_dict['CW-SV'], suptitle='CW-SV')

u.play(         bbs[which_var_value].ground_truth, max_length_seconds=max_length_seconds)
u.play(bbs[which_var_value].estimates_dict['Unprocessed'], max_length_seconds=max_length_seconds)
u.play(bbs[which_var_value].estimates_dict['Ideal'], max_length_seconds=max_length_seconds)
u.play(bbs[which_var_value].estimates_dict['CW'], max_length_seconds=max_length_seconds)
u.play(bbs[which_var_value].estimates_dict['CW-SV'], max_length_seconds=max_length_seconds)

pystoi.stoi(np.squeeze(bbs[which_var_value].ground_truth), np.squeeze(bbs[which_var_value].estimates_dict['Ideal']), 16000, extended=extended)
pystoi.stoi(np.squeeze(bbs[which_var_value].ground_truth), np.squeeze(bbs[which_var_value].estimates_dict['Unprocessed']), 16000, extended=extended)
pystoi.stoi(np.squeeze(bbs[which_var_value].ground_truth), np.squeeze(bbs[which_var_value].estimates_dict['CW']), 16000, extended=extended)
pystoi.stoi(np.squeeze(bbs[which_var_value].ground_truth), np.squeeze(bbs[which_var_value].estimates_dict['CW-SV']), 16000, extended=extended)
"""
