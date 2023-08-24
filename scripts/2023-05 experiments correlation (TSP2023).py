import argparse
import sys

from src.tsp2023 import TspScript

sys.path.append('..')
sys.path.append('../src')


if __name__ == '__main__':

    # use_multiple_processes_default = True if not g.debug_mode else False
    use_multiple_processes_default = False

    exp_names_all = ['target_correlation', 'noise_correlation', 'time_frames',
                     'snr', 'speech_time_frames', 'speech_snr', 'speech_nstft']
    exp_names_all_with_debug = exp_names_all + ['debug']

    parser = argparse.ArgumentParser(description='Run experiments for the TSP paper')
    parser.add_argument('--exp_name', type=str, default='debug', help=f'Experiment name, options: {exp_names_all}')
    parser.add_argument('--repeated_experiments_constant', type=float, default=1, help='Number of repeated experiments, e.g. 1e6, 1e8, 1e11')
    parser.add_argument('--use_multiple_processes', type=bool, default=use_multiple_processes_default, help='Use multiple processes')
    parser.add_argument('--target_noise_equal_variances', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    exp_name_temp = args.exp_name
    repeated_experiments_constant_ = args.repeated_experiments_constant
    use_multiple_processes_ = args.use_multiple_processes

    if exp_name_temp == 'all':
        exp_names = exp_names_all
    elif exp_name_temp not in exp_names_all_with_debug:
        raise ValueError(f"Unknown experiment name: {exp_name_temp}. Choose from {exp_names_all_with_debug}")
    else:
        exp_names = [exp_name_temp]

    results_dict_list = []
    for exp_name_ in exp_names:
        print(f"Running experiment: {exp_name_}")
        results_dict = TspScript.run_script(exp_name_,
                                             repeated_experiments_constant=repeated_experiments_constant_,
                                             use_multiple_processes=use_multiple_processes_,
                                             use_tex_labels_=False,
                                             target_noise_equal_variances=args.target_noise_equal_variances)
        results_dict_list.append(results_dict)

    if len(results_dict_list) == 1:
        # extract variables of dictionary to make it easier to work with
        # dict_keys(['err_mean_std_db_array', 'rtf_evaluators', 'cm', 'sh', 'atf_target', 'variances', 'err_mean_std_db_array_herm', 'settings'])
        rtf_evaluators = results_dict_list[0]['rtf_evaluators'][0]
        err_mean_std_db_array = results_dict_list[0]['err_mean_std_db_array']
        err_mean_std_db_array_herm = results_dict_list[0]['err_mean_std_db_array_herm']
        variances = results_dict_list[0]['variances']
        settings = results_dict_list[0]['settings']
        cm = results_dict_list[0]['cm']
        cov_manager = cm
        sh = results_dict_list[0]['sh']
        atf_target = results_dict_list[0]['atf_target']

        # TspScript.plot_errors([err_mean_std_db_array], [settings])[0].show()
        # TspScript.plot_errors([err_mean_std_db_array_herm], [settings])[0].show()





