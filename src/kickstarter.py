import copy
import subprocess
import sys
import time
import warnings
from itertools import repeat, zip_longest
from multiprocessing import cpu_count, Pool

sys.path.append('..')
sys.path.append('../src')

from src.beamforming_manager import BeamformingManager
import src.utils as u
from src.error_evaluator import ErrorEvaluator
from src.exp_data import ExpData
from src.data_loader import DataLoader
from src.exp_manager import ExperimentManager
from src.settings_manager import SettingsManager
import src.plot_manager as plot_manager
import src.global_constants as g


class Kickstarter:

    def __init__(self):
        u.set_printoptions_numpy()
        pass

    @classmethod
    def run_script(cls, exp_name__='debug', num_experiments_constant=1., use_multiple_processes=True,
                   use_tex_labels_=False, load_dir_name=None, target_noise_equal_variances=None):
        """ Run the experiments and save the results."""

        start_time = time.time()
        u.set_printoptions_numpy()
        results_dict_ = None
        # saving_prefixes = ['rmse', 'herm', 'bf_rmse']

        sm = SettingsManager()
        dl = DataLoader()

        # Load / generate settings
        settings_collection, num_experiments_constant = sm.read_hardcoded_settings(num_experiments_constant, exp_name__)
        settings_all_figures = sm.combine_common_and_specific_settings(settings_collection, num_experiments_constant)
        different_keys_across_figures, are_multiple_figs = sm.get_different_keys_across_figures(settings_all_figures)

        # Force target_noise_equal_variances as specified by the user
        if target_noise_equal_variances is not None:
            for sett in settings_all_figures:
                sett['target_noise_equal_variances'] = target_noise_equal_variances

        if load_dir_name is None:
            parent_dir_name = "talsp2025"
            out_dir_name = u.create_output_folder(parent_dir_name=parent_dir_name, child_dir_name=exp_name__)
        else:
            out_dir_name = load_dir_name

        for sett in settings_all_figures:
            sett['out_dir_name'] = out_dir_name

        if load_dir_name is None:
            # Run experiments
            results_dict_ = cls.run_experiment_parallelize(settings_all_figures, use_multiple_processes)
            results_dict_['settings'] = settings_all_figures

            if settings_all_figures[-1]['plot_correlation_histogram']:
                print(f"Skip saving and plotting accuracy because plot_correlation_histogram is True")
                return results_dict_

            errors_all_figures_all_metrics, settings_all_figures_all_metrics = \
                SettingsManager.group_data_for_saving_and_plotting(results_dict_,
                                                                   key_varying_elements=different_keys_across_figures)

            # Save experiments results (errors) and settings
            dl.save_data_wrapper(errors_all_figures_all_metrics, settings_all_figures_all_metrics,
                                 different_keys_across_figures, out_dir_name)

        else:
            # parent_dir_name example: '../../out2/run_experiments/time_frames-2023-07-04--12-25-11'
            # arrays example: '../../out2/run_experiments/time_frames-2023-07-04--12-25-11/arrays'
            errors_all_figures_all_metrics, settings_all_figures_all_metrics = dl.load_data(load_dir_name)
            _, figures_dir_name, _ = u.make_folders_errors_figures_settings(out_dir_name, use_tex_labels_)

        skip_saving_data = (g.debug_mode and not g.debug_save) or not g.release_save_plots
        if skip_saving_data:
            warnings.warn(f"Skipping saving data: {g.debug_mode=}, {g.debug_save=}, {g.release_save_plots=}")

        # Plot results
        _, figures_dir_name, _ = u.make_folders_errors_figures_settings(out_dir_name, use_tex_labels_)
        for err_all_figs_single_metric, settings_all_fig_single_metric in zip(errors_all_figures_all_metrics,
                                                                              settings_all_figures_all_metrics):

            if not (err_all_figs_single_metric and settings_all_fig_single_metric):
                continue

            prefix = settings_all_fig_single_metric[0]['saving_prefix']
            figures = plot_manager.plot_errors(err_all_figs_single_metric, settings_all_fig_single_metric,
                                               use_tex_labels_=use_tex_labels_,
                                               is_beamforming_error='bf' in prefix,
                                               show_plot=g.debug_mode or skip_saving_data)

            # Save figures
            target_folder = dl.save_data(figures_=figures,
                                         different_keys_across_figures=different_keys_across_figures,
                                         settings_figures=settings_all_fig_single_metric,
                                         fig_dir_name=figures_dir_name,
                                         omit_time_from_file_name=False, prepend_file_name=prefix)

        print(f"{num_experiments_constant = :.2e}")

        # if figures is not None and len(figures) > 0:
        #     figures[-1].show()

        return results_dict_

    @classmethod
    def run_experiment_parallelize(cls, settings_all_subplots, use_multiple_processes):
        """ Run the experiments in single or multiple processes. """

        atf, variances = None, None
        # warmup run is used to generate atf and variances of synthetic signals
        if settings_all_subplots[0]['needs_warmup_run']:
            atf, variances = cls.run_warmup_experiment(settings_all_subplots[0])

        if use_multiple_processes:
            how_many_processes = min(cpu_count(), len(settings_all_subplots))
            print(f"Running in {how_many_processes} processes")
            with Pool(how_many_processes) as p:
                results_all_figs = p.starmap(cls.run_experiment,
                                             [(sett, atf, variances) for sett in settings_all_subplots])
        else:
            print(f"Running in a single process")
            results_all_figs = map(cls.run_experiment, [sett__ for sett__ in settings_all_subplots],
                                   repeat(atf), repeat(variances))
        results_all_figs = list(results_all_figs)  # each element is a different figure (e.g. correlation_target)

        # store results_ as a dict of lists. Each key is given by the list above, e.g. 'err_mean_std_db_array'
        # result_keys should be copied from return arguments of ExperimentManager.run_experiment:
        results_keys = ['rtf_errors_dict', 'rtf_evaluators_dict',
                        'beamforming_errors_dict', 'beamforming_evaluators_dict', 'exp_data_dict']

        results_dict_all_figs = {key: [] for key in results_keys}
        for results_single_fig in results_all_figs:
            for key in results_keys:
                results_dict_all_figs[key].append(results_single_fig[results_keys.index(key)])

        return results_dict_all_figs

    @staticmethod
    def make_sound_open_folder(target_folder, exp_name__='', start_time=0.):
        total_time = int(time.time() - start_time)
        if sys.platform == 'darwin' and exp_name__ != 'debug':
            # open target_folder in Finder
            if target_folder is not None:
                subprocess.Popen(['open', target_folder])

            # make sound to notify that the experiment is done
            # if not g.debug_mode:
            #     if total_time > 60:
            #         os.system('say "hey boss, we are done"')

        # Print total time in hours, minutes, seconds
        print(f"Total time: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60}s")

    @classmethod
    def run_experiment(cls, settings, atf_target=None, variances=None):
        """ Run experiment with the given settings. The settings are modified in-place. """

        exp_data_dict, exp_settings_list, rtf_errors_dict, rtf_evaluators_dict, variation_factor_values = (
            cls.run_experiment_rtf_estimation(settings, atf_target, variances))

        beamforming_errors_dict, beamforming_evaluators_dict = {}, {}
        run_beamforming = exp_settings_list and 'beamforming_algorithm' in exp_settings_list[0] and \
                            exp_settings_list[0]['beamforming_algorithm'] is not None
        if run_beamforming:
            beamforming_errors_dict, beamforming_evaluators_dict = cls.run_experiment_beamforming(settings,
                                                                                                  exp_data_dict)

        # order of algorithms in "err_mean_std_db_array" follows exp_settings['algo_names']
        return rtf_errors_dict, rtf_evaluators_dict, beamforming_errors_dict, beamforming_evaluators_dict, exp_data_dict

    @classmethod
    def run_experiment_beamforming(cls, settings, exp_data_dict):
        """ Run the beamforming experiment with the given settings. """
        print("\n***** Running beamforming experiment *****\n")

        beamforming_err_list = []
        beamforming_eval_list = []

        exp_settings_list, variation_factor_values = SettingsManager.settings_to_settings_list(settings)

        for sett_single_variation, variation_value in zip_longest(exp_settings_list, variation_factor_values):
            processed_samples_realizations, processed_stft_realizations = (
                BeamformingManager.run_beamforming_all_realizations(exp_data_dict[str(variation_value)],
                                                                    sett_single_variation))

            to_eval = processed_stft_realizations if sett_single_variation['beamforming_metrics'] == ['RMSE'] \
                else processed_samples_realizations

            algo_names_bf = SettingsManager.get_algo_names_beamforming(sett_single_variation['algo_names'])
            bf_err, bf_eval = BeamformingManager.evaluate_beamformed_data_all_realizations(
                processed_realizations=to_eval,
                algo_names_bf=algo_names_bf,
                bf_metrics=sett_single_variation['beamforming_metrics'])

            beamforming_err_list.append(bf_err)
            beamforming_eval_list.append(bf_eval)

        beamforming_errors_dict = cls.aggregate_dictionaries_of_lists(beamforming_err_list)
        beamforming_evaluators_dict = cls.aggregate_dictionaries_of_lists(beamforming_eval_list)

        return beamforming_errors_dict, beamforming_evaluators_dict

    @classmethod
    def run_experiment_rtf_estimation(cls, settings, atf_target=None, variances=None):
        """ Run the RTF estimation experiment with the given settings. """
        print("\n***** Running RTF estimation experiment *****\n")

        if settings['gen_signals_freq_domain'] and len(settings['nstft']) > 1:
            raise NotImplementedError("Multiple nstft values not supported yet.")

        # Each entry of exp_data_dict corresponds to a beamforming metric (e.g. 'fwSNRseg')
        # Each entry is a list, one element per variation of the experiment (e.g. different SNRs).
        # Each element is np.ndarray, with shape:
        # (num_rtf_estimation_methods + 1 (ground truth), 1, 3=(mean, mean+std, mean-std))
        exp_data_dict = {}

        rtf_err_list = []
        rtf_eval_list = []
        other_parameters, variances = cls.generate_or_load_variances(settings, variances)
        exp_settings_list, variation_factor_values = SettingsManager.settings_to_settings_list(settings)
        for sett_single_variation, variation_value in zip_longest(exp_settings_list, variation_factor_values):

            # Single variation, all realizations. Estimate the RTFs.
            exp_data_dict[str(variation_value)] = ExperimentManager.run_experiment_single_variation(
                                                                            sett=sett_single_variation,
                                                                            variation_factor_value=variation_value,
                                                                            atf_target=atf_target, **other_parameters)

            # Evaluate errors on the RTFs for each realization
            rtf_err, rtf_eval = cls.rtf_evaluate_errors(exp_data_dict[str(variation_value)], sett_single_variation)
            rtf_err_list.append(rtf_err)
            rtf_eval_list.append(rtf_eval)

        rtf_errors_dict = cls.aggregate_dictionaries_of_lists(rtf_err_list)
        rtf_evaluators_dict = cls.aggregate_dictionaries_of_lists(rtf_eval_list)

        return exp_data_dict, exp_settings_list, rtf_errors_dict, rtf_evaluators_dict, variation_factor_values

    @classmethod
    def aggregate_dictionaries_of_lists(cls, list_of_dict_of_lists):
        """
        Aggregate a list of dictionaries of lists into a dictionary of lists.
        Assume that all dictionaries have the same keys.
        For example, if the input is
        inp = [a, b],
        where
        a = {'l1': [1, 2], 'l2': [3, 4]} and b = {'l1': [5, 6], 'l2': [7, 8]},
        then the output is
        out = {'l1': [1, 2, 5, 6], 'l2': [3, 4, 7, 8]}.
        The order of the lists is preserved.
        """

        if len(list_of_dict_of_lists) == 0:
            return {}

        res = {}
        for k in list_of_dict_of_lists[0].keys():
            res[k] = list(dict_of_lists[k] for dict_of_lists in list_of_dict_of_lists)
        return res

    @classmethod
    def rtf_evaluate_errors(cls, exp_data, sett_single_variation):

        rtf_errors_dict = {metric_name: [] for metric_name in sett_single_variation['rtf_metrics']}
        rtf_evaluators_dict = {metric_name: [] for metric_name in sett_single_variation['rtf_metrics']}

        for metric_name in sett_single_variation['rtf_metrics']:
            if isinstance(exp_data, ExpData) and exp_data.rtf_estimates is not None:
                # rtfs_estimates_raw is a list, one element per realization.
                # Each element is a dictionary, one key per algorithm (e.g. 'CW'). Same for rtf_targets and loud_bins_masks
                err, rtf_eval = ErrorEvaluator.evaluate_errors_single_variation(exp_data.rtf_estimates,
                                                                                exp_data.rtf_targets,
                                                                                [metric_name],
                                                                                sett_single_variation['algo_names'],
                                                                                exp_data.loud_bins_masks)
                rtf_errors_dict[metric_name] = err
                rtf_evaluators_dict[metric_name] = rtf_eval

        return rtf_errors_dict, rtf_evaluators_dict

    @staticmethod
    def generate_or_load_variances(settings, variances=None):

        variation_factor_key = '-'.join(str(x) for x in settings['varying_factors'])
        other_parameters = {'variation_factor_key': variation_factor_key}
        if settings['gen_signals_freq_domain']:
            if variances is None:
                variances = SettingsManager.generate_variances(settings['target_noise_equal_variances'],
                                                               settings['num_mics_max'],
                                                               settings['nstft'][0] // 2 + 1)
                print(f"Generating new variances: {variances}")
            else:
                print(f"Using provided variances: {variances}")

            other_parameters.update({'variances_target': variances[0],
                                     'variances_noise': variances[1]})

        elif not settings['gen_signals_freq_domain'] and variances is not None:
            raise ValueError("variances must be None if gen_signals_freq_domain is False.")

        return other_parameters, variances

    @classmethod
    def run_warmup_experiment(cls, settings_single_subplot_):
        """ warm-up run to calculate variances and atf (synthetic signals only)"""

        sett_warmup = copy.deepcopy(settings_single_subplot_)
        sett_warmup['exp_name'] = 'warmup'
        sett_warmup['varying_factors'] = ['']
        sett_warmup['num_repeated_experiments'] = 1

        # _, _, _, _, atf, variances, _, _, _ = cls.run_experiment(sett_warmup)
        # res = cls.run_experiment(sett_warmup)

        _, variances = Kickstarter.generate_or_load_variances(sett_warmup)
        sett_warmup_list, _ = SettingsManager.settings_to_settings_list(sett_warmup)
        sett_warmup = sett_warmup_list[0]
        atf_warmup, _ = ExperimentManager.run_experiment_single_variation(sett=sett_warmup)

        return atf_warmup, variances
