import copy
import warnings
import numpy as np

import src.global_constants as g
import src.utils as u
import src.config as cfg

no_correlation_value = 0.
low_correlation_value = 0.25
mid_correlation_value = 0.35
high_correlation_value = 0.75


class SettingsManager:
    """ Class to manage the settings of the experiment. """

    def __init__(self):
        pass

    @staticmethod
    def get_variation_key_values(original_settings, varying_factors_=None):
        """
        Returns the key and values of the varying factor. For example, if the varying factor is
        'num_mics', this function returns 'num_mics' and a list of values, such as [2, 4, 8].
        """

        if varying_factors_ is None:
            varying_factors_ = original_settings['varying_factors']

        variation_key = '-'.join([str(x) for x in varying_factors_])
        try:
            variation_values = u.get_by_path(original_settings, varying_factors_)
        except KeyError:
            if variation_key == '' or variation_key is None or variation_key == 'None':
                return '', [0]
            else:
                # if above did not work, try deleting the '0', which is there to access first element of list.
                varying_factor = [x for x in varying_factors_ if x != 0]
                variation_values = u.get_by_path(original_settings, varying_factor)

        if not isinstance(variation_values, list):
            variation_values = [variation_values]

        return variation_key, variation_values

    @staticmethod
    def generate_settings_for_multiple_variations(original_settings):
        new_settings_list = []

        _, variation_values = SettingsManager.get_variation_key_values(original_settings)

        if not variation_values:
            warnings.warn('No variation values found. Returning original settings.')
            return [original_settings], variation_values

        for variation_value in variation_values:
            new_settings = copy.deepcopy(original_settings)
            u.set_by_path(new_settings, original_settings['varying_factors'], variation_value)
            new_settings_list.append(new_settings)

        return new_settings_list, variation_values

    @staticmethod
    def varying_factors_to_list(settings):
        variation_key, variation_values = SettingsManager.get_variation_key_values(settings)
        if not isinstance(variation_values, list):
            settings[variation_key] = [variation_values]
        return settings

    @staticmethod
    def settings_to_settings_list(settings):
        settings = SettingsManager.assign_default_values(settings)
        settings = SettingsManager.non_varying_factors_list_to_scalar(settings)
        settings = SettingsManager.varying_factors_to_list(settings)
        exp_settings_list, variation_values = SettingsManager.generate_settings_for_multiple_variations(settings)
        return exp_settings_list, variation_values

    @staticmethod
    def validate(settings):

        # specify factor to vary in experiments
        assert 'varying_factors' in settings

        # overlap doesn't make sense if we generate in frequency domain
        assert not (settings['noverlap_percentage'] > 0 and settings['gen_signals_freq_domain'])

    @staticmethod
    def assign_default_values(settings):
        settings['gen_signals_freq_domain'] = settings.get('gen_signals_freq_domain', False)
        settings['plot_atf_vectors'] = settings.get('plot_atf_vectors', False)
        settings['add_identity_noise_noisy'] = settings.get('add_identity_noise_noisy', False)
        settings['varying_factors'] = settings.get('varying_factors', [''])
        settings['correlation_noise_type'] = settings.get('correlation_noise_type', 'freq')
        settings['correlation_target_type'] = settings.get('correlation_target_type', 'freq')
        settings['num_retained_eigva'] = int(float(settings.get('num_retained_eigva', -1)))
        settings['perc_active_noise_freq'] = float(settings.get('perc_active_noise_freq', 1))
        settings['perc_active_target_freq'] = float(settings.get('perc_active_target_freq', 1))
        settings['generate_single_frame_many_realizations'] = settings.get('generate_single_frame_many_realizations',
                                                                           False)
        settings['desired'] = settings.get('desired', [None])
        settings['alpha_cov_estimation'] = settings.get('alpha_cov_estimation', g.alpha_cov)

        settings['noise_estimate_perturbation_amount'] = float(settings.get('noise_estimate_perturbation_amount', 0))
        noverlap_perc = settings.get('noverlap_percentage', 0)
        settings['noverlap_percentage'] = [float(x) for x in noverlap_perc] if isinstance(noverlap_perc,
                                                                                          list) else float(
            noverlap_perc)

        settings['rir_settings'] = SettingsManager.get_rir_settings_from_settings_or_default(settings)

        settings['max_relative_difference_loud_bins'] = int(float(settings.get('max_relative_difference_loud_bins',
                                                                               g.max_relative_difference_loud_bins_default)))

        settings['needs_warmup_run'] = settings.get('needs_warmup_run', True)
        settings['correlation_noise_type'] = settings.get('correlation_noise_type', '')
        settings['correlation_target_type'] = settings.get('correlation_target_type', '')

        settings['num_neighbours_target'] = settings.get('num_neighbours_target', -1)
        settings['grid_spacing_target'] = settings.get('grid_spacing_target', -1)
        settings['num_neighbours_noise'] = settings.get('num_neighbours_noise', -1)
        settings['grid_spacing_noise'] = settings.get('grid_spacing_noise', -1)
        settings['correlation_target_pattern'] = settings.get('correlation_target_pattern', 'equicorrelated')

        settings['beamforming_algorithm'] = settings.get('beamforming_algorithm', None)
        settings['beamforming_metrics'] = settings.get('beamforming_metrics', [])

        settings['correlation_noise_pattern'] = settings.get('correlation_noise_pattern', 'equicorrelated')
        settings['processed_freq_range_hz'] = settings.get('processed_freq_range_hz', [0, g.fs / 2])

        settings['plot_correlation_histogram'] = settings.get('plot_correlation_histogram', False)

        return settings

    @staticmethod
    def get_rir_settings_from_settings_or_default(settings=None):

        if settings is None:
            settings = {}

        rir_sett = settings.get('rir_settings', {})

        rir_sett['rtf_type'] = rir_sett.get('rtf_type', 'real')
        rir_sett['noise_angle'] = rir_sett.get('noise_angle', 0)
        rir_sett['noise_distance'] = rir_sett.get('noise_distance', 0)
        rir_sett['target_angle'] = rir_sett.get('target_angle', 0)
        rir_sett['target_distance'] = rir_sett.get('target_distance', 0)
        rir_sett['room_size'] = rir_sett.get('room_size', None)
        rir_sett['rir_corpus'] = rir_sett.get('rir_corpus', '')
        rir_sett['num_nonzero_samples_rir_target'] = int(float(rir_sett.get('num_nonzero_samples_rir_target', -1)))
        rir_sett['num_nonzero_samples_rir_noise'] = int(float(rir_sett.get('num_nonzero_samples_rir_noise', -1)))

        return rir_sett

    @staticmethod
    # Only one factor is allowed to vary at a time. It takes on the values of the corresponding list in YAML file.
    # All other factors must be scalar. This function takes the first value of the lists and sets it as the current factor value.
    def non_varying_factors_list_to_scalar(settings):
        factors_names_to_be_converted = ['nstft', 'correlated_noise_snr', 'noise_estimate_perturbation_amount',
                                         'correlation_noise', 'correlation_target', 'duration_output_sec',
                                         'duration_output_frames', 'single_frame_snr', 'AR_coefficient',
                                         'amplitude_centered_around_zero', 'num_retained_eigva',
                                         'duration_output_frames', 'noverlap_percentage']

        factors_names_to_be_converted = filter(lambda name: name not in settings['varying_factors'],
                                               factors_names_to_be_converted)

        for factor_name in factors_names_to_be_converted:
            if factor_name in settings and isinstance(settings[factor_name], list):
                settings[factor_name] = settings[factor_name][0]

        return settings

    @staticmethod
    def generate_variances(target_noise_equal_variances, num_mics, num_freqs):
        """CW-SV seems to suffer if variance of noise is very low for all microphones for a certain frequency."""

        if target_noise_equal_variances:
            target_variance_range = (0.5, 0.5)
            noise_variance_range = (0.5, 0.5)
        else:
            target_variance_range = (g.eps, 5e-1)
            noise_variance_range = (g.eps, 5e-1)

        # # Different size because target is point source: power at different microphones determined by transfer function
        variances_target = g.rng.uniform(target_variance_range[0], target_variance_range[1], size=num_freqs)
        variances_noise = g.rng.uniform(noise_variance_range[0], noise_variance_range[1], size=num_mics * num_freqs)

        return variances_target, variances_noise

    @staticmethod
    def get_algo_names_beamforming(rtf_algo_names):
        algo_names_bf = copy.deepcopy([x for x in rtf_algo_names if not u.is_crb(x)])
        algo_names_bf.append('Ideal')
        algo_names_bf.append('Unprocessed')
        return algo_names_bf

    @staticmethod
    def filter_correlation_type(corr_type):
        if corr_type is None or corr_type == '' or corr_type == 'none' or corr_type == 'None':
            return None
        elif (('frequency' in corr_type and 'space' in corr_type) or ('freq' in corr_type and 'space' in corr_type)
              or ('fs' in corr_type) or ('f+s' in corr_type) or ('frequency+space' in corr_type)):
            return 'frequency+space'
        elif 'frequency' in corr_type or 'freq' in corr_type or 'f' in corr_type:
            return 'frequency'
        elif 'space' in corr_type or 's' in corr_type:
            return 'space'
        else:
            raise ValueError('Unknown correlation type')

    @staticmethod
    def convert_algo_names_correlation_type(current_names_, sett_):

        noise_corr_type = SettingsManager.filter_correlation_type(sett_['correlation_noise_type'])

        if noise_corr_type == 'frequency':
            current_names_ = [f"{name}" for name in current_names_]
            # current_names_ = [f"{name} (f)" for name in current_names_]
        elif noise_corr_type == 'space':
            current_names_ = [f"{name} (s)" for name in current_names_]
        elif noise_corr_type == 'frequency+space':
            current_names_ = [f"{name} (f+s)" for name in current_names_]
        elif noise_corr_type is None:
            pass
        else:
            raise ValueError(f"Unknown correlation type: {noise_corr_type}")

        return current_names_

    @classmethod
    def get_saving_prefix(cls, key, is_beamforming_data):
        if 'rmse' in key.lower():
            saving_prefix = 'rmse'
        elif 'herm' in key.lower():
            saving_prefix = 'herm'
        elif 'fwsnr' in key.lower():
            saving_prefix = 'fwsnr'
        elif 'stoi' in key.lower():
            saving_prefix = 'stoi'
        else:
            saving_prefix = key.lower()
            warnings.warn(f"Unknown prefix corresponding to error metric: {key}")

        if is_beamforming_data:
            saving_prefix = 'bf_' + saving_prefix

        return saving_prefix

    @classmethod
    def group_data_for_saving_and_plotting(cls, results_dict_, key_varying_elements):
        """
        Group the data in the results_dict_ into figures, and prepare the settings for saving and plotting.
        Different FIGURES correspond to different simulation parameters: for example correlation target can be low in
        a figure, and high in another figure. They have nothing to do with metrics.
        """

        if not isinstance(key_varying_elements, list):
            raise ValueError("key_varying_elements must be a list.")
        if len(key_varying_elements) > 1:
            raise NotImplementedError("Not implemented for more than one varying element.")

        rtf_err_all_metrics_all_figs, rtf_sett_all_metrics_all_figs = \
            cls.group_data_for_saving_and_plotting_inner(results_dict_['rtf_errors_dict'],
                                                         results_dict_['settings'], key_varying_elements[0],
                                                         is_beamforming_data=False)

        beamforming_errors_all_metrics_all_figs, beamforming_settings_all_metrics_all_figs = \
            cls.group_data_for_saving_and_plotting_inner(results_dict_['beamforming_errors_dict'],
                                                         results_dict_['settings'], key_varying_elements[0],
                                                         is_beamforming_data=True)

        errors_all_metrics_all_figs = rtf_err_all_metrics_all_figs + beamforming_errors_all_metrics_all_figs
        settings_all_metrics_all_figs = rtf_sett_all_metrics_all_figs + beamforming_settings_all_metrics_all_figs

        return errors_all_metrics_all_figs, settings_all_metrics_all_figs

    @staticmethod
    def group_data_for_saving_and_plotting_inner(errors_all_figs, settings_all_figures, key_varying_element,
                                                 is_beamforming_data=False):
        """ Group the errors_ into figures, and prepare the settings for saving and plotting."""

        error_first_fig = errors_all_figs[0]
        if error_first_fig == {}:
            return [], []

        error_metrics = list(error_first_fig.keys())
        value_varying_elements_across_figs = []
        for settings_single_figure in settings_all_figures:
            try:
                value_varying_elements_across_figs.append(settings_single_figure[key_varying_element])
            except KeyError:
                var_key, var_value = SettingsManager.get_variation_key_values(settings_single_figure)
                value_varying_elements_across_figs.append(var_value)

        # settings_all_metrics_all_figs = [copy.deepcopy(settings_all_figures) for _ in range(len(error_metrics))]
        settings_all_metrics_all_figs = [[[] for _ in value_varying_elements_across_figs] for _ in error_metrics]
        errors_all_metrics_all_figs = [[[] for _ in value_varying_elements_across_figs] for _ in error_metrics]
        bf_or_rtf_metric = 'beamforming_metrics' if is_beamforming_data else 'rtf_metrics'

        # Group errors by metric, and by figure. Each figure is a different plot.
        for fig_idx, errors_single_fig in enumerate(errors_all_figs):
            for key in errors_single_fig.keys():  # errors_single_fig is a dict
                metric_idx = error_metrics.index(key)  # Find the index corresponding to the metric
                if len(settings_all_figures) > 1:
                    fig_idx_ordered = value_varying_elements_across_figs.index(
                        settings_all_figures[fig_idx][key_varying_element])
                else:
                    fig_idx_ordered = 0
                errors_all_metrics_all_figs[metric_idx][fig_idx_ordered] = errors_single_fig[key]

                settings_all_metrics_all_figs[metric_idx][fig_idx_ordered] = copy.deepcopy(
                    settings_all_figures[fig_idx_ordered])
                settings_all_metrics_all_figs[metric_idx][fig_idx_ordered][bf_or_rtf_metric] = error_metrics[metric_idx]
                settings_all_metrics_all_figs[metric_idx][fig_idx_ordered]['saving_prefix'] = \
                    SettingsManager.get_saving_prefix(key, is_beamforming_data)

                assert 'saving_prefix' in settings_all_metrics_all_figs[metric_idx][fig_idx_ordered]

        # Convert to numpy arrays
        for metric_idx, results_single_metric in enumerate(errors_all_metrics_all_figs):
            for fig_idx_ordered, errors_single_fig in enumerate(results_single_metric):
                errors_all_metrics_all_figs[metric_idx][fig_idx_ordered] = np.array(errors_single_fig)

        assert 'saving_prefix' in settings_all_metrics_all_figs[0][0]
        assert isinstance(settings_all_metrics_all_figs[0], list)
        assert isinstance(settings_all_metrics_all_figs[0][0], dict)

        return errors_all_metrics_all_figs, settings_all_metrics_all_figs

    @staticmethod
    def get_different_keys_across_figures(settings_all_figures):
        """
        Find the keys that are different across the figures (e.g. correlation noise).
        If only one figure, return the keys that have different values within the fig (e.g. SNR).
        :param settings_all_figures:
        :return: key names that are different across the figures, and a boolean indicating whether the keys are
        different across the figures (True) or within the figure (False).
        """

        if len(settings_all_figures) <= 1:
            var_key, var_value = SettingsManager.get_variation_key_values(settings_all_figures[0])
            ret = var_key if isinstance(var_key, list) else [var_key]
            return ret, False

        # Find the keys that are different across the figures.
        all_keys = [s.keys() for s in settings_all_figures]
        different_keys_across_figures = list(set(all_keys[0]).symmetric_difference(*all_keys[1:]))
        if len(different_keys_across_figures) != 0:
            raise ValueError(
                f"Why do different figures have different number of settings? Problems with {different_keys_across_figures}")

        # knowing that all dicts in settings_all_figures have the same keys, find the keys that have different values
        different_values_across_figures = []
        for key in settings_all_figures[0].keys():
            values = [s[key] for s in settings_all_figures]
            if not all([v == values[0] for v in values]):
                different_values_across_figures.append(key)

        if len(different_values_across_figures) == 0:
            raise ValueError(f"different_values_across_figures is empty.")

        return different_values_across_figures, True

    @staticmethod
    def combine_common_and_specific_settings(exp_details_collection, repeated_experiments_constant=1.):
        """ Generate the settings for the experiments, by combining the original settings with the exp_details."""

        experiment_settings_original, exp_common, exp_figures = exp_details_collection
        settings_figures = []

        for exp_specific in exp_figures:  # each exp_detail_3 is a different plot
            sett = experiment_settings_original | exp_common | exp_specific  # the union of d1 and d2
            if 'num_repeated_experiments' not in sett:
                sett[
                    'num_repeated_experiments'] = SettingsManager.calculate_repeated_experiments_from_montecarlo_constant(
                    sett['duration_output_frames'], repeated_experiments_constant)
            settings_figures.append(sett)

        return settings_figures

    @staticmethod
    def read_hardcoded_settings(repeated_experiments_constant=1., exp_name=None):

        if 'speech' not in exp_name:
            cfg_name = "config_TALSP2025_synthetic.yaml"
        else:
            cfg_name = "config_TALSP2025_real.yaml"
        print(f"{cfg_name=}")

        experiment_settings_original = cfg.load_configuration(cfg_name)

        correlation_percentage_list = [0., 0.25, 0.5, 0.75, 0.95]
        # correlation_percentage_list = [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.95]
        # correlation_percentage_list = [0., 0.5, 1.]

        exp_details_figures = [{}, ]  # different figures

        exp_common = dict()

        if exp_name == 'target_correlation':
            exp_common = {  # shared settings.
                'varying_factors': ['correlation_target'],
                'correlation_target': correlation_percentage_list,
            }

            exp_details_figures = [  # different figures
                {'correlation_noise': low_correlation_value},
                {'correlation_noise': high_correlation_value}
            ]

        elif exp_name == 'noise_correlation':

            exp_common = {  # shared settings.
                'varying_factors': ['correlation_noise'],
                'correlation_noise': correlation_percentage_list,
            }

            exp_details_figures = [  # different figures
                {'correlation_target': low_correlation_value,
                 'correlation_noise_type': 'frequency', },
                {'correlation_target': high_correlation_value,
                 'correlation_noise_type': 'frequency', },
            ]

        elif exp_name == 'time_frames':

            if repeated_experiments_constant > 1:
                repeated_experiments_constant *= 100
            exp_common = {  # shared settings.
                'varying_factors': ['duration_output_frames'],
                'duration_output_frames': list(np.logspace(np.log10(10), np.log10(5000), 5).astype(int)),
                'correlation_target': high_correlation_value,
            }

            exp_details_figures = [
                {'correlation_noise_pattern': 'white',
                 'num_neighbours_noise': 0
                 },
                {'correlation_noise_pattern': 'neighbouring',
                 'num_neighbours_noise': 0
                 },
            ]

        elif exp_name == 'snr':

            exp_common = {  # shared settings.
                'varying_factors': ['noises_info', 0, 'snr'],
                'noises_info':
                    [{'snr': [-10, -5, 0, 5], 'names': ['white']}],
                'add_identity_noise_noisy': True,
            }

            exp_details_figures = [
                # {'correlation_target': no_correlation_value},
                {'correlation_target': high_correlation_value},
                # {'correlation_target': mid_correlation_value},
                {'correlation_target': low_correlation_value},
                # {'duration_output_frames': 5000, },
            ]

        elif exp_name == 'debug':
            exp_common = {  # shared settings.
                'varying_factors': ['noises_info', 0, 'snr'],
                # 'varying_factors': ['duration_output_frames'],
                'algo_names': ['CW-SV', 'CW-EV-SV', 'CW', 'CRB_unconditional'],
                'num_repeated_experiments': 1,
                'correlation_noise': 0.,
                'correlation_target': 0.9,
                'noises_info':
                    [{'snr': [0], }],
                # 'noises_info':
                #     [{'snr': [-10, -5, 0, 5, 10], }],
                # 'duration_output_frames': list(np.logspace(np.log10(1), np.log10(100), 6).astype(int)),
                'duration_output_frames': [100],
            }

        if not exp_common:
            exp_common = experiment_settings_original['exp_common'][exp_name]
        exp_common['exp_name'] = exp_name

        settings_collection = [experiment_settings_original, exp_common, exp_details_figures]

        return settings_collection, repeated_experiments_constant

    @staticmethod
    def calculate_repeated_experiments_from_montecarlo_constant(num_frames, repeated_experiments_constant):
        if isinstance(num_frames, int):
            num_frames_for_counting_montecarlo = num_frames
        elif isinstance(num_frames, list):
            num_frames_for_counting_montecarlo = np.mean(np.array(num_frames))
        else:
            raise ValueError(
                f"Cannot determine num_frames_for_counting_montecarlo from {num_frames=}")

        min_repeated_experiments = 10 if repeated_experiments_constant > 1 else 2
        num_repeated_exp = max(min_repeated_experiments,
                               int(repeated_experiments_constant / (20 * num_frames_for_counting_montecarlo ** 2)))
        return num_repeated_exp

    # @staticmethod
    # def get_settings_for_experiment(exp_name, repeated_experiments_constant=1.):
    #     settings_collection, repeated_experiments_constant = SettingsManager.read_hardcoded_settings(
    #         repeated_experiments_constant, exp_name)
    #     settings_figures = SettingsManager.combine_common_and_specific_settings(settings_collection,
    #                                                                            repeated_experiments_constant)
    #     return settings_figures

    @staticmethod
    def rtf_algos_need_wideband_covariances(algo_names):
        return any([algo_name in ['CRB_unconditional', 'CRB_conditional', 'CW-SV', 'CW-SV-orig-phase'] for algo_name in
                    algo_names])
