import copy
import warnings

import numpy as np
from prettytable import PrettyTable

import src.bound_manager as bound_manager
import src.cov_manager as cov_manager
import src.global_constants as g
import src.rtf_estimator as rtf_estimator
import src.signal_generator as signal_generator
import src.utils as u
from src.rtf_evaluator import RtfEvaluator, RtfErrorAverager
from itertools import zip_longest

sh = None


class ExperimentManager:
    def __init__(self, **kwargs):
        self.methods = kwargs['algo_names']
        self.avg_time_frames = kwargs['avg_time_frames']

        self.duration_output_frames = kwargs.get('duration_output_frames', None)
        self.duration_output_sec = -1
        # if self.duration_output_frames is None:
        #     self.duration_output_sec = float(kwargs['duration_output_sec'])

        self.num_mics_max = kwargs['num_mics_max']
        self.noises_info = kwargs.get('noises_info', [])
        if self.noises_info is not None:
            for noise_info in self.noises_info:
                if isinstance(noise_info['snr'], list):
                    noise_info['snr'] = noise_info['snr'][0]

                if 'noise_volumes_per_mic' not in noise_info:
                    noise_info['noise_volumes_per_mic'] = g.rng.uniform(size=(self.num_mics_max, 1))

        self.nstft = kwargs['nstft']
        if 'noverlap_percentage' in kwargs:
            self.noverlap = int(kwargs['nstft'] * kwargs['noverlap_percentage'])
        self.sig_generator = None

    def initialize_signal_generator(self, **kwargs):
        assert (self.avg_time_frames or self.duration_output_sec < 20 or self.duration_output_frames < 2000)
        sig_generator = signal_generator.SignalGenerator(**kwargs)


        sig_generator.noises_info = self.noises_info

        return sig_generator

    def initialize_rtf_estimator(self, rtfs_ground_truth_stft, exp_settings):
        re = rtf_estimator.RtfEstimator(g.idx_ref_mic)
        re.rtfs_gt = rtfs_ground_truth_stft
        re.methods = self.methods

        re.flag_estimate_signal_threshold = exp_settings.get('flag_estimate_signal_threshold', False)
        re.flag_scree_method = exp_settings.get('flag_scree_method', False)
        re.flag_keep_num_freqs_eigenvectors = exp_settings.get('flag_keep_num_freqs_eigenvectors', False)
        re.flag_mdl_criterion = exp_settings.get('flag_mdl_criterion', False)

        return re

    @staticmethod
    def run_experiment(settings_, atf_target=None, variances=None):
        """ Run experiment with the given settings. The settings are modified in-place. """

        settings = SettingsManager.assign_default_values(settings_)

        u.set_printoptions_numpy()
        variation_factor_key = '-'.join(str(x) for x in settings['varying_factors'])
        err_mean_std_db_list_mse = []
        err_mean_std_db_list_herm = []
        rtf_evaluators = dict()
        cm = None

        if settings['gen_signals_freq_domain'] and len(settings['nstft']) > 1:
            raise NotImplementedError("Multiple nstft values not supported yet.")

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

        exp_settings_list, variation_factor_values = SettingsManager.settings_to_settings_list(settings)
        for exp_settings, variation_factor_value in zip_longest(exp_settings_list, variation_factor_values):
            atf_target, cm, data_to_evaluate_error = \
                ExperimentManager.run_experiment_single_variation(sett=exp_settings,
                                                                  variation_factor_value=variation_factor_value,
                                                                  atf_target=atf_target, **other_parameters)

            for metric_name in exp_settings['metric_names']:
                rtfs_estimates_raw, rtf_targets, loud_bins_masks = data_to_evaluate_error
                if rtfs_estimates_raw is not None:
                    err, rtf_eval = ExperimentManager.evaluate_errors_single_variation(rtfs_estimates_raw,
                                                                                       rtf_targets, loud_bins_masks,
                                                                                       [metric_name],
                                                                                       exp_settings['algo_names'])
                    if 'mse' in metric_name.lower():
                        err_mean_std_db_list_mse.append(err)
                        rtf_evaluators[str(variation_factor_value)] = rtf_eval
                    elif 'herm' in metric_name.lower():
                        err_mean_std_db_list_herm.append(err)

        # order of algorithms in "err_mean_std_db_array" follows exp_settings['algo_names']
        err_mean_std_db_array = np.array(err_mean_std_db_list_mse)
        err_mean_std_db_array_herm = np.array(err_mean_std_db_list_herm)

        # sh = signal_generator.SignalHolder(stimuli_stft, sg)

        return err_mean_std_db_array, rtf_evaluators, cm, sh, atf_target, variances, err_mean_std_db_array_herm

    @staticmethod
    def run_experiment_single_variation(sett=None, variation_factor_value=None, atf_target=None, variances_noise=None,
                                        variances_target=None, variation_factor_key=None):

        force_new_atf = False
        cov_mask_target = None
        alpha_cov_estimation = 0  # covariance smoothing only for real speech
        SettingsManager.validate(sett)
        rtfs_estimates_raw_all_realizations = []
        loud_bins_masks_all_realizations = []
        rtf_targets_all_realizations = []
        exp_manager = ExperimentManager(**sett)
        sg = exp_manager.initialize_signal_generator(**sett)
        cm = cov_manager.CovarianceManager(sett["add_identity_noise_noisy"], sg.get_stft_shape()[-1])
        sett['correlation_noise_type'] = cm.filter_correlation_type(sett['correlation_noise_type'])
        sett['correlation_target_type'] = cm.filter_correlation_type(sett['correlation_target_type'])

        new_random_rtf_every_realization = False
        if 'real' not in sett['rtf_type'] and 'debug' not in sett['rtf_type'] and 'once' not in sett['rtf_type']:
            new_random_rtf_every_realization = True

        num_repeated_experiments = sett['num_repeated_experiments'] if not (g.debug_mode and g.debug_plots) else 1
        for ii in range(num_repeated_experiments):
            if ii % 500 == 0:
                print(f"{variation_factor_key} {variation_factor_value}, "
                      f"realization {ii + 1}/{num_repeated_experiments}")
            # check that ATF has correct number of frequency bins
            if atf_target is not None and atf_target.shape[1] != sg.get_stft_shape()[1]:
                force_new_atf = True
                warnings.warn(f"Cannot reuse ATF, because the number of frequency bins don't match: "
                              f"{atf_target.shape[1]} != {sg.get_stft_shape()[1]}. Generating new ATF.")

            # generate new atf/rtf unless mode is 'random-once' (for Cramer-Rao bound simulations)
            if atf_target is None or force_new_atf:
                if ii == 0:
                    print(f"{sett['rtf_type'] = }")
                atf_target = sg.generate_atf(sett['rtf_type'])
                force_new_atf = False

            rtf_target = sg.generate_rtf_from_atf(atf_target)
            rtf_estimator_ = exp_manager.initialize_rtf_estimator(rtf_target, sett)

            if '' in sett['varying_factors']:
                warnings.warn("No varying factors. Probably warm-up run or debugging. Results will be discarded.")
                return atf_target, cm, (None, None, None)

            if cov_mask_target is None and sett['covariance_target_type'] != 'equicorrelated':
                cov_mask_target = cm.generate_covariance_mask(sg.get_stft_shape(), sett['covariance_target_type'],
                                                                        num_neighbours=sett['num_neighbours'],
                                                              checkerboard_spacing=sett['checkerboard_spacing'])

            if sett['gen_signals_freq_domain']:
                """ Generate true covariance matrices for the target and noise signals in frequency domain."""

                rs_dry_cov_k, cm.cov_dry_oracle = cm.generate_target_covariance(sett['correlation_target'],
                                                                                sg.get_stft_shape(),
                                                                                sett['correlation_target_type'],
                                                                                sett['perc_active_target_freq'],
                                                                                variances_target,
                                                                                sett['covariance_target_type'],
                                                                                covariance_mask=cov_mask_target)

                # cm.phi_xx_bf converges to cm.phi_dry_bf_true for infinite number of realizations
                A = np.diag(atf_target.flatten('F'))
                cm.phi_wet_bf_true = A @ cm.cov_dry_oracle @ A.conj().T
                ref_power_wet = cm.compute_snr_from_covariance_matrices(cm.phi_wet_bf_true)
                correlated_noise_snr = sett['noises_info'][0]['snr'] - ref_power_wet
                cm.phi_vv_bf_true = \
                    cm.generate_noise_covariance(sg.get_stft_shape(),
                                                 snr_db=correlated_noise_snr,
                                                 noise_corr_coefficient=float(sett['correlation_noise']),
                                                 corr_type=sett['correlation_noise_type'],
                                                 percentage_active_bins=float(sett['perc_active_noise_freq']),
                                                 variances=variances_noise,
                                                 covariance_type=sett['covariance_noise_type'],)

                # add noise floor to noise covariance matrix (sensor noise)
                cm.phi_vv_bf_true += cm.generate_noise_covariance(sg.get_stft_shape(),
                                                                  snr_db=g.white_noise_floor_db - ref_power_wet,
                                                                  variances=1.0)

                if g.debug_mode and ii == 0:
                    print(f"{cm.compute_snr_from_covariance_matrices(cm.phi_wet_bf_true, cm.phi_vv_bf_true):.1f} "
                          f"dB wet SNR")

                stimuli_stft = \
                    sg.generate_gaussian_signals_from_covariance_matrices(rs_dry_cov_k, cm.phi_vv_bf_true,
                                                                          atf_target, sg.get_stft_shape())

            else:
                """ Load/generate the target and noise signals in time domain."""
                if ii == 0:
                    print(f"RTF estimation methods: {rtf_estimator_.methods}.")
                    if 'random' in sett['rtf_type'] or 'deterministic' in sett['rtf_type']:
                        print("Multiplication ATF-target signal in frequency domain")
                    else:
                        print("Convolution RIR-target signal in time domain")

                stimuli_samples, stimuli_stft = sg.generate_signal_samples(atf_target)

                if not sett['generate_single_frame_many_realizations'] and \
                        not sett['avg_time_frames']:
                    alpha_cov_estimation = sett['alpha_cov_estimation']

            if sg.noise_estimate_perturbation_amount > 0:
                stimuli_stft['mix'] = sg.perturb_noise_stft_with_wgn(sg.noise_estimate_perturbation_amount,
                                                                     stimuli_stft['mix'])

            phase_correction = None
            if not sett['gen_signals_freq_domain']:
                # find element in stimuli_stft stft with the largest number of frames
                key_max_frames = max(stimuli_stft, key=lambda x: stimuli_stft[x].shape[-1])

                # phase correction computed for the stft with the largest number of frames
                phase_correction = cm.compute_correction_term(sg.nstft, sg.noverlap, stimuli_stft[key_max_frames].shape)

            # only pass this argument for performance bounds - needed to estimate DRY covariance
            desired_dry_stft_ref = None
            if any('CRB' in x for x in sett['algo_names']) and cm.cov_dry_oracle is None:
                desired_dry_stft_ref = stimuli_stft['desired_dry']

            if ii % 500 == 0:
                print(f"Estimating covariances from {int(stimuli_stft['mix'].shape[-1])} frames, "
                      f"smoothing {alpha_cov_estimation}.")
                if cm.add_identity_noise_noisy:
                    print("Adding identity matrix to noise and noisy CPSD matrices")

            """
            Cut the stimuli_stft to only the first K frequency bins to improve inversion of covariance matrices.
            K should correspond to 2Khz, which is the highest frequency of interest for speech.
            """

            if not sett['gen_signals_freq_domain']:
                k_freq = 1500
                K = sg.get_frequency_bins_from_frequencies(k_freq)
                K = K if K > 1 else sg.get_frequency_bins_from_frequencies(g.fs // 2)
                for key, stimulus in stimuli_stft.items():
                    stimuli_stft[key] = stimulus[:, :K, ...]
                if phase_correction is not None:
                    phase_correction = phase_correction[:K, ...]
                rtf_target = rtf_target[..., :K]
                if cov_mask_target is not None:
                    cov_mask_target = cov_mask_target[:K, :K]
                if desired_dry_stft_ref is not None:
                    desired_dry_stft_ref = desired_dry_stft_ref[:, :K, ...]

                """normalize stimuli_stft to avoid numerical issues. first find maximum value across elements"""
                max_value = np.max(np.abs(list(stimuli_stft.values())))
                for key in stimuli_stft.keys():
                    stimuli_stft[key] = stimuli_stft[key] / max_value

            cm.estimate_covariances(stimuli_stft, exp_manager.avg_time_frames, alpha_cov_estimation,
                                    phase_correction, desired_dry_stft_ref)

            if sett['use_true_covariance']:
                if not sett['gen_signals_freq_domain']:
                    raise ValueError("True covariance matrices can only be used when generating signals in the "
                                     "frequency domain. Change \'use_true_covariance\' to False.")

                cm.cov_noise = cm.phi_vv_bf_true[..., np.newaxis]
                if sett['add_identity_noise_noisy']:
                    cm.cov_noise = cm.cov_noise + g.diagonal_loading * np.identity(cm.cov_noise.shape[0])[..., np.newaxis]

                # update narrowband covariance matrices
                cm.copy_single_freq_covariances_from_bifreq_covariances((sg.get_stft_shape()[:2] + (1,)))
            cm.cov_wet_gevd = cm.estimate_cov_wet_gevd(cm.cov_noisy, cm.cov_noise, sg.get_stft_shape())

            # debug, plot estimated and "true" covariance matrices
            if g.debug_mode and ii == 0 and g.debug_plots:
                amp_range = (np.min(u.log_pow(cm.cov_noisy)), np.max(u.log_pow(cm.cov_noisy)))  # (-50, 30)
                cm.plot_cov(amp_range=amp_range)
                cm.plot_cov(true_cov=True, amp_range=amp_range)

            rtfs_estimates_raw = rtf_estimator_.estimate_rtf(cm, sg.get_stft_shape(), sett['num_retained_eigva'],
                                                             covariance_mask_target=cov_mask_target)

            if (new_random_rtf_every_realization and ii < max(10, num_repeated_experiments // 1000)) or \
                    (not new_random_rtf_every_realization and ii < 3):
                """ Calculate performance bounds. Only do this for the first few iterations,
                    since it is computationally expensive."""
                bounds = bound_manager.BoundManager.calculate_bounds(sett['algo_names'], atf_target, cm,
                                                                     sg.get_stft_shape(), stimuli_stft['desired_dry'])
                rtfs_estimates_raw.update(bounds)

            if not rtfs_estimates_raw:  # empty result dictionary
                break

            # New ATF should be generated for next Montecarlo realization, unless we are calculating CRB.
            # In this case, we want to use the same ATF for all realizations.
            if new_random_rtf_every_realization:
                if any([u.is_crb(algo) for algo in sett['algo_names']]):
                    warnings.warn('ATF should be generated only once, because CRB is being calculated. ')
                force_new_atf = True

            rtfs_estimates_raw_all_realizations.append(rtfs_estimates_raw)
            rtf_targets_all_realizations.append(rtf_target)

            if ii % 500 == 0:
                rtfs_estimates_clean = RtfEvaluator.clean_up_rtf_estimates(rtfs_estimates_raw, clip_estimates=True)
                rtf_evaluator = RtfEvaluator(ground_truth=rtf_target, estimates=rtfs_estimates_clean,
                                             metric_names=sett['metric_names'])
                rtf_evaluator.evaluate_errors_single_realization(plot_rtf_realization=False)

            loud_bins_masks_all_realizations.append(None)
            if not sett['gen_signals_freq_domain']:
                # LOWER max_relative_difference means LESS bins are evaluated
                loud_bins_masks_all_realizations[-1] = RtfEvaluator.find_loud_bins_masks(stimuli_stft['desired_dry'],
                                                                                         max_relative_difference=sett[
                                                                                       'max_relative_difference_loud_bins'],
                                                                                         print_log=False)

        data_to_evaluate_errors = (rtfs_estimates_raw_all_realizations, rtf_targets_all_realizations,
                                   loud_bins_masks_all_realizations)

        return atf_target, cm, data_to_evaluate_errors

    @staticmethod
    def evaluate_errors_single_variation(rtfs_estimates_raw_all_realizations, rtf_targets_all_realizations,
                                         loud_bins_masks_all_realizations, metric_names, algo_names):

        errors_all_realizations = []
        rtf_evaluator = None
        for rtf_target, rtfs_estimates_raw, loud_bin_mask in zip(rtf_targets_all_realizations,
                                                                 rtfs_estimates_raw_all_realizations,
                                                                 loud_bins_masks_all_realizations):
            rtfs_estimates = RtfEvaluator.clean_up_rtf_estimates(rtfs_estimates_raw, clip_estimates=True)
            rtf_evaluator = RtfEvaluator(ground_truth=rtf_target, estimates=rtfs_estimates,
                                         metric_names=metric_names)
            if loud_bin_mask is not None:
                rtf_evaluator.loud_bins_mask = loud_bin_mask

            err_single_realization = rtf_evaluator.evaluate_errors_single_realization(plot_rtf_realization=False,
                                                                                      print_realization_error=False)

            errors_all_realizations.append(err_single_realization)

        err_averager = RtfErrorAverager(errors_all_realizations, algo_names, metric_names)

        err_array = err_averager.build_error_array_from_dict(err_averager.err_list)
        err_mean_std = err_averager.compute_error_mean_std_over_realizations(err_array)
        err_mean_std_db = err_averager.convert_errors_to_db(err_mean_std)
        err_mean_std_db = err_averager.clean_up_errors_array(err_mean_std_db)

        return err_mean_std_db, rtf_evaluator


class SettingsManager:
    """ Class to manage the settings of the experiment. """

    def __init__(self):
        pass

    @staticmethod
    def get_variation_key_values(original_settings):
        """
        Returns the key and values of the varying factor. For example, if the varying factor is
        'num_mics', this function returns 'num_mics' and a list of values, such as [2, 4, 8].
        """
        variation_key = '-'.join([str(x) for x in original_settings['varying_factors']])
        try:
            variation_values = u.get_by_path(original_settings, original_settings['varying_factors'])
        except KeyError:
            if variation_key == '' or variation_key is None or variation_key == 'None':
                return '', [0]
            else:
                # if above did not work, try deleting the '0', which is there to access first element of list.
                varying_factor = [x for x in original_settings['varying_factors'] if x != 0]
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
        settings['noverlap_percentage'] = float(settings.get('noverlap_percentage', 0))
        settings['rtf_type'] = settings.get('rtf_type', 'real')
        settings['num_nonzero_samples_rir'] = int(float(settings.get('num_nonzero_samples_rir', -1)))
        settings['max_relative_difference_loud_bins'] = int(float(settings.get('max_relative_difference_loud_bins',
                                                                               g.max_relative_difference_loud_bins_default)))

        settings['needs_warmup_run'] = settings.get('needs_warmup_run', True)
        settings['correlation_noise_type'] = settings.get('correlation_noise_type', '')
        settings['correlation_target_type'] = settings.get('correlation_target_type', '')
        settings['num_neighbours'] = settings.get('num_neighbours', np.inf)
        settings['checkerboard_spacing'] = settings.get('checkerboard_spacing', np.inf)
        settings['covariance_target_type'] = settings.get('covariance_target_type', 'equicorrelated')

        return settings

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


# def print_errors_summary(settings, errors_array):
#     variation_factor_name, var_fact_values = SettingsManager.get_variation_key_values(settings)
#     for metric_idx, metric_name in enumerate(settings['metric_names']):
#         for var_fact_idx, var_fact_value in enumerate(var_fact_values):
#             print(f"{variation_factor_name} = {var_fact_value}")
#             for algo_idx, algo in enumerate(settings['algo_names']):
#                 mean = errors_array[var_fact_idx, algo_idx, metric_idx, 0]
#                 std = errors_array[var_fact_idx, algo_idx, metric_idx, 1]
#                 report = f"{metric_name} {algo} {mean:.3f} ± {std:.3f}"
#                 print(report)
#             print()


def print_errors_table_from_settings(settings, errors_array):
    if not errors_array.any():
        return

    variation_factor_name, var_fact_values = SettingsManager.get_variation_key_values(settings)
    metric_names = settings['metric_names']
    algo_names = settings['algo_names']
    print(f"Varying {variation_factor_name}")

    print_errors_table(errors_array, var_fact_values, algo_names, metric_names)


def print_errors_table(errors_array, var_fact_values, algo_names, metric_names):
    print("----------")
    for metric_idx, metric_name in enumerate(metric_names):
        # reorder elements
        var_fact_values = [float(ele) for ele in var_fact_values]
        order = np.argsort(var_fact_values)[:len(errors_array)]
        var_fact_values = np.asarray(var_fact_values)[order]
        errors_array = errors_array[order]

        tab = PrettyTable()
        tab.add_column("Algorithms", algo_names)
        for var_fact_idx, var_fact_value in enumerate(var_fact_values):
            tab.add_column(f"{var_fact_value:2.2f}", list(errors_array[var_fact_idx, :, metric_idx, 0]))
            # std = errors_array[var_fact_idx, algo_idx, metric_idx, 1]
        tab.float_format = '.3'
        print(tab)
