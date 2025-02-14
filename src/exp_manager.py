import copy
import warnings

import numpy as np
from prettytable import PrettyTable

# from src import atf_plotter
import src.cov_manager as cov_manager
import src.global_constants as g
import src.rtf_estimator as rtf_estimator
import src.signal_generator as signal_generator
import src.utils as u
from src.exp_data import ExpData
from src.settings_manager import SettingsManager
from src.error_evaluator import ErrorEvaluator
import src.cov_generator as covariance_generator
from src.bound_manager import BoundManager as bm

sh = None
f0_harmonics_bins = None


class ExperimentManager:
    def __init__(self, **kwargs):
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
        assert (self.avg_time_frames or self.duration_output_sec < 20 or self.duration_output_frames < 2000)

    @staticmethod
    def initialize_rtf_estimator(rtf_estimation_algorithms, rtfs_ground_truth_stft=None, exp_settings=None):

        if exp_settings is None:
            exp_settings = {}

        re = rtf_estimator.RtfEstimator(g.idx_ref_mic)
        re.rtfs_gt = rtfs_ground_truth_stft
        re.methods = rtf_estimation_algorithms

        re.flag_estimate_signal_threshold = exp_settings.get('flag_estimate_signal_threshold', False)
        re.flag_scree_method = exp_settings.get('flag_scree_method', False)
        re.flag_keep_num_freqs_eigenvectors = exp_settings.get('flag_keep_num_freqs_eigenvectors', False)
        re.flag_mdl_criterion = exp_settings.get('flag_mdl_criterion', False)

        return re

    @classmethod
    def run_experiment_single_variation(cls, sett=None, variation_factor_value=None, atf_target=None,
                                        variances_noise=None,
                                        variances_target=None, variation_factor_key=None):

        force_new_atf = False
        cov_mask_target = None
        processed_freq_mask = None
        alpha_cov_estimation = 0  # covariance smoothing only for real speech
        SettingsManager.validate(sett)
        rtfs_estimates_raw_all_realizations = []
        loud_bins_masks_all_realizations = []
        processed_bins_masks_all_realizations = []
        rtf_targets_all_realizations = []
        rtf_nb_all_freqs_all_realizations = []
        stimuli_realizations = []
        covariance_managers_realizations = []
        exp_manager = ExperimentManager(**sett)
        exp_manager.sig_generator = signal_generator.SignalGenerator(**sett)
        exp_manager.sig_generator.noises_info = exp_manager.noises_info
        sg = exp_manager.sig_generator

        cm_no_phase_corr = None
        stimuli_stft_no_corr = None
        results_without_phase_correction = any(['orig-phase' in x for x in sett['algo_names']])
        if results_without_phase_correction:
            cm_no_phase_corr = cov_manager.CovarianceManager(sett["add_identity_noise_noisy"])

        cg = covariance_generator.CovarianceGenerator()
        sett['correlation_noise_type'] = SettingsManager.filter_correlation_type(sett['correlation_noise_type'])
        sett['correlation_target_type'] = SettingsManager.filter_correlation_type(sett['correlation_target_type'])
        rtf_type = sett['rir_settings']['rtf_type']

        new_random_rtf_every_realization = False
        if 'real' not in rtf_type and 'debug' not in rtf_type and 'once' not in rtf_type:
            new_random_rtf_every_realization = True

        if not sett['generate_single_frame_many_realizations'] and not sett['avg_time_frames']:
            alpha_cov_estimation = sett['alpha_cov_estimation']

        num_bins_orig = sg.stft_shape[1]
        processed_freq_range_hz = sett['processed_freq_range_hz']

        # RTF will only be estimated for the frequencies in the range (k_freq_min, k_freq_max)
        is_speech_signal = not sett['gen_signals_freq_domain']
        if is_speech_signal:
            processed_freq_mask = u.compute_mask_frequency_range(processed_freq_range_hz, sg.nstft)

        num_repeated_experiments = sett['num_repeated_experiments'] \
            if not g.debug_mode else min(sett['num_repeated_experiments'], 2)

        for ii in range(num_repeated_experiments):
            cm = cov_manager.CovarianceManager(sett["add_identity_noise_noisy"])
            cm_narrowband = cov_manager.CovarianceManager(sett["add_identity_noise_noisy"])
            print_log_realization = ii % 500 == 0 if not g.debug_mode else True
            if print_log_realization:
                print(f"{variation_factor_key} {variation_factor_value}, real. {ii + 1}/{num_repeated_experiments}")

            flag_calculate_crb = np.any([u.is_crb(x) for x in sett['algo_names']]) and (
                    (new_random_rtf_every_realization and ii < max(10, num_repeated_experiments // 1000)) or
                    (not new_random_rtf_every_realization and ii < 3))

            # check that ATF has correct number of frequency bins
            if atf_target is not None and atf_target.shape[1] != sg.stft_shape[1]:
                force_new_atf = True
                warnings.warn(f"Cannot reuse ATF, because the number of frequency bins don't match: "
                              f"{atf_target.shape[1]} != {sg.stft_shape[1]}. Generating new ATF.")

            # generate new atf/rtf unless mode is 'random-once' (for Cramer-Rao bound simulations)
            if atf_target is None or force_new_atf:
                if ii == 0:
                    print(f"{ rtf_type = }")
                atf_target = sg.rir_manager.compute_or_generate_acoustic_transfer_function(rtf_type)
                force_new_atf = False

            if '' in sett['varying_factors']:
                warnings.warn("No varying factors. Probably warm-up run or debugging. Results will be discarded.")
                return atf_target, (variances_target, variances_noise)

            rtf_target = sg.rir_manager.generate_rtf_from_atf(atf_target)
            rtf_estimator_ = ExperimentManager.initialize_rtf_estimator(sett['algo_names'], rtf_target, sett)
            rtf_estimator_nb = ExperimentManager.initialize_rtf_estimator(['CW'])

            if sett['correlation_target_pattern'] != 'equicorrelated':
                cov_mask_target = cg.generate_covariance_mask(sg.stft_shape[1],
                                                              sett['correlation_target_pattern'],
                                                              num_neighbours=sett['num_neighbours_target'],
                                                              grid_spacing=sett['grid_spacing_target'])

            if sett['gen_signals_freq_domain']:
                cm.cov_dry_oracle_ref_mic, cm.cov_dry_oracle, cm.cov_wet_gt, cm.cov_noise_gt = \
                    cg.generate_covariance_matrices_freq_domain(variances_noise, variances_target,
                                                                atf_target=atf_target, stft_shape=sg.stft_shape,
                                                                sett=sett, cov_mask_target=cov_mask_target)

                stimuli_stft = \
                    sg.generate_gaussian_signals_from_covariance_matrices(cm.cov_dry_oracle_ref_mic,
                                                                          cm.cov_noise_gt,
                                                                          atf_target, sg.stft_shape)
                stimuli_samples = sg.stimuli_samples_from_stft(stimuli_stft)

            else:
                """ Load/generate the target and noise signals in time domain."""
                if print_log_realization:
                    print(f"RTF estimation methods: {rtf_estimator_.methods}.")
                    if 'random' in rtf_type or 'deterministic' in rtf_type:
                        print("Multiplication ATF-target signal in frequency domain")
                    else:
                        print("Convolution RIR-target signal in time domain")

                stimuli_samples, stimuli_stft = sg.generate_signal_samples(atf_target)

            if sg.noise_estimate_perturbation_amount > 0:
                stimuli_stft['mix'] = sg.perturb_noise_stft_with_wgn(sg.noise_estimate_perturbation_amount,
                                                                     stimuli_stft['mix'])

            """normalize volume to avoid numerical issues. first find maximum value across elements"""
            # max_value = np.max([np.max(np.abs(stimuli_stft[key])) for key in stimuli_stft.keys()])
            # for key in stimuli_stft.keys():
            #     stimuli_stft[key] = 1e-1 * stimuli_stft[key] / max_value

            # stimuli_stft_all_freqs_no_corr = copy.deepcopy(stimuli_stft)
            # stimuli_samples_original = copy.deepcopy(stimuli_samples)

            # only use for performance bounds - needed to estimate DRY covariance if oracle dry does not exist
            desired_dry_stft_ref = None
            if sett.get('plot_correlation_histogram', False) or (
                     any('CRB' in x for x in sett['algo_names']) and cm.cov_dry_oracle is None):
                desired_dry_stft_ref = stimuli_stft['desired_dry']

            len_seconds = sg.frames_to_samples(stimuli_stft['mix'].shape[-1], sg.nstft, sett['noverlap_percentage']) / g.fs
            if print_log_realization:
                print(f"Estimating covariances from {int(stimuli_stft['mix'].shape[-1])} frames (={len_seconds:.2f}s)")
                if cm.add_identity_noise_noisy:
                    print("Adding identity matrix to noise and noisy CPSD matrices")

            stimuli_stft_all_freqs_no_corr = copy.deepcopy(stimuli_stft)
            stimuli_stft_all_freqs_phase_corr = copy.deepcopy(stimuli_stft)
            stimuli_samples_original = copy.deepcopy(stimuli_samples)
            rtf_target_all_freqs = copy.deepcopy(rtf_target)
            if is_speech_signal:

                # find element in stimuli_stft stft with the largest number of frames and compute phase correction
                key_max_frames = max(stimuli_stft, key=lambda x: stimuli_stft[x].shape[-1])
                phase_correction = cm.compute_phase_correction_stft(stimuli_stft[key_max_frames].shape, sg.noverlap)

                stimuli_stft_all_freqs_phase_corr = sg.apply_phase_correction(stimuli_stft_all_freqs_phase_corr,
                                                                                    phase_correction)

                # Filter all data so that only freq range of interest is processed (mainly to save computation time)
                if processed_freq_mask is not None:
                    for key, stimulus in stimuli_stft.items():
                        stimuli_stft[key] = sg.apply_bin_mask(stimulus, processed_freq_mask, num_bins_orig)
                    phase_correction = sg.apply_bin_mask(phase_correction[np.newaxis], processed_freq_mask,
                                                         num_bins_orig)
                    rtf_target = sg.apply_bin_mask(rtf_target, processed_freq_mask, num_bins_orig)
                    cov_mask_target = sg.apply_bin_mask(cov_mask_target, processed_freq_mask, num_bins_orig)
                    desired_dry_stft_ref = sg.apply_bin_mask(desired_dry_stft_ref, processed_freq_mask, num_bins_orig)

                stimuli_stft_no_corr = copy.deepcopy(stimuli_stft)
                stimuli_stft = sg.apply_phase_correction(stimuli_stft, phase_correction)
                # stimuli_stft = sg.remove_mean(stimuli_stft)
                # stimuli_stft, vad_dict = sg.filter_stimuli_stft_oracle_vad(stimuli_stft, silence_threshold=5e-6)

            cm.estimate_all_covariances(stimuli_stft, exp_manager.avg_time_frames, alpha_cov_estimation, None,
                                        desired_dry_stft_ref, sett['use_true_noise_covariance'])

            if is_speech_signal:
                cm_narrowband.estimate_all_covariances_narrowband(stimuli_stft_all_freqs_phase_corr,
                                                                  sett['use_true_noise_covariance'])

            # debug, plot estimated and "true" covariance matrices
            # u.plot_matrix(cm.cov_wet_gevd, amp_range=(-120, -0), freq_range_hz=(0, 1500), stft_shape=stimuli_stft['wet'].shape)
            # if ii == 0:
            #     ExperimentManager.debug_plots(cm, sett['out_dir_name'], variation_factor_key, variation_factor_value)

            rtfs_estimates_raw = rtf_estimator_.estimate_rtf(cm, sg.stft_shape, sett['num_retained_eigva'],
                                                             covariance_mask_target=cov_mask_target)
            if is_speech_signal:
                rtfs_estimates_nb = rtf_estimator_nb.estimate_rtf(cm_narrowband, sg.stft_shape)
                rtfs_estimates_nb['Ideal'] = copy.deepcopy(rtf_target_all_freqs[..., np.newaxis])

                if 'CW' in rtfs_estimates_raw and 'CW' in rtfs_estimates_nb and \
                        not np.allclose(rtfs_estimates_raw['CW'], rtfs_estimates_nb['CW'][:, processed_freq_mask]):
                    if not np.allclose(cm.nb_cov_noisy, cm_narrowband.nb_cov_noisy):
                        raise ValueError("Narrowband, noisy covariance matrices differ between narrowband and wideband covariance estimators.")
                    if not np.allclose(cm.nb_cov_noise, cm_narrowband.nb_cov_noise):
                        raise ValueError("Noise covariance matrices differ between narrowband and wideband covariance estimators.")
                    raise ValueError("CW estimates differ between narrowband and wideband covariance estimators.")
            else:
                rtfs_estimates_nb = np.array([])

            if results_without_phase_correction:
                cm_no_phase_corr.estimate_all_covariances(stimuli_stft_no_corr, exp_manager.avg_time_frames,
                                                          alpha_cov_estimation, None, desired_dry_stft_ref,
                                                          sett['use_true_noise_covariance'])
                rtfs_estimates_raw_2 = rtf_estimator_.estimate_rtf(cm_no_phase_corr, sg.stft_shape,
                                                                   sett['num_retained_eigva'],
                                                                   covariance_mask_target=cov_mask_target)

                # Append "-no-corr" to keys in rtf_estimates_raw_2
                rtfs_estimates_raw_2 = {key + "-orig-phase": value for key, value in rtfs_estimates_raw_2.items()}
                rtfs_estimates_raw = {**rtfs_estimates_raw, **rtfs_estimates_raw_2}

            if flag_calculate_crb:
                crb = bm.calculate_bounds(sett['algo_names'], atf_target, cm, sg.stft_shape, stimuli_stft['desired_dry'])
                rtfs_estimates_raw.update(crb)

            if not rtfs_estimates_raw:  # empty result dictionary
                break

            # New ATF should be generated for next Montecarlo realization, unless we are calculating CRB.
            # In this case, we want to use the same ATF for all realizations.
            if new_random_rtf_every_realization:
                if any([u.is_crb(algo) for algo in sett['algo_names']]):
                    warnings.warn('ATF should be generated only once, because CRB is being calculated. ')
                force_new_atf = True

            # Loud bins mask is computed for SELECTED FREQUENCIES (e.g. only 100-1500Hz, ie 46 bins out of 257)
            # LOWER max_relative_difference means LESS bins are evaluated
            loud_bins_mask = ErrorEvaluator.find_loud_bins_masks(stimuli_stft['wet'],
                                                                 sett['max_relative_difference_loud_bins'],
                                                                 print_log=False)

            if print_log_realization:
                rtfs_estimates_clean = ErrorEvaluator.clean_up_rtf_estimates(rtfs_estimates_raw, clip_estimates=True)
                rtf_evaluator = ErrorEvaluator(rtf_target, rtfs_estimates_clean, sett['rtf_metrics'], loud_bins_mask)
                _, err = rtf_evaluator.evaluate_errors_single_realization(plot_rtf_realization=False)

                # if g.debug_mode and g.debug_show_plots:
                #     plot_manager.plot_hermitian_angle_per_frequency_and_psd(stimuli_stft, loud_bins_mask,
                #                                                             processed_freq_mask, err, sg.nstft)

            loud_bins_masks_all_realizations.append(loud_bins_mask)
            processed_bins_masks_all_realizations.append(processed_freq_mask)
            stimuli_realizations.append({'stft': stimuli_stft_all_freqs_no_corr, 'samples': stimuli_samples_original})
            covariance_managers_realizations.append(cm)
            # rtfs_estimates_raw_all_realizations.append(rtfs_estimates_clean)
            rtfs_estimates_raw_all_realizations.append(rtfs_estimates_raw)
            rtf_targets_all_realizations.append(rtf_target)
            if is_speech_signal:
                rtf_nb_all_freqs_all_realizations.append(rtfs_estimates_nb)
            # u.plot_matrix(cm.cov_dry_oracle, amp_range=(-120, -0))

        if sett.get('plot_correlation_histogram', False):
            cls.plot_correlation_histogram(cm, covariance_managers_realizations, len_seconds,
                                           loud_bins_masks_all_realizations, sett, sg)

        exp_data = ExpData(rtfs_estimates_raw_all_realizations, rtf_targets_all_realizations,
                           rtf_nb_all_freqs_all_realizations,
                           loud_bins_masks_all_realizations, covariance_managers_realizations, stimuli_realizations,
                           processed_bins_masks_all_realizations)

        return exp_data

    @staticmethod
    def plot_correlation_histogram(cm, covariance_managers_realizations, len_seconds,
                                   loud_bins_masks_all_realizations, sett, sg):
        covs = []
        for cm, mask in zip(covariance_managers_realizations, loud_bins_masks_all_realizations):
            # Mask is 1D and of size cov_dr_oracle.shape[0]/num_mics_max
            # Make it 2D (and symmetric) to remove columns and rows which correspond to non-loud bins
            mask_2d = np.bool_(np.kron(np.outer(mask, mask), np.ones((sg.num_mics_max, sg.num_mics_max))))
            new_shape = np.sum(mask) * sg.num_mics_max
            cov_filtered = cm.cov_dry_oracle[..., 0][mask_2d].reshape((new_shape, new_shape), order='F')
            covs.append(cov_filtered)
            # covs.append(cm.cov_dry_oracle[..., 0])
        title = 'White noise' if 'white' in sett['desired'] else 'Speech'
        cm.estimate_correlation_all_covariances(covs, sg.num_mics_max, name='Dry', duration_seconds=len_seconds,
                                                title=title)

    @staticmethod
    def debug_plots(cm, out_dir_name, variation_factor_key, variation_factor_value, force_show=False):

        amp_range = (np.maximum(-160, np.min(u.log_pow(cm.cov_noisy))), np.max(u.log_pow(cm.cov_noisy)))  # (-50, 30)
        if g.debug_mode or force_show:
            if g.debug_show_plots or force_show:
                cm.plot_cov(amp_range=amp_range, show_single_mic=True, )
                cm.plot_cov(true_cov=True, amp_range=amp_range, f0_harmonics_bins=f0_harmonics_bins,
                            show_single_mic=False)

        elif g.release_save_plots:
            f1 = cm.plot_cov(amp_range=amp_range, show_plot=False)
            f2 = cm.plot_cov(true_cov=True, amp_range=amp_range, f0_harmonics_bins=f0_harmonics_bins,
                             show_plot=False, show_single_mic=False)
            name_appendix = f"_cov_{variation_factor_key}={variation_factor_value}"
            out_dir = u.check_create_folder(out_dir_name, 'cov_plots')
            u.save_figure(f1, "Emp" + name_appendix, out_dir_name=out_dir)
            u.save_figure(f2, "EmpGt" + name_appendix, out_dir_name=out_dir)

    @staticmethod
    async def debug_plots_async(cm, out_dir_name, variation_factor_key, variation_factor_value):
        ExperimentManager.debug_plots(cm, out_dir_name, variation_factor_key, variation_factor_value)


def print_errors_table_from_settings(settings, errors_array):
    if not errors_array.any():
        return

    variation_factor_name, var_fact_values = SettingsManager.get_variation_key_values(settings)
    rtf_metrics = settings['rtf_metrics']
    algo_names = settings['algo_names']
    print(f"Varying {variation_factor_name}")

    print_errors_table(errors_array, var_fact_values, algo_names, rtf_metrics)


def print_errors_table(errors_array, var_fact_values, algo_names, rtf_metrics):
    print("----------")
    for metric_idx, metric_name in enumerate(rtf_metrics):
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

