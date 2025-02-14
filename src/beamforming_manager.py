import copy
import warnings

import numpy as np
import scipy
from matplotlib import pyplot as plt

import src.global_constants as g
import src.utils as u
import src.signal_generator as signal_gen
from src.error_evaluator import ErrorEvaluator
from src.exp_data import ExpData


class BeamformingManager:
    """ Class for managing beamforming operations."""

    def __init__(self, exp_data_realization: ExpData):
        """ Initialize BeamformingManager with data from a single realization. """

        self.rtfs_estimates = copy.deepcopy(exp_data_realization.rtf_estimates)
        self.rtfs_estimates['Ideal'] = exp_data_realization.rtf_targets[..., np.newaxis]
        self.selected_bins_mask = exp_data_realization.selected_bins_mask
        self.loud_bins_mask = exp_data_realization.loud_bins_masks
        self.cov_holder = exp_data_realization.cov_managers

        self.stimuli_stft = copy.deepcopy(exp_data_realization.stimuli['stft'])

        self.cov_holder.estimate_all_covariances_narrowband(self.stimuli_stft)

        num_freqs_covariance = self.cov_holder.nb_cov_noise.shape[2]
        num_freqs_selected = np.sum(self.selected_bins_mask)
        if num_freqs_covariance != num_freqs_selected:
            self.rtfs_estimates = self.replace_estimates_for_missing_or_quiet_freqs(exp_data_realization.rtf_nb_all_freqs,
                                                                                    self.rtfs_estimates,
                                                                                    self.selected_bins_mask, self.loud_bins_mask)

    @staticmethod
    def evaluate_beamformed_data_all_realizations(processed_realizations, algo_names_bf, bf_metrics,
                                                  loud_bins_masks=None):
        """ Evaluate beamforming errors for all realizations. """

        target_realizations = [x['Target'] for x in processed_realizations]
        processed_realizations_no_target = copy.deepcopy(processed_realizations)
        for realization in processed_realizations_no_target:
            realization.pop('Target')

        beamforming_errors_dict = {metric_name: [] for metric_name in bf_metrics}
        beamforming_evaluators_dict = {metric_name: [] for metric_name in bf_metrics}

        for beamforming_metric in bf_metrics:
            err, bf_eval = (
                ErrorEvaluator.evaluate_errors_single_variation(estimates_realizations=processed_realizations_no_target,
                                                                ground_truth_realizations=target_realizations,
                                                                err_metrics=[beamforming_metric],
                                                                algo_names=algo_names_bf,
                                                                needs_cleanup=False,
                                                                loud_bins_masks_all_realizations=loud_bins_masks))

            beamforming_errors_dict[beamforming_metric] = err
            beamforming_evaluators_dict[beamforming_metric] = bf_eval

        return beamforming_errors_dict, beamforming_evaluators_dict

    @staticmethod
    def select_bins_stimuli_stft(stimuli_original, selected_bins_mask):
        stimuli_stft_filtered = copy.deepcopy(stimuli_original)
        for key, stimulus in stimuli_original.items():
            stimuli_stft_filtered[key] = stimuli_stft_filtered[key][..., selected_bins_mask, :]
            assert stimuli_original[key].ndim == 3
            assert stimuli_stft_filtered[key].ndim == 3
        return stimuli_stft_filtered

    def mvdr_calculate_weights(self, rtf, noisy_cov):
        """ A consolidated perspective... eq. (43)"""
        num_mics, num_freqs, _ = rtf.shape
        weights = np.zeros_like(rtf)

        # k0 = 20  # maximum allowed condition number of noise covariance matrix
        # k0 = 200  # maximum allowed condition number of noise covariance matrix
        # k0 = 200  # maximum allowed condition number of noise covariance matrix
        # loading_factor = 1e-8
        for kk in range(num_freqs):
            # COMPACT NOISE COVARIANCE MATRIX MODEL FOR MVDR BEAMFORMING
            # eigenvals = np.linalg.eigvalsh(noisy_cov[..., kk, 0])
            # loading_factor = max((np.max(eigenvals) - k0 * np.min(eigenvals)) / (k0 - 1), loading_eps)
            # rx_inv_rtf = scipy.linalg.solve(noisy_cov[..., kk, 0] + loading_factor * np.identity(num_mics), rtf[:, kk],
            #                                 assume_a='pos')
            # weights[:, kk] = rx_inv_rtf / (rtf[:, kk].conj().T @ rx_inv_rtf)

            # weights[:, kk, 0] = scipy.linalg.solve(cov_wet[:, :, kk, 0] + mu * cov_noise[:, :, kk, 0],
            #                                         cov_wet[:, 0, kk, 0], assume_a='pos')
            cov_nb_kk_inv_rtf = scipy.linalg.solve(noisy_cov[:, :, kk, 0], rtf[:, kk], assume_a='pos')
            weights[:, kk] = cov_nb_kk_inv_rtf / (rtf[:, kk].conj().T @ cov_nb_kk_inv_rtf)

        return weights

    def sdw_mwf_calculate_weights(self, rtf, cov_wet, cov_noise, mu=1):
        """ A consolidated perspective... eq. (42)"""
        num_mics, num_freqs, _ = rtf.shape
        weights = np.zeros_like(rtf)
        weights_mvdr = self.mvdr_calculate_weights(rtf, cov_noise)

        for kk in range(num_freqs):
            cov_noise_nb_kk_inv_rtf = scipy.linalg.solve(cov_noise[:, :, kk, 0], rtf[:, kk], assume_a='pos')
            noise_remaining_pow = 1. / np.real((rtf[:, kk].conj().T @ cov_noise_nb_kk_inv_rtf))
            var_early = np.real(cov_wet[g.idx_ref_mic, g.idx_ref_mic, kk, 0])
            wf_scaling = np.real(var_early / (var_early + float(noise_remaining_pow)))
            weights[:, kk] = weights_mvdr[:, kk] * np.clip(wf_scaling, a_min=1e-1, a_max=1)

        return weights

    def sdw_mwf_rank_one_calculate_weights(self, rtf, noise_cov, clean_cov, mu=1):
        """ Performance Analysis of Multichannel Wiener Filter-Based Noise Reduction in Hearing Aids Under
        Second Order Statistics Estimation Errors, equation (10)"""

        num_mics, num_freqs = noise_cov.shape[0], noise_cov.shape[2]
        weights = np.zeros((num_mics, num_freqs), dtype=complex)

        unit_vector = np.zeros((num_mics,))
        unit_vector[g.idx_ref_mic] = 1

        for kk in range(num_freqs):
            rv_inv_rtf = scipy.linalg.solve(noise_cov[..., kk, 0] + 1e-6 * np.identity(num_mics),
                                            rtf[:, kk], assume_a='pos')
            Ps = clean_cov[g.idx_ref_mic, g.idx_ref_mic, kk, 0]
            A_ref_star = rtf[:, kk].conj().T @ unit_vector
            rho = Ps * rtf[:, kk].conj().T @ rv_inv_rtf
            weights[:, kk] = np.squeeze(rv_inv_rtf * (Ps * A_ref_star) / (mu + rho))

        return weights

    def apply_beamformers(self, weights_dict):

        filtered_signals_dict = {key: np.array([]) for key in weights_dict.keys()}
        for bf_name, bf_weights in weights_dict.items():

            num_mics, num_freqs, num_frames = self.stimuli_stft['mix'].shape
            filtered_signal_arr = np.zeros((num_freqs, num_frames), dtype=complex)
            for kk in range(num_freqs):
                filtered_signal_arr[kk, :] = bf_weights[:, kk].conj().T @ self.stimuli_stft['mix'][..., kk, :]

            filtered_signals_dict[bf_name] = filtered_signal_arr[np.newaxis, ...]

        return filtered_signals_dict

    def calculate_beamformer_weights(self, beamforming_algorithm_name):
        """ Calculate beamformer weights for all RTFs in rtfs_estimates. """

        weights_dict = {}
        num_mics, num_freqs, _ = self.rtfs_estimates['Ideal'].shape
        wet_matrix_not_provided = (self.cov_holder.nb_cov_wet_oracle is None or
                                   np.allclose(self.cov_holder.nb_cov_wet_oracle, 0))

        if 'sdw_mwf' in beamforming_algorithm_name and wet_matrix_not_provided:
            warnings.warn(
                f"{beamforming_algorithm_name} beamformer requires a non-zero cov_wet_oracle.Using mvdr instead")
            beamforming_algorithm_name = 'mvdr'

        if beamforming_algorithm_name == 'mvdr':
            for rtf_estimate_name, rtf_estimate in self.rtfs_estimates.items():
                if not u.is_crb(rtf_estimate_name):
                    weights_dict[rtf_estimate_name] = self.mvdr_calculate_weights(rtf_estimate,
                                                                                  # self.cov_holder.nb_cov_noisy)
                                                                                  self.cov_holder.nb_cov_noise)
        elif beamforming_algorithm_name == 'sdw_mwf':
            mu = 1
            for rtf_estimate_name, rtf_estimate in self.rtfs_estimates.items():
                if not u.is_crb(rtf_estimate_name):
                    weights_dict[rtf_estimate_name] = self.sdw_mwf_calculate_weights(rtf_estimate,
                                                                                     self.cov_holder.nb_cov_wet_oracle,
                                                                                     self.cov_holder.nb_cov_noise,
                                                                                     mu=mu)
        elif beamforming_algorithm_name == 'sdw_mwf_r1':
            mu = 1
            for rtf_estimate_name, rtf_estimate in self.rtfs_estimates.items():
                if not u.is_crb(rtf_estimate_name):
                    weights_dict[rtf_estimate_name] = self.sdw_mwf_rank_one_calculate_weights(rtf=rtf_estimate,
                                                                                              noise_cov=self.cov_holder.nb_cov_noise,
                                                                                              clean_cov=self.cov_holder.nb_cov_wet_oracle,
                                                                                              mu=mu)

        elif beamforming_algorithm_name == 'mean':
            for rtf_estimate_name, rtf_estimate in self.rtfs_estimates.items():
                if not u.is_crb(rtf_estimate_name):
                    weights_dict[rtf_estimate_name] = np.ones((num_mics, num_freqs)) / num_mics

        elif beamforming_algorithm_name == 'ref_mic':
            for rtf_estimate_name, rtf_estimate in self.rtfs_estimates.items():
                if not u.is_crb(rtf_estimate_name):
                    weights_dict[rtf_estimate_name] = np.zeros((num_mics, num_freqs))
                    weights_dict[rtf_estimate_name][g.idx_ref_mic] = 1

        else:
            raise NotImplementedError(f"Unknown beamforming algorithm {beamforming_algorithm_name}")

        # Keeps reference microphone only
        weights_dict['Unprocessed'] = np.zeros_like(weights_dict['Ideal'])
        weights_dict['Unprocessed'][g.idx_ref_mic] = 1

        return weights_dict

    @staticmethod
    def run_beamforming_all_realizations(exp_data: ExpData, sett):
        # Compute output of beamforming algorithm for all realizations
        sg = signal_gen.SignalGenerator(**sett)
        processed_stft_realizations = []
        processed_samples_realizations = []
        # a_masked_all = []
        play = False

        exp_data_realizations_list = [ExpData(*x) for x in zip(*exp_data.get_all_fields())]

        for idx, exp_data_realization in enumerate(exp_data_realizations_list):

            bm = BeamformingManager(exp_data_realization)

            bf_weights_dict = bm.calculate_beamformer_weights(sett['beamforming_algorithm'])
            processed_stft = bm.apply_beamformers(bf_weights_dict)
            processed_stft['Target'] = bm.stimuli_stft['early']

            if processed_stft['Target'].shape[1] == exp_data_realization.selected_bins_mask.shape[0]:
                processed_samples = sg.stimuli_samples_from_stft(processed_stft)
            else:
                # Add zeros where the frequencies have been filtered out (selected_bins_mask)
                num_bins_total = exp_data_realization.selected_bins_mask.shape[0]
                num_frames = processed_stft['Target'].shape[-1]
                processed_stft_padded = {key: np.zeros((1, num_bins_total, num_frames), complex) for key in
                                         processed_stft.keys()}
                for key, value in processed_stft.items():
                    processed_stft_padded[key][:, exp_data_realization.selected_bins_mask, :] = value
                processed_samples = sg.stimuli_samples_from_stft(processed_stft_padded)

            if play:
                u.play(processed_samples['Target'], g.fs)
                u.play(processed_samples['CW'], g.fs)
                u.play(processed_samples['CW-SV'], g.fs)
                u.play(processed_samples['Ideal'], g.fs)
                u.play(processed_samples['Unprocessed'], g.fs)

                amp_range = (-50, 50)
                u.plot_matrix(processed_stft['Target'], title='Target', amp_range=amp_range)
                u.plot_matrix(processed_stft['Ideal'], title='Ideal', amp_range=amp_range)
                u.plot_matrix(processed_stft['CW'], title='CW', amp_range=amp_range)
                u.plot_matrix(processed_stft['CW-SV'], title='CW-SV', amp_range=amp_range)
                u.plot_matrix(processed_stft['Unprocessed'], title='Unprocessed', amp_range=amp_range)

            processed_samples_realizations.append(processed_samples)
            processed_stft_realizations.append(processed_stft)

            # if g.debug_mode and idx == 0:  # Plot beampattern
            #     selected_bins = exp_data_realization[6]
            #     Beampattern.plot_beampatterns(selected_bins, bf_weights_dict, sett, sg.rir_manager.inter_mic_distance)

        # Normalize
        for idx in range(len(processed_stft_realizations)):
            for key, value in processed_stft_realizations[idx].items():
                processed_stft_realizations[idx][key] = value / np.max(np.abs(value))

        for idx in range(len(processed_samples_realizations)):
            for key, value in processed_samples_realizations[idx].items():
                processed_samples_realizations[idx][key] = value / np.max(np.abs(value))

        # a_masked = {}
        # for idx_realization, loud_bin_mask in enumerate(exp_data.loud_bins_masks):
        #     for key, value in processed_stft_realizations[idx_realization].items():
        #         # a_masked[key] = round(np.mean(np.abs(value[:, loud_bin_mask] - processed_stft_realizations[idx_realization]['Target'][:, loud_bin_mask])), 5)
        #         a_masked[key] = round(np.mean(np.abs(value - processed_stft_realizations[idx_realization]['Target']) ** 2), 5)
        #     a_masked_all.append(a_masked)
        #
        # a_masked_avg = {}
        # for key in a_masked.keys():
        #     a_masked_avg[key] = np.mean([x[key] for x in a_masked_all])
        # print(a_masked_avg)

        # u.plot_matrix(processed_stft['Target'][:, :50], title="Target", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['CW'][:, :50], title="CW", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['CW-SV'][:, :50], title="CW-SV", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['Ideal'][:, :50], title="Ideal", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['Unprocessed'][:, :50], title="Unprocessed", amp_range=(-40, 30))
        #
        # u.plot_matrix(processed_stft['Target'][:, :] - processed_stft['CW'][:, :], title="Target - CW", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['Target'][:, :] - processed_stft['CW-SV'][:, :], title="Target - CW-SV", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['Target'][:, :] - processed_stft['Ideal'][:, :], title="Target - Ideal", amp_range=(-40, 30))
        # u.plot_matrix(processed_stft['Target'][:, :] - processed_stft['Unprocessed'][:, :], title="Target - Unprocessed", amp_range=(-40, 30))
        #
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, :] - processed_stft['CW'][:, :]) / np.abs(processed_stft['CW'][:, :])) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, :] - processed_stft['CW-SV'][:, :]) / np.abs(processed_stft['CW-SV'][:, :])) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, :] - processed_stft['Ideal'][:, :]) / np.abs(processed_stft['Ideal'][:, :])) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, :] - processed_stft['Unprocessed'][:, :]) / np.abs(processed_stft['Unprocessed'][:, :])) = }")
        #
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, 1:] - processed_stft['CW'][:, 1:]) ** 2) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, 1:] - processed_stft['CW-SV'][:, 1:]) ** 2) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, 1:] - processed_stft['Ideal'][:, 1:]) ** 2) = }")
        # print(f"{np.sum(np.abs(processed_stft['Target'][:, 1:] - processed_stft['Unprocessed'][:, 1:]) ** 2) = }")

        return processed_samples_realizations, processed_stft_realizations

    @staticmethod
    def replace_estimates_for_missing_or_quiet_freqs(rtf_nb_all_freqs, rtf_estimates, selected_bin_mask, loud_bins_masks):
        # Fill missing values in self.rtf_estimates with the values in exp_data.rtf_nb_all_freqs.
        # Where possible, fill the item corresponding to same key in both dictionaries.
        # If key not present in exp_data.rtf_nb_all_freqs, fill with key 'CW'.

        rtf_all_freqs_new = copy.deepcopy(rtf_estimates)

        selected_and_loud_bins_mask = copy.deepcopy(selected_bin_mask)
        selected_and_loud_bins_mask[selected_bin_mask] = loud_bins_masks

        if 'CW' not in rtf_nb_all_freqs.keys() or 'Ideal' not in rtf_nb_all_freqs.keys():
            raise ValueError('rtf_nb_all_freqs must contain a "CW" and an "Ideal" key.')

        for key in rtf_estimates.keys():
            copy_from = 'Ideal' if key == 'Ideal' else 'CW'
            rtf_all_freqs_new[key] = copy.deepcopy(rtf_nb_all_freqs[copy_from])
            rtf_all_freqs_new[key][:, selected_and_loud_bins_mask] = copy.deepcopy(rtf_estimates[key][:, loud_bins_masks])

        return rtf_all_freqs_new


class Beampattern:
    def __init__(self, _inter_mic_distance, _num_freqs, _fs=g.fs, _selected_freqs=None):

        # Parameters
        self.num_freqs = _num_freqs
        self.num_freqs_real = self.num_freqs // 2 + 1
        self.freqs_real = np.fft.rfftfreq(self.num_freqs, 1 / _fs)[..., np.newaxis]

        if _selected_freqs is not None:
            self.num_freqs_real = np.sum(_selected_freqs)
            self.freqs_real = self.freqs_real[_selected_freqs]

        self.max_degrees = 180

        self.fs = _fs
        self.d = _inter_mic_distance
        self.c = 343
        self.num_spatial_angles = int(self.max_degrees / 1)

        self.spatial_angles = np.atleast_2d(
            np.linspace(np.pi, 2 * np.pi, self.num_spatial_angles))  # [rad] 0 to 180 degrees

    def compute_beampattern(self, W, bin_mask=None):
        """Compute beampattern for a given set of weights"""

        num_freqs_real = self.num_freqs_real if bin_mask is None else np.sum(bin_mask)
        freqs_real = self.freqs_real if bin_mask is None else self.freqs_real[bin_mask]

        D = np.zeros((num_freqs_real, self.num_spatial_angles), dtype='complex128')

        num_mics = W.shape[0]
        assert num_mics % 2 == 1
        mic_idx = np.arange(- (num_mics - 1) / 2, (num_mics - 1) / 2 + 1)
        # mic_idx = np.arange(1, num_mics + 1)

        # Compute beampattern, from Eq. (33) McCowan 2001 "Mic arrays: a tutorial"
        for n, w_n in zip(mic_idx, W):
            # w_n is W for single microphone, all frequencies. n is the index of the microphone.
            coeff = 1j * 2 * np.pi * freqs_real * n * self.d * np.cos(self.spatial_angles) / self.c
            D += w_n * np.exp(coeff)

        return D

    def plot_bf_3d(self, bps):
        fig, axs = plt.subplots(1, 2, squeeze=False, subplot_kw=dict(projection='3d'), figsize=(11, 4))

        x, y = np.meshgrid(np.arange(bps.shape[-2]), np.arange(bps.shape[-1]))

        for ax, bp in zip(axs.flatten(), bps):
            # Plot the surface.
            surf = ax.plot_surface(x, y, u.log_pow(bp.T, thr=1E-3), cmap='plasma',
                                   linewidth=0, rstride=10, cstride=10)

            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel('Arrival angle [deg]')
            ax.set_zlabel('Gain [dB]')

            many_xlabels = np.array([(round(x, -2) / 1000) for x in self.freqs_real.flatten()])
            xlocs = np.linspace(0, x.shape[1] - 1, 5).astype('i8')

            xlabels = many_xlabels[xlocs]
            ax.set_xticks(xlocs)
            ax.set_xticks(np.linspace(0, x.shape[1] - 1, 13).astype('i8'), minor=True)
            ax.set_xticklabels(xlabels)

            ax.set_yticks(np.linspace(0, self.max_degrees, 5))
            ax.set_yticks(np.linspace(0, self.max_degrees, 13), minor=True)

            ax.view_init(40, -130)

        return fig, axs

    @staticmethod
    def plot_beampatterns(selected_bins, bf_weights_dict, sett, inter_mic_distance):
        # do an AND operation. problem is, loud_bins only has values for the selected bins.
        loud_and_selected = copy.deepcopy(selected_bins)
        loud_and_selected[selected_bins] = True
        bp = Beampattern(_num_freqs=sett['nstft'],
                         _inter_mic_distance=inter_mic_distance, _selected_freqs=loud_and_selected)
        beampatterns = {}
        beampatterns_avg = {}  # sum over frequencies

        for key, weights in bf_weights_dict.items():
            beampatterns[key] = bp.compute_beampattern(weights)
            if key != 'Unprocessed':
                beampatterns_avg[key] = np.mean(np.abs(beampatterns[key]), axis=0)

        Beampattern.plot_beampatterns_individual_freqs(beampatterns, sett['processed_freq_range_hz'],
                                                       bp.freqs_real.flatten())
        Beampattern.plot_beampatterns_avg(beampatterns_avg, bp.max_degrees)

    @staticmethod
    def plot_beampatterns_individual_freqs(beampatterns, processed_freq_range_hz, list_all_freqs_hz):
        # f, ax = bp.plot_bf_3d(np.array([beampatterns['Ideal'], beampatterns['CW']]))
        desired_freqs_hz = np.array([400 + 100 * i for i in range(20)])

        desired_freqs_hz = desired_freqs_hz[desired_freqs_hz < processed_freq_range_hz[-1]]
        desired_freqs_hz = desired_freqs_hz[processed_freq_range_hz[0] < desired_freqs_hz]
        desired_freqs_hz = desired_freqs_hz[:6]
        if len(desired_freqs_hz) > 0:
            desired_freqs_bins = (
                u.get_frequency_bins_from_frequencies(desired_freqs_hz, list_all_freqs_hz=list_all_freqs_hz))
            ll = [np.abs(beampatterns['Ideal'][desired_freqs_bins, :]),
                  np.abs(beampatterns['CW'][desired_freqs_bins, :]),
                  np.abs(beampatterns['CW-SV'][desired_freqs_bins, :])]
            tt = ['Ideal', 'CW', 'CW-SV']
            f1 = u.plot(ll, titles=tt, time_axis=False, show=False)
            for ax in f1.get_axes():
                ax.legend([f'{freq / 1e3 :.2f}kHz' for freq in desired_freqs_hz])
                ax.set_xlabel('Arrival angle [deg]')
                ax.set_ylabel('Gain [dB]')

                # minor ticks every 15, major ticks every 30
                ax.set_xticks(np.arange(0, 180, 15))
                ax.set_xticks(np.arange(0, 180, 30), minor=True)

                ax.grid(which='major', color='#CCCCCC')
                ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
                ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

            f1.show()

    @staticmethod
    def plot_beampatterns_avg(beampatterns_avg, max_degrees=180):
        f = u.plot(list(beampatterns_avg.values()), titles=list(beampatterns_avg.keys()), time_axis=False,
                   show=False)
        ax = f.get_axes()[0]

        # x-axis goes from 0 to 180
        ax.set_xticks(np.linspace(0, max_degrees, 10))
        ax.set_xticks(np.linspace(0, max_degrees, 19), minor=True)
        ax.set_xticklabels(np.linspace(0, 180, 10).astype('i8'))
        ax.set_xlabel('Arrival angle [deg]')
        ax.set_ylabel('Gain [dB]')
        ax.legend()
        f.show()
