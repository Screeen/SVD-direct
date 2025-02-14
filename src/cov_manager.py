import copy
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy

import hfhd.hd as hd

import src.global_constants as g
import src.utils as u
from src.rtf_estimator import RtfEstimator


class CovarianceManager:
    def __init__(self, add_id_noise_noisy=False):

        # narrowband covariances
        self.nb_cov_wet_oracle = None
        self.nb_cov_noise = None
        self.nb_cov_noisy = None

        # wideband empirical covariances -- estimated from data
        self.cov_noise = None
        self.cov_noisy = None
        self.cov_wet_gevd = None

        self.cov_wet_oracle = None
        self.cov_noise_oracle = None
        self.cov_dry_oracle = None  # used for CRBs
        self.cov_dry_oracle_ref_mic = None

        # used for EVD-SVD
        self.cov_noise_sqrt = None
        self.eigve_cov_wet = None

        # wideband ground-truth covariances -- used to generate synthetic data
        self.cov_noisy_gt = None
        self.cov_wet_gt = None
        self.cov_noise_gt = None  # used for CRBs

        self.add_identity_noise_noisy = add_id_noise_noisy

    @staticmethod
    def estimate_cov(x_stft: np.ndarray, wideband=False, avg_time_frames_=False,
                     warning_level='error', subtract_mean=False, add_identity=False,
                     alpha=0.95, phase_correction=None) -> np.ndarray:

        # if subtract_mean:
        #     x_stft -= np.mean(x_stft, axis=-1, keepdims=True)

        if not wideband:
            x_cpsd = CovarianceManager.estimate_cov_narrowband_inner(x_stft, avg_time_frames_, alpha)
            if add_identity:
                x_cpsd = x_cpsd + g.diagonal_loading * np.identity(x_cpsd.shape[0])[..., np.newaxis, np.newaxis]
        else:
            x_cpsd, is_singular = CovarianceManager.estimate_cov_wideband_inner(x_stft, avg_time_frames_, alpha,
                                                                                warning_level, phase_correction)
            if add_identity:
                x_cpsd = x_cpsd + g.diagonal_loading * np.identity(x_cpsd.shape[0])[..., np.newaxis]

        return x_cpsd

    @staticmethod
    def estimate_cov_wideband_inner(x_stft_, avg_time_frames_=True, alpha=g.alpha_cov,
                                    warning_level='warning', correction_term=None):
        """
        Estimate the joint spectral-spatial cross-power spectral density (CPSD) matrix for a given STFT matrix.
        :param x_stft_: STFT matrix of shape (num_mics, num_freqs, num_frames)
        :param avg_time_frames_: if True, average over time frames
        :param alpha: smoothing factor for the covariance matrix
        :param warning_level: if 'error', raise an error if the covariance matrix is singular
        :param correction_term: if not None, multiply the STFT matrix by this term. Accounts for frame-delay in STFT
        :return: CPSD matrix of shape (num_mics*num_freqs, num_mics*num_freqs, num_frames)
        """

        assert x_stft_.size > 0

        if x_stft_.ndim == 2:
            warnings.warn(f"estimate_cov_wideband_inner: input signal was 2D {x_stft_.shape=}. "
                          f"Assuming that only 1 microphone is present")
            x_stft_ = x_stft_[np.newaxis, ...]

        error_msg = None
        num_mics, num_freqs, num_frames_input = x_stft_.shape
        num_freqs_mics = num_mics * num_freqs
        num_frames_cpsd = 1 if avg_time_frames_ else num_frames_input

        # produce long vector [x_freq1_mic1,...,x_freq1_micM, x_freq2_mic1, ...., x_freqK_micM]
        x_stft = np.reshape(x_stft_, (num_freqs_mics, num_frames_input), order='f')

        if avg_time_frames_ and num_frames_input < num_freqs_mics:
            error_msg = f"numTimeFrames = {num_frames_input} < num_mics*num_freqs = {num_freqs_mics}: " \
                        f"covariance matrix will be singular"
            if warning_level == 'error':
                raise ValueError(error_msg)
        x_cpsd = np.zeros((num_freqs_mics, num_freqs_mics, num_frames_cpsd), dtype=complex)
        if avg_time_frames_:
            x_cpsd[..., 0] = (x_stft @ x_stft.conj().T) / num_frames_input
        else:
            x_cpsd[..., 0] = x_stft[..., 0, np.newaxis] @ x_stft[..., 0, np.newaxis].conj().T
            for tt in range(1, num_frames_input):
                x_cpsd_1frame = x_stft[..., tt][..., np.newaxis]
                x_cpsd[..., tt] = alpha * x_cpsd[..., tt - 1] + (1 - alpha) * x_cpsd_1frame @ x_cpsd_1frame.conj().T

        if error_msg is not None and warning_level == 'warning':
            warnings.warn(error_msg)

        is_singular = error_msg is not None
        return x_cpsd, is_singular

    @staticmethod
    def estimate_cov_narrowband_inner(x_stft, avg_time_frames_, alpha):
        num_mics, num_freqs, num_frames_input = x_stft.shape
        num_frames_cpsd = 1 if avg_time_frames_ else num_frames_input
        x_cpsd = np.zeros((num_mics, num_mics, num_freqs, num_frames_cpsd), dtype=complex)
        if avg_time_frames_:
            for kk in range(num_freqs):
                x_cpsd[..., kk, 0] = x_stft[:, kk, :] @ x_stft[:, kk, :].conj().T
            x_cpsd /= num_frames_input
        else:
            for kk in range(num_freqs):
                x_cpsd[..., kk, 0] = x_stft[..., kk, 0] @ x_stft[..., kk, 0].conj().T
                for tt in range(1, num_frames_cpsd):
                    windowed = x_stft[:, kk, tt, np.newaxis]
                    x_cpsd[..., kk, tt] = alpha * x_cpsd[..., kk, tt - 1] + \
                                          (1 - alpha) * (windowed @ windowed.conj().T)
        return x_cpsd

    @staticmethod
    def estimate_cov_loop_impl(x_stft, avg_time_frames_):
        num_mics, num_freqs, num_frames_input = x_stft.shape
        x_cpsd = np.zeros((num_mics, num_mics, num_freqs, num_frames_input), dtype=complex)
        assert avg_time_frames_

        for tt in range(num_frames_input):
            for kk in range(num_freqs):
                frame = x_stft[:, kk, tt, np.newaxis]
                x_cpsd[..., kk, tt] = frame @ frame.conj().T

        x_cpsd = np.mean(x_cpsd, axis=-1, keepdims=True)

        return x_cpsd

    def estimate_all_covariances_narrowband(self, stimuli_stft, use_true_noise_covariance=False):
        """
        Estimates the covariances from the stft of the stimuli.
        :param stimuli_stft:  dictionary with the stft of the stimuli
        :return: None
        """

        add_id = self.add_identity_noise_noisy

        self.nb_cov_noise = self.estimate_cov(stimuli_stft['noise'], avg_time_frames_=True,
                                              add_identity=add_id)
        self.nb_cov_noisy = self.estimate_cov(stimuli_stft['mix'], avg_time_frames_=True,
                                              add_identity=add_id)

        wet_oracle_key = 'early' if 'early' in stimuli_stft else 'wet'
        self.nb_cov_wet_oracle = self.estimate_cov(stimuli_stft[wet_oracle_key], avg_time_frames_=True,
                                                   add_identity=False)

        if use_true_noise_covariance:
            self.cov_noise = self.cov_noise_gt[..., np.newaxis]
            if add_id:
                self.cov_noise = self.cov_noise + g.diagonal_loading * np.identity(self.cov_noise.shape[0])[..., None]

    def estimate_all_covariances(self, stimuli_stft, avg_time_frames=True, alpha_=g.alpha_cov, phase_correction=None,
                                 dry_stft=None, use_true_noise_covariance=False, num_offdiag_bins_dict=None):
        """
        Estimates the covariances from the stft of the stimuli.
        :param num_offdiag_bins_dict: dictionary with the number of off-diagonal bins to retain for each covariance mat.
        :param stimuli_stft:  dictionary with the stft of the stimuli
        :param avg_time_frames: if True, average over time frames
        :param alpha_: smoothing factor for the covariance matrix
        :param phase_correction: if not None, multiply the STFT matrix by this term. Accounts for frame-delay in STFT
        :param dry_stft: if not None, use this to estimate the dry covariance
        :param use_true_noise_covariance: if True, use the true noise covariance instead of estimating it
        :return: None
        """

        if num_offdiag_bins_dict is None:
            num_offdiag_bins_dict = {'noisy': -1, 'noise': -1, 'wet_gevd': -1}
        mix_stft_shape = stimuli_stft['mix'].shape
        num_mics, num_selected_freqs, num_time_frames = mix_stft_shape
        add_id = self.add_identity_noise_noisy
        s_bf = dict(wideband=True, avg_time_frames_=avg_time_frames,
                    warning_level='warning' if g.debug_mode else '', alpha=alpha_, phase_correction=phase_correction)

        # Only for plots and comparison: oracle empirical covariances
        wet_oracle_key = 'early' if 'early' in stimuli_stft else 'wet'
        self.cov_wet_oracle = self.estimate_cov(stimuli_stft[wet_oracle_key], **s_bf, add_identity=False)
        if g.debug_mode and g.debug_show_plots:
            self.cov_noise_oracle = self.estimate_cov(stimuli_stft['noise'], **s_bf, add_identity=add_id)

        # Estimate empirical noise and noisy covariances
        self.cov_noisy = self.estimate_cov(stimuli_stft['mix'], **s_bf, add_identity=add_id)
        self.cov_noise = self.estimate_cov(stimuli_stft['noise'], **s_bf, add_identity=add_id)

        # Set some of the off-diagonal blocks to zero
        self.cov_noisy = u.ForceToZeroOffBlockDiagonal(self.cov_noisy, num_mics, num_offdiag_bins_dict['noisy'])
        self.cov_noise = u.ForceToZeroOffBlockDiagonal(self.cov_noise, num_mics, num_offdiag_bins_dict['noise'])

        if use_true_noise_covariance:
            # TODO Remove this option, there is no need to call this function with True noise covariance available
            if self.cov_noise_gt is None:
                raise ValueError("True covariance matrices can only be used when generating signals in the "
                                 "frequency domain. Change \'use_true_noise_covariance\' to False.")
            self.cov_noise = self.cov_noise_gt[..., np.newaxis]
            if add_id:
                self.cov_noise = self.cov_noise + g.diagonal_loading * np.identity(self.cov_noise.shape[0])[..., None]

        self.cov_wet_gevd, self.eigve_cov_wet = self.estimate_cov_wet_gevd(self.cov_noisy, self.cov_noise,
                                                                           mix_stft_shape, 'cholesky')

        self.cov_wet_gevd = np.atleast_3d(self.cov_wet_gevd)
        self.cov_noise = np.atleast_3d(self.cov_noise)
        self.cov_noisy = np.atleast_3d(self.cov_noisy)

        self.cov_wet_gevd = u.ForceToZeroOffBlockDiagonal(self.cov_wet_gevd, num_mics, num_offdiag_bins_dict['wet_gevd'])

        if dry_stft is not None:
            self.cov_dry_oracle_ref_mic = self.estimate_cov(dry_stft[g.idx_ref_mic][np.newaxis, ...], **s_bf)
            self.cov_dry_oracle = np.kron(self.cov_dry_oracle_ref_mic[..., 0], np.ones((num_mics, num_mics)))
            self.cov_dry_oracle = self.cov_dry_oracle[..., np.newaxis]

        if self.cov_wet_gt is not None and self.cov_noise is not None:
            self.cov_noisy_gt = self.cov_wet_gt + self.cov_noise_gt

        # instead of estimating covariances again, just copy block-diagonal of bifrequency covariances to
        # narrowband covariances
        cov_shape = mix_stft_shape[:2] + (self.cov_noise.shape[-1],)
        self.copy_single_freq_covariances_from_bifreq_covariances(cov_shape)

    def copy_single_freq_covariances_from_bifreq_covariances(self, stft_shape, cov_noisy_emp=None):
        """Even when cross-freq components are evaluated, diagonal elements correspond.
        So instead of recomputing them, just copy from block-diagonal of bifrequency covariances"""

        num_mics, num_freqs, num_frames = stft_shape
        self.nb_cov_wet_oracle = np.zeros((num_mics, num_mics, num_freqs, num_frames), dtype=complex, order='F')
        self.nb_cov_noise = np.zeros_like(self.nb_cov_wet_oracle)
        self.nb_cov_noisy = np.zeros_like(self.nb_cov_wet_oracle)

        for tt in range(num_frames):
            for km in range(0, num_freqs * num_mics, num_mics):
                wideband_index = slice(km, km + num_mics), slice(km, km + num_mics), tt
                narrowband_index = Ellipsis, km // num_mics, tt

                # if self.cov_wet_oracle is not None:
                #     self.nb_cov_wet_oracle[narrowband_index] = self.cov_wet_oracle[wideband_index]
                self.nb_cov_noise[narrowband_index] = self.cov_noise[wideband_index]

                if cov_noisy_emp is not None:
                    self.nb_cov_noisy[narrowband_index] = cov_noisy_emp[wideband_index]
                else:
                    self.nb_cov_noisy[narrowband_index] = self.cov_noisy[wideband_index]

    def remove_identity(self, phi):
        if self.add_identity_noise_noisy:
            return phi - g.diagonal_loading * np.identity(phi.shape[0])[..., np.newaxis]
        else:
            return phi

    def plot_cov(self, true_cov=False, amp_range=None, time_frame=-1, f0_harmonics_bins=None, show_plot=True,
                 show_single_mic=False):

        font_size = 'x-large'

        if true_cov:
            x, v, y = self.cov_wet_gt, self.cov_noise_gt, self.cov_noisy_gt
            if all([x is None, v is None]):
                x = self.cov_wet_oracle
                v = self.cov_noise_oracle
            titles = ['Wet ground truth', 'Noise ground truth', 'Mix ground truth', 'Dry ground truth']

        else:
            x, v, y = self.cov_wet_gevd, self.remove_identity(self.cov_noise), \
                self.remove_identity(self.cov_noisy)
            titles = ['Est. wet (GEVD)', 'Est. noise', 'Est. mix', 'True dry']

        if show_single_mic:
            # transform x, v, y using get_spectral_covariance_from_spectral_spatial_covariance
            num_mics = self.nb_cov_noisy.shape[0]
            x = self.get_spectral_covariance_from_spectral_spatial_covariance(x, num_mics)
            v = self.get_spectral_covariance_from_spectral_spatial_covariance(v, num_mics)
            y = self.get_spectral_covariance_from_spectral_spatial_covariance(y, num_mics)

        if x is not None and x.ndim == 3:
            x = x[..., time_frame]
        if v is not None and v.ndim == 3:
            v = v[..., time_frame]
        if y is not None and y.ndim == 3:
            y = y[..., time_frame]

        if amp_range is None:
            if all([x is not None, v is not None, y is not None]):
                min_amplitude = max(-50, min(np.min(u.log_pow(x)), np.min(u.log_pow(v)), np.min(u.log_pow(y))))
                max_amplitude = max(np.max(u.log_pow(x)), np.max(u.log_pow(v)), np.max(u.log_pow(y)))
                amp_range = (min_amplitude, max_amplitude)
            else:
                amp_range = (-50, 15)

        fig_opt = dict(constrained_layout=True)
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', **fig_opt)

        # plot wet covariance
        fig, im = u.plot_matrix(x, axes[0, 0], title=titles[0], xy_label='', amp_range=amp_range, show_colorbar=False)

        if f0_harmonics_bins is None:
            if v is not None:
                # plot noise covariance
                fig, im = u.plot_matrix(v, axes[0, 1], title=titles[1], xy_label='', amp_range=amp_range,
                                        show_colorbar=False)
            else:
                axes[0, 1].set_visible(False)
        else:
            # add a red 'x' on the diagonal of the plot at the harmonic frequencies of the fundamental frequency
            fig, im = u.plot_matrix(x, axes[0, 1], title=titles[0], xy_label='', amp_range=amp_range,
                                    show_colorbar=False)
            f0_harmonics_bins = f0_harmonics_bins
            for f0_harmonic_bin in f0_harmonics_bins:
                axes[0, 1].scatter(f0_harmonic_bin, f0_harmonic_bin, s=30, facecolors='none', edgecolors='r')

        if y is not None:
            fig, im = u.plot_matrix(y, axes[1, 0], title=titles[2], xy_label='', amp_range=amp_range,
                                    show_colorbar=False)
        else:
            axes[-1, 0].set_visible(False)

        if true_cov and self.cov_dry_oracle is not None:
            fig, im = u.plot_matrix(self.cov_dry_oracle, axes[1, 1],
                                    title=titles[3], xy_label='', amp_range=amp_range, show_colorbar=False)
        else:
            axes[-1, -1].set_visible(False)

        cb = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, use_gridspec=False)
        cb.set_label('Magnitude [dBm]', size=font_size)
        cb.ax.tick_params(labelsize=font_size)

        # make sure all axes are y inverted
        for ax in axes.flatten():
            if not ax.yaxis_inverted():
                ax.invert_yaxis()

        if show_plot:
            fig.show()

        return fig

    @staticmethod
    def estimate_correlation_all_covariances(covariance_matrices, num_mics, name='', duration_seconds=0, title=''):
        # Estimate the average, minimum and maximum correlation coefficients for the "spectral" covariance matrices,
        # which are obtained using the method get_spectral_covariance_from_spectral_spatial_covariance.
        # The correlation coefficient is evaluated as: rho_f(k1, k2) = cov(k1, k2) / sqrt(cov(k1, k1) * cov(k2, k2))

        how_many = 8
        show_plot = True

        correlation_coefficients = []
        for cov in covariance_matrices:
            cov = CovarianceManager.get_spectral_covariance_from_spectral_spatial_covariance(cov, num_mics)
            corr = CovarianceManager.estimate_correlation(cov, how_many)
            correlation_coefficients.append(corr['corr_coeffs'])

        if show_plot:
            x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            avg_heights = np.zeros(len(x_ticks) - 1)
            for corr_coeffs in correlation_coefficients:
                heights, bins = np.histogram(corr_coeffs.flatten(), bins=x_ticks)
                heights = heights / sum(heights)
                avg_heights += heights
            avg_heights /= len(correlation_coefficients)

            if title == '':
                title = f'Occurrences of correlation coefficients for {name} (over {duration_seconds:.2f} s)'

            CovarianceManager.plot_histogram_correlation_coefficients(x_ticks, avg_heights, title)

    @staticmethod
    def estimate_correlation(cov, how_many=5):

        num_freqs = cov.shape[0]

        # compute correlation coefficients (only for the upper triangular part)
        corr_coeffs = np.ones((num_freqs, num_freqs)) * np.nan
        for k1 in range(num_freqs):
            for k2 in range(k1 + 1, num_freqs):
                corr_coeffs[k1, k2] = np.abs(cov[k1, k2] / np.sqrt(cov[k1, k1] * cov[k2, k2]))
        avg_corr = np.nanmean(corr_coeffs)

        # find top 5 max correlation coefficients and their locations
        # replace nan with -1 to avoid sorting issues
        corr_coeffs[np.isnan(corr_coeffs)] = -1
        max_corr_top5 = np.sort(corr_coeffs.flatten())[-how_many:]
        max_corr_top5_locations = np.unravel_index(np.argsort(corr_coeffs, axis=None)[-how_many:], corr_coeffs.shape)

        corr_dict = {
            'avg': avg_corr,
            'max': max_corr_top5,
            'max_location': max_corr_top5_locations,
            'corr_coeffs': corr_coeffs
        }

        return corr_dict

    @staticmethod
    def plot_histogram_correlation_coefficients(x_ticks, heights, title=''):
        # using corr_coeffs, make a histogram of the correlation coefficients
        # y axis is the percentage of the total number of data points
        u.set_plot_options(True)
        fig, ax = plt.subplots(figsize=(3.5, 2.))
        # font_size = '12'

        ax.bar(x_ticks[:-1], heights, width=x_ticks[1] - x_ticks[0], align='edge', edgecolor='white')
        ax.set_xticks(x_ticks)

        # the y axis is the percentage of the total number of data points
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0, 1.1, 0.2)])
        ax.grid(axis='y')

        ax.set_title(title)
        ax.set_xlabel(r'Correlation coefficient')
        ax.set_ylabel(r'Occurrences [\%]')

        plt.show()
        title_save = title.replace(' ', '_').replace('(', '').replace(')', '')
        title_save = title_save + datetime.now().strftime("_%Y%m%d_%H%M%S")
        u.save_figure(fig, title_save)

    @staticmethod
    def project_covariance_into_psd_cone(r, warn_if_not_psd=True):
        """ Project into the PSD cone by eigenvalue decomposition: a PSD matrix has non-negative eigenvalues. """
        eigs, eigve = np.linalg.eigh(r)
        if any(eigs < 0):
            if warn_if_not_psd:
                warnings.warn('Covariance matrix is not positive definite. Projecting into PSD cone')
            eigs[eigs < 0] = 0
            r = eigve @ np.diag(eigs) @ eigve.conj().T
        return r

    @staticmethod
    def compute_phase_correction_stft(stft_shape, overlap):
        """
        Compute the correction term to account for delay in STFT. See for example Equation 3 in
        "Fast computation of the spectral correlation" by Antoni, 2017.

        :param stft_shape: shape of the stft matrix, e.g. (num_mics, num, num_frames)
        :param overlap: a number between 0 and win_len-1, where win_len = (num_freqs_real-1)*2
        :return:
        """

        (_, num_freqs_real, num_frames) = stft_shape

        if num_frames == 1:
            return np.ones((num_freqs_real, 1))

        win_len = (num_freqs_real - 1) * 2  # delay correction depends on window size = win_len
        shift_samples = win_len - overlap

        # normalized frequencies in [0, 0.5] (real part)
        frequencies = np.arange(0, num_freqs_real)[:, np.newaxis] / win_len
        time_frames = np.arange(0, shift_samples * num_frames, shift_samples)[np.newaxis, :]
        correction_term = np.exp(-2j * np.pi * frequencies * time_frames)

        return correction_term

    @staticmethod
    def estimate_cov_wet_gevd(cov_noisy, cov_noise, rtfs_shape, modality='gevd'):

        num_mics, num_selected_freqs, num_time_frames = rtfs_shape
        num_estimated_covariances = cov_noisy.shape[-1]
        eigve_cov_wet = None
        cov_wet_est = np.zeros_like(cov_noisy)
        max_rank_cov_wet = min(num_selected_freqs, num_time_frames)

        for tt in range(num_estimated_covariances):

            noisy = cov_noisy[..., tt]
            noise = cov_noise[..., tt] if cov_noise is not None else None

            if modality == 'subtraction':
                cov_wet_est[..., tt] = noisy - noise
                cov_wet_est[..., tt] = CovarianceManager.project_covariance_into_psd_cone(cov_wet_est[..., tt])

            elif modality == 'cholesky':
                # noise_temp = u.ForceToZeroOffBlockDiagonal(noise, block_size=num_mics, max_distance_diagonal=0)
                noise_temp = noise
                noise_sqrt = np.linalg.cholesky(noise_temp)
                noise_sqrt_inv = np.linalg.inv(noise_sqrt)
                whitened_noisy = noise_sqrt_inv @ noisy @ noise_sqrt_inv.conj().T
                eigenvals, eigves = scipy.linalg.eigh(whitened_noisy, check_finite=True)
                eigenvals, eigves = RtfEstimator.sort_eigenvectors_get_major(eigenvals, eigves, max_rank_cov_wet)
                if np.any(eigenvals > 1):
                    eigenvals = eigenvals - 1
                else:
                    warnings.warn(f"All eigenvalues of cov_wet, {modality = }, are less than 1. ")
                eigenvals = np.maximum(g.eps, eigenvals.real)
                dewhitened_eigve = noise_sqrt @ eigves
                eigve_cov_wet = dewhitened_eigve @ np.diagflat(np.sqrt(eigenvals))
                cov_wet_est[..., tt] = eigve_cov_wet @ eigve_cov_wet.conj().T

            elif modality == 'gevd':
                eigenvals, eigves = scipy.linalg.eigh(noisy, noise, check_finite=False, driver='gvd')

                # keep only eigenvectors corresponding to eigenvalues > 1 and largest num_freqs eigenvalues
                eigenvals, eigves_right = RtfEstimator.sort_eigenvectors_get_major(eigenvals, eigves, max_rank_cov_wet)
                if np.any(eigenvals > 1):
                    eigenvals = eigenvals - 1
                else:
                    warnings.warn(f"All eigenvalues of cov_wet, {modality = }, are less than 1. ")
                eigenvals = np.maximum(0, eigenvals.real)

                # Several ways to get the left eigenvectors
                eigves_left = noise @ eigves_right
                eigve_cov_wet = eigves_left @ np.diagflat(np.sqrt(eigenvals))
                cov_wet_est[..., tt] = eigve_cov_wet @ eigve_cov_wet.conj().T

                # this should be identical, but it contains a matrix inverse, so it is probably slower
                # cov_wet_est[..., tt] = noise @ eigves[:, -num_freqs:] @ np.diagflat(eigenvals[-num_freqs:]) @ np.linalg.inv(eigves)[-num_freqs:]

            elif modality == 'gevd-nb':
                # if set to 1, then only the first eigenvalue is retained. Very similar results to narrowband CW
                max_rank_cov_wet = 1

                for kk in range(num_selected_freqs):
                    noisy_kk = noisy[kk * num_mics:(kk + 1) * num_mics, kk * num_mics:(kk + 1) * num_mics]
                    noise_kk = noise[kk * num_mics:(kk + 1) * num_mics, kk * num_mics:(kk + 1) * num_mics]

                    eigenvals, eigves = scipy.linalg.eigh(noisy_kk, noise_kk)
                    eigenvals, eigves = RtfEstimator.sort_eigenvectors_get_major(eigenvals, eigves, max_rank_cov_wet,
                                                                                 squeeze=False)
                    eigves_left = noise_kk @ eigves  # eigves_left = np.linalg.inv(eigves).conj().T

                    # only needed if more than 1 eigenvalue is retained
                    eigenvals = eigenvals - 1
                    eigenvals = np.maximum(0, eigenvals.real)

                    res = eigves_left @ np.diagflat(eigenvals) @ eigves_left.conj().T
                    cov_wet_est[kk * num_mics:(kk + 1) * num_mics, kk * num_mics:(kk + 1) * num_mics, tt] = res

            else:
                raise ValueError(f"Unknown modality: {modality}")

        return cov_wet_est, eigve_cov_wet

    @staticmethod
    def transform_cov_real_imaginary_parts_to_complex_cov(cov_real_imag_concatenated):
        # https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts

        if cov_real_imag_concatenated.ndim != 2:
            raise ValueError(f'cov_real_imag must be a 2D matrix, but {cov_real_imag_concatenated.shape=}')

        split_point = cov_real_imag_concatenated.shape[0] // 2
        cov_real_real = cov_real_imag_concatenated[:split_point, :split_point]
        cov_imag_imag = cov_real_imag_concatenated[split_point:, split_point:]
        cov_real_imag = cov_real_imag_concatenated[:split_point, split_point:]
        cov_imag_real = cov_real_imag_concatenated[split_point:, :split_point]
        cov_complex = cov_real_real + cov_imag_imag + 1j * (cov_imag_real - cov_real_imag)

        return cov_complex

    @staticmethod
    def estimate_support_sparse_covariance(stft_data_3d, use_skggm=False):
        stft_shape = stft_data_3d.shape
        num_mics, _, num_frames = stft_shape
        stft_data_2d = stft_data_3d[g.idx_ref_mic]

        # compute PSD and remove mean
        stft_data_2d = np.abs(stft_data_2d.copy())
        stft_data_2d -= np.mean(stft_data_2d, axis=-1, keepdims=True)

        cov_sparse_single_mic, _, _ = CovarianceManager.estimate_sparse_cov_internal(stft_data_2d, use_skggm=True)
        support_cov_sparse_multi_mic = np.array(np.kron(cov_sparse_single_mic, np.ones((num_mics, num_mics))) > 0)

        return support_cov_sparse_multi_mic

    @staticmethod
    def estimate_sparse_cov_internal(stft_data, use_skggm=False, prior=None):
        """
        Estimate sparse covariance matrix using QuicGraphicalLasso _lib
        :param stft_data: ndarray of shape (num_freqs, num_frames)
        :param use_skggm: use skggm instead of NERCOME
        :param prior: prior for the regularization parameter
        :return:
        """
        from inverse_covariance import QuicGraphicalLasso
        if use_skggm:
            model = QuicGraphicalLasso(
                lam=0.3 if prior is None else prior,
                tol=1e-6,
                init_method="cov",
                verbose=0,
                auto_scale=True,
                max_iter=100,
            )
            model.fit(stft_data.T)
            try:
                cov_sparse = model.covariance_
                cov_inv_sparse = model.precision_
                cov_empirical = model.sample_covariance_
            except AttributeError:
                cov_sparse = model.estimator_.covariance_
                cov_inv_sparse = model.estimator_.precision_
                cov_empirical = model.estimator_.sample_covariance_
            if not np.all(np.isfinite(cov_sparse)):
                warnings.warn(f"Sparse covariance is not finite. Using empirical covariance instead.")
                cov_sparse = cov_empirical

            return cov_sparse, cov_inv_sparse, cov_empirical

        else:

            cov = hd.nercome(stft_data, m=None, M=50)

            # I don't think this works. The results don't really change if we change parameters
            # num_mics_freqs = stft_data.shape[0] // 2
            # poet_res = poet.POET(stft_data, K=num_mics_freqs - 2, thres='soft')
            # cov = poet_res.SigmaY

            # cov = hd.nonlinear_shrinkage(stft_data)

            return cov, None, None

    @staticmethod
    def estimate_sparse_cov_split_real_imag(stft_data_3d, cov_mask_prior=None, use_skggm=False,
                                            cov_or_prec='cov'):

        if cov_mask_prior is not None:
            # Form a block matrix, each block is cov_mask_prior replicated
            cov_mask_prior = np.tile(cov_mask_prior, (2, 2)).astype(float)

        stft_data_real = np.real(stft_data_3d).reshape((-1, stft_data_3d.shape[-1]), order='F')
        stft_data_imag = np.imag(stft_data_3d).reshape((-1, stft_data_3d.shape[-1]), order='F')
        if cov_or_prec == 'cov':
            stft_data_concat = np.concatenate((stft_data_real, stft_data_imag), axis=0)
        elif cov_or_prec == 'prec':
            stft_data_norm = np.abs(stft_data_3d).reshape((-1, stft_data_3d.shape[-1]), order='F') ** 2
            stft_data_concat = np.concatenate((stft_data_real / stft_data_norm, -stft_data_imag / stft_data_norm), axis=0)
        else:
            raise ValueError(f"Unknown cov_or_prec: {cov_or_prec}")

        cov_sparse, cov_inv_sparse, cov_empirical = (CovarianceManager.
                                                     estimate_sparse_cov_internal(stft_data_concat, use_skggm=use_skggm,
                                                                                  prior=cov_mask_prior))

        cov_sparse = CovarianceManager.transform_cov_real_imaginary_parts_to_complex_cov(cov_sparse)[..., np.newaxis]
        if cov_empirical is not None:
            cov_empirical = CovarianceManager.transform_cov_real_imaginary_parts_to_complex_cov(cov_empirical)[
                ..., np.newaxis]
        if cov_inv_sparse is not None:
            cov_inv_sparse = CovarianceManager.transform_cov_real_imaginary_parts_to_complex_cov(cov_inv_sparse)[
                ..., np.newaxis]

        return cov_sparse, cov_inv_sparse, cov_empirical

    @staticmethod
    def create_symmetric_block_matrix_from_diag_values(block_values, block_size, num_blocks_per_row, filling_value=0.):
        """
         Block values is a 1d vector of size num_blocks_per_row. If shorter, fill with zeros.
         for example, if block size = 2, num_blocks_per_row = 3, block_values = [1, 2, 0], then the resulting matrix is
         [[1, 1, 2, 2, 0, 0],
          [1, 1, 2, 2, 0, 0],
          [2, 2, 1, 1, 2, 2],
          [2, 2, 1, 1, 2, 2]
          [0, 0, 2, 2, 1, 1],
          [0, 0, 2, 2, 1, 1]]

          We first create a Toeplitz matrix, then use np.kron to create the block matrix.
         """

        if num_blocks_per_row < len(block_values):
            raise ValueError('The number of blocks per row is larger than the length of block values.')

        block_values = np.array(block_values)
        if block_values.size < num_blocks_per_row:
            block_values = np.concatenate(
                (block_values, filling_value * np.ones(num_blocks_per_row - block_values.size)))

        # Create a Toeplitz matrix
        toeplitz_matrix = scipy.linalg.toeplitz(block_values)

        # Create a block matrix
        block_matrix = np.kron(toeplitz_matrix, np.ones((block_size, block_size)))

        return block_matrix

    @staticmethod
    def get_spectral_covariance_from_spectral_spatial_covariance(cov_spectral_spatial, num_mics):
        """
        Extract the spectral covariance matrix from the spectral-spatial covariance matrix.
        For example, for num_mics = 2 and num_freqs = 3, we would have to extract the X and discard the O, as in:

        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0
        ---------------------
        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0
        ---------------------
        X 0 | X 0 | X 0
        0 0 | 0 0 | 0 0

        Create a mask that has True where the X are and False where the O are. Then use the mask to extract the X.
        """
        if cov_spectral_spatial is None:
            return None

        return cov_spectral_spatial[::num_mics, ::num_mics]

    @classmethod
    def estimate_sparse_cov(cls, stimulus_stft, dense_cov=None, estimate_support=True, use_skggm=False,
                            cov_or_prec='cov', non_sparse_penalty=0.3):

        if estimate_support:
            sparse_support = cls.estimate_support_sparse_covariance(stimulus_stft, use_skggm=use_skggm)
            sparse_cov = dense_cov * sparse_support[..., np.newaxis]
            inv_cov = None
            emp_cov = None
        else:
            num_mics, num_freqs, num_frames = stimulus_stft.shape
            cov_mask_prior = np.ones((num_mics * num_freqs, num_mics * num_freqs))
            cov_mask_prior = cov_mask_prior - u.ForceToZeroOffBlockDiagonal(cov_mask_prior, block_size=num_mics,
                                                                            max_distance_diagonal=0)
            cov_mask_prior = non_sparse_penalty * cov_mask_prior
            sparse_cov, inv_cov, emp_cov = cls.estimate_sparse_cov_split_real_imag(stimulus_stft, use_skggm=use_skggm,
                                                                                   cov_mask_prior=cov_mask_prior, cov_or_prec=cov_or_prec)

        return sparse_cov, inv_cov, emp_cov
