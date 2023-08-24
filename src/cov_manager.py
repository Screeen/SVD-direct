import warnings

import scipy
from numba import njit

import numpy as np
import matplotlib.pyplot as plt

import src.global_constants as g
import src.utils as u


class CovarianceManager:
    def __init__(self, add_id_noise_noisy=False, num_frames_covariance_estimation=1):

        self.cov_wet_gevd = np.empty(())
        self.nb_cov_wet_oracle = np.empty(())
        self.nb_cov_noise = np.empty(())
        self.nb_cov_noisy = np.empty(())
        self.cov_wet_oracle = np.empty(())
        self.cov_noise = np.empty(())
        self.cov_noisy = np.empty(())

        self.cov_dry_oracle = None  # used for CRBs

        # phi_xx_bf should ideally converge to phi_wet_bf_true for infinite number of realizations
        self.phi_wet_bf_true = None

        # phi_vv_bf should converge to phi_vv_bf_true
        self.phi_vv_bf_true = None  # used for CRBs

        self.add_identity_noise_noisy = add_id_noise_noisy
        self.num_frames_covariance_estimation = num_frames_covariance_estimation

    @staticmethod
    def estimate_cpsd_wrapper(x_stft: np.array, with_crossfreq=False, avg_time_frames_=False,
                              warning_level='error', subtract_mean=False, add_identity=False,
                              alpha=0.95, phase_correction=None) -> np.array:

        # print(f"Calculate CPSD. Use cross-frequencies = {with_crossfreq}")
        # x_stft = np.copy(x_stft_input)

        # if subtract_mean:
        #     x_stft -= np.mean(x_stft, axis=-1, keepdims=True)

        if not with_crossfreq:
            x_cpsd = CovarianceManager.estimate_cpsd(x_stft, avg_time_frames_, alpha)
            if add_identity:
                x_cpsd = x_cpsd + g.diagonal_loading * np.identity(x_cpsd.shape[0])[..., np.newaxis, np.newaxis]
        else:
            x_cpsd, is_singular = CovarianceManager.estimate_cpsd_bifreq(x_stft, avg_time_frames_, alpha,
                                                                         warning_level, phase_correction)
            if add_identity:
                x_cpsd = x_cpsd + g.diagonal_loading * np.identity(x_cpsd.shape[0])[..., np.newaxis]

        return x_cpsd

    @staticmethod
    def check_matrix_is_hermitian_psd(x_cpsd):
        valid = True
        x_cpsd = np.squeeze(x_cpsd)
        if not u.is_hermitian(x_cpsd):
            warnings.warn("x_cpsd was not Hermitian")
            valid = False
        if not u.is_positiveSemiDefinite(x_cpsd):
            warnings.warn("x_cpsd was not PSD")
            valid = False
        return valid

    @staticmethod
    def suppress_negative_or_small(x_cpsd: np.array, tol):

        z = 0
        m1 = x_cpsd.real < tol
        m2 = x_cpsd.T.real < tol
        m = np.logical_or(m1, m2)
        x_cpsd[m] = z + x_cpsd[m].imag * 1j

        mi1 = x_cpsd.imag < tol
        mi2 = x_cpsd.T.imag < tol
        mi = np.logical_or(mi1, mi2)
        x_cpsd[mi] = x_cpsd[mi].real + z * 1j

        return x_cpsd

    @staticmethod
    def estimate_cpsd_bifreq(x_stft, avg_time_frames_=True, alpha=g.alpha_cov,
                             warning_level='warning', correction_term=None):
        """
        Estimate the joint spectral-spatial cross-power spectral density (CPSD) matrix for a given STFT matrix.
        :param x_stft: STFT matrix of shape (num_mics, num_freqs, num_frames)
        :param avg_time_frames_: if True, average over time frames
        :param alpha: smoothing factor for the covariance matrix
        :param warning_level: if 'error', raise an error if the covariance matrix is singular
        :param correction_term: if not None, multiply the STFT matrix by this term. Accounts for frame-delay in STFT
        :return: CPSD matrix of shape (num_mics*num_freqs, num_mics*num_freqs, num_frames)
        """

        assert x_stft.size > 0

        if x_stft.ndim == 2:
            print(f"estimate_cpsd_bifreq: input signal was 2D (shape {x_stft.shape}). "
                              f"Assuming that only 1 microphone is present")
            x_stft = x_stft[np.newaxis, ...]

        error_msg = None
        num_mics, num_freqs, num_frames_input = x_stft.shape
        num_freqs_mics = num_mics * num_freqs
        num_frames_cpsd = 1 if avg_time_frames_ else num_frames_input

        if correction_term is not None:
            for mm in range(num_mics):
                x_stft[mm] = x_stft[mm] * correction_term[..., :num_frames_input]

        # produce long vector [x_freq1_mic1,...,x_freq1_micM, x_freq2_mic1, ...., x_freqK_micM]
        x_stft = np.reshape(x_stft, (num_freqs_mics, num_frames_input), order='f')

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

        if error_msg is not None:
            if warning_level == 'warning':
                warnings.warn(error_msg)

        is_singular = error_msg is not None
        return x_cpsd, is_singular

    @staticmethod
    def estimate_cpsd(x_stft, avg_time_frames_, alpha):
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
    def estimate_cpsd_loop_impl(x_stft, avg_time_frames_):
        num_mics, num_freqs, num_frames_input = x_stft.shape
        x_cpsd = np.zeros((num_mics, num_mics, num_freqs, num_frames_input), dtype=complex)
        assert avg_time_frames_

        for tt in range(num_frames_input):
            for kk in range(num_freqs):
                frame = x_stft[:, kk, tt, np.newaxis]
                x_cpsd[..., kk, tt] = frame @ frame.conj().T

        x_cpsd = np.mean(x_cpsd, axis=-1, keepdims=True)

        return x_cpsd

    # def generate_synthetic_covariances(self, cm, atfs, noise_stft_shape):
    #
    #     # dimensions
    #     num_mics, num_freqs, num_frames = noise_stft_shape
    #     num_freqs_mics = num_freqs * num_mics
    #
    #     # noise
    #     cm.phi_vv_bf = np.diag(SignalGenerator.generate_circular_gaussian(num_freqs_mics))
    #     cm.phi_vv_bf = u.herm(cm.phi_vv_bf) @ cm.phi_vv_bf
    #     cm.phi_vv_bf = cm.phi_vv_bf + np.identity(num_freqs_mics, complex)
    #     if self.exp_settings['limit_correlation_synthetic_covariances']:
    #         num_freqs_to_keep_vv = int(self.exp_settings['correlation_noise'] * num_freqs)
    #         cm.phi_vv_bf = u.ForceToZeroOffBlockDiagonal(cm.phi_vv_bf, num_freqs_to_keep_vv, num_mics)
    #     cm.phi_vv_bf = cm.phi_vv_bf[..., np.newaxis]
    #
    #     # clean signal, generate powers directly (no estimation error)
    #     s_d = SignalGenerator.generate_circular_gaussian(num_freqs)
    #     s_tilde = np.kron(s_d, np.ones(num_mics))
    #     A = np.diag(atfs.flatten('F'))
    #     s = A @ s_tilde
    #     s = np.atleast_2d(s).T
    #
    #     cm.phi_xx_bf = s @ u.herm(s)
    #     cm.phi_xx_bf = cm.phi_xx_bf[..., np.newaxis]
    #     if self.exp_settings['limit_correlation_synthetic_covariances']:
    #         num_freqs_to_keep_xx = int(self.exp_settings['correlation_target'] * num_freqs)
    #         cm.phi_xx_bf = u.ForceToZeroOffBlockDiagonal(cm.phi_xx_bf, num_freqs_to_keep_xx, num_mics)
    #
    #     # mix
    #     cm.phi_yy_bf = cm.phi_vv_bf + cm.phi_xx_bf
    #
    #     cm.copy_single_freq_covariances_from_bifreq_covariances(noise_stft_shape)

    def estimate_covariances(self, stimuli_stft, avg_time_frames, alpha_=g.alpha_cov, phase_correction=None,
                             dry_stft=None):
        """
        Estimates the covariances from the stft of the stimuli.
        :param stimuli_stft:  dictionary with the stft of the stimuli
        :param avg_time_frames: if True, average over time frames
        """

        mix_stft_shape = stimuli_stft['mix'].shape
        add_identity_noise_noisy = self.add_identity_noise_noisy

        s_bf = dict(with_crossfreq=True, avg_time_frames_=avg_time_frames, warning_level='warning', alpha=alpha_,
                    phase_correction=phase_correction)
        self.cov_wet_oracle = CovarianceManager.estimate_cpsd_wrapper(stimuli_stft['desired_wet'],
                                                                      **s_bf, add_identity=False)
        self.cov_noisy = CovarianceManager.estimate_cpsd_wrapper(stimuli_stft['mix'], **s_bf,
                                                                 add_identity=add_identity_noise_noisy)

        self.cov_noise = CovarianceManager.estimate_cpsd_wrapper(stimuli_stft['noise'], **s_bf,
                                                                 add_identity=add_identity_noise_noisy)

        if dry_stft is not None:
            num_mics = mix_stft_shape[0]
            self.cov_dry_oracle = CovarianceManager.estimate_cpsd_wrapper(dry_stft, **s_bf)
            self.cov_dry_oracle = np.kron(self.cov_dry_oracle[..., 0], np.ones((num_mics, num_mics), complex))[
                ..., np.newaxis]

        # instead of estimating covariances again, just copy block-diagonal of bifrequency covariances to
        # narrowband covariances
        cov_shape = mix_stft_shape[:2] + (self.cov_noise.shape[-1],)
        self.copy_single_freq_covariances_from_bifreq_covariances(cov_shape)

    def copy_single_freq_covariances_from_bifreq_covariances(self, stft_shape):
        """Even when cross-freq components are evaluated, diagonal elements correspond.
        So instead of recomputing them, just copy from block-diagonal of bifrequency covariances"""

        num_mics, num_freqs, num_frames = stft_shape
        self.nb_cov_wet_oracle = np.zeros((num_mics, num_mics, num_freqs, num_frames), dtype=complex, order='F')
        self.nb_cov_noise = np.zeros_like(self.nb_cov_wet_oracle, order='F')
        self.nb_cov_noisy = np.zeros_like(self.nb_cov_wet_oracle, order='F')

        for tt in range(num_frames):
            for km in range(0, num_freqs * num_mics, num_mics):
                self.nb_cov_wet_oracle[..., km // num_mics, tt] = self.cov_wet_oracle[km:km + num_mics, km:km + num_mics, tt]
                self.nb_cov_noise[..., km // num_mics, tt] = self.cov_noise[km:km + num_mics, km:km + num_mics, tt]
                self.nb_cov_noisy[..., km // num_mics, tt] = self.cov_noisy[km:km + num_mics, km:km + num_mics, tt]

    def remove_identity(self, phi):
        if self.add_identity_noise_noisy:
            return phi - g.diagonal_loading * np.identity(phi.shape[0])[..., np.newaxis]
        else:
            return phi

    def plot_cov(self, true_cov=False, amp_range=None, time_frame=-1):

        font_size = 'x-large'

        if true_cov:
            x, v, y = self.phi_wet_bf_true, self.phi_vv_bf_true, None
            if all([x is None, v is None]):
                return
            titles = ['True wet', 'True noise', None, 'True dry']

        else:
            x, v, y = self.cov_wet_gevd, self.remove_identity(self.cov_noise), \
                      self.remove_identity(self.cov_noisy)
            titles = ['Est. wet (GEVD)', 'Est. noise', 'Est. mix', 'True dry']

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

        fig_opt = dict(figsize=(6, 6), constrained_layout=True)
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', **fig_opt)

        u.plot_matrix(x, axes[0, 0], title=titles[0], xy_label='', amp_range=amp_range, show_colorbar=False)
        fig, im = u.plot_matrix(v, axes[0, 1], title=titles[1], xy_label='', amp_range=amp_range, show_colorbar=False)

        if y is not None:
            fig, im = u.plot_matrix(y, axes[1, 0], title=titles[2], xy_label='', amp_range=amp_range, show_colorbar=False)
        else:
            axes[-1, 0].set_visible(False)

        if true_cov and self.cov_dry_oracle is not None:
            fig, im = u.plot_matrix(self.cov_dry_oracle, axes[1, 1],
                                    title=titles[3], xy_label='', amp_range=amp_range, show_colorbar=False)
        else:
            axes[-1, -1].set_visible(False)

        cb = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, )
        cb.set_label('Magnitude [dBm]', size=font_size)
        cb.ax.tick_params(labelsize=font_size)

        # make sure all axes are y inverted
        for ax in axes.flatten():
            if not ax.yaxis_inverted():
                ax.invert_yaxis()

        fig.show()

        return fig

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
    def generate_covariance(corr_coefficient: float, cov_shape, variances, covariance_type='equicorrelated'):

        if isinstance(variances, float) or isinstance(variances, int):
            # warnings.warn(f'generate_covariance: variances is a scalar, assuming all variances are equal to {variances}')
            variances = np.ones(shape=cov_shape[0]) * variances
        elif variances is None or (isinstance(variances, np.ndarray) and variances.ndim > 2):
            raise ValueError('variances must be a scalar or a vector')

        if covariance_type == 'ar(1)':
            raise NotImplementedError
            # r[i, j] = corr_coefficient ** abs(i - j) * variance
            # r = np.zeros(shape=cov_shape, dtype=complex)
            # for i in range(cov_shape[0]):
            #     for j in range(cov_shape[1]):
            #         r[i, j] = corr_coefficient ** abs(i - j) * variance

            # implement code above more efficiently, given that r is symmetric and Toeplitz
            # r = corr_coefficient ** np.abs(np.arange(cov_shape[0]) - np.arange(cov_shape[1])[:, None]) * variance
            # assert np.allclose(r, r1)
        elif covariance_type == 'white':
            r = np.identity(cov_shape[0], dtype=complex) * variances[0]
        else:  # if covariance_type == 'equicorrelated':
            r = CovarianceManager.generate_covariance_equicorrelated(variances, corr_coefficient, cov_shape)

        return r

    @staticmethod
    def project_covariance_into_psd_cone(r):
        """ Project into the PSD cone by eigenvalue decomposition: a PSD matrix has non-negative eigenvalues. """
        eigs, eigve = np.linalg.eigh(r)
        if any(eigs < 0):
            warnings.warn('Covariance matrix is not positive definite. Projecting into PSD cone')
            eigs[eigs < 0] = 0
            r = eigve @ np.diag(eigs) @ eigve.conj().T
        return r

    @staticmethod
    def generate_covariance_equicorrelated(variances, corr_coefficient, cov_shape):
        # r = g.rng.uniform(0, corr_coefficient * np.sqrt(variance), size=cov_shape)
        # r = g.rng.uniform(-corr_coefficient * np.sqrt(variance), corr_coefficient * np.sqrt(variance), size=cov_shape)
        # r = g.rng.normal(loc=corr_coefficient * np.sqrt(variance), scale=0.01, size=cov_shape)
        # r = corr_coefficient * np.sqrt(variance) * np.ones(shape=cov_shape)
        # r = r + 1j * r
        # r = np.triu(r).conj() + np.tril(r)  # diagonal is doubled
        # np.fill_diagonal(r, np.sqrt(g.rng.uniform(variance - variance / 2, variance + variance / 2, size=cov_shape[0])))
        # r = r @ r.conj().T
        # r = variance * r / np.mean(np.diag(r))  # need to compensate otherwise variance is higher for higher correlation
        # Like in Stoica's paper
        # r = corr_coefficient * variance * np.ones(shape=cov_shape, dtype=complex)
        # np.fill_diagonal(r, variance)
        r = np.diag(variances).astype(complex)
        # all cross-elements are found as np.sqrt(r[i, i] * r[j, j]), that is an upper bound on the cross-correlation,
        # and then multiplied by corr_coefficient. Diagonal elements are preserved.
        if corr_coefficient != 0:

            # corr_coefficient_mat = g.rng.normal(loc=corr_coefficient, scale=0.001, size=cov_shape)
            corr_coefficient_mat = np.ones(cov_shape, dtype=complex) * corr_coefficient
            # corr_coefficient_mat[corr_coefficient_mat < 0] = corr_coefficient_mat[corr_coefficient_mat < 0] * -1
            # corr_coefficient_mat[corr_coefficient_mat > 1] = 1 - g.eps
            # corr_coefficient_mat = np.triu(corr_coefficient_mat).conj() + np.tril(corr_coefficient_mat)

            for ii in range(cov_shape[0]):
                for jj in range(ii + 1, cov_shape[1]):
                    r[ii, jj] = np.sqrt(r[ii, ii] * r[jj, jj]) * corr_coefficient_mat[ii, jj]
                    r[jj, ii] = r[ii, jj].conj()

        return r

    @staticmethod
    def generate_target_covariance(sig_corr_coefficient, stft_shape, corr_type='freq', percentage_active_bins=1.,
                                   variances=None, covariance_type='equicorrelated', covariance_mask=None):
        # Signal covariance
        (num_mics_, num_freqs_, num_frames) = stft_shape
        corr_type = CovarianceManager.filter_correlation_type(corr_type)

        if (corr_type == 'frequency') or (corr_type is None and sig_corr_coefficient == 0):
            rs_K = CovarianceManager.generate_covariance(sig_corr_coefficient, (num_freqs_, num_freqs_), variances,
                                                         covariance_type=covariance_type)
        else:
            raise ValueError(f"Target speaker is a point source. Only frequency correlation can be arbitrary. "
                             f"User requested {corr_type} correlation with {sig_corr_coefficient} correlation coefficient.")

        # Apply random or deterministic mask to covariance matrix
        if percentage_active_bins < 1:
            rs_K = CovarianceManager.apply_mask_to_spectral_covariance(rs_K, percentage_active_bins)

        if covariance_mask is not None:
            assert covariance_mask.shape == rs_K.shape
            rs_K = rs_K * covariance_mask

        rs_K = CovarianceManager.project_covariance_into_psd_cone(rs_K)

        rs_KM_crb = np.kron(rs_K, np.ones((num_mics_, num_mics_), complex))

        return rs_K, rs_KM_crb

    @staticmethod
    def generate_noise_covariance(stft_shape, snr_db=+np.inf, noise_corr_coefficient=0, corr_type='freq',
                                  percentage_active_bins=1., variances=None, covariance_type='equicorrelated'):
        """
        Generate the true noise covariance matrix with given correlation coefficient and variance.

        :param snr_db:
        :param percentage_active_bins:
        :param noise_corr_coefficient: the correlation coefficient of the noise
        :param stft_shape: the shape of the STFT
        :param corr_type: can be 'freq_space', 'freq', 'space'
        :param variances: the variance of the noise (diagonal elements of the covariance matrix)

        :return: covariance matrix
        :rtype: np.ndarray
        """

        (num_mics_, num_freqs_, num_frames) = stft_shape
        num_mics_freqs = num_mics_ * num_freqs_
        corr_type = CovarianceManager.filter_correlation_type(corr_type)

        # noise correlated across frequency AND space or uncorrelated
        if (corr_type is None and noise_corr_coefficient == 0) or corr_type == 'frequency+space':
            rv = CovarianceManager.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                       variances, covariance_type=covariance_type)

        elif corr_type == 'frequency':
            # noise correlated across frequency, not across space
            # rv_freq = CovarianceManager.generate_covariance(noise_corr_coefficient, noise_var, (num_freqs_, num_freqs_))
            # rv = np.kron(rv_freq, np.identity(num_mics_))
            rv = CovarianceManager.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                       variances, covariance_type=covariance_type)

            # set to 0 all sub-diagonals which are not multiples of num_mics_
            for i in range(num_mics_freqs):
                for j in range(i + 1, num_mics_freqs):
                    if (j - i) % num_mics_ != 0:
                        rv[i, j] = 0
                        rv[j, i] = 0

        elif corr_type == 'space':
            # noise correlated across space BUT NOT frequency
            rv = CovarianceManager.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                       variances, covariance_type=covariance_type)
            rv = u.ForceToZeroOffBlockDiagonal(rv, 0, num_mics_)

        else:
            raise ValueError(f'Unknown corr_type: {corr_type} but correlation is {noise_corr_coefficient}')

        rv = CovarianceManager.project_covariance_into_psd_cone(rv)

        # rescale covariance matrices to match the desired SNR.
        # noise_power_original = np.var(noise_samples)
        snr_linear = u.db_to_linear(snr_db)
        noise_power_original = np.real(np.mean(np.diag(rv)))
        reference_power = 1
        power_rescaling_coefficient = (reference_power / (snr_linear * noise_power_original))
        rv *= power_rescaling_coefficient

        # name not exact: we sometimes mask only one microphone with this implementation
        if percentage_active_bins < 1:
            raise NotImplementedError("perc_active_noise_freq < 1 not implemented yet")
            # cm.phi_vv_bf_true = cm.apply_mask_to_spectral_covariance(cm.phi_vv_bf_true,
            #                                                          exp_settings['perc_active_noise_freq'])

        if np.alltrue(rv == 0):
            raise ValueError('Noise covariance is all zeros')

        return rv

    @staticmethod
    def compute_snr_from_covariance_matrices(reference_cov=None, noise_cov=None):

        reference_pow = 1 if reference_cov is None else np.mean(np.abs(np.diag(reference_cov)))
        noise_pow = 1 if noise_cov is None else np.mean(np.abs(np.diag(noise_cov)))

        effective_snr_db = u.linear_to_db(reference_pow / noise_pow)

        return effective_snr_db

    @staticmethod
    @njit(cache=True)
    def compute_correction_term(nstft, overlap, stft_shape):
        """
        Compute the correction term for the covariance matrix.
        :param nstft:
        :param overlap: a number between 0 and nstft-1
        :param stft_shape:
        :return:
        """

        (_, num_freqs, num_frames) = stft_shape

        # Use nstft because delay correction depends on window size = nstft, not num_freqs (= nstft/2+1)
        lag = np.zeros(num_frames, np.int32)
        for tt in range(1, num_frames):
            lag[tt] = tt * (nstft - overlap)

        correction_term = np.zeros((num_freqs, num_frames), np.complex128)
        for kk in range(num_freqs):
            for tt in range(num_frames):
                correction_term[kk, tt] = np.exp(-1j * 2 * np.pi * (kk / num_freqs) * lag[tt])

        return correction_term

    @staticmethod
    def apply_mask_to_spectral_covariance(r_freq, percentage_active_freqs):
        """
        Apply a mask to the covariance matrix. The mask is a binary matrix with 1s in the active frequencies and 0s in the
        inactive frequencies.
        :param r_freq: the covariance matrix
        :param percentage_active_freqs: the percentage of frequencies that are active
        :return: the filtered covariance matrix
        """

        num_freqs = r_freq.shape[0]
        num_freqs_active = int(np.round(percentage_active_freqs * num_freqs))

        # extract the indices of the frequencies that are active
        active_freqs = g.rng.choice(num_freqs, num_freqs_active, replace=False)

        # set the covariance of the inactive frequencies to zero
        r_freq_filtered = np.zeros_like(r_freq)
        r_freq_filtered[active_freqs, :] = r_freq[active_freqs, :]
        r_freq_filtered[:, active_freqs] = r_freq[:, active_freqs]

        return r_freq_filtered

    @staticmethod
    def estimate_cov_wet_gevd(cov_noisy, cov_noise, rtfs_shape):

        num_mics, num_freqs, _ = rtfs_shape
        cov_wet_est = np.zeros_like(cov_noisy)
        for tt in range(cov_wet_est.shape[-1]):

            noisy = cov_noisy[..., tt]
            noise = cov_noise[..., tt]
            eigenvals, eigves = scipy.linalg.eigh(noisy,
                                                  noise + np.eye(noise.shape[0]) * g.eps,
                                                  check_finite=False, driver='gvd')
            if np.all(eigenvals < 1):
                warnings.warn(f"All eigenvalues of cov_wet_gevd are less than 1. "
                              f"Algorithms will not work properly. ")

            # Several ways to get the left eigenvectors
            # keep only eigenvectors corresponding to eigenvalues > 1 and largest num_freqs eigenvalues
            eigves_left = noise @ eigves
            eigves_left_signal = eigves_left[:, -num_freqs:]
            eigenvals_signal = np.diag(np.maximum(0, eigenvals[-num_freqs:] - 1))
            # eigenvals_signal = np.diag(np.maximum(0, eigenvals[-num_freqs:]))  # sometimes subtracting one results in all 0 eigenvalues
            cov_wet_est[..., tt] = eigves_left_signal @ eigenvals_signal @ eigves_left_signal.conj().T

            # this should be identical, but it contains a matrix inverse, so it is probably slower
            # phi_ss = noise @ eigves[:, -num_freqs:] @ eigenvals_signal @ np.linalg.inv(eigves)[-num_freqs:]

        # return cov_noisy

        return cov_wet_est

    @staticmethod
    def generate_covariance_mask(stft_shape, covariance_type='neighbouring', num_neighbours=np.inf,
                                 checkerboard_spacing=np.inf):

        num_mics, num_freqs, _ = stft_shape
        # TODO change for time-varying covariance matrices

        if covariance_type == 'neighbouring':
            mask = np.ones((num_freqs, num_freqs), dtype=bool, order='F')
            mask = u.ForceToZeroOffBlockDiagonal(mask, num_neighbours, block_size=1)
        elif covariance_type == 'checkerboard':
            mask = CovarianceManager.generate_checkerboard_mask(num_freqs, grid_spacing=checkerboard_spacing)
        elif covariance_type == 'neighbouring+checkerboard':
            mask1 = CovarianceManager.generate_checkerboard_mask(num_freqs, grid_spacing=checkerboard_spacing)
            mask2 = np.ones((num_freqs, num_freqs), dtype=bool, order='F')
            mask2 = u.ForceToZeroOffBlockDiagonal(mask2, num_neighbours, block_size=1)
            mask = mask1 | mask2
        else:
            raise NotImplementedError(f"{covariance_type = } not implemented yet")

        return mask

    @staticmethod
    def generate_checkerboard_mask(num_freqs, grid_spacing=2):
        # grid_spacing = 2 --> resulting matrix is [1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]
        # grid_spacing = 3 --> resulting matrix is [1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]
        # generate vector [1, 0, 0, 1, 0, 0, 1, 0, 0, ...]. Must be longer or shorter depending on grid_spacing
        grid_vector = np.zeros(num_freqs, dtype=bool)
        grid_vector[::grid_spacing] = True
        mask = scipy.linalg.toeplitz(grid_vector)  # make toeplitz matrix from vector
        return mask
