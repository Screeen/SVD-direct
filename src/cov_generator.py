import numpy as np
import scipy.linalg

from numba import njit

from src.cov_manager import CovarianceManager
from src.settings_manager import SettingsManager
import src.utils as u
import src.global_constants as g


class CovarianceGenerator:
    def __init__(self):
        pass

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
            r = np.identity(cov_shape[0], dtype=complex) * variances[-1]

            # diagonal, white over mics, but not over freqs (for WB is as easy as completely white, maybe even a bit easier? If true cov. is available.)
            # variances_freqs = variances[:variances.size // 4]
            # r1 = np.diag(variances_freqs)
            # r = np.kron(r1, np.identity(4)).astype(complex)

            # diagonal, white over freqs, but not over mics (harder for WB and for narrowband, but WB suffers more)
            # variances_mics = variances[:2]
            # r1 = np.diag(variances_mics)
            # r = np.kron(np.identity(cov_shape[0] // 2), r1).astype(complex)

            # diagonal, non-white. Much harder for WB.
            # r = np.diagflat(variances).astype(complex)

        else:  # if covariance_type == 'equicorrelated':
            r = CovarianceGenerator.generate_covariance_equicorrelated(variances, corr_coefficient, cov_shape)

        return r

    @staticmethod
    def generate_covariance_equicorrelated(variances, corr_coefficient, cov_shape):
        """
        Generate a covariance matrix with given correlation coefficient and variance.
        All cross-elements are found as np.sqrt(r[i, i] * r[j, j]), that is an upper bound on the cross-correlation,
        and then multiplied by corr_coefficient. Diagonal elements are preserved.
        :param variances: the variance (diagonal elements of the covariance matrix)
        :param corr_coefficient: the correlation coefficient (between -1 and 1), determines off-diagonal elements
        :param cov_shape: the shape of the covariance matrix
        :return: generated covariance matrix
        """

        if cov_shape[0] != cov_shape[1]:
            raise ValueError(f'cov_shape must be a tuple of two equal integers, but is {cov_shape}')

        if len(cov_shape) != 2:
            raise ValueError(f'cov_shape must be a tuple of two integers, but is {cov_shape}')

        if len(variances) != cov_shape[0]:
            raise ValueError(f'variances must have length {cov_shape[0]}')

        r = np.diag(variances).astype(complex)

        if corr_coefficient != 0:
            corr_coefficient_mat = np.ones(cov_shape, dtype=complex) * corr_coefficient
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
        corr_type = SettingsManager.filter_correlation_type(corr_type)

        if (corr_type == 'frequency') or (corr_type is None and sig_corr_coefficient == 0):
            rs_K = CovarianceGenerator.generate_covariance(sig_corr_coefficient, (num_freqs_, num_freqs_), variances,
                                                           covariance_type=covariance_type)
        else:
            raise ValueError(f"Target speaker is a point source. Only frequency correlation can be arbitrary. "
                             f"User requested {corr_type} correlation with {sig_corr_coefficient} correlation coefficient.")

        # Apply random or deterministic mask to covariance matrix
        if percentage_active_bins < 1:
            rs_K = CovarianceGenerator.apply_mask_to_spectral_covariance(rs_K, percentage_active_bins)

        if covariance_mask is not None:
            assert covariance_mask.shape == rs_K.shape
            rs_K = rs_K * covariance_mask

        rs_K = CovarianceManager.project_covariance_into_psd_cone(rs_K, warn_if_not_psd=False)

        rs_KM_crb = np.kron(rs_K, np.ones((num_mics_, num_mics_), complex))

        return rs_K, rs_KM_crb

    @staticmethod
    def generate_noise_covariance(stft_shape, snr_db=+np.inf, noise_corr_coefficient=0., corr_type='freq',
                                  percentage_active_bins=1., variances=None, covariance_type='equicorrelated',
                                  covariance_mask=None):
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
        corr_type = SettingsManager.filter_correlation_type(corr_type)

        # noise correlated across frequency AND space or uncorrelated
        if (corr_type is None and noise_corr_coefficient == 0) or corr_type == 'frequency+space':
            rv = CovarianceGenerator.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                         variances, covariance_type=covariance_type)

        elif corr_type == 'frequency':
            # noise correlated across frequency, not across space
            # rv_freq = CovarianceGenerator.generate_covariance(noise_corr_coefficient, noise_var, (num_freqs_, num_freqs_))
            # rv = np.kron(rv_freq, np.identity(num_mics_))
            rv = CovarianceGenerator.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                         variances, covariance_type=covariance_type)

            # set to 0 all sub-diagonals which are not multiples of num_mics_
            for i in range(num_mics_freqs):
                for j in range(i + 1, num_mics_freqs):
                    if (j - i) % num_mics_ != 0:
                        rv[i, j] = 0
                        rv[j, i] = 0

        elif corr_type == 'space':
            # noise correlated across space BUT NOT frequency
            rv = CovarianceGenerator.generate_covariance(noise_corr_coefficient, (num_mics_freqs, num_mics_freqs),
                                                         variances, covariance_type=covariance_type)
            rv = u.ForceToZeroOffBlockDiagonal(rv, num_mics_, 0)

        else:
            raise ValueError(f'Unknown corr_type: {corr_type} but correlation is {noise_corr_coefficient}')

        # Apply random or deterministic mask to covariance matrix
        if covariance_mask is not None:
            assert covariance_mask.shape == rv.shape
            rv = rv * covariance_mask

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

    @classmethod
    def generate_covariance_matrices_freq_domain(cls, variances_noise, variances_target, atf_target, stft_shape, sett,
                                                 cov_mask_target=None):

        """ Generate true covariance matrices for the target and noise signals in frequency domain."""
        cov_dry_oracle_single_mic, \
            cov_dry_oracle = cls.generate_target_covariance(sett['correlation_target'],
                                                            stft_shape,
                                                            sett['correlation_target_type'],
                                                            sett['perc_active_target_freq'],
                                                            variances_target,
                                                            sett['correlation_target_pattern'],
                                                            covariance_mask=cov_mask_target)

        cov_mask_noise = None
        if sett['correlation_target_pattern'] != 'equicorrelated':
            cov_mask_noise = cls.generate_covariance_mask(stft_shape[1] * stft_shape[0],
                                                          sett['correlation_noise_pattern'],
                                                          num_neighbours=sett['num_neighbours_noise'] * stft_shape[0],
                                                          grid_spacing=sett['grid_spacing_noise'] * stft_shape[0])

        # cm.phi_xx_bf converges to cm.phi_dry_bf_true for infinite number of realizations
        A = np.diag(atf_target.flatten('F'))
        phi_wet_bf_true = A @ cov_dry_oracle @ A.conj().T
        ref_power_wet = cls.compute_snr_from_covariance_matrices(phi_wet_bf_true)
        correlated_noise_snr = sett['noises_info'][0]['snr'] - ref_power_wet
        phi_vv_bf_true = \
            cls.generate_noise_covariance(stft_shape,
                                          snr_db=correlated_noise_snr,
                                          noise_corr_coefficient=float(sett['correlation_noise']),
                                          corr_type=sett['correlation_noise_type'],
                                          percentage_active_bins=float(sett['perc_active_noise_freq']),
                                          variances=variances_noise,
                                          covariance_type=sett['correlation_noise_pattern'],
                                          covariance_mask=cov_mask_noise)

        # add noise floor to noise covariance matrix (sensor noise)
        phi_vv_bf_true += cls.generate_noise_covariance(stft_shape,
                                                        snr_db=g.white_noise_floor_db - ref_power_wet,
                                                        variances=1.0)

        return cov_dry_oracle_single_mic, cov_dry_oracle, phi_wet_bf_true, phi_vv_bf_true

    @staticmethod
    def generate_covariance_mask(cov_size, covariance_type_='neighbouring', num_neighbours=-1,
                                 grid_spacing=-1):

        # TODO change for time-varying covariance matrices

        # if there is a '+' in covariance_type, then we need to generate two masks and combine them
        covariance_type_list = covariance_type_.split('+')

        mask = np.zeros((cov_size, cov_size), dtype=bool, order='F')
        mask1 = np.zeros((cov_size, cov_size), dtype=bool, order='F')
        for covariance_type in covariance_type_list:
            if covariance_type == 'neighbouring':
                all_ones = np.ones((cov_size, cov_size), dtype=bool, order='F')
                mask1 = u.ForceToZeroOffBlockDiagonal(all_ones, block_size=1, max_distance_diagonal=num_neighbours)
            elif covariance_type == 'stripes':
                if grid_spacing > 1:
                    mask1 = CovarianceGenerator.generate_stripes_mask(cov_size, grid_spacing=grid_spacing)
            elif covariance_type == 'grid':
                if grid_spacing > 1:
                    mask1 = CovarianceGenerator.generate_grid_mask(cov_size, grid_spacing)
            elif covariance_type == 'white':
                mask1 = np.diag(np.ones(cov_size, dtype=bool))
            elif covariance_type == 'equicorrelated':  # do not mask
                mask1 = np.ones((cov_size, cov_size), dtype=bool, order='F')
            else:
                raise NotImplementedError(f"{covariance_type = } not implemented yet")
            mask = mask.astype(bool) | mask1.astype(bool)

        return mask

    @staticmethod
    def compute_snr_from_covariance_matrices(reference_cov=None, noise_cov=None):

        reference_pow = 1 if reference_cov is None else np.mean(np.abs(np.diag(reference_cov)))
        noise_pow = 1 if noise_cov is None else np.mean(np.abs(np.diag(noise_cov)))

        effective_snr_db = u.linear_to_db(reference_pow / noise_pow)

        return effective_snr_db

    @staticmethod
    @njit(cache=True)
    def generate_stripes_mask(num_freqs, grid_spacing=2):

        # grid_spacing = 2 --> resulting matrix is [1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]
        # grid_spacing = 3 --> resulting matrix is [1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]
        # generate vector [1, 0, 0, 1, 0, 0, 1, 0, 0, ...]. Must be longer or shorter depending on grid_spacing
        grid_vector = np.zeros(num_freqs, dtype=bool)
        grid_vector[::grid_spacing] = True
        mask = scipy.linalg.toeplitz(grid_vector)  # make toeplitz matrix from vector
        return mask

    @staticmethod
    @njit(cache=True)
    def generate_grid_mask(num_freqs, grid_spacing=2):
        mask = np.zeros((num_freqs, num_freqs))
        for n1 in range(1, num_freqs):
            for n2 in range(1, num_freqs):
                if (n1 % grid_spacing == 0) and (n2 % grid_spacing == 0):
                    mask[n1, n2] = 1

        return mask

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
