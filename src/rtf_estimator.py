import copy
import warnings

import numpy as np
import numpy.linalg as npl
import scipy
from numba import njit

import src.global_constants as g
import src.utils as u
from src.utils import plot_matrix as plm

from scipy.stats import gmean
from scipy.optimize import minimize


class RtfEstimator:
    def __init__(self, idx_reference_mic_=0):
        # self.oracle_correlation_and_labelling = oracle_correlation_and_labelling
        self.index_reference_mic = idx_reference_mic_

        self.num_correlated_freqs = 0
        self.rtfs_gt = None
        self.flag_estimate_signal_threshold = None
        self.flag_scree_method = None
        self.flag_keep_num_freqs_eigenvectors = None
        self.flag_mdl_criterion = None

        self.methods = []

    def estimate_rtf_covariance_subtraction(self, clean_speech_cpsd, use_first_column=True) -> np.array:

        print("estimate_rtf_covariance_subtraction...")

        num_mics, _, num_freqs, num_time_frames = clean_speech_cpsd.shape
        rtfs = np.ones((num_mics, num_freqs, num_time_frames), dtype=complex)
        for tt in range(num_time_frames):
            for kk in range(num_freqs):
                rtfs[..., kk, tt] = self.covariance_subtraction_internal(clean_speech_cpsd[..., kk, tt],
                                                                         use_first_column)

        return rtfs

    def covariance_subtraction_internal(self, clean_speech_cpsd, use_first_column=False) -> np.array:
        if not np.alltrue(np.diag(clean_speech_cpsd) >= 0):
            return np.ones((clean_speech_cpsd.shape[0],), dtype=complex) * np.nan
        else:
            if use_first_column:
                return self.covariance_subtraction_first_column(clean_speech_cpsd)
            else:
                return self.covariance_subtraction_eigve(clean_speech_cpsd)

    # @njit(cache=True, fastmath=True)
    @staticmethod
    def estimate_rtf_covariance_whitening(noise_cpsd, noisy_cpsd, use_cholesky=True) -> np.array:
        """
            1) Markovich, Shmulik, Sharon Gannot, and Israel Cohen. ‘Multichannel Eigenspace Beamforming in a Reverberant Noisy
            Environment With Multiple Interfering Speech Signals’. 2009

            2) Markovich-Golan, Shmulik, and Sharon Gannot. ‘Performance Analysis of the Covariance Subtraction Method for
            Relative Transfer Function Estimation and Comparison to the Covariance Whitening Method’. 2015
            """

        num_mics, _, num_freqs, num_time_frames = noise_cpsd.shape
        # self.InitSelectionVector(num_mics)

        rtfs = np.ones((num_mics, num_freqs, num_time_frames), dtype=np.complex128)
        time_frames = range(num_time_frames)
        if use_cholesky:
            for tt in time_frames:
                if num_time_frames > 1:
                    print(f"Processing time frame {tt + 1}/{num_time_frames}", end='\r')
                for kk in range(num_freqs):
                    rtfs[..., kk, tt] = RtfEstimator.covariance_whitening_cholesky(noise_cpsd[..., kk, tt],
                                                                                   noisy_cpsd[..., kk, tt])
        else:
            for tt in time_frames:
                if num_time_frames > 1:
                    print(f"Processing time frame {tt + 1}/{num_time_frames}", end='\r')
                for kk in range(num_freqs):
                    rtfs[..., kk, tt] = RtfEstimator.covariance_whitening_generalized_eig(noise_cpsd[..., kk, tt],
                                                                                          noisy_cpsd[..., kk, tt],
                                                                                          hermitian_matrices=True)

        return rtfs

    @staticmethod
    def covariance_whitening_generalized_eig(noise_cpsd, noisy_cpsd, hermitian_matrices=False, normalize_rtf=True):

        if not hermitian_matrices:
            eigenvals, eigves = \
                scipy.linalg.eig(noisy_cpsd, noise_cpsd, check_finite=False, left=False, right=True)
        else:
            try:
                eigenvals, eigves = \
                    scipy.linalg.eigh(noisy_cpsd, noise_cpsd, check_finite=False, driver='gvd')
            except np.linalg.LinAlgError:
                return np.zeros_like(noise_cpsd[0, ...])

        _, max_right_eigve = RtfEstimator.sort_eigenvectors_get_major(eigenvals, eigves)
        rtf = noise_cpsd @ max_right_eigve
        if normalize_rtf:
            rtf = RtfEstimator.normalize_to_1(rtf)

        return np.squeeze(rtf)

    @staticmethod
    def covariance_whitening_generalized_eig_explicit_inversion(noise_cpsd, noisy_cpsd):

        # even if A, B are Hermitian, B^{-1} A is NOT hermitian!
        eigenvals, eigves = \
            scipy.linalg.eig(np.linalg.inv(noise_cpsd) @ noisy_cpsd, check_finite=False)

        _, max_eigve = RtfEstimator.sort_eigenvectors_get_major(eigenvals, eigves)
        max_eigve = noise_cpsd @ max_eigve
        rtf = RtfEstimator.normalize_to_1(max_eigve)

        return np.squeeze(rtf)

    @staticmethod
    def covariance_whitening_cholesky(noise_cpsd, noisy_cpsd) -> np.array:
        # `noise_cpsd` must be Hermitian (symmetric if real-valued) and positive-definite

        _, maj_eigve_noisy_whitened, noise_cpsd_sqrt = RtfEstimator.get_eigenvectors_whitened_noisy_cov(noise_cpsd,
                                                                                                        noisy_cpsd)
        rtf = noise_cpsd_sqrt @ maj_eigve_noisy_whitened  # transform back from whitened domain
        rtf = RtfEstimator.normalize_to_1(rtf)

        return np.squeeze(rtf)

    @staticmethod
    # @njit((numba.c16[:, :], numba.c16[:, :], numba.i8), cache=True, fastmath=True)
    def get_eigenvectors_whitened_noisy_cov(noise_cpsd, noisy_cpsd, how_many=1):
        noise_cpsd_sqrt, noisy_cpsd_whitened = RtfEstimator.whiten_covariance(noise_cpsd, noisy_cpsd)
        eigva_noisy_whitened, eigve_noisy_whitened = np.linalg.eigh(noisy_cpsd_whitened)
        maj_eigva, maj_eigve_whitened = RtfEstimator.sort_eigenvectors_get_major(eigva_noisy_whitened,
                                                                                 eigve_noisy_whitened, how_many)

        return maj_eigva, maj_eigve_whitened, noise_cpsd_sqrt

    # @njit((numba.c16[:, :], numba.c16[:, :]), cache=True, fastmath=True)  # somehow it is slower with numba
    @staticmethod
    def whiten_covariance(noise_cpsd, noisy_cpsd):
        """
        1) Perform Cholesky decomposition on noise_cpsd: noise_cpsd = L @ L.conj().T
        2) Calculate whitened covariance R_white = L^-1 @  noisy_cpsd @ (L^(-1))^H
        :param noise_cpsd: noise spatial covariance
        :param noisy_cpsd: noisy spatial covariance
        :return: Cholesky factor L, whitened noisy spatial covariance
        """
        noise_cpsd_sqrt = np.linalg.cholesky(noise_cpsd)
        noise_cpsd_sqrt_inv = np.linalg.inv(noise_cpsd_sqrt)
        noisy_cpsd_whitened = noise_cpsd_sqrt_inv @ noisy_cpsd @ noise_cpsd_sqrt_inv.conj().T
        # assert u.is_hermitian(noisy_cpsd_whitened)
        return noise_cpsd_sqrt, noisy_cpsd_whitened

    @staticmethod
    # @njit(numba.complex128[:](numba.complex128[:]), cache=True, fastmath=True)
    def normalize_to_1(eigve_single_column):
        # normalize vector to get 1 at reference microphone
        if np.abs(eigve_single_column[g.idx_ref_mic]) < g.eps:
            eigve_normalized = np.zeros_like(eigve_single_column)
        else:
            eigve_normalized = eigve_single_column / eigve_single_column[g.idx_ref_mic]

        return eigve_normalized

    @staticmethod
    def normalize_to_1_eigenvector_matrix(eigve_matrix, num_freqs):
        num_freqs_mics, num_candidates = eigve_matrix.shape
        eigve_matrix_3d = eigve_matrix.reshape((-1, num_freqs, num_candidates), order='F')
        we_normalized = np.zeros_like(eigve_matrix_3d, dtype=np.complex128, order='F')
        for kk in range(num_freqs):
            we_normalized[:, kk] = RtfEstimator.normalize_to_1_fat_block(eigve_matrix_3d[:, kk])
        we_normalized = we_normalized.reshape((num_freqs_mics, num_candidates), order='F')

        return we_normalized

    @staticmethod
    def normalize_to_1_fat_block(eigve_matrix):
        num_mics, num_candidates = eigve_matrix.shape
        we_normalized = np.zeros_like(eigve_matrix, dtype=np.complex128, order='F')
        for cc in range(num_candidates):
            we_normalized[:, cc] = RtfEstimator.normalize_to_1(eigve_matrix[:, cc])

        return we_normalized

    @staticmethod
    def get_internal_loop_range(mic_freq_outer_idx, max_neighbours, num_freqs, num_mics):
        return np.arange(
            start=max(mic_freq_outer_idx - num_mics * max_neighbours, 0),
            stop=min(mic_freq_outer_idx + num_mics * max_neighbours + 1, num_freqs * num_mics),
            step=num_mics)

    # @njit((numba.c16[:], numba.c16[:, :], numba.i8), cache=True, fastmath=True)  # problematic because of np.squeeze
    @staticmethod
    def sort_eigenvectors_get_major(eigva, eigve, num_to_keep=1):
        """
        Return eigenvector corresponding to eigenvalue with maximum norm. if eigenvalues are not ALL finite, return NaN
        """

        if num_to_keep == -1:
            num_to_keep = len(eigva)  # keep all eigenvectors

        if not np.all(np.isfinite(eigva)):
            return np.ones_like(eigva)[:num_to_keep] * np.nan, np.ones_like(eigve)[:, :num_to_keep] * np.nan

        # Sort eigenvalues and eigenvectors in ascending order
        idx_largest_eigvas_sorted = np.argsort(np.real(eigva))
        eigva, eigve = eigva[idx_largest_eigvas_sorted], eigve[:, idx_largest_eigvas_sorted]

        return np.squeeze(eigva[-num_to_keep:]), np.squeeze(eigve[:, -num_to_keep:])

    @staticmethod
    def covariance_subtraction_first_column(phi_xx_tt, reference_mic=g.idx_ref_mic):
        return np.squeeze(phi_xx_tt[:, reference_mic] / (g.eps + phi_xx_tt[reference_mic, reference_mic]))

    @staticmethod
    # for single source, RTF corresponds to eigenvector corresponding to largest eigenvalue
    def covariance_subtraction_eigve(phi_xx_tt):
        eigva, eigve = scipy.linalg.eigh(phi_xx_tt, check_finite=False)
        _, rtf = RtfEstimator.sort_eigenvectors_get_major(eigva, eigve)
        rtf = RtfEstimator.normalize_to_1(rtf)
        return rtf

    @staticmethod
    def get_indices(k1, k2, num_mics):
        i1 = slice(k1, k1 + num_mics)
        i2 = slice(k2, k2 + num_mics)
        i1_int = k1 // num_mics
        i2_int = k2 // num_mics
        return i1, i1_int, i2, i2_int

    def estimate_eigenvectors_bifreq(self, phi_vv_bf, phi_yy_bf, noise_stft_shape, sub_or_whiten='sub',
                                     thr_override=None, num_retained_eigva=-1):

        raise NotImplementedError
        assert (phi_yy_bf.ndim == phi_vv_bf.ndim == 3)
        if sub_or_whiten == 'sub':
            noise_sqrt = None
            thr_subtraction = 1e-3 if thr_override is None else thr_override
            wa, we_signal_3d = self.estimate_eigenvectors_bifreq_subtraction(noise_stft_shape, phi_vv_bf,
                                                                             phi_yy_bf, thr=thr_subtraction,
                                                                             num_retained_eigva=num_retained_eigva)
        elif sub_or_whiten == 'whiten':
            thr_whitening = 1e-6 if thr_override is None else thr_override
            wa, we_signal_3d, _, noise_sqrt = self.estimate_eigenvectors_bifreq_whitening(noise_stft_shape,
                                                                                          phi_vv_bf, phi_yy_bf,
                                                                                          thr=1 + thr_whitening,
                                                                                          num_retained_eigva=num_retained_eigva,
                                                                                          plot_eigve=False)
        elif sub_or_whiten == 'gevd':
            thr_gevd = 1e-6 if thr_override is None else thr_override
            num_mics, num_freqs, _ = noise_stft_shape

            try:
                eigenvals, eigves = scipy.linalg.eigh(phi_yy_bf[..., 0],
                                                      phi_vv_bf[..., 0], check_finite=False, driver='gvd')
            except np.linalg.LinAlgError:
                eigenvals, eigves = scipy.linalg.eig(np.linalg.pinv(phi_vv_bf[..., 0]) @ phi_yy_bf[..., 0])
                eigenvals, eigves = self.sort_eigenvectors_get_major(eigenvals, eigves, -1)

            eigves_left = phi_vv_bf[..., 0] @ eigves

            # keep only eigenvectors corresponding to eigenvalues > 1 and largest num_freqs eigenvalues
            num_retained_eigva = self.select_eigenvectors_signal_subspace(eigenvals, noise_stft_shape,
                                                                          num_retained_eigva, thr=1 + thr_gevd)
            eigves_left_signal = eigves_left[:, -num_retained_eigva:]
            eigenvals_signal = np.diag(np.maximum(0, eigenvals[-num_retained_eigva:] - 1))
            phi_ss_est = eigves_left_signal @ eigenvals_signal @ eigves_left_signal.conj().T

            # perform eigendecomposition of phi_ss_est
            wa_s, we_s = np.linalg.eigh(phi_ss_est)
            wa_s = wa_s[-num_retained_eigva:]
            we_s = we_s[:, -num_retained_eigva:]

            # reshape and check
            we_s_3d = we_s.reshape((num_freqs, num_mics, -1))
            assert np.any(np.isnan(we_s_3d)) or np.allclose(we_s[:num_mics], we_s_3d[0])
            assert np.any(np.isnan(we_s_3d)) or np.allclose(we_s[-num_mics:], we_s_3d[-1])

            return wa_s, we_s_3d, None
        else:
            raise ValueError(f"sub_or_whiten is {sub_or_whiten}")
        return wa, we_signal_3d, noise_sqrt

    def estimate_eigenvectors_bifreq_subtraction(self, stft_shape, phi_vv, phi_yy, thr=1e-3, num_retained_eigva=-1):
        mode_str = 'cov subtraction'
        print(f"EstimateRtf_FullBifreq in {mode_str} mode")

        phi_xx = np.squeeze(phi_yy - phi_vv)
        eigva, eigve = scipy.linalg.eigh(phi_xx)

        num_retained_eigva = self.select_eigenvectors_signal_subspace(eigva, stft_shape, num_retained_eigva, thr)

        wa_signal, we_signal = RtfEstimator.sort_eigenvectors_get_major(eigva, eigve, num_retained_eigva)
        num_mics, num_freqs, _ = stft_shape

        assert False  # correct reshaping is probably (num_mics, num_freqs, num_retained_eigva)! Check
        we_signal_3d = we_signal.reshape((num_freqs, num_mics, num_retained_eigva))

        return wa_signal, we_signal_3d

    def estimate_eigenvectors_bifreq_whitening(self, stft_shape, phi_vv_bf, phi_yy_bf, thr,
                                               num_retained_eigva=-1, plot_eigve=False):

        mode_str = 'cov whitening'
        print(f"EstimateRtf_FullBifreq in {mode_str} mode")
        assert (phi_yy_bf.shape[-1] == phi_vv_bf.shape[-1] == 1)  # only implemented for single time-frame
        num_mics, num_freqs, _ = stft_shape

        # Whiten R_y and perform eigendecomposition. Eigenvectors are sorted.
        wa_white, we_white, noise_cpsd_sqrt = self.get_eigenvectors_whitened_noisy_cov(phi_vv_bf[..., 0],
                                                                                       phi_yy_bf[..., 0], -1)

        # Keep only (eigva, eigve) pairs that span signal subspace
        num_retained_eigva = self.select_eigenvectors_signal_subspace(wa_white, stft_shape, num_retained_eigva, thr)
        wa_signal_white = np.maximum(0, wa_white[-num_retained_eigva:] - 1)
        we_signal_white = we_white[:, -num_retained_eigva:]

        # reconstruct phi_ss_est from whitened eigenvectors
        phi_ss_est_whitened = (we_signal_white @ np.diag(wa_signal_white) @ we_signal_white.conj().T)
        phi_ss_est = noise_cpsd_sqrt @ phi_ss_est_whitened @ noise_cpsd_sqrt.conj().T

        # perform eigendecomposition of phi_ss_est
        wa_white, we_signal_nonwhite = np.linalg.eigh(phi_ss_est)
        wa_signal_white = wa_white[-num_retained_eigva:]
        we_signal_nonwhite = we_signal_nonwhite[:, -num_retained_eigva:]

        # reshape and check
        we_signal_nonwhite_3d = we_signal_nonwhite.reshape((num_freqs, num_mics, -1))
        assert np.any(np.isnan(we_signal_nonwhite_3d)) or np.allclose(we_signal_nonwhite[:num_mics],
                                                                      we_signal_nonwhite_3d[0])
        assert np.any(np.isnan(we_signal_nonwhite_3d)) or np.allclose(we_signal_nonwhite[-num_mics:],
                                                                      we_signal_nonwhite_3d[-1])

        if plot_eigve:
            plm(we_white[..., -num_freqs:], title="Possible signal eigenvectors")

        return wa_signal_white, we_signal_nonwhite_3d, we_signal_nonwhite, noise_cpsd_sqrt

    def select_eigenvectors_signal_subspace(self, eigva, stft_shape, num_retained_eigva, thr):

        _, num_freqs, num_snapshots = stft_shape

        # allow_more_eigenvectors_thank_rank only makes sense for debugging.
        # Theory says no more eigenvectors than rank = num_freqs should be retained.
        allow_more_eigenvectors_thank_rank = False
        if allow_more_eigenvectors_thank_rank:
            num_max_retained_eigva = len(eigva)
            warnings.warn("Watch out, allowing more eigenvectors than reasonable into signal partition.")
        else:
            num_max_retained_eigva = num_freqs

        # user did not specify number of eigvas to retain: determine automatically
        if num_retained_eigva < 0:
            if self.flag_keep_num_freqs_eigenvectors:
                num_retained_eigva = num_freqs

            elif self.flag_estimate_signal_threshold:
                print(f"Ignore eigenvectors with eigenvalues less than {thr}")
                num_retained_eigva = self.num_eigenvectors_to_keep(eigva,
                                                                   admitted_range=(1, num_freqs), thr=thr)

            elif self.flag_scree_method:
                eigvas = np.maximum(eigva - 1, g.log_pow_threshold)
                second_diff = (np.gradient(np.gradient(eigvas)))
                num_retained_eigva = len(eigva) - np.argmax(second_diff)

            elif self.flag_mdl_criterion:
                if num_snapshots == 1:
                    warnings.warn("Only one snapshot for MDL criterion, DOUBLE CHECK IF PASSING RIGHT ARGUMENTS")
                eigvas = np.maximum(eigva[::-1], g.log_pow_threshold)
                num_retained_eigva = self.minimize_mdl_criterion(eigvas, num_snapshots)

            else:  # discard eigenvalues with negative values
                print(f"Ignore eigenvectors with eigenvalues less than {np.floor(thr)}")
                num_retained_eigva = self.num_eigenvectors_to_keep(eigva, admitted_range=(1, num_freqs),
                                                                   thr=np.floor(thr))

        # Whatever the method, we cannot have more signal eigenvectors than the number of frequencies
        num_retained_eigva = int(np.clip(num_retained_eigva, a_min=1, a_max=num_max_retained_eigva))
        # print(f"Keeping {num_retained_eigva} eigenvectors")

        return num_retained_eigva

    @staticmethod
    def num_eigenvectors_to_keep(wa, admitted_range, thr=1e-6):
        num_retained_eigva = min(admitted_range[1], max(admitted_range[0], int(np.sum([wa > thr]))))
        return num_retained_eigva

    @staticmethod
    def full_bifreq_rtf_from_eigve(we_max_mask, we_signal):
        num_freqs, num_mics, _ = we_signal.shape
        rtfs = np.zeros((num_mics, num_freqs), dtype=complex)
        for kk in range(num_freqs):
            rtf = we_signal[kk, :, we_max_mask[kk]]
            rtfs[:, kk] = RtfEstimator.normalize_to_1(rtf)
        return rtfs

    # find maximum per each column. Each column (=eigenvector) can only have one maximum, EXCEPT the column
    # corresponding to the largest eigenvector
    @staticmethod
    def find_rtf_labelling(we_signal_reshaped, assume_oracle_ordering=False):

        we_signal_norm = np.linalg.norm(we_signal_reshaped, axis=1)  # microphone dim consumed
        if assume_oracle_ordering:
            assert False
            num_freqs, num_retained_eigenvectors = we_signal_norm.shape
            # assume low freqs of desired signal are correlated, high freqs are not
            if num_freqs == num_retained_eigenvectors:
                we_max_mask = np.arange(stop=num_freqs)[::-1]
            else:
                we_max_mask = np.zeros((num_freqs,), int)
                num_correlated_freqs = num_freqs - num_retained_eigenvectors
                we_max_mask[:num_correlated_freqs] = num_retained_eigenvectors - 1
                we_max_mask[num_correlated_freqs:] = np.arange(num_retained_eigenvectors)[::-1]
        else:
            we_max_mask = np.nanargmax(we_signal_norm, axis=1)

        return we_max_mask

    @staticmethod
    def find_rtf_labelling_brute_force(we, rtfs_gt=None):

        num_freqs, num_mics, num_candidates = we.shape
        error_matrix = np.zeros((num_freqs, num_candidates), float)
        for kk in range(num_freqs):
            for cc in range(num_candidates):
                candidate_rtf = RtfEstimator.normalize_to_1(we[kk, :, cc])
                # error_matrix[kk, cc] = np.linalg.norm(candidate_rtf - rtfs_gt[:, kk])
                error_matrix[kk, cc] = u.HermitianAngle(rtfs_gt[..., kk, np.newaxis], candidate_rtf[..., np.newaxis])
        try:
            min_error_mask = np.nanargmin(error_matrix, axis=1)
        except ValueError:
            print("All estimates are NaN in find_rtf_labelling_brute_force")
            min_error_mask = np.zeros(num_freqs, dtype=int)

        if g.debug_mode:
            plm(error_matrix, title="Error matrix")

        return min_error_mask

    @staticmethod
    def find_rtf_labelling_average(we, wa=None):

        return RtfEstimator.find_rtf_labelling_weighted_average(we)

        num_freqs, num_mics, num_candidates = we.shape
        for kk in range(num_freqs):
            for cc in range(num_candidates):
                we[kk, ..., cc] = RtfEstimator.normalize_to_1(we[kk, ..., cc])

        weighted = False
        if weighted and wa is not None:
            rtfs = np.average(we, weights=np.atleast_1d(wa).flatten(), axis=-1)
        else:
            rtfs = np.nanmean(we, axis=-1)  # average across candidates

        rtfs = np.transpose(rtfs)
        return rtfs

    @staticmethod
    def find_rtf_labelling_weighted_average(we):
        num_freqs, num_mics, num_candidates = we.shape
        we_magnitude = np.linalg.norm(we, axis=1, keepdims=True)  # shape will be (num_freqs, num_candidates)
        # we_magnitude = np.ones_like(we_magnitude)

        # we_normalized = RtfEstimator.normalize_to_1_eigenvector_matrix(we)
        # we_normalized = we / we_magnitude
        rtfs = np.zeros((num_freqs, num_mics), dtype=complex)
        for kk in range(num_freqs):
            temp = np.atleast_2d(we_magnitude[kk, ...]) @ we[kk, ...].T
            rtfs[kk] = RtfEstimator.normalize_to_1(temp.flatten())
        rtfs = np.transpose(rtfs)
        return rtfs

    @staticmethod
    def find_largest_rank1_collection(we, eigva_ratio_thr=2.):

        f = RtfEstimator.find_subblock_largest_principal_vector(we)

        num_mics, num_candidates = f.shape
        eigva_ratio_curr = 1e8
        for cc in range(num_candidates):
            f_sel = f[:, cc:]
            eigva, eigve = np.linalg.eigh(f_sel @ f_sel.conj().T)
            eigva_ratio_next = eigva[-1] / np.sum(eigva[:-1])
            variation = eigva_ratio_next / eigva_ratio_curr
            if variation > eigva_ratio_thr or (eigva[-1] > 0.1 and eigva_ratio_next > 5):
                return num_candidates - cc
            eigva_ratio_curr = eigva_ratio_next

        warnings.warn("Check find_largest_rank1_collection!")
        return 1

    @staticmethod
    def find_subblock_largest_principal_vector(f):
        # first, find and select block-row which has largest norm in principal eigenvector
        num_freqs, num_mics, num_candidates = f.shape
        norms = np.zeros(num_freqs)
        for kk in range(num_freqs):
            norms[kk] = np.linalg.norm(f[kk, :, -1])
        k_best = np.argmax(norms)
        f = copy.deepcopy(f[k_best])
        return f

    @staticmethod
    def find_rtf_labelling_svd(we, oracle_rtf=None):

        num_freqs, num_mics, num_candidates = we.shape
        rtfs = np.zeros((num_freqs, num_mics), dtype=complex)

        if oracle_rtf is None:  # default
            for kk in range(num_freqs):
                eigva, eigve = np.linalg.eigh(we[kk] @ we[kk].conj().T)
                rtfs[kk] = RtfEstimator.normalize_to_1(eigve[:, -1])

        else:  # oracle matching (debug)
            for kk in range(num_freqs):
                eigva, eigve = np.linalg.eigh(we[kk] @ we[kk].conj().T)
                error = np.zeros(eigve.shape[-1])
                oracle_rtf_kk = RtfEstimator.normalize_to_1(oracle_rtf[:, kk])
                for cc in range(eigve.shape[-1]):
                    candidate_rtf = RtfEstimator.normalize_to_1(eigve[:, cc])
                    # error[cc] = np.linalg.norm(candidate_rtf - oracle_rtf_kk)  # mse
                    error[cc] = u.HermitianAngle(oracle_rtf_kk[..., np.newaxis],
                                                 candidate_rtf[..., np.newaxis])  # Herm angle

                rtfs[kk] = RtfEstimator.normalize_to_1(eigve[:, np.nanargmin(error)])

        rtfs = np.transpose(rtfs)
        return rtfs

    def estimate_rtf(self, cov_holder, stft_shape, num_retained_eigva=-1, covariance_mask_target=None):
        """
        Estimate the RTF using a variety of methods.

        Parameters:
        - cov_holder: object containing the bifrequency covariances.
        - stft_shape: shape of the STFT data.
        - num_retained_eigva: number of eigenvectors to retain when using eigenvector-based methods.

        Returns:
        - named_estimates: dictionary of RTF estimates, where the keys are the names of the methods used and the values are the estimates.
        """

        named_estimates = dict()
        num_mics, _, num_freqs, num_time_frames = cov_holder.nb_cov_noise.shape

        num_mics_freqs = num_mics * num_freqs
        if num_retained_eigva > num_mics_freqs:
            warnings.warn(f"num_retained_eigva: {num_retained_eigva} but num_mics_freqs: {num_mics_freqs}!")
            return named_estimates

        if 'CS' in self.methods:
            named_estimates['CS'] = self.estimate_rtf_covariance_subtraction(
                cov_holder.nb_cov_noisy - cov_holder.nb_cov_noise,
                use_first_column=False)
        if 'CW' in self.methods:
            named_estimates['CW'] = self.estimate_rtf_covariance_whitening(cov_holder.nb_cov_noise,
                                                                           cov_holder.nb_cov_noisy,
                                                                           use_cholesky=False)
        if any('CS-EV' in method_name for method_name in self.methods):
            if 'CS-EV-top1' in self.methods:
                was, wes, _ = self.estimate_eigenvectors_bifreq(cov_holder.cov_noise, cov_holder.cov_noisy, stft_shape,
                                                                'sub')
                we_max_mask = self.find_rtf_labelling_brute_force(wes, self.rtfs_gt)
                named_estimates['CS-EV-top1'] = self.full_bifreq_rtf_from_eigve(we_max_mask, wes)
            if 'CS-EV-SV' in self.methods:
                was, wes, _ = self.estimate_eigenvectors_bifreq(cov_holder.cov_noise, cov_holder.cov_noisy, stft_shape,
                                                                'sub', num_retained_eigva=num_retained_eigva)
                named_estimates['CS-EV-SV'] = self.find_rtf_labelling_svd(wes)

        # if any('CW-EV' in method_name for method_name in self.methods):
        #     wa_cw, we_cw, noise_sqrt = self.estimate_eigenvectors_bifreq(cov_holder.phi_vv_bf, cov_holder.phi_yy_bf,
        #                                                                  stft_shape,
        #                                                                  'gevd',
        #                                                                  num_retained_eigva=num_retained_eigva)
        #     normalize_vectors_to_unit_norm = False
        #     if normalize_vectors_to_unit_norm:
        #         print("Normalizing sub-block eigenvectors to have unit norm")
        #         we_cw = we_cw / np.linalg.norm(we_cw, axis=1, keepdims=True)
        #
        #     if 'CW-EV-avg' in self.methods:
        #         wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
        #
        #         num_freqs, num_mics, num_candidates = we.shape
        #         magnitude_matrix = np.zeros((num_freqs, num_candidates), float)
        #         for kk in range(num_freqs):
        #             for cc in range(num_candidates):
        #                 magnitude_matrix[kk, cc] = u.squared_euclidean_norm(we[kk, :, cc])
        #
        #         sorted_indices = magnitude_matrix.argsort(axis=-1)
        #
        #         # now sort the eigenvectors according to the magnitude matrix
        #         for kk in range(num_freqs):
        #             we[kk] = we[kk, :, sorted_indices[kk]].T
        #
        #         if g.debug_mode:
        #             plm(magnitude_matrix, title="Magnitude matrix")
        #
        #         named_estimates['CW-EV-avg'] = self.find_rtf_labelling_svd(we)
        #     if 'CW-EV-top1' in self.methods:
        #         wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
        #         we_max_mask = self.find_rtf_labelling_brute_force(we, self.rtfs_gt)
        #         named_estimates['CW-EV-top1'] = self.full_bifreq_rtf_from_eigve(we_max_mask, we)
        #     if 'CW-EV-SV' in self.methods:
        #         wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
        #         named_estimates['CW-EV-SV'] = self.find_rtf_labelling_svd(we)
        #     if 'CW-EV-SV-oracle' in self.methods:
        #         wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
        #         named_estimates['CW-EV-SV-oracle'] = self.find_rtf_labelling_svd(we, self.rtfs_gt)
        #     if 'CW-EV-SV-scaled' in self.methods:
        #         wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
        #         we_scaled = np.abs(np.atleast_1d(wa)[np.newaxis, np.newaxis, :]) * we
        #         named_estimates['CW-EV-SV-scaled'] = self.find_rtf_labelling_svd(we_scaled)
        #
        # if 'CW-EV-SV' in self.methods:
        #     named_estimates['CW-EV-SV'] = self.estimate_rtf_cw_ev_sv(cov_holder)

        if 'CW-SV' in self.methods:
            named_estimates['CW-SV'] = self.estimate_rtf_cw_sv(cov_holder,
                                                               rtf_target=self.rtfs_gt if g.debug_mode else None)
        if 'CW-SV-mask' in self.methods:
            named_estimates['CW-SV-mask'] = self.estimate_rtf_cw_sv(cov_holder,
                                                                    rtf_target=self.rtfs_gt if g.debug_mode else None,
                                                                    covariance_mask_target_=covariance_mask_target)

        if 'CS-SV' in self.methods:

            print("CS-SV")
            phi_ss = np.squeeze(cov_holder.cov_noisy - cov_holder.cov_noise)
            assert False  # correct reshaping is probably (num_mics, num_freqs, num_retained_eigva)! Check
            phi_ss = phi_ss.reshape((num_freqs, num_mics, -1))

            rtfs = np.zeros((num_freqs, num_mics), dtype=complex)
            for kk, phi_ss_kk in enumerate(phi_ss):
                left_sv, s_vals, vh = np.linalg.svd(phi_ss_kk, full_matrices=False)
                rtfs[kk] = left_sv[:, 0]
                rtfs[kk] = RtfEstimator.normalize_to_1(left_sv[:, 0])

            named_estimates['CS-SV'] = np.transpose(rtfs)

        return named_estimates

    @staticmethod
    def minimize_mdl_criterion(eigvas, num_snapshots):
        """Determine the number of signal eigenvalues using MDL criterion.

        The MDL criterion is based on information theory and is described in Wax and Kailath's
        paper "Detection of Signals by Information Theoretic Criteria".

        Args:
            eigvas (ndarray): The set of eigenvalues to evaluate.
            num_snapshots (int): The number of snapshots used in the calculation of the eigenvalues.

        Returns:
            num_signal_eigva (int): The number of signal eigenvalues in the given set of eigenvalues.

        """
        num_candidates = len(eigvas)
        score = np.zeros((num_candidates,))

        for mk in range(num_candidates):
            score[mk] = RtfEstimator.mdl_criterion(eigvas, mk, num_snapshots)
        num_signal_eigva = np.argmin(score)
        return num_signal_eigva

    @staticmethod
    def mdl_criterion(eigenvalues, num_selected, num_snapshots):
        """
        Calculate the Minimum Description Length (MDL) criterion for selecting a number of eigenvalues.

        Parameters
        ----------
        eigenvalues : numpy array
            The array of eigenvalues to select from.
        num_selected : int
            The number of eigenvalues to select.
        num_snapshots : int
            The number of snapshots used in calculating the eigenvalues.

        Returns
        -------
        float
            The MDL criterion value.
        """
        num_total = len(eigenvalues)
        mean_geometric = gmean(eigenvalues[num_selected:])
        mean_arithmetic = np.mean(eigenvalues[num_selected:])
        first_term = -(num_total - num_selected) * num_snapshots * np.log(g.eps + (mean_geometric / mean_arithmetic))
        second_term = 0.5 * num_selected * (2 * num_total - num_selected) * np.log(g.eps + num_snapshots)
        return first_term + second_term

    @staticmethod
    def estimate_rtf_cw_sv(cov_holder_, rtf_target=None, rtfs_shape=None, covariance_mask_target_=None):
        """
        Estimates the RTF of a sound source using the "wideband covariance whitening SVD-direct" method

        :param rtf_target: The oracle RTF of the sound source in the shape of (num_mics, num_freqs, num_time_frames).
        :param rtfs_shape: The shape of the RTF to be estimated. If None, the shape of the covariance matrices is used.
        :param cov_holder_:  An object that holds the noisy/noise only (phi_yy_bf, phi_vv_bf) covariance matrices
        :return: The estimated RTF of the sound source in the shape of (num_mics, num_freqs, num_time_frames)
        """

        if rtfs_shape is None:
            rtfs_shape = cov_holder_.nb_cov_noisy.shape[1:]
        num_mics, num_freqs, num_time_frames = rtfs_shape
        rtfs = np.zeros(rtfs_shape, dtype=np.complex128)

        mask = None
        if covariance_mask_target_ is not None:
            mask = np.kron(covariance_mask_target_, np.ones((num_mics, num_mics), dtype=bool))
            mask = mask.reshape((num_mics, num_freqs, num_mics * num_freqs), order='F')

        for tt in range(num_time_frames):
            phi_ss = cov_holder_.cov_wet_gevd[..., tt]

            """
            # Filter elements of phi_ss that are so low as to be considered estimation error
            powers_ratio = RtfEstimator.estimate_correlation_coeff_power_ratio(phi_ss, rtfs_shape)
            powers_ratio_mask = powers_ratio > 0.4
            powers_ratio_mask_mics_freqs = np.kron(powers_ratio_mask, np.ones((num_mics, num_mics)))
            phi_ss = phi_ss * powers_ratio_mask_mics_freqs
            """

            phi_ss = phi_ss.reshape((num_mics, num_freqs, num_mics * num_freqs), order='F')

            if mask is not None:
                for mm in range(mask.shape[0]):
                    for kk in range(mask.shape[1]):
                        phi_ss[mm, kk, ~mask[mm, kk]] = 0

            for kk in range(num_freqs):
                left_sv, s_vals, vh = np.linalg.svd(phi_ss[:, kk], full_matrices=False)
                rtfs[:, kk, tt] = RtfEstimator.normalize_to_1(left_sv[:, 0])

                # temp_mat = np.column_stack((left_sv[:, 0], vh[0, kk * num_mics:(kk + 1) * num_mics].conj()))
                # rtfs[:, kk, tt] = np.linalg.svd(temp_mat, full_matrices=False)[0][:, 0]
                # rtfs[:, kk, tt] = RtfEstimator.normalize_to_1(rtfs[:, kk, tt])

            """
            # equivalent to the above, but using CPD tensor decomposition
            import tensorly
            phi_ss = phi_ss.reshape((num_freqs, num_mics, num_freqs*num_mics), order='c')
            cpd = tensorly.decomposition.CP(1)
            for kk in range(num_freqs):
                phi_ss_kk = np.array(np.split(phi_ss[kk], num_freqs, axis=-1))
                cpd_decomp = cpd.fit_transform(phi_ss_kk)
                rtfs[:, kk, tt] = np.squeeze(RtfEstimator.normalize_to_1(cpd_decomp.factors[1][:, 0]))
            """

        return rtfs

    @staticmethod
    def estimate_correlation_coeff_power_ratio(phi_ss, rtfs_shape):

        num_mics, num_freqs, num_time_frames = rtfs_shape
        phi_ss_reshape = phi_ss.reshape((num_mics, num_freqs, num_mics, num_freqs), order='F')

        # Calculate the reference power for each frequency (the power of the block diagonal)
        ref_powers = np.zeros((num_freqs,))
        for kk in range(num_freqs):
            ref_powers[kk] = np.linalg.norm(phi_ss_reshape[:, kk, :, kk], ord='fro') + g.eps

        # Calculate the ratio of the power of each off-diagonal block to the power of the corresponding diagonal blocks
        powers_ratio = np.zeros((num_freqs, num_freqs))
        for kk1 in range(num_freqs):
            for kk2 in range(num_freqs):
                power = (np.linalg.norm(phi_ss_reshape[:, kk1, :, kk2], ord='fro'))
                powers_ratio[kk1, kk2] = power / np.sqrt(ref_powers[kk1] * ref_powers[kk2])

        return powers_ratio

    @staticmethod
    def get_diagonal_offdiagonal_power_ratio(freq_idx, num_mics, cov):

        mask_keep_sides = np.ones(cov.shape[-1], dtype=bool)
        mask_keep_sides[freq_idx * num_mics:(freq_idx + 1) * num_mics] = False
        mask_keep_center = np.bitwise_not(mask_keep_sides)

        cov_center = np.where(mask_keep_center, cov[:, freq_idx, :], 0)
        cov_sides = np.where(mask_keep_sides, cov[:, freq_idx, :], 0)

        norm_central_block = np.linalg.norm(cov_center)
        norm_off_central_block_no_rep = np.linalg.norm(cov_sides) + g.eps

        ratio = norm_central_block / norm_off_central_block_no_rep

        return ratio

    @staticmethod
    def get_diagonal_offdiagonal_power_ratio_global(num_mics, cov):

        mask_keep_center = np.ones(cov.shape, dtype=bool)
        mask_keep_center = u.ForceToZeroOffBlockDiagonal(mask_keep_center, 0, num_mics)
        mask_keep_sides = np.bitwise_not(mask_keep_center)

        cov_center = np.where(mask_keep_sides, cov, 0)
        cov_sides = np.where(mask_keep_center, cov, 0)

        norm_central_block = np.linalg.norm(cov_center)
        norm_off_central_block_no_rep = np.linalg.norm(cov_sides) + g.eps

        ratio = norm_central_block / norm_off_central_block_no_rep

        return ratio

    @staticmethod
    def estimate_rtf_cw_sv_cholesky(cov_holder_, rtfs_shape=None):

        warnings.warn("this method fails for non-white noise! DO NOT USE!")
        if rtfs_shape is None:
            rtfs_shape = cov_holder_.nb_cov_noisy.shape[1:]
        num_mics, num_freqs, num_time_frames = rtfs_shape
        rtfs = np.zeros(rtfs_shape, dtype=complex)

        for tt in range(num_time_frames):
            if num_time_frames > 1:
                print(f"Processing time frame {tt + 1}/{num_time_frames}", end='\r')

            noise_cpsd_sqrt, phi_ss_white = \
                RtfEstimator.whiten_covariance(cov_holder_.cov_noise[..., tt], cov_holder_.cov_noisy[..., tt])
            phi_ss_white = phi_ss_white - np.identity(num_mics * num_freqs)

            # reshape so that we can iterate over frequency
            assert False  # correct reshaping is probably (num_mics, num_freqs, num_retained_eigva)! Check
            phi_ss_white = phi_ss_white.reshape((num_freqs, num_mics, num_mics * num_freqs))

            # estimate RTF as major left sing. vec. of each M x KM block
            for kk, phi_ss_kk in enumerate(phi_ss_white):
                left_sv, s_vals, vh = np.linalg.svd(phi_ss_kk, full_matrices=False)
                rtfs[:, kk, tt] = left_sv[:, 0]

            assert False  # correct reshaping is probably (num_mics, num_freqs, num_retained_eigva)! Check
            rtfs_reshape = np.reshape(rtfs, (num_mics * num_freqs, -1))
            rtfs_reshape = noise_cpsd_sqrt @ rtfs_reshape
            rtfs = np.reshape(rtfs_reshape, rtfs.shape)

            for kk, phi_ss_kk in enumerate(phi_ss_white):
                rtfs[:, kk, tt] = RtfEstimator.normalize_to_1(rtfs[:, kk, tt])

        return rtfs

    def estimate_rtf_cw_ev_sv(self, cov_holder, rtfs_shape=None):

        if rtfs_shape is None:
            rtfs_shape = cov_holder.nb_cov_noisy.shape[1:]
        num_mics, num_freqs, num_time_frames = rtfs_shape
        rtfs = np.zeros((num_mics, num_freqs, num_time_frames), dtype=np.complex128)

        for tt in range(num_time_frames):
            eigenvals, we_s = scipy.linalg.eigh(cov_holder.cov_wet_gevd[..., tt])

            # keep only eigenvectors corresponding to eigenvalues > 1 and largest num_freqs eigenvalues
            # print(eigenvals_gevd)
            # num_retained_eigva = self.select_eigenvectors_signal_subspace(np.maximum(0, eigenvals_gevd),
            #                                                               (1, num_freqs,
            #                                                                cov_holder.num_frames_covariance_estimation),
            #                                                               -1, thr=1 + 1e-6)
            num_retained_eigva = num_freqs
            we_s = we_s[:, -num_retained_eigva:]

            # reshape and check
            we_s_3d = we_s.reshape((num_freqs, num_mics, -1))
            assert np.any(np.isnan(we_s_3d)) or np.allclose(we_s[:num_mics], we_s_3d[0])
            assert np.any(np.isnan(we_s_3d)) or np.allclose(we_s[-num_mics:], we_s_3d[-1])
            rtfs[..., tt] = self.find_rtf_labelling_svd(we_s_3d)

        return rtfs

    @staticmethod
    def comm_mat(m, n):
        # determine permutation applied by K
        w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

        # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
        return np.eye(m * n)[w, :]

    @staticmethod
    def vec_lower_triangular(X):
        return np.reshape(X[np.tril_indices(X.shape[0])], (-1, 1), order='F')

    @staticmethod
    def unvec_lower_triangular(x, n):
        # transform vectorized lower triangular matrix back to matrix
        # x is a vector of length n(n+1)/2
        # n is the number of rows/columns of the matrix

        # create empty matrix
        X = np.zeros((n, n), order='F', dtype=x.dtype)

        # fill lower triangular part of matrix
        X[np.tril_indices(n)] = x[:, 0]

        return X

    @staticmethod
    def vec(X):
        return np.reshape(X, (-1, 1), order='F')

    @staticmethod
    def unvec(x):
        num_cols = np.sqrt(len(x))
        assert np.isclose(num_cols, np.round(num_cols))
        num_cols = int(num_cols)
        return np.reshape(x, (num_cols, num_cols), order='F')

    @staticmethod
    def get_diag(X):
        return np.reshape(np.diag(X), (-1, 1), order='F')

    @staticmethod
    def compute_avg_power_over_freq(B, num_mics, num_freqs):
        avg_power = np.zeros((num_freqs,))
        b = np.diag(B)
        for kk in range(num_freqs):
            avg_power[kk] = np.real(np.mean(b[kk * num_mics: (kk + 1) * num_mics]))
        return avg_power

    @staticmethod
    def normalize_by_ref_mic(B, num_mics_, num_freqs_):
        # normalize A so that the first diagonal element is 1, first + num_freqs is 1, etc.
        for kk in range(num_freqs_):
            B[kk * num_mics_: (kk + 1) * num_mics_, kk * num_mics_: (kk + 1) * num_mics_] = \
                B[kk * num_mics_: (kk + 1) * num_mics_, kk * num_mics_: (kk + 1) * num_mics_] / \
                B[kk * num_mics_, kk * num_mics_]
        return B

    @staticmethod
    def compute_normalization_by_ref_mic(B, num_mics_, num_freqs_):
        """
        B is diagonal with elements b1, b2, ..b_num_mics, c1, c2, .., c_num_mics, ....,
        and its total elements are num_freqs*num_mics.
        then normalization matrix C is such that B @ C is diagonal with elements
        1, b2/b1, ..., b_num_mics/b1, 1, c2/c1, ..., c_num_mics/c1, ....
        """
        b = RtfEstimator.get_diag(B)
        c = np.zeros_like(b)
        for kk in range(num_freqs_):
            c[kk * num_mics_: (kk + 1) * num_mics_] = 1 / b[kk * num_mics_]
        C = np.diagflat(c)
        return C

    @staticmethod
    @njit(cache=True)
    def enforce_cauchy_schwartz_inequality(phi):
        # precalculate the diagonal elements of phi_ss_bar
        phi_diag = np.maximum(g.eps, np.diag(phi).real)

        # use prior knowledge that phi_ss_bar is hermitian to improve computational speed of loop above
        for kk1 in range(phi.shape[0]):
            for kk2 in range(kk1 + 1, phi.shape[1]):
                off_diag_mod = np.abs(phi[kk1, kk2])
                max_mod = np.sqrt(phi_diag[kk1] * phi_diag[kk2])
                if off_diag_mod > max_mod:
                    phi[kk1, kk2] = phi[kk1, kk2] * (max_mod / off_diag_mod)
                    phi[kk2, kk1] = np.conj(phi[kk1, kk2])

        return phi
