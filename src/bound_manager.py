import copy

import numpy as np
import numpy.linalg as npl
from scipy import linalg as spl

import src.global_constants as g
import scipy
import warnings


class BoundManager:
    """
    This class contains all the methods for computing the bounds on the estimation error.
    """
    @staticmethod
    def compute_derivative_g_by_theta(atf):
        """See eq. (3.30) Kay"""
        num_mics, num_freqs = atf.shape
        mat = np.zeros((num_mics, num_mics, num_freqs), dtype=atf.dtype)
        for kk in range(num_freqs):
            mat[..., kk] = BoundManager.compute_derivative_g_by_theta_single_freq(np.squeeze(atf[:, kk]))
        return mat

    @staticmethod
    def compute_derivative_g_by_theta_single_freq(atf):
        # ATF for a single frequency
        num_mics = len(atf)
        g_by_theta_single_freq = np.zeros((num_mics, num_mics), dtype=complex)
        norm_ref_mic_inv = 1. / atf[g.idx_ref_mic] ** 2
        ref_mic_inv = 1. / atf[g.idx_ref_mic]
        for ii in range(1, num_mics):  # row index, skip first row (all zeros)
            for jj in range(num_mics):  # column index
                if jj == g.idx_ref_mic:  # first column
                    g_by_theta_single_freq[ii, jj] = -atf[ii] * norm_ref_mic_inv
                elif ii == jj:  # diagonal
                    g_by_theta_single_freq[ii, jj] = ref_mic_inv

        return g_by_theta_single_freq

    """ 
    estimate (time-domain) variance from freq domain signal, independently per each microphone
    check also: https://github.com/scipy/scipy/commit/672619a7d7812d7efc4584ef4104d54b56fa4cdd
    (commit 672619a7d7812d7efc4584ef4104d54b56fa4cdd scipy)
    """

    @staticmethod
    def power_from_stft(x_stft, avg_freqs=False):
        df = (g.fs / 2) / (x_stft.shape[1] - 1)
        # same as np.mean(noise_stft*noise_stft.conj()), as in Kay eq. (15.8)
        x_pow = np.mean(np.real(x_stft) ** 2 + np.imag(x_stft) ** 2, axis=-1, keepdims=True, dtype=float)
        if avg_freqs:
            x_pow = (x_pow[0] + x_pow[-1] + 2 * np.sum(x_pow[1:-1])) * df
        else:
            x_pow = x_pow * df * x_pow.shape[1] * 2
        return np.squeeze(x_pow)

    @staticmethod
    def crb_conditional(atf_target, noise_cov, stft_shape, dry_stft, remove_identity):

        num_mics, num_freqs, num_time_frames = stft_shape

        if noise_cov.ndim < 3:
            noise_cov = noise_cov[..., np.newaxis]

        if remove_identity:
            assert False
            for tt in range(noise_cov.shape[-1]):
                noise_cov[..., tt] = noise_cov[..., tt] - np.identity(num_freqs * num_mics)

        rv_true_inv = npl.pinv(noise_cov[..., 0])
        fisher = np.zeros_like(noise_cov[..., 0])
        for tt in range(num_time_frames):
            capital_s = np.diag(dry_stft[..., tt].flatten('F'))
            if capital_s.shape != rv_true_inv.shape:
                capital_s = np.kron(capital_s, np.identity(num_mics))
            fisher = fisher + (capital_s.conj().T @ rv_true_inv @ capital_s)

        crb_theta = npl.pinv(fisher)
        crb_theta = BoundManager.crb_atf_to_rtf(crb_theta, atf_target)

        return crb_theta

    @staticmethod
    def crb_unconditional(atf, rs, rv, stft_shape, ry_est):
        """ Compute the CRB for the unconditional case"""

        small_eigvas_threshold = 1e-10
        A = np.diag(atf.flatten('F'))
        Ah = A.conj().T
        num_mics, num_freqs, num_frames = stft_shape
        N = num_mics * num_freqs

        rs, rv, ry_est = np.squeeze(rs), np.squeeze(rv), np.squeeze(ry_est)

        def unit_matrix(m, k):
            E = np.zeros((N, N), complex, 'F')
            E[m, k] = 1
            return E

        rx_true_inv = npl.pinv(A @ rs @ Ah + rv, hermitian=True, rcond=small_eigvas_threshold)

        fisher = np.zeros((N, N), complex, 'F')

        # # Loop over the upper triangular part of the matrix
        for mm in range(N):
            Gm = unit_matrix(mm, mm) @ rs @ Ah
            rx_gm = rx_true_inv @ Gm
            for kk in range(mm, N):
                Fk = A @ rs @ unit_matrix(kk, kk)
                rx_fk = rx_true_inv @ Fk
                # fast trace: np.trace(a.dot(b)) == np.sum(inner1d(a, b.T))
                # fisher[mm, kk] = np.sum(np.inner(rx_fk, rx_gm))
                # needs checking
                fisher[mm, kk] = np.trace(rx_fk @ rx_gm)

        fisher_bl = np.zeros_like(fisher)
        for mm in range(N):
            Gm = unit_matrix(mm, mm) @ rs @ Ah
            for kk in range(0, N):
                Gk = unit_matrix(kk, kk) @ rs @ Ah
                fisher_bl[mm, kk] = np.trace(rx_true_inv @ Gk @ rx_true_inv @ Gm)

        # Get the indices of the upper triangular part of the matrix.
        # Set the values of the lower triangular part to the complex conjugates
        # of the corresponding values in the upper triangular part.
        idx = np.triu_indices(N)
        fisher[idx[1], idx[0]] = fisher[idx[0], idx[1]].conj()

        fisher_tr = fisher_bl.conj().T
        fisher_tl = fisher.conj()
        fisher_br = fisher
        big_fisher = np.block([[fisher_tl, fisher_tr], [fisher_bl, fisher_br]])
        big_crb = np.linalg.pinv(big_fisher, hermitian=True, rcond=small_eigvas_threshold) / num_frames
        crb_atf = big_crb[:N, :N]

        atf = np.reshape(np.diag(A), (num_mics, num_freqs), order='f')
        crb_rtf = BoundManager.crb_atf_to_rtf(crb_atf, atf)

        return crb_rtf

    @staticmethod
    def crb_unconditional_kronecker_form(atf, rs, rv, stft_shape, ry_est):

        small_eigvas_threshold = 1e-10
        A = np.diag(atf.flatten('F'))

        num_mics, num_freqs, num_frames = stft_shape
        N = num_mics * num_freqs

        rs, rv, ry_est = np.squeeze(rs), np.squeeze(rv), np.squeeze(ry_est)
        rx_true_inv = npl.pinv(A @ rs @ A.conj().T + rv, hermitian=True, rcond=small_eigvas_threshold)

        Ja = scipy.linalg.khatri_rao(A.conj() @ rs.T, np.identity(N))
        Ja_star = scipy.linalg.khatri_rao(np.identity(N), A @ rs)
        # Jrs = np.kron(A.conj(), A)
        # Jrs_star = np.zeros_like(Jrs)
        # Jrv = np.identity(N*N)
        # Jrv_star = np.zeros_like(Jrv)
        J = np.concatenate((Ja, Ja_star), axis=1)  # concatenate horizontally
        # J = np.concatenate((Ja, Ja_star, Jrs, Jrs_star), axis=1)  # concatenate horizontally
        # J = np.concatenate((Ja, Ja_star, Jrs, Jrs_star, Jrv, Jrv_star), axis=1)  # concatenate horizontally

        big_fisher = J.conj().T @ (np.kron(rx_true_inv.T, rx_true_inv)) @ J
        big_crb = np.linalg.pinv(big_fisher, hermitian=True, rcond=small_eigvas_threshold) / num_frames

        # The top right block determines the CRB for the ATF
        crb_atf = big_crb[:N, :N]
        crb_rtf = BoundManager.crb_atf_to_rtf(crb_atf, atf)

        return crb_rtf

    @staticmethod
    def calculate_bounds(algo_names, atf_target, cm, stft_shape, desired_dry_stft, rs_km=None):

        bounds = dict()

        if not any('CRB' in x for x in algo_names):
            return bounds

        if atf_target.size > 100:
            warnings.warn(f"The CRB calculation is very slow for large arrays "
                          f"({stft_shape[0]} mics, {stft_shape[1]} freqs). Skipping.")
            return bounds

        if cm.cov_noise_gt is not None:
            noise_cov = cm.cov_noise_gt
        else:
            warnings.warn("Using estimated noise cov instead of true one in 'crb_conditional'")
            noise_cov = cm.cov_noise

        if cm.cov_dry_oracle is not None:
            dry_cov = cm.cov_dry_oracle
        else:
            warnings.warn("Using estimated dry cov instead of true one in 'crb_unconditional'")
            dry_cov = cm.phi_dry_bf

        for algo_name in algo_names:
            if 'CRB_conditional' in algo_name:
                bounds[algo_name] = BoundManager.crb_conditional(atf_target, noise_cov,
                                                                         stft_shape, desired_dry_stft,
                                                                         remove_identity=False)

            if 'CRB_unconditional' in algo_name:
                # b1 = BoundManager.crb_unconditional(atf_target, dry_cov, rv=noise_cov,
                #                                                      stft_shape=stft_shape, ry_est=cm.phi_yy_bf)
                bounds[algo_name] = BoundManager.crb_unconditional_kronecker_form(atf_target, dry_cov, rv=noise_cov,
                                                                                  stft_shape=stft_shape, ry_est=cm.cov_noisy)

        return bounds

    @staticmethod
    def crb_atf_to_rtf(crb, atf_target):

        num_mics_, num_freqs_ = atf_target.shape

        # Derivative of transformation from ATF to RTF
        g_dt = [BoundManager.compute_derivative_g_by_theta_single_freq(atf_target[:, kk])
                for kk in range(num_freqs_)]
        g_dt = spl.block_diag(*g_dt)

        # Transform CRB from ATF to RTF
        crb = g_dt @ crb @ g_dt.conj().T

        # extract diagonal, convert to float, ensure positivity, reshape
        crb = np.asarray(np.real(np.diag(crb)), dtype=np.float64)
        crb = np.maximum(0, crb)
        crb = np.reshape(crb, (num_mics_, num_freqs_), order='f')

        return crb
