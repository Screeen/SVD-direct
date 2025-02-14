import unittest
import numpy as np
import sys

sys.path.append('..')
sys.path.append('../src')

from src import settings_manager
import src.cov_manager as cov_manager
import src.global_constants as g
import src.rtf_estimator as rtf_estimator
import src.signal_generator as signal_generator
import src.utils as u
from src.utils import is_positiveDefinite, is_hermitian
import warnings
from src.cov_generator import CovarianceGenerator

nstft_test = 32
noverlap_test = nstft_test // 2
noverlap_percentage_test = 0.5
num_mics_test = 3
duration_sec_test = 0.1
num_freqs_test = 1 + nstft_test // 2
num_mics_freqs_test = num_mics_test * num_freqs_test
win_name_test = 'hann'
noise_corr_test = 0.5
extra_settings_test = {'noise_estimate_perturbation_amount': 0.0,
                       'desired': []}

estimate_cov_sett_test = dict(wideband=True,
                              avg_time_frames_=True,
                              warning_level='warning')

extra_settings_test['rir_settings'] = settings_manager.SettingsManager.get_rir_settings_from_settings_or_default()
extra_settings_test['rir_settings']['rtf_type'] = 'random-once-small'

sg_test_settings = dict(num_mics_max=num_mics_test, duration_noise_sec=duration_sec_test,
                        duration_output_sec=duration_sec_test,
                        nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                        **extra_settings_test)


def get_off_diag_indices(N):
    return np.where(~np.eye(N, dtype=bool))


def generate_hermitian_spd_matrix(d=6):
    # https://math.stackexchange.com/questions/267300/positive-definite-matrix-must-be-hermitian
    Sigma_k = generate_random_complex_matrix(d)
    Sigma_k = Sigma_k @ Sigma_k.T.conj()
    Sigma_k = Sigma_k + np.identity(d)
    return Sigma_k


def generate_random_complex_matrix(d=6):
    return g.rng.standard_normal((d, d)) + g.rng.standard_normal((d, d)) * 1j


def generate_rank1_matrix(d=6, return_matrix_only=False, symmetric=False, full_rank=False, pos_def=False):
    v1 = g.rng.standard_normal((d, 1)) + g.rng.standard_normal((d, 1)) * 1j
    v1 = v1 / v1[0]

    if symmetric:
        matrix = v1 @ v1.T.conj()
    else:
        v2 = g.rng.standard_normal((d, 1)) + g.rng.standard_normal((d, 1)) * 1j
        v2 = v2 / v2[0]
        matrix = v1 @ v2.T.conj()

    if full_rank:
        matrix = matrix + g.rng.standard_normal((d,)) * np.identity(d)

    if pos_def:
        matrix = matrix.conj().T @ matrix

    assert matrix.shape[0] == d
    assert matrix.shape[1] == d
    if not return_matrix_only:
        return np.squeeze(v1), matrix
    else:
        return matrix


def generate_covariance_wgn(atf=None, estimate_cpsd_settings=None, return_samples=False):
    if estimate_cpsd_settings is None:
        estimate_cpsd_settings = dict(wideband=True,
                                      avg_time_frames_=True,
                                      warning_level='warning')

    num_samples = int(duration_sec_test * g.fs)

    if atf is not None:
        s_samples = g.rng.standard_normal((1, num_samples,))
        _, _, s_stft = u.stft(s_samples, nstft_test, noverlap_test, win_name_test)
        s_stft = s_stft * atf[..., np.newaxis]
    else:
        s_samples = g.rng.standard_normal((num_mics_test, num_samples,))
        _, _, s_stft = u.stft(s_samples, nstft_test, noverlap_test, win_name_test)

    rd = cov_manager.CovarianceManager.estimate_cov(s_stft, **estimate_cpsd_settings)

    if return_samples:
        return rd, s_stft
    else:
        return rd


def get_test_covariance_matrices():
    # sg = signal_generator.SignalGenerator(**sg_test_settings)
    identity = np.eye(num_mics_test * num_freqs_test)[..., np.newaxis]
    noises_cov_list = [identity,
                       generate_covariance_wgn() + identity,
                       generate_hermitian_spd_matrix(num_mics_freqs_test)[..., np.newaxis],
                       CovarianceGenerator.generate_covariance(noise_corr_test, (num_mics_freqs_test,
                                                                                 num_mics_freqs_test),
                                                               1)[
                           ..., np.newaxis]]
    noise_cov_list_description = ['identity', 'wgn + identity', 'spd', 'correlated']
    return noise_cov_list_description, noises_cov_list


def bifreq_estimation_template(false_test_rtf_true_test_cov):
    sg = signal_generator.SignalGenerator(**sg_test_settings)
    rm = sg.rir_manager
    noise_cov_list_description, noises_cov_list = get_test_covariance_matrices()
    cm = cov_manager.CovarianceManager

    for rn, noise_cov_description in zip(noises_cov_list, noise_cov_list_description):
        print(f"{noise_cov_description = }")
        atf = rm.compute_or_generate_acoustic_transfer_function(atf_type='random')
        rtf = rm.generate_rtf_from_atf(atf)

        rd = generate_covariance_wgn(rtf)
        rx = rd + rn

        cm.cov_noise = rn
        cm.cov_wet_oracle = rd
        cov_wet_oracle_block_diag = u.ForceToZeroOffBlockDiagonal(cm.cov_wet_oracle, num_mics_test, 0)
        cm.cov_noisy = rx

        cov_noise_sqrt = np.linalg.cholesky(rn[..., 0])
        cov_noise_sqrt_inv = np.linalg.inv(cov_noise_sqrt)
        cov_noisy_whitened = cov_noise_sqrt_inv @ rx[..., 0] @ cov_noise_sqrt_inv.conj().T
        c = (cov_noisy_whitened, cov_noise_sqrt)

        if false_test_rtf_true_test_cov:
            # check r_s and cm.cov_wet_oracle are equal
            rd_est1, _ = cm.estimate_cov_wet_gevd(rx, rn, sg.stft_shape, modality='gevd')
            rd_est2, _ = cm.estimate_cov_wet_gevd(rx, rn, sg.stft_shape, modality='cholesky')
            rd_est3, _ = cm.estimate_cov_wet_gevd(rx, rn, sg.stft_shape, modality='gevd-nb')

            np.testing.assert_almost_equal(rd_est1, cm.cov_wet_oracle)
            np.testing.assert_almost_equal(rd_est2, cm.cov_wet_oracle)
            np.testing.assert_almost_equal(rd_est3, cov_wet_oracle_block_diag)

        else:
            cm.cov_wet_gevd, _ = cm.estimate_cov_wet_gevd(rx, rn, sg.stft_shape)
            re = rtf_estimator.RtfEstimator()
            re.rtfs_gt = rtf

            # wa_cw, we_cw, _ = re.estimate_eigenvectors_bifreq(rn, rx, sg.stft_shape, 'whiten')
            # oracle procedure
            # wa, we = copy.deepcopy(wa_cw), copy.deepcopy(we_cw)
            # we_max_mask = re.find_rtf_labelling_brute_force(we, re.rtfs_gt)
            # est1 = re.full_bifreq_rtf_from_eigve(we_max_mask, we)
            # np.testing.assert_almost_equal(rtf, np.squeeze(est1), decimal=5)

            est2 = re.estimate_rtf_cw_ev_sv(cm, rtfs_shape=(num_mics_test, num_freqs_test, 1))
            np.testing.assert_almost_equal(rtf, np.squeeze(est2), decimal=5)

            est3 = re.estimate_rtf_cw_sv(cm, rtfs_shape=(num_mics_test, num_freqs_test, 1))
            np.testing.assert_almost_equal(rtf, np.squeeze(est3), decimal=5)

            # this method fails for non-white noise! do not use it!
            # est4 = re.estimate_rtf_cw_sv_cholesky(cm, rtfs_shape=(num_mics_test, num_freqs_test, 1))
            # np.testing.assert_almost_equal(rtf, np.squeeze(est4), decimal=5)


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


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        u.set_printoptions_numpy()

    @staticmethod
    def test_generate_spd_matrix():
        # check shape
        a = generate_hermitian_spd_matrix(3)
        np.testing.assert_equal(a.shape, (3, 3))

        # check positive definite
        eigvas = np.linalg.eigvals(a)
        np.testing.assert_array_less(0, eigvas)

    def test_covariance_whitening_single_freq(self):
        num_mics = 2
        self.test_covariance_whitening_single_freq_impl(num_mics, (True, True))

    def test_covariance_whitening_single_freq_impl(self, num_mics=2, symmetric_rs_rn=(True, True)):

        Rx, Rs, Rn, s = self.generate_cpsds_inner(num_mics=num_mics, symmetric_rs_rn=symmetric_rs_rn)

        self.assertTrue(check_matrix_is_hermitian_psd(Rx))
        self.assertTrue(check_matrix_is_hermitian_psd(Rn))
        self.assertTrue(check_matrix_is_hermitian_psd(Rx))

        re = rtf_estimator.RtfEstimator()

        s_est1 = re.covariance_whitening_cholesky(Rn, Rx)
        np.testing.assert_almost_equal(s, s_est1)

        s_est2 = re.covariance_whitening_generalized_eig(Rn, Rx,
                                                         hermitian_matrices=symmetric_rs_rn[0] and symmetric_rs_rn[1])
        np.testing.assert_almost_equal(s, s_est2)

        s_est3 = re.covariance_whitening_generalized_eig_explicit_inversion(Rn, Rx)
        np.testing.assert_almost_equal(s, s_est3)

    def test_whitening_vs_subtraction(self):

        num_mics = 3
        Rx, Rs, Rn, s = self.generate_cpsds_outer(num_mics, 1)
        Rx, Rs, Rn, s = np.squeeze(Rx), np.squeeze(Rs), np.squeeze(Rn), np.squeeze(s)
        is_hermitian(Rn)
        is_positiveDefinite(Rn)

        re = rtf_estimator.RtfEstimator()
        c1 = re.covariance_whitening_cholesky(Rn, Rx)
        np.testing.assert_almost_equal(s, c1)

        c2 = re.covariance_whitening_generalized_eig(Rn, Rx)
        np.testing.assert_almost_equal(s, c2)

        c3 = re.covariance_subtraction_first_column(Rx - Rn)
        np.testing.assert_almost_equal(s, c3)

        c4 = re.covariance_subtraction_eigve(Rx - Rn)
        np.testing.assert_almost_equal(s, c4)

    def test_subtraction_outer(self):

        num_mics = 5
        num_freqs = 60
        Rx, Rs, Rn, s = self.generate_cpsds_outer(num_mics, num_freqs)

        re = rtf_estimator.RtfEstimator()
        c2 = re.estimate_rtf_covariance_subtraction(Rx - Rn)
        c2 = np.squeeze(c2)
        np.testing.assert_almost_equal(s, c2)

        c3 = re.estimate_rtf_covariance_subtraction(Rx - Rn, use_first_column=False)
        c3 = np.squeeze(c3)
        np.testing.assert_almost_equal(s, c3)

    def test_whitening_outer(self):

        num_mics = 3
        num_freqs = 2
        for tt in range(10000):
            Rx, Rs, Rn, s = self.generate_cpsds_outer(num_mics, num_freqs)

            re = rtf_estimator.RtfEstimator()
            cc = re.estimate_rtf_covariance_whitening(noise_cpsd=Rn, noisy_cpsd=Rx, use_cholesky=True)
            cd = re.estimate_rtf_covariance_whitening(noise_cpsd=Rn, noisy_cpsd=Rx, use_cholesky=False)

            cc, cd = np.squeeze(cc), np.squeeze(cd)
            np.testing.assert_almost_equal(s, cc)
            np.testing.assert_almost_equal(s, cd)

    def generate_cpsds_outer(self, num_mics, num_freqs):

        Rn = np.zeros((num_mics, num_mics, num_freqs), dtype=complex)
        s = np.zeros((num_mics, num_freqs), dtype=complex)
        Rs = np.zeros_like(Rn)
        Rx = np.zeros_like(Rn)

        for kk in range(num_freqs):
            Rx[..., kk], Rs[..., kk], Rn[..., kk], s[..., kk] = self.generate_cpsds_inner(num_mics)

        Rn = Rn[..., np.newaxis]
        Rx = Rx[..., np.newaxis]

        return Rx, Rs, Rn, s

    def generate_cpsds_inner(self, num_mics, symmetric_rs_rn=(True, True)):

        s, Rs = generate_rank1_matrix(num_mics, symmetric=symmetric_rs_rn[0])

        if symmetric_rs_rn[1]:
            Rn = generate_hermitian_spd_matrix(num_mics)
        else:
            # make sure it is not positive definite
            Rn = generate_random_complex_matrix(num_mics) - 3 * np.identity(num_mics, dtype=complex)

        Rx = Rn + Rs
        return Rx, Rs, Rn, s

    def test_rolling_vs_mean_covariance_estimation(self):

        cm = cov_manager.CovarianceManager()
        num_samples = 5 * int(g.fs)
        noise_samples = g.rng.standard_normal((num_samples,))
        f, t, noise_stft = u.stft(noise_samples, nstft_test, noverlap_test, win_name_test)
        noise_stft = noise_stft[np.newaxis, ...]

        c = cm.estimate_cov(noise_stft, wideband=False, avg_time_frames_=False)
        c_avg = cm.estimate_cov(noise_stft, wideband=False, avg_time_frames_=True)
        np.testing.assert_allclose(c[..., -1].flatten(), c_avg.flatten(), atol=1e-1)

        c_bf = cm.estimate_cov(noise_stft, wideband=True, avg_time_frames_=False)
        c_bf_avg = cm.estimate_cov(noise_stft, wideband=True, avg_time_frames_=True)
        np.testing.assert_allclose(c_bf[..., -1].flatten(), c_bf_avg.flatten(), atol=1e-1)

        # even when cross-freq components are evaluated, diagonal elements should correspond
        for kk in range(nstft_test // 2):
            np.testing.assert_allclose(c_avg[..., kk, -1].flatten(), c_bf_avg[kk, kk].flatten(), atol=1e-2)

    def test_estimation_empirical_covariance(self):
        num_mics = 2
        durationNoiseSec1 = 1
        sg = signal_generator.SignalGenerator(num_mics, duration_noise_sec=durationNoiseSec1,
                                              duration_output_sec=durationNoiseSec1,
                                              nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                                              **extra_settings_test)
        v = sg.load_and_convolve_noise_samples('pink', sg.duration_output_samples,
                                               dir_point_source=False, same_volume_all_mics=False)
        V = sg.stft(v)
        cm = cov_manager.CovarianceManager()
        Ra = cm.estimate_cov(V, avg_time_frames_=True)
        Rb = cm.estimate_cov_loop_impl(V, avg_time_frames_=True)
        np.testing.assert_allclose(Ra, Rb)

    # can sporadically fail, just re run
    def test_noise_uncorrelated_covariance(self):
        num_mics = 2
        durationNoiseSec2 = 200
        sg = signal_generator.SignalGenerator(num_mics, duration_noise_sec=durationNoiseSec2,
                                              duration_output_sec=durationNoiseSec2,
                                              nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                                              **extra_settings_test)

        v = sg.load_and_convolve_noise_samples('white', sg.duration_output_samples, dir_point_source=False,
                                               same_volume_all_mics=False)
        V = sg.stft(v)
        cm = cov_manager.CovarianceManager()
        Rv = cm.estimate_cov(V, avg_time_frames_=True)

        # x_cpsd = np.zeros((num_mics, num_mics, num_freqs, num_frames_input), dtype=complex)
        for kk in range(Rv.shape[-2]):
            for mm1 in range(num_mics):
                Rv[mm1, :, kk, 0] = abs(Rv[mm1, :, kk, 0]) / abs(Rv[mm1, mm1, kk, 0])

        max_relative_correlation_offdiag_elements = 0.2  # ideally should be 0 for very long avg times
        for kk in range(Rv.shape[-2]):
            ii, jj = get_off_diag_indices(num_mics)
            np.testing.assert_array_less(Rv[ii, jj, kk, 0], max_relative_correlation_offdiag_elements)

    def test_error_metrics(self):

        a = np.ones((2, 3), dtype=complex)
        a1 = a + (0.1 - 0.2 * 1j)
        e1 = u.MSE_normalized_dB(a, a1)
        e11 = u.MSE(a, a1)

        a2 = np.zeros_like(a)
        e2 = u.MSE_normalized_dB(a, a2)
        e22 = u.MSE(a, a2)

        a3 = np.ones_like(a) * -100
        e3 = u.MSE_normalized_dB(a, a3)
        e33 = u.MSE(a, a3)

        # calculate mean of all these quantities in a line
        e1, e2, e3 = np.mean(e1), np.mean(e2), np.mean(e3)
        e11, e22, e33 = np.mean(e11), np.mean(e22), np.mean(e33)

        self.assertLess(e1, e2)
        self.assertLess(e11, e22)

        self.assertLess(e2, e3)
        self.assertLess(e22, e33)

    # this test doesn't make sense: a matrix with all positive entries can have negative eigenvalues!
    # consider e.g. [1, 2; 2, 1]
    # def test_cpsd_correction(self):
    #     a = generate_rank1_matrix(15, return_matrix_only=True, symmetric=True)
    #     b = cov_manager.CovarianceManager.suppress_negative_or_small(a, tol=1e-16)
    #
    #     np.testing.assert_array_less(np.zeros_like(b.real), b.real)
    #     np.testing.assert_array_less(np.zeros_like(b.imag), b.imag)
    # self.assertTrue(u.is_hermitian(b))
    # self.assertTrue(u.is_positiveSemiDefinite(b))

    def test_covariance_estimation_is_linear(self):

        cm = cov_manager.CovarianceManager()
        num_samples = 100 * int(g.fs)
        signal1_samples = g.rng.standard_normal((1, num_samples,))
        _, _, signal1_stft = u.stft(signal1_samples, nstft_test, noverlap_test, win_name_test)
        signal2_samples = g.rng.standard_normal((1, num_samples,))
        _, _, signal2_stft = u.stft(signal2_samples, nstft_test, noverlap_test, win_name_test)

        for cross_freq_covariance in [False, True]:
            settings = dict(wideband=cross_freq_covariance, avg_time_frames_=True, warning_level='warning')
            c1 = cm.estimate_cov(signal1_stft, **settings)
            c2 = cm.estimate_cov(signal2_stft, **settings)
            c_sum = c1 + c2
            c_cov_sum = cm.estimate_cov(signal1_stft + signal2_stft, **settings)
            np.testing.assert_allclose(c_sum, c_cov_sum, atol=1e-3)

    def test_bifreq_estimation_rtf(self):
        bifreq_estimation_template(False)

    def test_bifreq_estimation_cov_at_receivers(self):
        bifreq_estimation_template(True)

    def test_minimum_description_length_algorithm(self):

        # experiment 1 original paper
        test_eigvas = np.array([21.23, 2.17, 1.43, 1.09, 1.05, 0.94, 0.73])
        test_num_snapshots = 100
        q = rtf_estimator.RtfEstimator.minimize_mdl_criterion(test_eigvas, num_snapshots=test_num_snapshots)
        self.assertEqual(q, 2)

    def test_generate_correlated_samples(self):

        num_mics = 2
        num_freqs = 2
        num_mics_freqs = num_mics * num_freqs
        num_frames_list = [int(1e3), int(100e3)]

        cpx_data_list = [False, True]

        sg1 = signal_generator.SignalGenerator(num_mics, duration_noise_sec=duration_sec_test,
                                               duration_output_sec=duration_sec_test,
                                               nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                                               **extra_settings_test)

        for num_frames in num_frames_list:
            for cpx_data in cpx_data_list:
                for r in (generate_hermitian_spd_matrix(num_mics_freqs), np.eye(num_mics_freqs)):
                    # generate random SPD matrix (correlation matrix)

                    s0 = sg1.generate_correlated_signal(r, (num_mics_freqs, num_frames),
                                                        cpx_data=cpx_data)  # generate correlated samples
                    r_est = s0 @ s0.conj().T / (num_frames - 1)  # estimate covariance matrix

                    # s0_aug = np.concatenate((s0, s0.conj()), axis=0)
                    # r_aug_est = s0_aug @ s0_aug.conj().T / (num_frames - 1)  # estimate covariance matrix

                    required_precision = 0 if num_frames < 10000 else 1
                    np.testing.assert_almost_equal(r, r_est, decimal=required_precision)

    @staticmethod
    def enforce_cauchy_schwartz_inequality_slow(phi):
        # precalculate the diagonal elements of phi_ss_bar
        phi_diag = np.diag(phi).real

        for kk1 in range(phi.shape[0]):
            for kk2 in range(phi.shape[1]):
                if kk1 != kk2:
                    off_diag_mod = np.abs(phi[kk1, kk2])
                    max_mod = np.sqrt(phi_diag[kk1] * phi_diag[kk2])
                    if off_diag_mod > max_mod:
                        phi[kk1, kk2] = phi[kk1, kk2] * (max_mod / off_diag_mod)
        return phi

    def test_enforce_cauchy_schwartz_inequality(self):
        phi_ss_bar_ = generate_hermitian_spd_matrix(6)
        phi_ss_bar_1 = self.enforce_cauchy_schwartz_inequality_slow(phi_ss_bar_)
        phi_ss_bar_2 = rtf_estimator.RtfEstimator.enforce_cauchy_schwartz_inequality(phi_ss_bar_)
        np.testing.assert_almost_equal(phi_ss_bar_1, phi_ss_bar_2)

    def test_verify_complex_covariance_estimation(self):

        cov_list_description, cov_list = get_test_covariance_matrices()
        num_frames = num_mics_test * num_freqs_test * 1000

        for cov_gt, cov_description in zip(cov_list, cov_list_description):
            print(f"{cov_description = }")

            # Generate the data
            stft_data = signal_generator.SignalGenerator.generate_correlated_signal(cov_gt[..., 0],
                                                                                    (num_freqs_test * num_mics_test,
                                                                                     num_frames))
            stft_data_2d = np.reshape(stft_data, (num_mics_freqs_test, -1), order='F')

            # Estimate sample covariance matrix
            cov_emp = (stft_data_2d @ stft_data_2d.conj().T) / num_frames

            # Split real and imaginary parts and concatenate them, then use
            # https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts
            # to reconstruct the covariance matrix of the complex random vector
            stft_data_real = np.real(stft_data_2d)
            stft_data_imag = np.imag(stft_data_2d)
            stft_data_concat = np.concatenate((stft_data_real, stft_data_imag), axis=0)
            cov_concat = cov_manager.CovarianceManager.estimate_cov(stft_data_concat[np.newaxis],
                                                                    **estimate_cov_sett_test)
            cov_concat = cov_concat[..., 0]
            cov_emp_2 = cov_manager.CovarianceManager.transform_cov_real_imaginary_parts_to_complex_cov(cov_concat)
            np.testing.assert_almost_equal(cov_emp, cov_emp_2)

            # Estimate sample covariance matrix
            cov_emp_3 = cov_manager.CovarianceManager.estimate_cov(stft_data_2d[np.newaxis],
                                                                   **estimate_cov_sett_test)
            cov_emp_3 = cov_emp_3[..., 0]
            np.testing.assert_almost_equal(cov_emp, cov_emp_3)

            # Empirical and ground truth covariance should also be similar, but there still be substantial differences
            # due to the finite number of samples
            np.testing.assert_allclose(cov_emp, cov_gt[..., 0], atol=3)

            # Check that the empirical covariance is positive definite
            self.assertTrue(u.is_positiveDefinite(cov_emp))

            # Check that the empirical covariance is hermitian
            self.assertTrue(u.is_hermitian(cov_emp))

    def test_verify_inverse_complex_covariance_estimation(self):
        """ Verify that the inverse of the complex covariance matrix can be obtained from the
        real and imaginary parts of the data."""

        num_frames = 500
        cov_gt = generate_hermitian_spd_matrix(num_mics_freqs_test)
        cov_gt_inv = np.linalg.inv(cov_gt)
        stft_data = signal_generator.SignalGenerator.generate_correlated_signal(cov_gt,
                                                                                (num_freqs_test * num_mics_test,
                                                                                 num_frames))
        stft_data_2d = np.reshape(stft_data, (num_mics_freqs_test, -1), order='F')
        cov_emp = (stft_data_2d @ stft_data_2d.conj().T) / num_frames
        cov_emp_inv = np.linalg.inv(cov_emp)

        stft_data_real = np.real(stft_data_2d)
        stft_data_imag = np.imag(stft_data_2d)
        stft_data_concat = np.concatenate((stft_data_real, stft_data_imag), axis=0)
        cov_concat = cov_manager.CovarianceManager.estimate_cov(stft_data_concat[np.newaxis],
                                                                **estimate_cov_sett_test)
        cov_concat = cov_concat[..., 0]
        cov_emp_2 = cov_manager.CovarianceManager.transform_cov_real_imaginary_parts_to_complex_cov(cov_concat)
        cov_emp_inv_2 = np.linalg.inv(cov_emp_2)

        np.testing.assert_almost_equal(cov_emp_inv, cov_emp_inv_2)

        # Empirical and ground truth inverse covariance should also be similar
        np.testing.assert_allclose(cov_emp_inv, cov_gt_inv, atol=1)

        # Check that the empirical inverse covariance is positive definite
        self.assertTrue(u.is_positiveDefinite(cov_emp_inv))

        # Check that the empirical inverse covariance is hermitian
        self.assertTrue(u.is_hermitian(cov_emp_inv))

    def test_whiten_covariance_or_whiten_samples(self):
        """ Verify that the whitening of the covariance matrix and the whitening of the samples
        are equivalent."""

        # Generate the data
        rd_gt = generate_hermitian_spd_matrix(num_mics_freqs_test)
        rv_gt = generate_hermitian_spd_matrix(num_mics_freqs_test)

        num_frames = num_mics_test * num_freqs_test * 100
        sig_shape = (num_freqs_test * num_mics_test, num_frames)
        d = signal_generator.SignalGenerator.generate_correlated_signal(rd_gt, sig_shape)
        v = signal_generator.SignalGenerator.generate_correlated_signal(rv_gt, sig_shape)
        x = d + v

        # Whiten the covariance matrix
        rv_emp = cov_manager.CovarianceManager.estimate_cov(v, **estimate_cov_sett_test)[..., 0]
        rx_emp = cov_manager.CovarianceManager.estimate_cov(x, **estimate_cov_sett_test)[..., 0]
        rv_sqrt, rx_whitened_1 = rtf_estimator.RtfEstimator.whiten_covariance(rv_emp, rx_emp)
        rv_sqrt_inv = np.linalg.inv(rv_sqrt)
        np.testing.assert_almost_equal(rv_sqrt_inv @ rv_emp @ rv_sqrt_inv.conj().T, np.identity(num_mics_freqs_test))

        # Whiten the samples
        x_white = rv_sqrt_inv @ x
        rx_whitened_2 = cov_manager.CovarianceManager.estimate_cov(x_white, **estimate_cov_sett_test)[..., 0]
        np.testing.assert_almost_equal(rx_whitened_1, rx_whitened_2)

        # First invert, then do the Cholesky. Useful if rv_inv is already available
        # Unfortunately, this does not work, because the inverse of the Cholesky decomposition is not the Cholesky
        # decomposition of the inverse, not even if we take the Hermitian transpose of the Cholesky decomposition of
        # the inverse.
        # rv_inv = np.linalg.inv(rv_emp)
        # rv_sqrt_inv_3 = np.linalg.cholesky(rv_inv)
        #
        # np.testing.assert_almost_equal(rv_inv, rv_sqrt_inv.conj().T @ rv_sqrt_inv)
        # np.testing.assert_almost_equal(rv_sqrt_inv_3.conj().T @ rv_emp @ rv_sqrt_inv_3, np.identity(num_mics_freqs_test))
        #
        # x_white_3 = rv_sqrt_inv_3.conj().T @ x
        # rx_whitened_3 = cov_manager.CovarianceManager.estimate_cov(x_white_3, **estimate_cov_sett_test)[..., 0]
        # np.testing.assert_almost_equal(rx_whitened_1, rx_whitened_3)

        #  these are all different from rx_whitened_1
        # rx_whitened_3 = rv_sqrt_inv_2.conj().T @ rx_emp @ rv_sqrt_inv_2
        # rx_whitened_4 = rv_sqrt_inv_2 @ rx_emp @ rv_sqrt_inv_2.conj().T
        # rx_whitened_5 = rv_sqrt_2 @ rx_emp @ rv_sqrt_2.conj().T
        # rx_whitened_6 = rv_sqrt_2.conj().T @ rx_emp @ rv_sqrt_2
        pass
        # np.testing.assert_almost_equal(rx_whitened_1, rx_whitened_3)

    # def test_cholesky_of_inverse(self):
    #     rv = 0.1 * np.ones((3, 3)) + np.diagflat([1, 2, 3])
    #     rv_sqrt = np.linalg.cholesky(rv)
    #     np.testing.assert_almost_equal(rv_sqrt @ rv_sqrt.conj().T, rv)
    #
    #     rv_inv = np.linalg.inv(rv)
    #     rv_inv_sqrt = np.linalg.cholesky(rv_inv)
    #     np.testing.assert_almost_equal(rv_inv_sqrt @ rv_inv_sqrt.conj().T, rv_inv)
    #
    #     np.testing.assert_almost_equal(np.linalg.inv(rv_sqrt).conj().T, rv_inv_sqrt)

    def test_phase_correction(self):

        def phase_correction_loop_impl(stft_shape, overlap):
            (_, num_real_freqs, num_frames) = stft_shape
            num_all_freqs = (num_real_freqs - 1) * 2

            # Use nstft because delay correction depends on window size = nstft, not num_freqs (= nstft/2+1)
            lag = np.zeros(num_frames, np.int32)
            for tt in range(1, num_frames):
                lag[tt] = tt * (num_all_freqs - overlap)

            correction_term = np.zeros((num_real_freqs, num_frames), np.complex128)
            for kk in range(num_real_freqs):
                for tt in range(num_frames):
                    correction_term[kk, tt] = np.exp(-2j * np.pi * (kk / num_all_freqs) * lag[tt])

            return correction_term

        mics_freqs_or_frames_list = [2, 5, 10]
        overlap_percentages = [0.5, 0.75]

        for mics in mics_freqs_or_frames_list:
            for freqs in mics_freqs_or_frames_list:
                for frames in mics_freqs_or_frames_list:
                    for overlap_perc in overlap_percentages:
                        stft_shape_test = (mics, freqs, frames)
                        overlap_test = int(overlap_perc * 2 * (stft_shape_test[-1] - 1))
                        cc = cov_manager.CovarianceManager.compute_phase_correction_stft(stft_shape_test, overlap_test)
                        cc1 = phase_correction_loop_impl(stft_shape_test, overlap_test)
                        np.testing.assert_almost_equal(cc, cc1)


if __name__ == '__main__':
    unittest.main()
