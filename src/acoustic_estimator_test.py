import unittest
import numpy as np

import sys
sys.path.append('..')
sys.path.append('../src')

import cov_manager
import src.global_constants as g
import rtf_estimator
import signal_generator
import src.utils as u
from utils import is_positiveDefinite, is_hermitian
import copy

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
                       'desired': [],
                       'rtf_type': ''}

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


def generate_covariance_wgn(atf=None, estimate_cpsd_settings=None):
    if estimate_cpsd_settings is None:
        estimate_cpsd_settings = dict(with_crossfreq=True,
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

    c1 = cov_manager.CovarianceManager.estimate_cpsd_wrapper(s_stft, **estimate_cpsd_settings)

    return c1


def test_bifreq_estimation_template(false_test_rtf_true_test_cov):
    sg = signal_generator.SignalGenerator(num_mics_test, duration_noise_sec=duration_sec_test,
                                          duration_output_sec=duration_sec_test,
                                          nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                                          **extra_settings_test)
    identity = np.eye(num_mics_test * num_freqs_test)[..., np.newaxis]
    noises_cov_list = [identity,
                       generate_covariance_wgn() + identity,
                       generate_hermitian_spd_matrix(num_mics_freqs_test)[..., np.newaxis],
                       cov_manager.CovarianceManager.generate_covariance(noise_corr_test, (num_mics_freqs_test,
                                                                                           num_mics_freqs_test), 1)[
                           ..., np.newaxis]]

    noise_cov_list_description = ['identity', 'wgn + identity', 'spd', 'correlated']

    cm = cov_manager.CovarianceManager()

    for r_n, noise_cov_description in zip(noises_cov_list, noise_cov_list_description):
        print(f"{noise_cov_description = }")
        atf = sg.generate_atf(atf_type='random')
        rtf = sg.generate_rtf_from_atf(atf)

        r_s = generate_covariance_wgn(rtf)
        r_x = r_s + r_n

        cm.cov_noise = r_n
        cm.cov_wet_oracle = r_s
        cm.cov_noisy = r_x
        cm.cov_wet_gevd = cm.estimate_cov_wet_gevd(r_x, r_n, sg.get_stft_shape())

        if false_test_rtf_true_test_cov:
            # check r_s and cm.cov_wet_oracle are equal
            np.testing.assert_almost_equal(r_s, cm.cov_wet_oracle)
            np.testing.assert_almost_equal(cm.cov_wet_gevd, cm.cov_wet_oracle)

        else:
            re = rtf_estimator.RtfEstimator()
            re.rtfs_gt = rtf

            # wa_cw, we_cw, _ = re.estimate_eigenvectors_bifreq(r_n, r_x, sg.get_stft_shape(), 'whiten')
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

        self.assertTrue(cov_manager.CovarianceManager.check_matrix_is_hermitian_psd(Rx))
        self.assertTrue(cov_manager.CovarianceManager.check_matrix_is_hermitian_psd(Rn))
        self.assertTrue(cov_manager.CovarianceManager.check_matrix_is_hermitian_psd(Rx))

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

        c = cm.estimate_cpsd_wrapper(noise_stft, with_crossfreq=False, avg_time_frames_=False)
        c_avg = cm.estimate_cpsd_wrapper(noise_stft, with_crossfreq=False, avg_time_frames_=True)
        np.testing.assert_allclose(c[..., -1].flatten(), c_avg.flatten(), atol=1e-2)

        c_bf = cm.estimate_cpsd_wrapper(noise_stft, with_crossfreq=True, avg_time_frames_=False)
        c_bf_avg = cm.estimate_cpsd_wrapper(noise_stft, with_crossfreq=True, avg_time_frames_=True)
        np.testing.assert_allclose(c_bf[..., -1].flatten(), c_bf_avg.flatten(), atol=1e-2)

        # even when cross-freq components are evaluated, diagonal elements should correspond
        for kk in range(nstft_test // 2):
            np.testing.assert_allclose(c_avg[..., kk, -1].flatten(), c_bf_avg[kk, kk].flatten(), atol=1e-2)

    def test_cpsd(self):
        num_mics = 2
        durationNoiseSec1 = 1
        sg = signal_generator.SignalGenerator(num_mics, duration_noise_sec=durationNoiseSec1,
                                              duration_output_sec=durationNoiseSec1,
                                              nstft=nstft_test, noverlap_percentage=noverlap_percentage_test,
                                              **extra_settings_test)
        v = sg.load_and_convolve_noise_samples('pink', sg.duration_output_samples,
                                               dir_point_source=False, same_volume_all_mics=False)
        _, _, V = sg.stft(v)
        cm = cov_manager.CovarianceManager()
        Ra = cm.estimate_cpsd_wrapper(V, avg_time_frames_=True)
        Rb = cm.estimate_cpsd_loop_impl(V, avg_time_frames_=True)
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
        _, _, V = sg.stft(v)
        cm = cov_manager.CovarianceManager()
        Rv = cm.estimate_cpsd_wrapper(V, avg_time_frames_=True)

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

    def test_covariance_estimation_bf_linear(self):

        cm = cov_manager.CovarianceManager()
        num_samples = 100 * int(g.fs)
        signal1_samples = g.rng.standard_normal((1, num_samples,))
        _, _, signal1_stft = u.stft(signal1_samples, nstft_test, noverlap_test, win_name_test)
        signal2_samples = g.rng.standard_normal((1, num_samples,))
        _, _, signal2_stft = u.stft(signal2_samples, nstft_test, noverlap_test, win_name_test)

        for cross_freq_covariance in [False, True]:
            settings = dict(with_crossfreq=cross_freq_covariance, avg_time_frames_=True, warning_level='warning')
            c1 = cm.estimate_cpsd_wrapper(signal1_stft, **settings)
            c2 = cm.estimate_cpsd_wrapper(signal2_stft, **settings)
            c_sum = c1 + c2
            c_cov_sum = cm.estimate_cpsd_wrapper(signal1_stft + signal2_stft, **settings)
            np.testing.assert_allclose(c_sum, c_cov_sum, atol=1e-3)

    def test_bifreq_estimation_rtf(self):
        test_bifreq_estimation_template(False)

    def test_bifreq_estimation_cov_at_receivers(self):
        test_bifreq_estimation_template(True)

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


if __name__ == '__main__':
    unittest.main()
