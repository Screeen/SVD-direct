""""
Compare the speed of the covariance whitening (CW) and SVD-direct algorithms for RTF estimation
using simulated data.
"""

import src.global_constants as g
import src.utils as u
import numpy as np
from src.acoustic_estimator_test import generate_hermitian_spd_matrix
import scipy
import time


def cw(rx_list, rv_list, rtfs):

    for kk, (rx_, rv_) in enumerate(zip(rx_list, rv_list)):
        eigenvals, eigves = \
            scipy.linalg.eigh(rx_, rv_, check_finite=False)
        max_right_eigve = eigves[:, -1]
        rtfs[kk] = rv_ @ max_right_eigve

    return rtfs


def svd_direct(noisy, noise, rtfs, num_mics, num_freqs, max_retained_=np.inf):

    max_rank_cov_wet = min(num_freqs, max_retained_)
    eigenvals, eigves_right = scipy.linalg.eigh(noisy, noise, check_finite=False,
                                                subset_by_index=[num_mics * num_freqs - max_rank_cov_wet,
                                                                    num_mics * num_freqs - 1])

    # keep the largest max_rank_cov_wet eigenvectors
    # eigves_right = eigves_right[:, -max_rank_cov_wet:]
    # eigenvals = np.maximum(0, eigenvals.real)[-max_rank_cov_wet:]

    # Several ways to get the left eigenvectors
    eigves_left = noise @ eigves_right
    eigve_cov_wet = eigves_left @ np.diagflat(np.sqrt(eigenvals))
    phi_ss = eigve_cov_wet @ eigve_cov_wet.conj().T
    phi_ss = phi_ss.reshape((num_mics, num_freqs, num_mics * num_freqs), order='F')

    for kk in range(num_freqs):
        left_sv, s_vals, vh = np.linalg.svd(phi_ss[:, kk], full_matrices=False)
        rtfs[kk] = left_sv[:, 0]

    return rtfs


def svd_direct2(noisy, noise, rtfs, num_mics, num_freqs, a):
    max_rank_cov_wet = num_freqs
    noise_sqrt = np.linalg.cholesky(noise)
    noise_sqrt_inv = np.linalg.inv(noise_sqrt)
    whitened_noisy = noise_sqrt_inv @ noisy @ noise_sqrt_inv.conj().T
    eigenvals, eigves = scipy.linalg.eigh(whitened_noisy, check_finite=True)

    eigves = eigves[:, -max_rank_cov_wet:]
    eigenvals = np.maximum(0, eigenvals.real)[-max_rank_cov_wet:]

    dewhitened_eigve = noise_sqrt @ eigves
    eigve_cov_wet = dewhitened_eigve @ np.diagflat(np.sqrt(eigenvals))

    phi_ss = eigve_cov_wet @ eigve_cov_wet.conj().T
    phi_ss = phi_ss.reshape((num_mics, num_freqs, num_mics * num_freqs), order='F')

    for kk in range(K):
        left_sv, s_vals, vh = np.linalg.svd(phi_ss[:, kk], full_matrices=False)
        rtfs[kk] = left_sv[:, 0]

    return rtfs


u.set_printoptions_numpy()
K = 120
M = 4
K_direct = K

# def generate_covariance_equicorrelated(variances, corr_coefficient, cov_shape):
variances = g.rng.uniform(0.5, 1., (M*K_direct,))
corr_coeff = 0
cov_shape = (M*K_direct, M*K_direct)

rd = generate_hermitian_spd_matrix(M*K)
rv = np.diag(variances)
rx = rd + rv

rd_blocks = [generate_hermitian_spd_matrix(M) for _ in range(K)]
rv_blocks = [np.diag(variances[k*M:(k+1)*M]) for k in range(K)]
rx_blocks = [rd_blocks[k] + rv_blocks[k] for k in range(K)]

rtfs_cw = np.zeros((K, M), dtype=np.complex128)
# a = cw(rx_blocks, rv_blocks, rtfs_cw)

rtfs_svd = np.zeros((K_direct, M), dtype=np.complex128)
# b = svd_direct(rx, rv, rtfs_svd, M, K)

# Now we want to compare the speed of the two algorithms. We will use the same data over and over again for both
# algorithms. We will also use the same number of iterations to get a good average time.
num_iter = 50

mk_direct = M * K_direct
rx_direct = rx[:mk_direct, :mk_direct]
rv_direct = rv[:mk_direct, :mk_direct]
max_retained = min(10, K_direct)

cw(rx_blocks, rv_blocks, rtfs_cw)
start = time.time()
for _ in range(num_iter):
    a = cw(rx_blocks, rv_blocks, rtfs_cw)
end = time.time()
avg_cw_time = (end - start) / num_iter

svd_direct(rx_direct, rv_direct, rtfs_svd, M, K_direct, max_retained)
start = time.time()
for _ in range(num_iter):
    b = svd_direct(rx_direct, rv_direct, rtfs_svd, M, K_direct, max_retained)
end = time.time()
avg_svd_time = (end - start) / num_iter

print(f"Average time for SVD-direct: {avg_svd_time}")
print(f"Average time for CW: {avg_cw_time}")
print(f"Ratio: {avg_svd_time / avg_cw_time:.2f}")
