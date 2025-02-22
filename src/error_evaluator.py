import copy
import warnings
from itertools import zip_longest

import numpy as np
import pysepm  # pip3 install https://github.com/schmiph2/pysepm/archive/master.zip
from matplotlib import pyplot as plt
from numpy import squeeze as sq

import src.global_constants as g
import src.utils as u
import pystoi


class ErrorAverager:
    """
    Compute the average error over a list of realizations. The error is computed as the difference between the estimated
    RTF and the true RTF. The error is computed for each frequency bin and each microphone.
    """
    def __init__(self, err_list, algo_names, rtf_metrics):

        self.err_list = err_list
        self.algo_names = algo_names
        self.rtf_metrics = rtf_metrics
        self.num_algorithms = len(self.algo_names)
        self.num_error_metrics = len(self.rtf_metrics)

    def build_error_array_from_dict(self, err_list):
        """
        Build a 3D array of errors from a list of dictionaries. The first dimension is the algorithm, the second dimension is
        the error metric, and the third dimension is the realization.
        Make sure that order of algorithms in array is the same as in self.algo_names.
        :param err_list: A list of dictionaries, where each dictionary has the following structure:
        {
            'algorithm_name': {
                'error_metric_1': error_value_1,
                'error_metric_2': error_value_2,
                ...
            },
            ...
        }
        :return: A 3D array of errors with dimensions: [num_algorithms, num_error_metrics, num_realizations]
        :rtype: np.ndarray
        """
        num_realizations = len(err_list)
        err_collection = np.nan * np.ones((self.num_algorithms, self.num_error_metrics, num_realizations))
        for idx_realization, realization in enumerate(err_list):
            for algo_name, algo_errors in realization.items():
                for idx_error_metric, (error_metric_name, error_value) in enumerate(algo_errors.items()):
                    if algo_name not in self.algo_names:
                        warnings.warn(f"Algorithm {algo_name} is not in {self.algo_names = }. "
                                      f"Not adding to error array.", RuntimeWarning)
                        continue
                    idx_algorithm = self.algo_names.index(algo_name)
                    if error_metric_name == 'Hermitian angle' and u.is_crb(algo_name):
                        err_collection[idx_algorithm, idx_error_metric, idx_realization] = 0
                    else:
                        err_collection[idx_algorithm, idx_error_metric, idx_realization] = error_value

        return err_collection

    def compute_error_mean_std_over_realizations(self, err_collection):
        """
        Compute the mean and standard deviation of the error over the realizations.
        Also compute the standard error of the mean (https://en.wikipedia.org/wiki/Standard_error.)
        and the 95% confidence interval.
        The confidence interval can be expressed in terms of probability with respect to a single theoretical
        (yet to be realized) sample: "There is a 95% probability that the 95% confidence interval calculated from a
        given future sample will cover the true value of the population parameter."

        :param err_collection: A 3D array of errors with dimensions: [num_algorithms, num_error_metrics, num_realizations]
        :return: A 3D array of errors with dimensions: [num_algorithms, num_error_metrics, 3]
        :rtype: np.ndarray
        """
        err_mean_std_ = np.zeros((self.num_algorithms, self.num_error_metrics, 3))
        for algo_idx, err_single_algo in enumerate(err_collection):
            for metric_idx, _ in enumerate(err_single_algo):

                # use np.nanmean to ignore NaN values (corresponding to missing data, especially the CRB)
                # CRB is not calculated for all realizations, to save time
                err_mean_std_[algo_idx, metric_idx, 0] = np.nanmean(err_collection[algo_idx, metric_idx])

                # mean +/- standard error of the mean
                if u.is_crb(self.algo_names[algo_idx]) or err_collection.shape[2] == 1:
                    stderr = 0
                else:
                    stderr = np.nanstd(err_collection[algo_idx, metric_idx], ddof=1) / np.sqrt(err_collection.shape[2])
                    if np.isnan(stderr):
                        warnings.warn(f"Standard error of the mean is NaN for algorithm {self.algo_names[algo_idx]} "
                                      f"and metric {self.rtf_metrics[metric_idx]}.")
                        stderr = 0

                # 95% confidence interval
                err_mean_std_[algo_idx, metric_idx, 1] = err_mean_std_[algo_idx, metric_idx, 0] - 1.96 * stderr
                err_mean_std_[algo_idx, metric_idx, 2] = err_mean_std_[algo_idx, metric_idx, 0] + 1.96 * stderr

                # set std error to mean to avoid huge faded region in plot. Due to conversion to dB, log(0) = -inf!
                if err_mean_std_[algo_idx, metric_idx, 1] < 0:
                    # err_mean_std_[algo_idx, metric_idx, 1] = 0
                    err_mean_std_[algo_idx, metric_idx, 1] = err_mean_std_[algo_idx, metric_idx, 0]

        return err_mean_std_

    def maybe_convert_errors_to_db(self, err_mean_std):
        """ Convert (some of) the errors to dB. """

        # find the indices of the error metrics that should be converted to dB
        convert_to_db = np.array(['dB' in s for s in self.rtf_metrics])

        # convert the errors to dB
        err_mean_std = copy.deepcopy(err_mean_std)

        if np.any(err_mean_std[:, convert_to_db, :] < 0):
            warnings.warn('There are negative errors that will be converted to dB: Double check the results.')
            err_mean_std[err_mean_std[:, convert_to_db, :] < 0] = g.eps

        err_mean_std[:, convert_to_db, :] = u.linear_to_db(err_mean_std[:, convert_to_db, :])

        return err_mean_std

    @staticmethod
    def clean_up_errors_array(err_array):
        """ Remove NaNs and clip errors to a minimum and maximum value. """

        err_mean_std = copy.deepcopy(err_array)

        # if there are nans, set them to the maximum error
        if np.any(np.isnan(err_mean_std)):
            warnings.warn('There are NaNs in the error array. Setting them to maximum error.')
            err_mean_std[np.isnan(err_mean_std)] = g.mse_db_max_error

        # clip errors to a minimum and maximum value
        if np.any(err_mean_std > g.mse_db_max_error):
            warnings.warn('There are errors that are larger than the maximum error. Clipping them.')

        err_mean_std = np.clip(err_mean_std, a_min=g.mse_db_min_error, a_max=g.mse_db_max_error)

        return err_mean_std


class ErrorEvaluator:
    """
    Evaluate the performance of the algorithms in the dictionary "estimates" with respect to the ground truth "ground_truth".
    The error is computed as the difference between the estimated RTF and the true RTF. The error is computed for each
    frequency bin and each microphone.
    """
    def __init__(self, ground_truth, estimates, metrics, loud_bins_mask=None):

        self.metrics = metrics
        self.ground_truth = ground_truth
        self.estimates_dict = estimates
        self.errors_dict = dict()

        if ground_truth.ndim < 2:
            raise ValueError(f"The ground truth must have at least 2 dimensions, but it has {ground_truth.ndim = }.")

        # Check that ground_truth and estimates have compatible shapes
        if not np.all([est.shape[:2] == ground_truth.shape[:2] for est in estimates.values()]):
            raise ValueError(f"The ground truth and estimates must have the same shape, but they have shapes "
                                f"{ground_truth.shape} and {[est.shape for est in estimates.values()]}.")

        if np.all([est.shape[-1] == 1 for est in estimates.values()]) and ground_truth.ndim == 2:
            self.estimates_dict = {key: np.squeeze(value, axis=-1) for key, value in estimates.items()}

        num_freqs = self.ground_truth.shape[1]
        if loud_bins_mask is None:
            self.loud_bins_mask = np.ones(num_freqs, dtype=bool)  # set to True for loud bins, False for quiet bins
        else:
            self.loud_bins_mask = loud_bins_mask

        self.errors_dict = {key: dict() for key in self.estimates_dict.keys()}

    @staticmethod
    def evaluate_errors_single_variation(estimates_realizations, ground_truth_realizations, err_metrics, algo_names,
                                         loud_bins_masks_all_realizations=None, needs_cleanup=True):

        errors_all_realizations = []
        evaluator = None
        if loud_bins_masks_all_realizations is None:
            loud_bins_masks_all_realizations = [None] * len(ground_truth_realizations)

        for ground_truth, estimates_raw, loud_bin_mask in zip_longest(ground_truth_realizations,
                                                                      estimates_realizations,
                                                                      loud_bins_masks_all_realizations):
            estimates = estimates_raw
            if needs_cleanup:
                estimates = ErrorEvaluator.clean_up_rtf_estimates(estimates_raw, clip_estimates=True)

            evaluator = ErrorEvaluator(ground_truth=ground_truth, estimates=estimates, metrics=err_metrics,
                                       loud_bins_mask=loud_bin_mask)

            err_single_realization, _ = evaluator.evaluate_errors_single_realization(plot_rtf_realization=False,
                                                                                     print_realization_error=False)

            errors_all_realizations.append(err_single_realization)

        err_averager = ErrorAverager(errors_all_realizations, algo_names, err_metrics)

        err_array = err_averager.build_error_array_from_dict(err_averager.err_list)
        err_mean_std = err_averager.compute_error_mean_std_over_realizations(err_array)
        err_mean_std_db = err_averager.maybe_convert_errors_to_db(err_mean_std)
        err_mean_std_db = err_averager.clean_up_errors_array(err_mean_std_db)

        return err_mean_std_db, evaluator

    def error_metric_single_realization(self, metric, metric_name=''):
        """
        Calculate error with "metric" for a single realization.
        Compute error for each algorithm (all frequencies and microphones) and store it in a dictionary

        Parameters:
        - metric (callable): A function to calculate the error between the ground truth and the estimate.
        - metric_name (str, optional): A name for the error metric to be used for printing.
        - print_realization_error (bool, optional): A flag to indicate if the error should be printed.
        - to_db (bool, optional): A flag to indicate if the error should be converted to dB.
        """

        errors_dict = {key: dict() for key in self.estimates_dict.keys()}
        for est_name, est_data in self.estimates_dict.items():
            est_data_masked = np.squeeze(est_data[:, self.loud_bins_mask])
            if not u.is_crb(est_name):
                e = metric(self.ground_truth[:, self.loud_bins_mask], est_data_masked, mean=False)
                errors_dict[est_name][metric_name] = e
            else:
                e = u.return_real_part_second_argument(est_data_masked, est_data_masked)
                errors_dict[est_name][metric_name] = e

        return errors_dict

    def evaluate_errors_single_realization(self, plot_rtf_realization=True, print_realization_error=True):
        """ Evaluate the errors for a single realization.
        The errors are calculated for all the algorithms in the dictionary. """

        # if self.loud_bins_mask is not None and not np.all(self.loud_bins_mask):
        #     print("Some bins are quiet. Applying mask to ground truth and estimates.")

        t = self.evaluate_err_single_realization_single_metric()
        self.errors_dict.update(t)

        # calculate the mean error (over everything BUT NOT over realizations)
        errors_dict_all_freqs = copy.deepcopy(self.errors_dict)
        for est_name, _ in self.estimates_dict.items():
            for metric_name, _ in self.errors_dict[est_name].items():
                if 'RMSE' in metric_name or 'RMSE dB' in metric_name:
                    # convert from 'squared error' to 'root mean squared error'
                    # note: we could also normalize, by dividing by np.abs(np.mean(self.rtf_ground_truth))
                    self.errors_dict[est_name][metric_name] = \
                        np.sqrt(np.mean(self.errors_dict[est_name][metric_name]))
                else:
                    # mean over all frequencies and microphones
                    self.errors_dict[est_name][metric_name] = np.mean(self.errors_dict[est_name][metric_name])

        if print_realization_error:
            string = ''
            with np.printoptions(precision=5):
                for est_name, est_result in self.estimates_dict.items():
                    for metric_name, _ in self.errors_dict[est_name].items():
                        if 'MSE' not in metric_name:
                            transformation = u.linear_to_db if 'MSE' in metric_name else lambda x: x
                            string += f"{metric_name} {est_name}: " \
                                      f"{transformation(self.errors_dict[est_name][metric_name]):.2f}, "
            print(string)

        if plot_rtf_realization:
            self.plot_rtfs_multiple()

        return self.errors_dict, errors_dict_all_freqs

    def evaluate_err_single_realization_single_metric(self):

        err_dict = {key: dict() for key in self.estimates_dict.keys()}
        if 'MSE' in self.metrics:
            err_dict_temp = self.error_metric_single_realization(u.MSE, 'MSE')
            err_dict.update(err_dict_temp)

        if 'RMSE' in self.metrics or 'RMSE dB' in self.metrics:
            err_dict_temp = self.error_metric_single_realization(u.MSE, 'RMSE')
            err_dict.update(err_dict_temp)

        if 'Hermitian angle' in self.metrics:
            err_dict_temp = self.error_metric_single_realization(u.hermitian_angle, 'Hermitian angle')
            err_dict.update(err_dict_temp)

        if 'fwSNRseg' in self.metrics:
            for est_name, est_result in self.estimates_dict.items():
                err_dict_temp = pysepm.fwSNRseg(sq(self.ground_truth), sq(est_result), g.fs)
                err_dict[est_name]['fwSNRseg'] = err_dict_temp if np.isfinite(err_dict_temp) else 0

        if 'STOI' in self.metrics:
            for est_name, est_result in self.estimates_dict.items():
                # def stoi(x, y, fs_sig, extended=False)
                # x (np.ndarray): clean original speech
                # err_dict_temp = pysepm.stoi(x=sq(self.ground_truth), y=sq(est_result), fs_sig=int(g.fs), extended=True)
                temp = pystoi.stoi(sq(self.ground_truth), sq(est_result), int(g.fs), extended=True)
                if temp < 0:
                    print("Stoi shold be positive!")
                err_dict[est_name]['STOI'] = temp if np.isfinite(temp) else 0

        if 'pesq' in self.metrics:
            for est_name, est_result in self.estimates_dict.items():
                temp = pypesq.pesq(ref=self.ground_truth, deg=sq(est_result), fs=int(g.fs))
                err_dict[est_name]['pesq'] = temp if np.isfinite(temp) else 0

        if 'llr' in self.metrics:
            for est_name, est_result in self.estimates_dict.items():
                err_dict[est_name]['llr'] = pysepm.llr(sq(self.ground_truth), sq(est_result), fs=int(g.fs))

        return err_dict

    @staticmethod
    def mask_rtfs(x, active_mask, hide_masked=False):
        """
        Mask the RTFs with the active mask. If hide_masked is True, the masked bins are set to zero.
        If hide_masked is False, the masked bins are set to the ground truth value.
        Returns the masked RTFs.
        """
        ground_truth_mask = copy.deepcopy(x)
        if hide_masked:
            ground_truth_mask = ground_truth_mask[:, active_mask]
        else:
            ground_truth_mask[:, np.bitwise_not(active_mask)] = 0

        return ground_truth_mask

    def plot_rtfs(self, mask=False, hide_masked=False, which_mic=1):
        assert which_mic < self.ground_truth.shape[0]

        gt_label = 'Ground truth'

        fig = plt.figure(num=None, figsize=(10, 7))
        axes = fig.subplots(nrows=1, ncols=2, sharex='all', sharey='all', squeeze=False)

        for plot_idx, (ax, is_real_part) in enumerate(zip(axes[0], [True, False])):

            ground_truth_mask = self.mask_rtfs(self.ground_truth, self.loud_bins_mask, hide_masked) if mask \
                else self.ground_truth

            x_values = np.arange(0, ground_truth_mask.shape[-1])
            for idx, (est_name, estimated_rtfs) in enumerate(self.estimates_dict.items()):

                if u.is_crb(est_name):
                    continue

                estimated_rtfs_mask = self.mask_rtfs(estimated_rtfs, self.loud_bins_mask, hide_masked) if mask \
                    else estimated_rtfs

                label = est_name + ErrorEvaluator.make_title_from_errors(self.errors_dict[est_name])
                if plot_idx == 1:
                    label = None
                    gt_label = None

                y = np.real(estimated_rtfs_mask[which_mic]) if is_real_part else np.imag(estimated_rtfs_mask[which_mic])
                ax.plot(x_values, y, c=g.colors[idx + 1], label=label, linewidth=0.9)
                # u.log_pow(estimated_rtfs[which_mic])
            y_gt = np.real(ground_truth_mask[which_mic]) if is_real_part else np.imag(ground_truth_mask[which_mic])
            ax.plot(x_values, y_gt, c=g.colors[0], label=gt_label)

            ylabel = "Real" if is_real_part else "Imaginary"
            ax.set_ylabel(ylabel)
            ax.set_xlabel('FFT bin')
            ax.grid(True, which='both')

        fig.legend(fontsize='large', loc="lower left")
        fig.suptitle("Comparison of RTF estimates in frequency domain", fontsize='x-large')
        fig.tight_layout()
        fig.show()
        return fig

    def plot_rtfs_multiple(self, which_mic=1):
        assert which_mic < self.ground_truth.shape[0]

        labels = ['Ground truth', 'Estimate']
        psd_ground_truth_kwargs = dict(c=g.colors[0], label=labels[0])

        num_plots = len(self.estimates_dict)
        fig = plt.figure(num=None, figsize=(8, int(4 * num_plots)))
        axes = fig.subplots(nrows=num_plots, ncols=1, sharex='all', sharey='all', squeeze=False)
        axes = axes[:, 0]

        est_name = 'CS'
        if est_name in self.estimates_dict.keys():
            axes[0].plot(u.log_pow(self.ground_truth[which_mic]), **psd_ground_truth_kwargs)
            axes[0].plot(u.log_pow(self.estimates_dict[est_name][which_mic]), c=g.colors[1], label=labels[1])
            axes[0].set_title(f'Cov. subtraction{self.make_title_from_errors(self.errors_dict[est_name])}')
            axes[0].legend()

        est_name = 'CW'
        if 'CW' in self.estimates_dict.keys():
            ax_num = 0 if 'CS' not in self.estimates_dict.keys() else min(1, len(axes) - 1)
            axes[ax_num].plot(u.log_pow(self.ground_truth[which_mic]), **psd_ground_truth_kwargs)
            axes[ax_num].plot(u.log_pow(self.estimates_dict[est_name][which_mic]), c=g.colors[2], label=labels[1])
            axes[ax_num].set_title(f'Cov. whitening{self.make_title_from_errors(self.errors_dict[est_name])}')
            axes[ax_num].legend()

        est_name = 'CS-bifreq'
        if est_name in self.estimates_dict.keys():
            ax_num = min(2, len(axes) - 1)
            axes[ax_num].plot(u.log_pow(self.ground_truth[which_mic]), **psd_ground_truth_kwargs)
            axes[ax_num].plot(u.log_pow(self.estimates_dict[est_name][which_mic]), c=g.colors[3], label=labels[1])
            axes[ax_num].set_title(
                f'Cov. bifreq. neigh.{self.make_title_from_errors(self.errors_dict[est_name])}')
            axes[ax_num].legend()

        est_name = 'CW-SV'
        if est_name in self.estimates_dict.keys():
            ax_num = min(3, len(axes) - 1)
            axes[ax_num].plot(u.log_pow(self.ground_truth[which_mic]), **psd_ground_truth_kwargs)
            axes[ax_num].plot(u.log_pow(self.estimates_dict[est_name][which_mic]), c=g.colors[4], label=labels[1])
            axes[ax_num].set_title(
                f'Cov. bifreq.{self.make_title_from_errors(self.errors_dict[est_name])}')
            axes[ax_num].legend()

        # axes[3].plot(u.log_pow(rtfs_ground_truth_stft[which_mic],  c=g.colors[0])
        # axes[3].plot(u.log_pow(rtfs_cs_nan_bifreq_stft[which_mic],  c=g.colors[4])
        # axes[3].set_title(f'cov subtraction nan bifreq, mse = {cs_nan_bifreq_mse}')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def make_title_from_errors(errors_num):
        errors_string = ""
        for metric_name, metric_value in errors_num.items():
            unit = ''
            if 'dB' in metric_name:
                unit = 'dB'
            errors_string += f", {metric_name} = {metric_value:.2f}{unit}"
        return errors_string

    # def ExportAllToMp3(self):
    #     date_time_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    #     out_dir_name = os.path.join("out", date_time_string)
    #     os.mkdir(out_dir_name)
    #     for estimator_name, estimator_result in self.named_estimates.items():
    #         file_name = os.path.join(out_dir_name, f"{estimator_name}.mp3")
    #         self.write_mp3(file_name, fs, u.normalize_volume(estimator_result))

    # Changes rtf_gt and rtf_est to contain only frequencies under evaluation
    def keep_selected_frequencies_only(self, bins_to_evaluate):

        if bins_to_evaluate is None:
            return
        print(f"Evaluate only bins {bins_to_evaluate} of est. RTF")

        # all frequencies are ignored, except "evaluated_frequency_bins" which are marked as false
        num_freqs = self.ground_truth.shape[-1]
        ignored_frequencies_mask = np.ones(num_freqs, dtype=bool)
        ignored_frequencies_mask[bins_to_evaluate] = False

        self.ground_truth, self.estimates_dict = \
            self.keep_selected_frequencies_only_from_mask(ignored_frequencies_mask)

    def keep_selected_frequencies_only_from_mask(self, ignored_frequencies_mask):
        self.ground_truth = np.delete(self.ground_truth, ignored_frequencies_mask, axis=-1)
        for name, est in self.estimates_dict.items():
            est = np.delete(est, ignored_frequencies_mask, axis=1)
            self.estimates_dict[name] = est

        return self.ground_truth, self.estimates_dict

    @staticmethod
    def find_loud_bins_masks(target_stft, max_relative_difference_db=60, print_log=True):
        """
        Find bins that are loud enough to be used for evaluation.
        The bins are too quiet if the maximum power in target_stft is more than max_relative_difference dB higher than
        the mean power in the bin.
        :param target_stft: STFT of the target signal
        :param max_relative_difference_db: maximum relative difference between max and mean power in dB
        :return: mask of bins that are loud enough
        """

        power = u.MSE_single(target_stft)
        max_power_overall = np.max(power)
        mean_power_per_frequency = np.mean(power, axis=(0, 2))
        differences_db = u.linear_to_db(max_power_overall) - u.linear_to_db(mean_power_per_frequency)
        quiet_bins = abs(differences_db) > max_relative_difference_db

        if np.all(quiet_bins):
            warnings.warn("All bins are quiet. Ignore mask and use all bins.")
            return np.zeros_like(quiet_bins)
        elif np.all(np.bitwise_not(quiet_bins)):
            pass
        # elif np.sum(~quiet_bins) > num_freqs // 2:
        #     warnings.warn(f"More than 1/2 of the bins are loud enough. Are we sure this is speech?")
        else:
            if print_log:
                print(f"Keeping {len(quiet_bins) - np.sum(quiet_bins)}/{len(quiet_bins)} bins that are loud enough")

        return np.bitwise_not(quiet_bins)

        # quiet_bins_percentage = np.sum(abs(power - max_power_overall) < max_relative_difference, axis=-1) / num_frames
        # quiet_bins_percentage = np.mean(quiet_bins_percentage, axis=0)  # avg over mics
        # return quiet_bins_percentage < percentage_active

    @staticmethod
    def clean_up_rtf_estimates(rtf_estimates_dict, clip_estimates=True):
        """Modify named_estimates. Replace the reference microphone value with 1+0j,
        average the estimates over time, and clip the values to fall within a specified range.
        If any NaN values are present, they will be replaced with the specified minimum or maximum values.

        Args:
            rtf_estimates_dict: A dictionary mapping algorithm names to their corresponding estimates.
            clip_estimates: If True, do not clip the estimates to fall within a specified range.
            CAN BE IMPORTANT FOR BOUNDS: if algorithms may seem to perform better than they actually do.

        Returns:
            None (modifies named_estimates in place)
        """

        rtf_cleaned_dict = copy.deepcopy(rtf_estimates_dict)

        for key, estimate_value in rtf_cleaned_dict.items():

            # Cramer-rao bounds should not be "cleaned up"
            if estimate_value is not None and not u.is_crb(key):

                if np.any(np.real(estimate_value) < g.rtf_min) or np.any(np.real(estimate_value) > g.rtf_max) or \
                        np.any(np.imag(estimate_value) < g.rtf_min) or np.any(np.imag(estimate_value) > g.rtf_max):
                    error_string = f"estimate for {key} is out of bounds!"
                    if clip_estimates:
                        error_string += f" Clipping to [{g.rtf_min}, {g.rtf_max}]."
                    warnings.warn(error_string, category=RuntimeWarning)

                if clip_estimates:  # clip real and imag part. np.nan is left untouched
                    rtf_cleaned_dict[key] = u.clip_cpx(rtf_cleaned_dict[key], a_min=g.rtf_min, a_max=g.rtf_max)

                if np.isnan(rtf_cleaned_dict[key]).any():
                    warnings.warn(f"nan are present in averaged estimate for {key}!")
                    rtf_cleaned_dict[key] = np.nan_to_num(rtf_cleaned_dict[key],
                                                          nan=g.rtf_min, posinf=g.rtf_max, neginf=g.rtf_min)

                # averaging at the end, otherwise NaNs will not be averaged out
                if estimate_value.ndim == 3 and estimate_value.shape[-1] > 1:
                    rtf_cleaned_dict[key] = ErrorEvaluator.average_over_time(estimate_value)

                rtf_cleaned_dict[key][g.idx_ref_mic] = 1 + 0j  # replace estimate with oracle (all 1s for RTF)

        return rtf_cleaned_dict

    @staticmethod
    def average_over_time(x):
        """ Average over time, discarding the initial samples if there are more samples than mics*freqs."""

        mics_times_freqs = x.shape[0] * x.shape[1]
        info_msg = f"micsTimesFreqs = {mics_times_freqs}, numTimeFrames = {x.shape[-1]}"

        # if there are more samples than mics*freqs, discard the initial samples.
        # during the first "mics_times_freqs" snapshots the covariance matrices are not full rank.
        if x.shape[-1] > mics_times_freqs:
            info_msg_2 = ": discard initial samples."
            print(info_msg + info_msg_2)
            return np.squeeze(np.nanmean(x[..., mics_times_freqs:], axis=-1, keepdims=False))
        else:
            info_msg_2 = ": keep all samples for averaging over time."
            print(info_msg + info_msg_2)
            return np.squeeze(np.nanmean(x, axis=-1, keepdims=False))
