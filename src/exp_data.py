class ExpData:
    """
    Container for experiment data. This is returned by ExperimentManager.run_experiment_single_variation
    and passed to ExperimentManager.evaluate_errors_single_variation
    Contains data from ALL realizations of the experiment.
    """

    def __init__(self, rtf_estimates, rtf_targets, rtf_nb_all_freqs, loud_bins_masks, cov_managers, stimuli, selected_bins_mask):
        self.rtf_estimates = rtf_estimates
        self.rtf_targets = rtf_targets
        self.rtf_nb_all_freqs = rtf_nb_all_freqs
        self.loud_bins_masks = loud_bins_masks
        self.cov_managers = cov_managers
        self.stimuli = stimuli
        self.selected_bins_mask = selected_bins_mask

        if isinstance(rtf_estimates, list):
            self._check_lists_lengths()

    def _check_lists_lengths(self):
        if not len(self.rtf_estimates) == len(self.rtf_targets) == len(self.loud_bins_masks) == \
               len(self.cov_managers) == len(self.stimuli) == len(self.selected_bins_mask):
            raise ValueError("ExpData initialization error: all lists must have the same length.")

    def get_all_fields(self):
        return (self.rtf_estimates, self.rtf_targets, self.rtf_nb_all_freqs,
                self.loud_bins_masks, self.cov_managers, self.stimuli, self.selected_bins_mask)
