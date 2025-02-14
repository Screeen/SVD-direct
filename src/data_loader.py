import os
import pickle
import time
import warnings

import src.global_constants as g
import src.utils as u
from src.settings_manager import SettingsManager


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def save_data(settings_figures, figures_=None, arrays_all_figures=None, fig_extension='pdf',
                  different_keys_across_figures=None, array_dir_name=None, fig_dir_name=None,
                  settings_dir_name=None, **kwargs):

        if (g.debug_mode and not g.debug_save) or not g.release_save_plots:
            return None

        if arrays_all_figures is None:
            arrays_all_figures = [None]

        if figures_ is None:
            figures_ = [None]

        i = 1
        file_names = []
        target_folder_ = None

        # finally, create the file name
        # example file name: 01_correlation_target=0.0_correlation_noise=0.0
        for settings_figure in settings_figures:

            prefix = f"{kwargs['prepend_file_name']}_" if 'prepend_file_name' in kwargs else ''
            file_name = f"{prefix}{i:02d}"

            if different_keys_across_figures is not None:
                for key_name in different_keys_across_figures:
                    if key_name in settings_figure and len(str(settings_figure[key_name])) < 50:
                        variation_name = f"-{settings_figure[key_name]}"
                        # Make variation name compatible with Arxiv
                        to_replace = {' ': '_', '=': '', ',': '', '.': '', '[': '', ']': ''}
                        for key, value in to_replace.items():
                            variation_name = variation_name.replace(key, value)
                    else:
                        # warnings.warn(f"Cannot use value of {key_name=} in file name, because it is too long")
                        variation_name = f""
                    file_name += f"_{key_name}{variation_name}"

            i += 1
            file_names.append(file_name)

        # save errors
        for arrays_all_subplots, file_name_ in zip(arrays_all_figures, file_names):
            if arrays_all_subplots is not None:
                target_folder_ = u.save_array(arrays_all_subplots, user_file_name=file_name_,
                                              out_dir_name=array_dir_name,
                                              append_time=False, **kwargs)
                time.sleep(g.sleeping_time_figure_saving)  # wait 1 second to avoid overwriting the file

        # save settings
        if settings_dir_name is not None:
            for settings_figure, file_name_ in zip(settings_figures, file_names):
                if settings_figure is not None:
                    target_folder_ = u.save_settings(settings_figure, user_file_name=file_name_,
                                                     out_dir_name=settings_dir_name,
                                                     append_time=False, **kwargs)
                    time.sleep(g.sleeping_time_figure_saving)

        # save figures
        for figure_, file_name_ in zip(figures_, file_names):
            if figure_ is not None:
                target_folder_ = u.save_figure(figure_, user_file_name=file_name_, out_dir_name=fig_dir_name,
                                               append_time=False, format=fig_extension, **kwargs)
                time.sleep(g.sleeping_time_figure_saving)

        return target_folder_

    @staticmethod
    def load_data(parent_dir_name):
        """
        Load data from files. This is mirror function to save_data(). It should load errors and settings from files
        """

        if not os.path.isdir(parent_dir_name):
            raise FileNotFoundError(f"Directory {parent_dir_name} does not exist")

        # Load settings
        settings_all_figures = [[], [], [], [], []]
        settings_dir_name = os.path.join(parent_dir_name, 'settings')

        if not os.path.isdir(settings_dir_name):
            raise FileNotFoundError(f"Directory {settings_dir_name} does not exist")

        if len(os.listdir(settings_dir_name)) == 0:
            raise FileNotFoundError(f"Directory {settings_dir_name=} is empty")
        for file_name in os.listdir(settings_dir_name):
            try:
                with open(os.path.join(settings_dir_name, file_name), 'rb') as f:
                    # skip if hidden file
                    if file_name.startswith('.'):
                        continue
                    if 'rmse' in file_name:
                        settings_all_figures[0].append(pickle.load(f))
                    elif 'herm' in file_name:
                        settings_all_figures[1].append(pickle.load(f))
                    elif 'bf_llr' in file_name:
                        settings_all_figures[2].append(pickle.load(f))
                    elif 'bf_stoi' in file_name:
                        settings_all_figures[3].append(pickle.load(f))
                    elif 'bf_fwsnr' in file_name:
                        settings_all_figures[4].append(pickle.load(f))
                    else:
                        warnings.warn(f"Unknown file name: {file_name}")
                        continue
            except FileNotFoundError:
                print(f"File {file_name} not found")

        # Load errors
        errors_all_figures = [[], [], [], [], []]
        array_dir_name = os.path.join(parent_dir_name, 'arrays')
        if len(os.listdir(array_dir_name)) == 0:
            raise FileNotFoundError(f"Directory {array_dir_name=} is empty")
        for file_name in os.listdir(array_dir_name):
            with open(os.path.join(array_dir_name, file_name), 'rb') as f:
                # skip if hidden file
                if file_name.startswith('.'):
                    continue
                if 'rmse' in file_name:
                    errors_all_figures[0].append(pickle.load(f))
                elif 'herm' in file_name:
                    errors_all_figures[1].append(pickle.load(f))
                elif 'bf_llr' in file_name:
                    errors_all_figures[2].append(pickle.load(f))
                elif 'bf_stoi' in file_name:
                    errors_all_figures[3].append(pickle.load(f))
                elif 'bf_fwsnr' in file_name:
                    errors_all_figures[4].append(pickle.load(f))
                else:
                    warnings.warn(f"Unknown file name: {file_name}")
                    continue

        return errors_all_figures, settings_all_figures

    @staticmethod
    def validate_data(errors_all_figures_, settings_all_figures_, is_beamforming_error=False):

        for err_single_fig, sett_single_fig in zip(errors_all_figures_, settings_all_figures_):

            if is_beamforming_error:
                algo_names = SettingsManager. \
                    get_algo_names_beamforming(sett_single_fig['algo_names'])
            else:
                algo_names = sett_single_fig['algo_names']

            # (num_x_values, num_algorithms=num_labels, num_metrics=num_subplots, 3=(mean, std+, std-))
            _, x_values = SettingsManager.get_variation_key_values(sett_single_fig)
            if err_single_fig.shape != (len(x_values), len(algo_names), 1, 3):
                raise ValueError(f"Expected errors_all_figures_ to have shape "
                                 f"({len(x_values) = }, {len(algo_names) = }, 1, 3), "
                                 f"but got {err_single_fig.shape} instead."
                                 f"{algo_names = }, {is_beamforming_error = }")

    @classmethod
    def save_data_wrapper(cls, errors_all_figures_all_metrics, settings_all_figures_all_metrics,
                          different_keys_across_figures, out_dir_name_='out'):

        array_dir_name, _, settings_dir_name = \
            u.make_folders_errors_figures_settings(out_dir_name=out_dir_name_)
        for errors_all_figures_, settings_all_figures_ in zip(errors_all_figures_all_metrics,
                                                              settings_all_figures_all_metrics):
            prefix = settings_all_figures_[0]['saving_prefix']
            is_beamforming_error = 'bf' in prefix
            cls.validate_data(errors_all_figures_, settings_all_figures_, is_beamforming_error)
            cls.save_data(arrays_all_figures=errors_all_figures_, settings_figures=settings_all_figures_,
                                different_keys_across_figures=different_keys_across_figures,
                                array_dir_name=array_dir_name,
                                settings_dir_name=settings_dir_name, omit_time_from_file_name=True,
                                prepend_file_name=prefix)