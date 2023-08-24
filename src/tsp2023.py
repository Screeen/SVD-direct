import copy
import os
import pickle
import subprocess
import sys
import time
import warnings
from itertools import repeat
from multiprocessing import cpu_count, Pool
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import src.config as cfg
import src.global_constants as g
import src.utils as u
from src.exp_manager import SettingsManager, ExperimentManager
import src.cov_manager as cov_manager
import src.plot_manager as plot_manager

low_correlation_value = 0.25
high_correlation_value = 0.75


class TspScript:

    @staticmethod
    def get_algo_names(sett_):
        current_names_ = sett_['algo_names_displayed'] if 'algo_names_displayed' in sett_ else sett_['algo_names']

        noise_corr_type = cov_manager.CovarianceManager.filter_correlation_type(sett_['correlation_noise_type'])

        if noise_corr_type == 'frequency':
            current_names_ = [f"{name}" for name in current_names_]
            # current_names_ = [f"{name} (f)" for name in current_names_]
        elif noise_corr_type == 'space':
            current_names_ = [f"{name} (s)" for name in current_names_]
        elif noise_corr_type == 'frequency+space':
            current_names_ = [f"{name} (f+s)" for name in current_names_]
        elif noise_corr_type is None:
            pass
        else:
            raise ValueError(f"Unknown correlation type: {noise_corr_type}")

        return current_names_

    @staticmethod
    def generate_settings(exp_details_collection, repeated_experiments_constant=1.):

        experiment_settings_original, exp_common, exp_details_figures_ooo, exp_details_columns_o, exp_details_rows_oo = \
            exp_details_collection

        settings_figures = []

        for exp_detail_3 in exp_details_figures_ooo:  # each exp_detail_3 is a different plot

            settings_plots = []

            for exp_detail_2 in exp_details_rows_oo:

                settings_subplots = []

                # Each exp_detail is a set a colored lines in the same plot
                for exp_detail_idx, exp_detail_1 in enumerate(exp_details_columns_o):
                    sett = experiment_settings_original | exp_common | exp_detail_3 | \
                           exp_detail_2 | exp_detail_1  # the union of d1 and d2

                    if 'num_repeated_experiments' not in sett:
                        if isinstance(sett['duration_output_frames'], int):
                            num_frames_for_counting_montecarlo = sett['duration_output_frames']
                        else:
                            num_frames_for_counting_montecarlo = np.mean(np.array(sett['duration_output_frames']))

                        min_repeated_experiments = 10 if repeated_experiments_constant > 1 else 2
                        sett['num_repeated_experiments'] = max(min_repeated_experiments,
                                                               int(repeated_experiments_constant / (
                                                                       20 * num_frames_for_counting_montecarlo ** 2)))

                    settings_subplots.append(sett)

                settings_plots.extend(settings_subplots)

            settings_figures.extend(settings_plots)

        return settings_figures

    @staticmethod
    def read_hardcoded_settings(repeated_experiments_constant=1., exp_name=None):

        if 'speech' not in exp_name:
            # cfg_name = "config_experiments_TSP2023_synthetic.yaml"
            cfg_name = "config_experiments_TSP2023_synthetic_2023_08_23.yaml"
        else:
            # cfg_name = "config_experiments_TSP2023_real.yaml"
            cfg_name = "config_experiments_TSP2023_real_2023_08_23.yaml"
        print(f"{cfg_name=}")
        experiment_settings_original = cfg.load_configuration(cfg_name)

        correlation_percentage_list = [0., 0.25, 0.5, 0.75, 0.95]
        # correlation_percentage_list = [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.95]
        # correlation_percentage_list = [0., 0.5, 1.]

        exp_details_figures_ooo = [{}, ]  # different figures
        exp_details_rows_oo = [{}, ]  # different rows in the same figure
        exp_details_columns_o = [{}, ]  # different columns in the same figure

        exp_common = dict()

        if exp_name == 'target_correlation':
            exp_common = {  # shared settings.
                'varying_factors': ['correlation_target'],
                'correlation_target': correlation_percentage_list,
            }

            exp_details_columns_o = [  # different rows
                {}
                #     {'correlation_noise': low_correlation_value},
                # {'correlation_noise': high_correlation_value}
            ]

            exp_details_rows_oo = [  # different columns
                {}
                # {'duration_output_frames': 10, },
                # {'duration_output_frames': 100, },
                # {'duration_output_frames': 1000, },
                # {'duration_output_frames': 5000, },
            ]

            exp_details_figures_ooo = [  # different figures
                {'correlation_noise': low_correlation_value},
                # {'correlation_noise': high_correlation_value}
            ]

        elif exp_name == 'noise_correlation':

            exp_common = {  # shared settings.
                'varying_factors': ['correlation_noise'],
                'correlation_noise': correlation_percentage_list,
            }

            exp_details_figures_ooo = [  # different figures
                {'correlation_target': low_correlation_value,
                 'correlation_noise_type': 'frequency', },
                {'correlation_target': high_correlation_value,
                 'correlation_noise_type': 'frequency', },
            ]

            exp_details_columns_o = [{}]
            exp_details_rows_oo = [{}]

        elif exp_name == 'time_frames':

            if repeated_experiments_constant > 1:
                repeated_experiments_constant *= 100
            exp_common = {  # shared settings.
                'varying_factors': ['duration_output_frames'],
                'duration_output_frames': list(np.logspace(np.log10(10), np.log10(5000), 5).astype(int)),
            }

            exp_details_columns_o = [
                {}
                #     {'correlation_target': low_correlation_value},
                #     {'correlation_target': high_correlation_value}
            ]

            exp_details_rows_oo = [
                {}
                #     {'correlation_noise': low_correlation_value},
                #     {'correlation_noise': high_correlation_value,
                #      'correlation_noise_type': 'frequency'}
            ]

            exp_details_figures_ooo = [
                {'correlation_target': low_correlation_value},
                {'correlation_target': high_correlation_value}
            ]

        elif exp_name == 'snr':

            exp_common = {  # shared settings.
                'varying_factors': ['noises_info', 0, 'snr'],
                'noises_info':
                    [{'snr': [-10, -5, 0, 10, 20], 'names': ['white']}],
                    # [{'snr': [20], 'names': ['white']}],
            }

            exp_details_columns_o = [
                {}
                #     {'correlation_target': low_correlation_value},
                #     {'correlation_target': high_correlation_value}
            ]

            exp_details_rows_oo = [  # different rows
                {}
                #     {'correlation_noise': low_correlation_value},
                #     {'correlation_noise': high_correlation_value}
                #     {'duration_output_frames': 50, },
                #     {'duration_output_frames': 5000, },
            ]

            exp_details_figures_ooo = [
                {'correlation_target': low_correlation_value},
                {'correlation_target': high_correlation_value}
                # {'duration_output_frames': 5000, },
            ]
        elif exp_name == 'speech_snr':
            exp_common = {  # shared settings.
                'varying_factors': ['noises_info', 0, 'snr'],
                'noises_info':
                    [
                        {
                            'names': ['male'],
                            'snr': [-10, -5, 0, 10, 20, 30],
                            # 'snr': [10],
                            'isDirectional': True,
                            'same_volume_all_mics': False
                        },
                        {'names': ['white'], 'snr': [40], 'same_volume_all_mics': True}
                    ],
                'add_identity_noise_noisy': True
            }

        elif exp_name == 'speech_time_frames':
            exp_common = {  # shared settings.
                'varying_factors': ['duration_output_frames'],
                'duration_output_frames': list(np.logspace(np.log10(100), np.log10(5000), 4).astype(int)),
                'add_identity_noise_noisy': True
            }

        elif exp_name == 'speech_nstft':
            exp_common = {  # shared settings.
                'varying_factors': ['nstft'],
                'nstft': [64, 128, 256, 512, 1024],
                'add_identity_noise_noisy': True
            }

        elif exp_name == 'debug':
            exp_common = {  # shared settings.
                'varying_factors': ['noises_info', 0, 'snr'],
                # 'varying_factors': ['duration_output_frames'],
                'algo_names': ['CW-SV', 'CW-EV-SV', 'CW', 'CRB_unconditional'],
                'num_repeated_experiments': 1,
                'correlation_noise': 0.,
                'correlation_target': 0.9,
                'noises_info':
                    [{'snr': [0], }],
                # 'noises_info':
                #     [{'snr': [-10, -5, 0, 5, 10], }],
                # 'duration_output_frames': list(np.logspace(np.log10(1), np.log10(100), 6).astype(int)),
                'duration_output_frames': [100],
            }

        exp_common['exp_name'] = exp_name

        settings_collection = [experiment_settings_original, exp_common, exp_details_figures_ooo,
                               exp_details_columns_o, exp_details_rows_oo]

        return settings_collection, repeated_experiments_constant

    @staticmethod
    def group_errors_per_figure(errors_per_subplot, settings_per_subplot, num_subplots_per_figure=1):

        num_subplots = len(errors_per_subplot)
        num_figures = len(errors_per_subplot) // num_subplots_per_figure

        if num_figures <= 1:
            errors_per_figure = [errors_per_subplot]
            settings_per_figure = [settings_per_subplot]
        else:
            # split in num_figures plots
            errors_per_figure = [errors_per_subplot[i:i + num_subplots_per_figure] for i in
                                 range(0, num_subplots, num_subplots_per_figure)]
            settings_per_figure = [settings_per_subplot[i:i + num_subplots_per_figure] for i in
                                   range(0, num_subplots, num_subplots_per_figure)]

        return errors_per_figure, settings_per_figure

    @staticmethod
    def get_different_keys_across_figures(exp_details_figures_ooo):
        different_keys_across_figures = set()
        for fig_setting in exp_details_figures_ooo:
            different_keys_across_figures = set(fig_setting.keys()).union(different_keys_across_figures)
        return different_keys_across_figures

    @staticmethod
    def plot_errors(errors_all_figures, settings_all_figures, use_tex_labels_=False, num_rows_fig=1, num_cols_fig=1):

        suptitle = ''
        u.set_plot_options(use_tex_labels_)
        y_min = np.inf
        y_max = -np.inf

        xscale_log = False

        plot_area_size = 2.75
        figure_size = (1 + num_cols_fig * plot_area_size, 1 + num_rows_fig * plot_area_size)
        figures = []

        for errors_all_subplots, settings_all_subplots in zip(errors_all_figures, settings_all_figures):

            fig_opt = dict(figsize=figure_size, constrained_layout=True, sharex=True, sharey=True)
            fig, axes = plt.subplots(nrows=num_rows_fig, ncols=num_cols_fig, **fig_opt, squeeze=False)
            if not isinstance(settings_all_subplots, list):
                settings_all_subplots = [settings_all_subplots]

            for errors_single_graph, sett, ax in zip(errors_all_subplots, settings_all_subplots, axes.flat):

                exp_name = sett['exp_name']
                sett = SettingsManager.assign_default_values(sett)
                _, x_values = SettingsManager.get_variation_key_values(sett)
                correlation_name = 'Noise' if 'correlation_noise' in sett['varying_factors'] else 'Target'
                correlation_name = correlation_name + f" {sett['correlation_noise_type']} corr."
                clean_algo_names = TspScript.get_algo_names(sett)

                if exp_name == 'target_correlation':
                    # x_values_transformed = [int(100 * x) for x in x_values]
                    x_values_transformed = [round(x, 2) for x in x_values]
                    if use_tex_labels_:
                        title = f"$L={sett['duration_output_frames']}, \\upsilon_f={sett['correlation_noise']}$"
                        # x_label = f"$\\rho_f\ [\%]$"
                        x_label = rf"$\rho_f$"
                    else:
                        title = f"{sett['duration_output_frames']} frames, " \
                                f"{sett['correlation_noise'] * 100:.0f}% noise corr."
                        x_label = f"{correlation_name} [%]"
                    # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

                elif exp_name == 'noise_correlation':

                    # x_values_transformed = [int(100 * x) for x in x_values]
                    x_values_transformed = [round(x, 2) for x in x_values]
                    if use_tex_labels_:
                        title = f"$L={sett['duration_output_frames']}, \\rho_f={sett['correlation_target']}$"
                        if sett['correlation_noise_type'] == 'frequency':
                            subscript = 'f'
                        elif sett['correlation_noise_type'] == 'space':
                            subscript = 's'
                        else:
                            subscript = 'f+s'
                        x_label = rf"$\upsilon_{subscript}$"
                    else:
                        title = f"{sett['duration_output_frames']} frames, " \
                                f"{sett['correlation_target'] * 100:.0f}% target corr."
                        x_label = f"{correlation_name} [%]"
                    # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

                elif exp_name == 'time_frames':
                    if use_tex_labels_:
                        title = f"$\\rho_f={sett['correlation_target']}, \\upsilon_f={sett['correlation_noise']}$"
                        x_label = "Number of frames"
                    else:
                        title = f"{sett['correlation_target'] * 100:.0f}% target corr., " \
                                f"{sett['correlation_noise'] * 100:.0f}% noise corr."
                        x_label = "Number of frames"
                    xscale_log = True
                    x_values_transformed = [int(x) for x in x_values]

                    # suptitle = f"Varying number of frames, {sett['noises_info'][0]['snr'][0]} dB SNR"

                elif exp_name == 'snr':
                    if use_tex_labels_:
                        title = rf"$\rho_f={sett['correlation_target']}, \upsilon_f={sett['correlation_noise']}$"
                        x_label = r"SNR [dB]"
                    else:
                        title = f"{sett['correlation_target'] * 100:.0f}% target corr., " \
                                f"{sett['correlation_noise'] * 100:.0f}% noise corr."
                        x_label = f"SNR [dB]"
                    x_values_transformed = [int(x) for x in x_values]

                elif exp_name == 'speech_snr':
                    title = f""
                    # title = f"Real speech"
                    x_label = f"SNR [dB]"
                    x_values_transformed = [int(x) for x in x_values]

                elif exp_name == 'speech_time_frames':
                    title = f""
                    # title = f"Real speech"
                    x_label = f"Number of frames"
                    x_values_transformed = [int(x) for x in x_values]
                    xscale_log = True

                else:
                    title = 'debug'
                    x_values_transformed = [int(x) for x in x_values]
                    x_label = f"SNR [dB]"
                    # suptitle = f"{correlation_name}, {sett['noises_info'][0]['snr'][0]} dB SNR"

                f = plot_manager.plot_errors_wrapper(x_values_transformed,
                                                     ax=ax,
                                                     errors_array=errors_single_graph,
                                                     suptitle=suptitle, title=title,
                                                     x_label=x_label,
                                                     metric_names=sett['metric_names'],
                                                     algo_names=clean_algo_names,
                                                     legend_num_cols=1,
                                                     show_plot=g.debug_mode,
                                                     font_size='x-large',
                                                     legend_font_size='medium',
                                                     xscale_log=xscale_log,
                                                     )

                y_min = min(y_min, ax.get_ylim()[0])
                y_max = max(y_max, ax.get_ylim()[1])

            figures.append(fig)

        for figure in figures:

            # figure = plot_manager.set_transparent_background_figure(figure)
            for ax_idx, ax in enumerate(figure.get_axes()):
                if abs(y_max - y_min) > 5:
                    ax.set_ylim(np.floor(y_min), np.ceil(y_max))
                else:  # round to 1 decimal
                    ax.set_ylim(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10)
                ax.label_outer()
                if ax_idx != 0:
                    try:
                        ax.get_legend().remove()
                    except AttributeError:
                        pass

        return figures

    @staticmethod
    def run_script(exp_name__='debug', repeated_experiments_constant=1., use_multiple_processes=True,
                   use_tex_labels_=False, load_dir_name=None, target_noise_equal_variances=None):

        # time the experiment
        start_time = time.time()
        u.set_printoptions_numpy()

        # Load / generate settings
        settings_collection, repeated_experiments_constant = TspScript.read_hardcoded_settings(
            repeated_experiments_constant, exp_name=exp_name__)
        settings_all_subplots = TspScript.generate_settings(settings_collection, repeated_experiments_constant)
        _, exp_common, exp_details_figures_ooo, exp_details_columns_o, exp_details_rows_oo = settings_collection
        different_keys_across_figures = TspScript.get_different_keys_across_figures(exp_details_figures_ooo)
        num_rows_fig = len(exp_details_rows_oo)  # number of rows in the figure (subplots)
        num_cols_fig = len(exp_details_columns_o)
        prefixes = ['rmse', 'herm']
        # store results_ as a dict of lists. Each key is given by the list above, e.g. 'err_mean_std_db_array'
        results_keys = ['err_mean_std_db_array', 'rtf_evaluators', 'cm', 'sh', 'atf_target', 'variances',
                        'err_mean_std_db_array_herm']
        results_dict_ = {key: [] for key in results_keys}

        if target_noise_equal_variances is not None:
            for sett in settings_all_subplots:
                sett['target_noise_equal_variances'] = target_noise_equal_variances

        parent_dir_name = Path(os.path.basename(__file__)).stem
        if load_dir_name is None:
            # run experiments
            atf, variances = None, None
            # warmup run is used to generate atf and variances of synthetic signals
            if settings_all_subplots[0]['needs_warmup_run']:
                atf, variances = TspScript.run_warmup_experiment(settings_all_subplots[0])
            if use_multiple_processes:
                how_many_processes = min(cpu_count(), len(settings_all_subplots))
                with Pool(how_many_processes) as p:
                    results_ = p.starmap(ExperimentManager.run_experiment,
                                         [(sett, atf, variances) for sett in settings_all_subplots])
            else:
                results_ = map(ExperimentManager.run_experiment, [sett__ for sett__ in settings_all_subplots],
                               repeat(atf),
                               repeat(variances))

            for res in results_:
                for key in results_keys:
                    results_dict_[key].append(res[results_keys.index(key)])

            results_dict_['settings'] = settings_all_subplots

            errors_all_subplots_mse = results_dict_['err_mean_std_db_array']
            errors_all_subplots_herm = results_dict_['err_mean_std_db_array_herm']

            # Save experiments results (errors)
            array_dir_name, figures_dir_name, settings_dir_name = u.create_save_folders(parent_dir_name=parent_dir_name,
                                                                                        child_dir_name=exp_name__,
                                                                                        use_tex_figures=use_tex_labels_)

            num_subplots_per_figure = num_rows_fig * num_cols_fig
            errors_all_figures_rmse, settings_all_figures_rmse = \
                TspScript.group_errors_per_figure(errors_all_subplots_mse, settings_all_subplots,
                                                  num_subplots_per_figure)
            for s in settings_all_figures_rmse:
                for ss in s:
                    ss['metric_names'] = ['RMSE dB']

            # This horrible code is because the Hermitian angle used to be in a different subplot,
            # and it was saved in the same error array as the MSE.
            # Now we want it in a completely different figure instead
            errors_all_figures_herm, settings_all_figures_herm = \
                TspScript.group_errors_per_figure(errors_all_subplots_herm,
                                                  copy.deepcopy(settings_all_subplots), num_subplots_per_figure)
            for z in settings_all_figures_herm:
                for zz in z:
                    zz['metric_names'] = ['Hermitian angle']

            errors_all_figures = [errors_all_figures_rmse, errors_all_figures_herm]
            settings_all_figures = [settings_all_figures_rmse, settings_all_figures_herm]

            for errors_all_figures_, settings_all_figures_, prefix in zip(errors_all_figures, settings_all_figures,
                                                                          prefixes):
                TspScript.save_data(arrays_all_figures=errors_all_figures_, settings_figures=settings_all_figures_,
                                    different_keys_across_figures=different_keys_across_figures,
                                    array_dir_name=array_dir_name,
                                    settings_dir_name=settings_dir_name, omit_time_from_file_name=True,
                                    prepend_file_name=prefix)
        else:
            # parent_dir_name example: '../../out2/2023-05 experiments correlation (TSP2023)/time_frames-2023-07-04--12-25-11'
            # arrays example: '../../out2/2023-05 experiments correlation (TSP2023)/time_frames-2023-07-04--12-25-11/arrays'
            errors_all_figures, settings_all_figures = TspScript.load_data(load_dir_name)
            _, figures_dir_name, _ = u.create_save_folders(out_dir_name=load_dir_name, use_tex_figures=use_tex_labels_)

        # Plot results
        target_folder = None
        figures = None
        for errors_all_figures_, settings_all_figures_, prefix in zip(errors_all_figures, settings_all_figures,
                                                                      prefixes):
            figures = TspScript.plot_errors(errors_all_figures_, settings_all_figures_, use_tex_labels_=use_tex_labels_,
                                            num_rows_fig=num_rows_fig, num_cols_fig=num_cols_fig)

            # Save figures
            target_folder = TspScript.save_data(figures_=figures,
                                                different_keys_across_figures=different_keys_across_figures,
                                                settings_figures=settings_all_figures_, fig_dir_name=figures_dir_name,
                                                omit_time_from_file_name=False, prepend_file_name=prefix)

        print(f"{exp_common = }")
        print(f"{repeated_experiments_constant = :.2e}")

        if load_dir_name is None:
            # open target_folder in Finder
            total_time = int(time.time() - start_time)
            if sys.platform == 'darwin' and exp_name__ != 'debug':
                if target_folder is not None and total_time > 60:
                    subprocess.Popen(['open', target_folder])
                    os.system('say "hey boss, we are done"')
                elif total_time > 10:
                    os.system('say "done"')

            # Print total time in hours, minutes, seconds
            print(f"Total time: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60}s")

        if figures is not None and len(figures) > 0:
            figures[-1].show()

        return results_dict_

    @staticmethod
    def run_warmup_experiment(settings_single_subplot_):
        """ warm-up run to calculate variances and atf """

        sett_warmup = copy.deepcopy(settings_single_subplot_)
        sett_warmup['exp_name'] = 'warmup'
        sett_warmup['varying_factors'] = ['']
        sett_warmup['num_repeated_experiments'] = 1
        _, _, _, _, atf, variances, _ = ExperimentManager.run_experiment(sett_warmup)

        return atf, variances

    @staticmethod
    def save_data(settings_figures, figures_=None, arrays_all_figures=None, fig_extension='pdf',
                  different_keys_across_figures=None, array_dir_name=None, fig_dir_name=None,
                  settings_dir_name=None, **kwargs):

        if g.debug_mode and not g.debug_save:
            warnings.warn(f"Skipping saving data because debug_mode is on")
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

            # if there are subplots, just use the first one
            sett = copy.deepcopy(settings_figure)
            if len(settings_figure) > 0:
                sett = settings_figure[0]

            # file_name = f"{sett['exp_name']}"
            # file_name = f"{settings_subplots[0]['exp_name']}_{i:02d}"
            prefix = f"{kwargs['prepend_file_name']}_" if 'prepend_file_name' in kwargs else ''
            file_name = f"{prefix}{i:02d}"

            if different_keys_across_figures is not None:
                for key_ in different_keys_across_figures:
                    value_ = sett[key_]
                    file_name += f"_{key_}={value_}"

            i += 1
            file_names.append(file_name)

        for arrays_all_subplots, file_name_ in zip(arrays_all_figures, file_names):
            if arrays_all_subplots is not None:
                arrays_all_subplots_np = copy.deepcopy(arrays_all_subplots)
                if isinstance(arrays_all_subplots_np, list):
                    arrays_all_subplots_np = arrays_all_subplots_np[0]
                target_folder_ = u.save_array(arrays_all_subplots_np, user_file_name=file_name_,
                                              out_dir_name=array_dir_name,
                                              append_time=False, **kwargs)
                time.sleep(g.sleeping_time_figure_saving)  # wait 1 second to avoid overwriting the file

        # do the same for settings
        if settings_dir_name is not None:
            for settings_figure, file_name_ in zip(settings_figures, file_names):
                if settings_figure is not None:
                    if isinstance(settings_figure, list):
                        settings_figure = settings_figure[0]
                    target_folder_ = u.save_settings(settings_figure, user_file_name=file_name_,
                                                     out_dir_name=settings_dir_name,
                                                     append_time=False, **kwargs)
                    time.sleep(g.sleeping_time_figure_saving)

        # do the same for the figures
        for figure_, file_name_ in zip(figures_, file_names):
            if figure_ is not None:
                if isinstance(figure_, list):
                    figure_ = figure_[0]
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
        settings_all_figures = [[], []]
        settings_dir_name = os.path.join(parent_dir_name, 'settings')

        if not os.path.isdir(settings_dir_name):
            raise FileNotFoundError(f"Directory {settings_dir_name} does not exist")

        for file_name in os.listdir(settings_dir_name):
            try:
                with open(os.path.join(settings_dir_name, file_name), 'rb') as f:
                    if 'rmse' in file_name:
                        settings_all_figures[0].append([pickle.load(f)])
                    elif 'herm' in file_name:
                        settings_all_figures[1].append([pickle.load(f)])
                    else:
                        raise ValueError(f"Unknown file name: {file_name}")
            except FileNotFoundError:
                print(f"File {file_name} not found")

        # Load errors
        errors_all_figures = [[], []]
        array_dir_name = os.path.join(parent_dir_name, 'arrays')
        for file_name in os.listdir(array_dir_name):
            with open(os.path.join(array_dir_name, file_name), 'rb') as f:
                if 'rmse' in file_name:
                    errors_all_figures[0].append([pickle.load(f)])
                elif 'herm' in file_name:
                    errors_all_figures[1].append([pickle.load(f)])
                else:
                    raise ValueError(f"Unknown file name: {file_name}")

        return errors_all_figures, settings_all_figures
