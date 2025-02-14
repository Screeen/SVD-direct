import sys
import argparse
import os
import warnings

sys.path.append('..')
sys.path.append('../src')

import src.global_constants as g
from src.kickstarter import Kickstarter

"""
This script converts to tex the results of the experiments in the folder specified by which_folder.
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export plots for the TSP paper')
    parser.add_argument('-t', '--target-folder', type=str, default=None,
                        help='Target folder, e.g. ~/Documents/TU Delft/out3/talsp2025/unequal')
    args = parser.parse_args()

    if args.target_folder is None:

        # find most recent folder in g.out_dir_experiments
        # list folders in g.out_dir_experiments/talsp2025 and sort by date
        parent_dir = os.path.join(g.out_dir_experiments, 'talsp2024')
        folder_list = os.listdir(parent_dir)
        folder_list.sort(key=lambda x: os.path.getmtime(os.path.join(parent_dir, x)))

        # filter out hidden folders
        folder_list = [folder for folder in folder_list if not folder.startswith('.')]

        args.target_folder = os.path.join(parent_dir, folder_list[-1])  # get most recent folder

        warnings.warn(f"Target folder for exporting Tex plots not specified. Using default: {args.target_folder}")

    # Here we only need an additional loop if args.target_folder contains multiple experiments.
    # E.g. if args.target_folder is ~/Documents/TU Delft/out3/talsp2025/unequal, then we need to loop over
    # all the sub-folders in ~/Documents/TU Delft/out3/talsp2025/unequal.

    if 'settings' in os.listdir(args.target_folder):
        # args.target_folder is the folder where the results are stored. Contains settings, arrays, figures.
        exp_name_ = args.target_folder.split('/')[-1].split('-')[0]
        results_dict = Kickstarter.run_script(exp_name_,
                                              load_dir_name=args.target_folder,
                                              use_tex_labels_=True)
    else:
        for exp_name_complete in os.listdir(args.target_folder):
            if not os.path.isdir(os.path.join(args.target_folder, exp_name_complete)):
                continue

            # load_dir is the folder where the results are stored. Contains settings, arrays, figures.
            load_dir_name = os.path.join(args.target_folder, exp_name_complete)
            exp_name_ = exp_name_complete.split('-')[0]
            results_dict = Kickstarter.run_script(exp_name_,
                                                  load_dir_name=load_dir_name,
                                                  use_tex_labels_=True)
