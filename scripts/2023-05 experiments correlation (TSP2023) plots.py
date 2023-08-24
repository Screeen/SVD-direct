import sys
import argparse
import os
import warnings

sys.path.append('..')
sys.path.append('../src')

from src.tsp2023 import TspScript

"""
This script converts to tex the results of the experiments in the folder specified by which_folder.
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export plots for the TSP paper')
    parser.add_argument('-t', '--target-folder', type=str, default=None, help='Target folder, e.g. ~/Documents/TU Delft/out3/tsp2023/unequal')
    args = parser.parse_args()

    if args.target_folder is None:
        args.target_folder = r'/Users/giovannibologni/Documents/TU Delft/out3/tsp2023/equal long'
        warnings.warn(f"Target folder for exporting Tex plots not specified. Using default: {args.target_folder}")

    for exp_name_complete in os.listdir(args.target_folder):
        if not os.path.isdir(os.path.join(args.target_folder, exp_name_complete)):
            continue

        # load_dir is the folder where the results are stored. Contains settings, arrays, figures.
        load_dir_name = os.path.join(args.target_folder, exp_name_complete)
        exp_name_ = exp_name_complete.split('-')[0]
        results_dict = TspScript.run_script(exp_name_,
                                             load_dir_name=load_dir_name,
                                             use_tex_labels_=True)
