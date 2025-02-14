"""
This script converts a dataset of .mat files to .wav files.

Source folder: your-source-path/datasets/Hadad
Target folder: your-source-path/datasets/Hadad_wav

Create folders in target folder so that the structure is the same as in source folder.

Each mat file is converted to a wav file with the same name.

"""

import os
from pathlib import Path
import scipy.io.wavfile as wavfile
import scipy.io as sio
from tqdm import tqdm
import src.utils as u

source_folder = 'source-path'
target_folder = 'target-path'

# First, recreate the folder structure
for root, dirs, files in os.walk(source_folder):
    for dir_ in dirs:
        target_dir = target_folder / Path(root).relative_to(source_folder) / dir_
        os.makedirs(target_dir, exist_ok=True)

for root, dirs, files in os.walk(source_folder):
    for file in tqdm(files):
        if file.endswith(".mat"):
            source_file = Path(root) / file
            target_file = target_folder / Path(root).relative_to(source_folder) / file.replace(".mat", ".wav")
            data_dict = sio.loadmat(str(source_file))
            rir = data_dict['impulse_response']
            rir = u.resample(rir.T, current_fs=48000, desired_fs=16000).T

            # if not os.path.exists(target_file):
            wavfile.write(target_file, 16000, rir)
            print(f"Output file name: {target_file.name}")
