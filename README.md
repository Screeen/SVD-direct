# SVD-direct
## Introduction
This code accompanies the paper *Wideband Relative Transfer Function (RTF) Estimation Exploiting Frequency Correlations*.
The paper is currently under review for the IEEE/ACM Transactions on Audio, Speech, and Language Processing (TALSP).

![vowel in time, frequency, and bifrequency domains](./pics/pic1.jpg)
*The /ä/ phoneme uttered by a male speaker. 
The left plot depicts the waveform, while the center plot shows the power spectral density (PSD). 
The peaks in the PSD are found at integer multiples of the fundamental frequency (harmonics).
The right plot shows the spectral correlation or bifrequency spectrum. 
The grid-like structure of peaks in the bifrequency spectrum, whose spacing is proportional to the fundamental frequency, indicates a correlation between frequency components*

In short, the paper proposes a new method for estimating the acoustic transfer function (ATF) of a room.
In multi-microphone beamforming, the ATF is used not only to estimate the direction-of-arrival of a source, but also
the effect of the early reflections on the signal. This is important for dereverberation and noise reduction.

In the proposed algorithm, we exploit the correlation between frequency components of the signal to improve the ATF estimate.

## Installation and first run
### Prerequisites
1. Python 3.9 installed
2. Only tested on MacOS 13.4.1, but should work on Linux as well.

### Preparation
1. Open a terminal and navigate to the root of the project.
2. Clone this repository with `git clone git@github.com:Screeen/SVD-direct.git` or download the zip file and extract it. 
3. Run `python3 -m venv env` to create a virtual environment.
4. Activate the virtual environment with `source env/bin/activate`
5. Install the required packages with `python3 -m pip install -r requirements.txt`
6. Run the unit tests with
```
cd src
python3 acoustic_estimator_test.py
```
If everything went well, you should see something like this:
```
WARNING! DEFAULT RANDOM SEED, experiments will give same result over and over.
noise_cov_description = 'identity'
noise_cov_description = 'wgn + identity'
noise_cov_description = 'spd'
noise_cov_description = 'correlated'
estimate_rtf_covariance_subtraction...
```

### Running the experiments
8. We are now ready to run the experiments. First lets do a quick round to make sure everything works:
```
cd ..
chmod +x scripts_bash/run_tsp2023
./scripts_bash/run_tsp2023 1 equal
```
Runs the **first experiment** ('equal variances') with *Montecarlo constant* 1 (which means very few Montecarlo iterations will be run).
The results will be stored in `../out/tsp2023/something_something`.

If you are ready to wait, you can run the full experiment with 
```
./scripts_bash/run_tsp2023 1e9 equal
```
This will take around an hour to complete.

9. To run the second experiment, type
```
./scripts_bash/run_tsp2023 1e9 unequal
```

11. You can also export plots with Tex fonts `python 2023-05\ experiments\ correlation\ \(TSP2023\)\ plots.py -t ~/Documents/TU\ Delft/out/tsp2023`
12. Unfortunately, because the real data used in the paper is proprietary, we cannot share it here. But you can still run the code on your own data.
In general, use this script to run the experiments:
```
python3 -m scripts.2023-05\ experiments\ correlation\ \(TSP2023\) --help
```
Which will give you this output:
```
Run experiments for the TSP paper

optional arguments:
 -h, --help            show this help message and exit
 --exp_name EXP_NAME   Experiment name, options: ['target_correlation', 'noise_correlation', 'time_frames', 'snr', 'speech_time_frames', 'speech_snr', 'speech_nstft']
 --repeated_experiments_constant REPEATED_EXPERIMENTS_CONSTANT
                       Number of repeated experiments, e.g. 1e6, 1e8, 1e11
 --use_multiple_processes USE_MULTIPLE_PROCESSES
                       Use multiple processes
 --target_noise_equal_variances, --no-target_noise_equal_variances
```

So, for example, to run the experiments on speech you could use this command:
```
python3 -m scripts.2023-05\ experiments\ correlation\ \(TSP2023\) --exp_name speech_snr --repeated_experiments_constant 1 --use_multiple_processes True
```

> [!NOTE]
> Any feedback is welcome: Text me here or at G.Bologni@tudelft.nl.
> And have fun with the code!
