import copy
import warnings

import colorednoise as cn
import numpy as np
import scipy
from scipy import signal

import src.global_constants as g
import src.utils as u
from src import rir_manager


def sinusoid(n, f, phase=0):
    return np.sin(2 * np.pi * f * n + phase)


def cosine(n, f, phase=0):
    return np.cos(2 * np.pi * f * n + phase)


class SignalGenerator:

    def __init__(self, num_mics_max, **kwargs):

        self.noise_estimate_perturbation_amount = kwargs['noise_estimate_perturbation_amount']
        self.nstft = kwargs['nstft']

        self.sounds = {}
        self.noises_info = []

        self.desired_sig_names = kwargs['desired']
        if not isinstance(self.desired_sig_names, list):
            self.desired_sig_names = [self.desired_sig_names]

        self.duration_output_frames = None
        self.duration_output_samples = None
        self.sig_shape = None

        self.noverlap = 0
        self.noverlap_percentage = kwargs['noverlap_percentage']

        if 'noverlap_percentage' in kwargs:
            if not 0 <= self.noverlap_percentage <= 1:
                raise ValueError(f"Invalid value for {self.noverlap_percentage = }. Should be in [0, 1].")
            self.noverlap = int(self.noverlap_percentage * self.nstft)

        self.window_name = g.window_function_name  # use default window (g.window_function_name)
        if self.noverlap == 0:
            print(f"Selecting rectangular window: overlap set to {self.noverlap}")
            self.window_name = 'rectangular'

        self.num_mics_max = num_mics_max

        window = signal.windows.get_window(self.window_name, self.nstft, fftbins=True)
        r_shift_samples = self.nstft - self.noverlap
        window = window / np.linalg.norm(window)  # normalize to unit energy
        if not signal.check_COLA(window, self.nstft, self.noverlap):
            raise ValueError('The window does not satisfy the COLA condition')
        window = np.sqrt(window)
        self.stft_obj = signal.ShortTimeFFT(hop=r_shift_samples, fs=g.fs, win=window, fft_mode='onesided')

        if 'duration_output_sec' in kwargs:
            self.duration_output_samples = self.seconds_to_stft_compatible_samples(kwargs['duration_output_sec'],
                                                                                   self.noverlap)
            _, _, self.duration_output_frames = self.get_stft_shape()

        elif 'duration_output_frames' in kwargs:
            equivalent_num_frames = kwargs['duration_output_frames']
            if equivalent_num_frames == 'auto':
                equivalent_num_frames = int(10 * num_mics_max * (self.nstft + 1))
            elif equivalent_num_frames == 'minimum':
                equivalent_num_frames = int(num_mics_max * (self.nstft // 2 + 1))
            else:
                assert equivalent_num_frames > 0

            # Desired number of samples does not depend on overlap
            self.duration_output_samples = SignalGenerator.frames_to_samples(equivalent_num_frames,
                                                                             self.nstft,
                                                                             0)

            # Actual number of frames depends on number of samples and overlap
            self.duration_output_frames = SignalGenerator.samples_to_frames(self.duration_output_samples,
                                                                            self.nstft,
                                                                            self.noverlap_percentage)
            # print(f"{equivalent_num_frames = } frames without overlap corresponds to a duration in seconds that gives "
            #       f"{self.duration_output_frames = } frames with overlap.")
        else:
            warnings.warn(f"Both 'duration_output_sec' and 'duration_output_frames' not specified."
                          f"Default: use signal of length 1s.")
            self.duration_output_samples = int(g.fs)

        self.exp_settings = kwargs

        rir_settings = self.exp_settings['rir_settings']
        rir_settings['num_mics_max'] = self.num_mics_max
        rir_settings['nstft'] = self.nstft

        self.rir_manager = rir_manager.RirManager(**rir_settings)

        self.dry_samples_cache = dict()

        # Make sure that we use same noise "type" for mix and for covariance estimation, e.g. same washing machine noise
        # when using "esc-50-selected"
        self.noise_name_current_montecarlo = ''

    @staticmethod
    def seconds_to_stft_compatible_samples(x_seconds, noverlap):
        if noverlap == 0:
            return int(np.ceil(x_seconds * g.fs))
        return int(noverlap * np.ceil(x_seconds * g.fs / noverlap))

    @staticmethod
    def frames_to_samples(num_frames, frame_len, overlap_percentage):
        assert (0 <= overlap_percentage <= 1)
        temp = frame_len * (num_frames - (num_frames - 1) * overlap_percentage)
        return int(np.ceil(temp))

    @staticmethod
    def samples_to_frames(num_samples, frame_len, overlap_percentage):
        assert (0 <= overlap_percentage <= 1)
        num_frames = (num_samples - frame_len * overlap_percentage) / (frame_len * (1 - overlap_percentage))
        return int(np.ceil(num_frames))

    def generate_cosine(self, num_realizations, num_samples, freq_hz=None):

        amplitude_centered_around_zero = self.exp_settings.get('amplitude_centered_around_zero', True)

        dry = np.zeros((num_realizations, num_samples))
        amplitude_range = (0.5, 1.0) if not amplitude_centered_around_zero else (-0.5, 0.5)

        time_axes = np.zeros((num_realizations, num_samples))
        time_axis_single = np.arange(num_samples) / g.fs
        time_axes = np.broadcast_to(time_axis_single, time_axes.shape)

        if freq_hz is None:
            # calculate sinusoidal frequencies
            generated_freqs_perc = self.exp_settings.get('frequencies_being_generated', None)
            if generated_freqs_perc is None:
                generated_freqs_perc = self.exp_settings.get('frequencies_being_evaluated', None)
            if generated_freqs_perc is None:
                raise AttributeError(
                    "Generating sin signal but no frequency specified under 'frequencies_being_evaluated'")
            generated_freqs_hz = self.get_frequencies_hz_from_percentages(generated_freqs_perc)
        else:
            generated_freqs_hz = freq_hz

        # generate random phase
        num_sins = len(generated_freqs_hz)
        phase_range = (0, self.exp_settings.get('single_frame_phase_range_end', 0))
        phi = g.rng.uniform(low=phase_range[0], high=phase_range[1], size=(num_sins, num_realizations, 1))

        # generate random amplitude
        amp = g.rng.uniform(low=amplitude_range[0], high=amplitude_range[1], size=(num_sins, num_realizations, 1))
        print(f"Generating {num_sins} sinusoids with amplitude in {amplitude_range} and phase in {phase_range}")

        for cos_idx, cos_freq_hz in enumerate(generated_freqs_hz):
            dry += cosine(time_axes, cos_freq_hz, phase=phi[cos_idx]) * amp[cos_idx]
        return dry

    @staticmethod
    def is_silent_sound(x, percent_quiet=0.5, threshold=1e-4):
        # If more than percent_quiet of the samples are below threshold, return True
        return x is None or np.mean(np.abs(x) < threshold) > percent_quiet

    def get_normalized_sound_from_name(self, sound_name, length_samples, directional_point_source):
        """
        :param sound_name:
        :param length_samples:
        :param directional_point_source:
        :return:
        """

        # interferer
        if sound_name == 'male_voice' or sound_name == 'interferer' or sound_name == 'male':
            if 'male' not in self.dry_samples_cache:
                out_dir_name = g.dataset_folder / "Anechoic"
                voice_fname = out_dir_name / (f"SI Harvard Word Lists Male_16khz" + ".wav")
                samplerate4, male_dry_samples = scipy.io.wavfile.read(voice_fname)
                self.dry_samples_cache['male'] = male_dry_samples.T

            dry = self.get_mono_clip_random_offset(self.dry_samples_cache['male'], length_samples)

        elif sound_name == 'esc-50-selected':
            out_dir_name = g.dataset_folder / "ESC-50-master" / "audio-selected"
            if self.noise_name_current_montecarlo == '':
                audio_list = list(out_dir_name.glob("*.wav"))
                if len(audio_list) == 0:
                    raise FileNotFoundError(f"No audio files found in {out_dir_name}")
                self.noise_name_current_montecarlo = g.rng.choice(audio_list)
            file_name = self.noise_name_current_montecarlo
            if file_name not in self.dry_samples_cache:
                self.dry_samples_cache[file_name] = self.read_make_float_resample(file_name)
            try:
                dry = self.get_mono_clip_random_offset(self.dry_samples_cache[file_name], length_samples)
            except ValueError as e:
                # Throw the error, plus show the file name that caused the error
                raise ValueError(f"{e} File name: {file_name}")

        # non-stationary noise
        elif sound_name == 'shots':
            out_dir_name = g.dataset_folder / "Anechoic"
            shots_fname = out_dir_name / (f"train_claps_mono" + ".wav")
            samplerate5, shots_dry_samples = scipy.io.wavfile.read(shots_fname)
            shots_dry_samples = u.signed16bitToFloat(shots_dry_samples).T
            shots_dry_samples = u.resample(shots_dry_samples, samplerate5)
            dry = self.get_mono_clip_random_offset(shots_dry_samples, length_samples)

        # babble noise
        elif sound_name == 'babble':
            out_dir_name = g.dataset_folder
            babble_fname = out_dir_name / f"party-crowd-daniel_simon.wav"
            samplerate5, babble_dry_samples = scipy.io.wavfile.read(babble_fname)
            babble_dry_samples = u.signed16bitToFloat(babble_dry_samples)
            if babble_dry_samples.ndim > 1 and babble_dry_samples.shape[0] > babble_dry_samples.shape[1]:
                babble_dry_samples = babble_dry_samples.T
            babble_dry_samples = u.resample(babble_dry_samples, samplerate5)
            dry = self.get_mono_clip_random_offset(babble_dry_samples, length_samples)

        elif sound_name == 'pink' or sound_name == 'brown':
            beta = 1 if sound_name == 'pink' else 2
            size = [length_samples] if directional_point_source else [self.num_mics_max, length_samples]
            dry = cn.powerlaw_psd_gaussian(beta, size=size)

        elif sound_name == 'female_voice' or sound_name == 'female':
            if 'female' not in self.dry_samples_cache:
                out_dir_name = g.dataset_folder / "Anechoic"
                voice_fname = out_dir_name / (f"SI Harvard Word Lists Female_16khz" + ".wav")
                samplerate3, female_dry_samples = scipy.io.wavfile.read(voice_fname)
                # voice_dry_samples = u.signed16bitToFloat(voice_dry_samples).T
                # voice_dry_samples = resample(voice_dry_samples, samplerate3)
                self.dry_samples_cache['female'] = female_dry_samples.T

            dry = self.get_mono_clip_random_offset(self.dry_samples_cache['female'], length_samples)

        elif sound_name == 'vowel' or sound_name == 'long_vowel':
            out_dir_name = g.dataset_folder
            file_name = out_dir_name / "long_a.wav"
            dry = self.read_make_float_resample(file_name)
            dry = self.get_mono_clip_random_offset(dry, length_samples)

        elif sound_name == 'washing_machine' or sound_name == 'washing_machine_1':
            file_name = g.dataset_folder / "ESC-50-master" / "audio-selected" / "1-23996-A-35.wav"
            if 'washing_machine_1' not in self.dry_samples_cache:
                self.dry_samples_cache['washing_machine_1'] = self.read_make_float_resample(file_name)
            dry = self.get_mono_clip_random_offset(self.dry_samples_cache['washing_machine_1'], length_samples)

        elif sound_name == 'washing_machine_2':
            file_name = g.dataset_folder / "ESC-50-master" / "audio-selected" / "1-21896-A-35.wav"
            if 'washing_machine_2' not in self.dry_samples_cache:
                self.dry_samples_cache['washing_machine_2'] = self.read_make_float_resample(file_name)
            dry = self.get_mono_clip_random_offset(self.dry_samples_cache['washing_machine_2'], length_samples)

        elif sound_name == 'wgn' or sound_name == 'white':
            size = [length_samples] if directional_point_source else [self.num_mics_max, length_samples]
            dry = g.rng.standard_normal(size)

        elif sound_name == 'correlated' or sound_name == 'all_sines' or sound_name == 'sinusoid':
            num_channels = 1 if directional_point_source else self.num_mics_max
            # fundamental_freq_hz = g.rng.integers(80, 250)  # DON'T do this! Otherwise noise and noise_mix have different frequencies
            fundamental_freq_hz = 100
            freqs_hz = [fundamental_freq_hz * i for i in range(1, 20)]
            dry = self.generate_cosine(num_channels, length_samples, freq_hz=freqs_hz)
            dry = np.squeeze(dry)

        elif sound_name == 'demo_target':
            file_name = g.dataset_folder / 'target-speech.wav'
            if 'demo_target' not in self.dry_samples_cache:
                self.dry_samples_cache['demo_target'] = self.read_make_float_resample(file_name)
            dry = self.get_mono_clip_random_offset(self.dry_samples_cache['demo_target'], length_samples)

        else:
            raise TypeError(f"invalid sound_name {sound_name}")

        if np.max(np.abs(dry)) < 1e-6:
            warnings.warn(
                f"Sound file '{sound_name}' is silent for length {length_samples / g.fs}s. Max value: {np.max(np.abs(dry))}")

        dry = dry - np.mean(dry)
        dry = SignalGenerator.normalize_length(length_samples, dry)
        dry = u.normalize_volume(dry, max_value=0.5)

        return dry

    @staticmethod
    def read_make_float_resample(file_name):
        current_fs, dry_samples = scipy.io.wavfile.read(file_name)
        dry_samples = u.signed16bitToFloat(dry_samples)
        if dry_samples.ndim > 1 and dry_samples.shape[0] > dry_samples.shape[1]:
            dry_samples = dry_samples.T
        dry = u.resample(dry_samples, current_fs, g.fs)
        return dry

    @staticmethod
    def normalize_length(desired_duration, dry):
        if dry.shape[-1] < desired_duration:
            dry = np.tile(dry, reps=desired_duration // dry.shape[-1])
        else:
            dry = dry[..., :desired_duration]  # trim
        return dry

    def generate_desired_signal(self, signal_names=None, atf_desired=None):

        sound_dry = np.zeros(self.duration_output_samples)
        assert len(signal_names) == 1
        for sound_name in signal_names:
            sound_dry = sound_dry + self.get_normalized_sound_from_name(sound_name, self.duration_output_samples,
                                                                        directional_point_source=True)

        # smoothen signal to avoid clicks at the end and at the beginning
        # sound_dry = self.fade_avoid_clicks(sound_dry)
        # sound_dry = u.smoothen_corners(sound_dry)
        # sound_dry = sound_dry - np.mean(sound_dry)
        # To avoid corner effects of STFT, we select the central portion of the signal, smoothen it, and then pre and post pad
        first_unaffected, last_unaffected = self.stft_obj.lower_border_end[0], \
        self.stft_obj.upper_border_begin(self.duration_output_samples)[0]
        sound_dry_central_portion = sound_dry[first_unaffected * 2:last_unaffected]
        sound_dry_central_portion = u.smoothen_corners(sound_dry_central_portion, alpha=1)
        sound_dry = np.pad(sound_dry_central_portion,
                           (first_unaffected * 2, self.duration_output_samples - last_unaffected))
        sound_dry = sound_dry[np.newaxis, :]
        # sound_dry = u.pad_or_trim_to_len(sound_dry, self.duration_output_samples)
        sound_dry_stft = self.stft(sound_dry)

        sound_wet, sound_wet_stft, early, early_stft = (
            self.apply_transfer_function(sound_dry, atf_desired, self.exp_settings['rir_settings']['rtf_type'],
                                         sound_dry_stft, self.rir_manager.rir_voice_samples))

        sound_dry = np.broadcast_to(sound_dry, sound_wet.shape)

        return np.squeeze(sound_wet), np.squeeze(sound_wet_stft), np.squeeze(sound_dry), np.squeeze(sound_dry_stft), \
            early, early_stft

    @staticmethod
    def fade_avoid_clicks(sound_dry):
        if sound_dry is None:
            return sound_dry

        if sound_dry.ndim == 1:
            sound_dry = sound_dry[np.newaxis, :]

        fade_length = int(min(sound_dry.shape[-1], (g.fs * 0.040)))
        fading_window = np.ones_like(sound_dry)
        fading_window[:, :fade_length] = np.linspace(0, 1, fade_length)
        fading_window[:, -fade_length:] = np.linspace(1, 0, fade_length)
        sound_dry = sound_dry * fading_window
        return np.squeeze(sound_dry)

    def apply_transfer_function(self, dry_samples, atf_desired, rtf_type, dry_stft, rir=None):

        if 'random' in rtf_type or 'deterministic' in rtf_type:
            raise NotImplementedError("How to do this after differencing wet/early?")
            wet_stft = dry_stft[np.newaxis] * atf_desired[..., np.newaxis]
            _, wet_samples = self.istft(wet_stft)
            wet_samples = u.pad_or_trim_to_len(wet_samples, dry_samples.shape[-1])
        else:
            # Dry signal (anechoic)
            dry_samples = np.atleast_2d(dry_samples)
            dry_stft_num_frames = dry_stft.shape[-1]

            # Reverberant signal
            wet_samples = SignalGenerator.convolve_with_rir(dry_samples, rir)
            wet_stft = self.stft(wet_samples)
            wet_stft = wet_stft[..., :dry_stft_num_frames]

            # Early reflections (single channel, it's the reference signal at the reference microphone)
            early_stft = dry_stft * atf_desired[g.idx_ref_mic, ..., np.newaxis]
            early_samples = self.istft(early_stft).real
            early_samples = early_samples[..., :wet_samples.shape[-1]]
            early_stft = self.stft(early_samples)
            # early_samples = SignalGenerator.convolve_with_rir(dry_samples, rir[g.idx_ref_mic, :self.nstft][np.newaxis, ...])
            # early_samples = u.pad_or_trim_to_len(early_samples, wet_samples.shape[-1])
            # early_stft = self.stft(early_samples)

            wet_samples = wet_samples[:self.num_mics_max, :self.duration_output_samples]
            early_samples = early_samples[:, :self.duration_output_samples]

            early_stft = early_stft[..., :self.duration_output_frames]
            wet_stft = wet_stft[..., :self.duration_output_frames]

            # u.plot([dry_samples, early_samples, wet_samples], time_axis=False, titles=['dry', 'early', 'wet'])
            # sel = slice(None), slice(0, 2500)
            # u.plot([dry_samples[sel], early_samples[sel], wet_samples[sel]], time_axis=False, titles=['dry', 'early', 'wet'])
            # #
            # u.plot_matrix(dry_stft[:, :50], title="Dry STFT", amp_range=(-40, 30))
            # u.plot_matrix(early_stft[:, :50], title="Early STFT", amp_range=(-40, 30))
            # u.plot_matrix(wet_stft[g.idx_ref_mic, :50], title="Wet STFT", amp_range=(-40, 30))

        return wet_samples, wet_stft, early_samples, early_stft

    def get_empty_time_stft_matrices(self):
        if self.duration_output_samples is not None:
            empty_mat_time = np.zeros((self.num_mics_max, self.duration_output_samples))
            empty_mat_stft = self.stft(empty_mat_time)
            return empty_mat_time, empty_mat_stft
        else:
            return np.zeros((self.num_mics_max, 1)), \
                np.zeros((self.num_mics_max, self.nstft // 2 + 1, self.duration_output_frames))

    def stft(self, x, **kwargs):
        # return u.stft(x, nstft=self.nstft, noverlap=self.noverlap, window=self.window_name)
        return self.stft_obj.stft(x, **kwargs)

    def istft(self, x, **kwargs):
        # return u.istft(x, nstft=self.nstft, noverlap=self.noverlap, window=self.window_name)
        return self.stft_obj.istft(x, **kwargs)

    def load_and_convolve_noise_samples(self, noise_name, num_samples, dir_point_source=False,
                                        same_volume_all_mics=False,
                                        noise_volumes_per_mic=1):

        if len(noise_name) == 0 or (len(noise_name) == 1 and '' in noise_name):
            return np.zeros((self.num_mics_max, num_samples)), 0

        noise_samples = self.get_normalized_sound_from_name(noise_name, num_samples, dir_point_source)

        if dir_point_source and same_volume_all_mics:
            raise ValueError(f"dir_point_source and same_volume_all_mics cannot be both True.")
        elif dir_point_source:
            noise_samples = self.convolve_with_rir(noise_samples[np.newaxis, ...], self.rir_manager.rir_noise_samples)
        elif not same_volume_all_mics:
            # scale different microphones differently
            noise_samples = noise_samples * noise_volumes_per_mic

        noise_samples = u.pad_or_trim_to_len(noise_samples, num_samples)

        return noise_samples

    @staticmethod
    def convolve_with_rir(dry, rir_samples):
        if rir_samples is not None:
            return signal.convolve(dry, rir_samples, mode='full')
        else:
            warnings.warn("No RIR specified. Returning dry signal.")
            return dry

    def generate_noise_samples_single_type(self, noise_info, num_samples=None):
        """ Generate noise samples of a single type. Corresponds to a single location in space, or sensor noise."""

        noise_samples_tot = np.zeros((self.num_mics_max, num_samples))
        assert (len(noise_info['names']) == 1)  # this for loop is probably superfluous, remove? TODO
        for noise_name in noise_info['names']:
            directional_point_source = noise_info.get('isDirectional', False)
            same_volume_all_mics = noise_info.get('same_volume_all_mics', False)
            noise_samples = self.load_and_convolve_noise_samples(noise_name, num_samples,
                                                                 directional_point_source, same_volume_all_mics,
                                                                 noise_volumes_per_mic=noise_info[
                                                                     'noise_volumes_per_mic'])
            noise_samples_tot += noise_samples

        return noise_samples_tot

    def generate_signal_samples(self, atf_desired):

        if self.desired_sig_names is None:
            raise ValueError(
                "No desired signal specified. Add \"desired\" to the yaml file. Example: desired: ['female']")
        elif len(self.desired_sig_names) > 1:
            # more than one name is specified. randomly pick one to use as the desired signal in this realization
            desired_sig_names_current = [g.rng.choice(self.desired_sig_names)]
        else:
            desired_sig_names_current = self.desired_sig_names

        wet_samples, wet_stft, desired_dry_samples, desired_dry_stft, early_samples, early_stft = \
            self.generate_desired_signal(desired_sig_names_current, atf_desired)

        # First realization: noise used to estimate noise only covariance
        # Should be long enough to guarantee that the matrix is invertible
        # reference_clean = desired_dry_samples  # use dry signal to calculate SNR (wet signal should be used in general)
        reference_clean = wet_samples  # use wet signal to calculate SNR

        if g.noise_estimation_time > 0:
            noise_length_samples = int(g.noise_estimation_time * g.fs)
            warnings.warn(f"Using {g.noise_estimation_time} seconds of noise to estimate noise covariance matrix")
        else:
            noise_length_samples = reference_clean.shape[-1]

        # First realization: noise used to estimate noise only covariance Rv (or Rn).
        # Longer estimation time means more accurate estimation of Rv.
        self.noise_name_current_montecarlo = ''
        noise_samples = self.generate_and_mix_noises_variable_snr(reference_clean,
                                                                  noise_length_samples=noise_length_samples)

        # Second realization: noise used to make noisy mix
        noise_samples_mix = self.generate_and_mix_noises_variable_snr(reference_clean,
                                                                      noise_length_samples=wet_samples.shape[-1])

        noise_stft_mix = self.stft(noise_samples_mix)
        mix_stft_ = noise_stft_mix + wet_stft
        mix_samples = self.istft(mix_stft_)

        stimuli_samples = {'wet': wet_samples,
                           'mix': mix_samples,
                           'noise': noise_samples,
                           'noise_mix': noise_samples_mix,
                           'desired_dry': desired_dry_samples,
                           'early': early_samples}

        # Pad signals so that they all same length
        # TODO: Probably a bug that signals are padded to the length of the noise used to estimate covariance?
        max_len = max([stimuli_samples[key].shape[-1] for key in stimuli_samples.keys()])
        for key in stimuli_samples.keys():
            stimuli_samples[key] = u.pad_or_trim_to_len(stimuli_samples[key], max_len)

        stimuli_stft = {'wet': wet_stft,
                        'mix': mix_stft_,
                        'noise': self.stft(noise_samples),
                        'noise_mix': self.stft(noise_samples_mix),  # if g.debug_mode else np.zeros_like(mix_stft_),
                        'desired_dry': self.stft(desired_dry_samples),
                        'early': early_stft}

        return stimuli_samples, stimuli_stft

    # Adding noise with a desired signal-to-noise ratio https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    def generate_and_mix_noises_variable_snr(self, reference_samples, noise_length_samples=None):

        if noise_length_samples is None:
            noise_length_samples = self.duration_output_samples
        elif noise_length_samples == reference_samples.shape[-1]:
            pass
        else:
            reference_samples = SignalGenerator.normalize_length(noise_length_samples, reference_samples)

        noises_samples = [self.generate_noise_samples_single_type(noise_info, noise_length_samples)
                          for noise_info in self.noises_info]

        # to ensure that each noise and input signals have the same power
        reference_power = np.var(reference_samples)

        noise_tot_samples = np.zeros((self.num_mics_max, noise_length_samples))
        for noise_samples, noise_info in zip(noises_samples, self.noises_info):
            if noise_samples.any() > g.eps:
                # adjust the power of the added noise to obtain the desired signal-to-noise ratio (SNR).
                noise_power_original = np.var(noise_samples)
                snr_linear = u.db_to_linear(float(noise_info['snr']))
                rescaling_coefficient = np.sqrt(reference_power / (snr_linear * noise_power_original))
                noise_tot_samples = noise_tot_samples + rescaling_coefficient * noise_samples

        # noise_tot_samples = self.fade_avoid_clicks(noise_tot_samples)
        # noise_tot_samples = u.smoothen_corners(noise_tot_samples)
        noise_tot_samples = noise_tot_samples - np.mean(noise_tot_samples, axis=1, keepdims=True)

        return noise_tot_samples

    def get_stft_shape(self):
        if self.sig_shape is not None:
            return self.sig_shape
        else:
            _, dummy_stft = self.get_empty_time_stft_matrices()
            self.sig_shape = dummy_stft.shape
            return self.sig_shape

    @property  # first decorate the getter method
    def stft_shape(self):  # This getter method name is *the* name
        if self.sig_shape is not None:
            return self.sig_shape
        else:
            _, dummy_stft = self.get_empty_time_stft_matrices()
            self.sig_shape = dummy_stft.shape
            return self.sig_shape

    @stft_shape.setter  # the property decorates with `.setter` now
    def stft_shape(self, value):  # name, e.g. "attribute", is the same
        self.sig_shape = value  # the "value" name isn't special

    @staticmethod
    def perturb_noise_stft_with_wgn(amount, noise_stft_):
        print(f"Perturbing noise with WGN of std dev {amount:.6f}")
        noise_stft_ += amount * u.circular_gaussian(noise_stft_.shape)
        return noise_stft_

    def get_frequencies_bins_from_percentages(self, freqs_percentages):
        if freqs_percentages and np.alltrue(np.array(freqs_percentages) > 0):
            freqs_bins = (1 + self.nstft // 2) * np.array(freqs_percentages)
            freqs_bins = np.asarray(np.floor(freqs_bins), dtype='int').flatten()
            # freqs_bins = np.asarray(np.round(freqs_bins), dtype='int').flatten()
            # freqs_bins = np.asarray([np.floor(freqs_bins), np.ceil(freqs_bins)], dtype='int').flatten()
            freqs_bins = np.unique(freqs_bins)
            return freqs_bins
        else:
            return None

    @staticmethod
    def get_frequencies_hz_from_percentages(freqs_percentages):
        if any(freqs_percentages):
            freqs_hz = [(g.fs / 2.) * freq_perc for freq_perc in freqs_percentages]
            return freqs_hz
        else:
            return None

    @staticmethod
    def generate_correlated_signal(r_true, sig_shape, cpx_data=True):
        """
        Generate a correlated signal from a given covariance matrix.
        The signal is generated by first generating white noise data, and then multiplying it with the square root of the covariance matrix.
        This method is based on the following reference: https://sefidian.com/2021/12/04/steps-to-sample-from-a-multivariate-gaussian-normal-distribution-with-python-code/
        """

        if np.alltrue(np.diag(r_true) == 0):
            warnings.warn("Real covariance matrix is all zeroes?")
            return np.zeros(sig_shape)

        if r_true.ndim != 2:
            raise ValueError(f"r_true must be a 2D matrix, but {r_true.shape = }")

        # Generate "white" noise data
        if cpx_data:
            v_white = u.circular_gaussian(sig_shape)
        else:
            v_white = g.rng.standard_normal(size=sig_shape)

        if np.allclose(r_true, np.identity(r_true.shape[0])):
            # If the covariance matrix is the identity matrix, the white noise data is already the correlated signal
            return v_white
        else:
            rv_true_sqrt = u.cholesky_or_svd(r_true)
            v = rv_true_sqrt @ v_white

        return v

    @staticmethod
    def generate_gaussian_signals_from_covariance_matrices(rs_K, rv_true_, atf_target, stft_shape):
        """
        Generate a clean signal, a wet signal, and a noise signal, all in the frequency domain.
        Generation is based on a given covariance matrix for the clean signal, and a given covariance matrix for the noise signal.

        Parameters:
        - dry (numpy array): an array with shape (num_freqs, num_mics, num_frames), representing the clean power spectral density of the signal
        - rv_true_ (numpy array): an array with shape (num_freqs*num_mics, num_freqs*num_mics), representing the true noise covariance
        - atf_target (numpy array): an array with shape (num_freqs, num_mics), representing the target acoustic transfer function
        """

        (_, _, num_frames) = stft_shape

        dry = SignalGenerator.generate_correlated_signal_outer(rs_K, uncorrelated_over_space=False,
                                                               stft_shape=stft_shape)
        transfer_function = np.diag(atf_target.flatten('F'))
        wet = transfer_function @ dry

        noise = SignalGenerator.generate_correlated_signal_outer(rv_true_, uncorrelated_over_space=True,
                                                                 stft_shape=stft_shape)
        noise_for_mix = SignalGenerator.generate_correlated_signal_outer(rv_true_, uncorrelated_over_space=True,
                                                                         stft_shape=stft_shape)

        dry = np.reshape(dry, stft_shape, order='F')
        noise = np.reshape(noise, stft_shape, 'F')
        wet = np.reshape(wet, stft_shape, 'F')
        noise_for_mix = np.reshape(noise_for_mix, stft_shape, 'F')

        assert noise_for_mix.shape[-1] == noise.shape[-1]  # otherwise need to check SNR rescaling

        mix = wet + noise_for_mix

        stimuli_stft = {'wet': wet, 'mix': mix, 'noise': noise, 'desired_dry': dry}

        return stimuli_stft

    @staticmethod
    def generate_correlated_signal_outer(r_true, uncorrelated_over_space, stft_shape):
        # uncorrelated_over_space: true if diffuse signal, false if point_source

        (num_mics_, num_freqs_, num_frames) = stft_shape

        if uncorrelated_over_space:
            # correlation simulated over both frequency and space. Different mic channels have different signals.
            x = SignalGenerator.generate_correlated_signal(r_true, (num_freqs_ * num_mics_, num_frames))

        else:
            # correlation only simulated over frequency
            x = SignalGenerator.generate_correlated_signal(r_true, (num_freqs_, num_frames))  # K x T

            # broadcast to M x K x T. All channels have the same signal. Used for dry point source.
            x = np.broadcast_to(x[np.newaxis, ...], stft_shape)  # M x K x T
            x = x.reshape((-1, num_frames), order='f')  # MK x T

        return x

    @staticmethod
    def get_rescaling_factor(reference_stft, noise_stft, snr_db):
        """
        Rescale the noise signal to a given SNR.
        The SNR is defined as the ratio of the power of the clean signal to the power of the noise signal.
        """

        snr_linear = u.db_to_linear(float(snr_db))

        reference_pow = np.mean(np.var(reference_stft, axis=(1, 2)))
        noise_pow = np.mean(np.var(noise_stft, axis=(1, 2)))

        rescaling_coefficient = np.sqrt(reference_pow / (snr_linear * noise_pow))

        return rescaling_coefficient

    def stimuli_samples_from_stft(self, stimuli_stft):
        """ Convert the STFT of the stimuli to the time domain. """
        stimuli_samples_new = {}
        for key in stimuli_stft.keys():
            stimuli_samples_new[key] = self.istft(stimuli_stft[key])
            stimuli_samples_new[key] = u.normalize_volume(stimuli_samples_new[key])

        return stimuli_samples_new

    def get_mono_clip_random_offset(self, sound_samples, desired_length_samples):

        # select left channel
        if sound_samples.ndim > 1 and sound_samples.shape[0] > 1:
            sound_samples = sound_samples[0]

        if desired_length_samples > sound_samples.shape[-1]:
            raise ValueError(
                f"Sound file of length {sound_samples.shape[-1]} samples ({sound_samples.shape[-1] // g.fs} s) "
                f"is too short for desired duration of {desired_length_samples} ({desired_length_samples // g.fs} s).")

        dry = None
        ii = 0
        max_iterations = 200
        while self.is_silent_sound(dry) and ii < max_iterations:
            max_offset = sound_samples.size - desired_length_samples
            starting_offset = g.rng.integers(0, max_offset)
            dry = sound_samples[starting_offset:starting_offset + desired_length_samples]
            ii = ii + 1

        if ii == max_iterations:
            raise ValueError(f"Could not find a non-silent sample")

        return dry

    def filter_stimuli_stft_oracle_vad(self, x_stft, silence_threshold=1e-6):
        """
        stimuli_stft['wet'] and stimuli_stft['desired_dry'] are non-zero only when signal is present.
        Same goes for stimuli_stft['noise']. stimuli_stft['mix'] is unmasked only when both noise and speech are present.

        Idea from "Low-rank Approximation Based Multichannel Wiener Filter Algorithms for Noise Reduction with
        Application in Cochlear Implants".

        Setting a HIGHER threshold means that MORE frames are considered 'active'.
        """

        def get_vad_active_frames(x):
            return np.mean(np.abs(x), axis=(0, 1)) > silence_threshold

        vad_active_frames = dict()
        vad_active_frames['early'] = get_vad_active_frames(x_stft['early'])
        vad_active_frames['wet'] = get_vad_active_frames(x_stft['wet'])
        vad_active_frames['noise'] = get_vad_active_frames(x_stft['noise'])
        vad_active_frames['noise_mix'] = get_vad_active_frames(x_stft['noise_mix'])
        vad_active_frames['mix'] = vad_active_frames['wet'] & vad_active_frames['noise_mix']

        # Raise an error if any of the items in vad_active_frames is all False, which means that the signal is silent
        for key, vad_active_frames_ in vad_active_frames.items():
            if not np.any(vad_active_frames_):
                raise ValueError(f"Signal is silent for {key = }")

        x_stft_filtered = copy.deepcopy(x_stft)
        x_stft_filtered['early'] = x_stft['early'][:, :, vad_active_frames['early']]
        x_stft_filtered['wet'] = x_stft['wet'][:, :, vad_active_frames['wet']]
        x_stft_filtered['desired_dry'] = x_stft['desired_dry'][:, :, vad_active_frames['wet']]
        x_stft_filtered['noise'] = x_stft['noise'][:, :, vad_active_frames['noise']]
        x_stft_filtered['mix'] = x_stft['mix'][:, :, vad_active_frames['mix']]
        x_stft_filtered['noise_mix'] = x_stft['noise_mix'][:, :, vad_active_frames['noise_mix']]

        return x_stft_filtered, vad_active_frames

    @staticmethod
    def apply_phase_correction_after_vad(stimuli_stft_vad, vad_dict, phase_correction):
        """
        Apply phase correction for estimation of inter-spectral correlations.
        Phase correction is needed to compensate delay of STFT time windows. Since the VAD cuts off some segments, phase correction for those segments needs to be cut off, too.
        """

        num_mics = stimuli_stft_vad['wet'].shape[0]

        # stimuli_stft_vad and vad_dict are dicts with same keys
        stimuli_stft_corrected = copy.deepcopy(stimuli_stft_vad)
        for key, vad_active_frames in vad_dict.items():
            if vad_active_frames is not None and phase_correction is not None:
                for mm in range(num_mics):
                    phase_correction_mm = phase_correction[..., :len(vad_active_frames)][..., vad_active_frames]
                    stimuli_stft_corrected[key][mm] = stimuli_stft_vad[key][mm] * phase_correction_mm

        return stimuli_stft_corrected

    @staticmethod
    def apply_phase_correction(stimuli_stft, phase_correction):
        """
        Apply phase correction for estimation of inter-spectral correlations.
        Phase correction is needed to compensate delay of STFT time windows.
        """

        stimuli_stft_corrected = copy.deepcopy(stimuli_stft)
        for key in stimuli_stft.keys():
            if phase_correction is not None:
                num_frames = stimuli_stft[key].shape[-1]
                stimuli_stft_corrected[key] = stimuli_stft[key] * phase_correction[..., :num_frames]

        return stimuli_stft_corrected

    @staticmethod
    def remove_mean(stimuli_stft_):
        # remove mean across time
        stimuli_stft = copy.deepcopy(stimuli_stft_)
        for key in stimuli_stft.keys():
            stimuli_stft[key] = stimuli_stft[key] - np.mean(stimuli_stft[key], axis=-1, keepdims=True)
        return stimuli_stft

    @staticmethod
    def apply_bin_mask(stft_target, bin_mask, num_bins_original):
        if stft_target is not None and stft_target.shape[1] == num_bins_original:
            stft_target = stft_target[:, bin_mask]
        return stft_target
