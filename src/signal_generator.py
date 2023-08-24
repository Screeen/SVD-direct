import warnings
from pathlib import Path

import colorednoise as cn
import numpy as np
import scipy

import src.global_constants as g
import src.utils as u

dataset_folder = Path(__file__).parent.parent / 'datasets'


def sinusoid(n, f, phase=0):
    return np.sin(2 * np.pi * f * n + phase)


def cosine(n, f, phase=0):
    return np.cos(2 * np.pi * f * n + phase)


class SignalGenerator:

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
            self.noverlap = int(self.noverlap_percentage * self.nstft)

        self.window_name = g.window_function_name  # use default window (g.window_function_name)
        if self.noverlap == 0:
            print(f"Selecting rectangular window: overlap set to {self.noverlap}")
            self.window_name = 'rectangular'

        self.num_mics_max = num_mics_max

        if 'duration_output_sec' in kwargs:
            self.duration_output_samples = self.seconds_to_stft_compatible_samples(kwargs['duration_output_sec'],
                                                                                   self.noverlap)
            _, _, self.duration_output_frames = self.get_stft_shape()
        elif 'duration_output_frames' in kwargs:
            self.duration_output_frames = kwargs['duration_output_frames']
            if self.duration_output_frames == 'auto':
                self.duration_output_frames = int(10 * num_mics_max * (self.nstft + 1))
            elif self.duration_output_frames == 'minimum':
                self.duration_output_frames = int(num_mics_max * (self.nstft // 2 + 1))
            else:
                assert self.duration_output_frames > 0
            print(f"{self.duration_output_frames = }")
            self.duration_output_samples = SignalGenerator.frames_to_samples(self.duration_output_frames,
                                                                             self.nstft,
                                                                             self.noverlap_percentage)
        else:
            warnings.warn(f"Both 'duration_output_sec' and 'duration_output_sec' not specified."
                          f"Default: use signal of length 1s.")
            self.duration_output_samples = int(g.fs)

        self.rir_voice_samples = None
        if kwargs['rtf_type'] == 'real':
            self.rir_voice_samples = self.load_rir_from_type('target', kwargs['rir_corpus'])
            self.rir_noise_samples = self.load_rir_from_type('noise', kwargs['rir_corpus'])

            num_samples_rir = self.nstft  # default behaviour (cut late reverberation)
            if kwargs['num_nonzero_samples_rir'] != -1:  # user manually set this value
                num_samples_rir = kwargs['num_nonzero_samples_rir']

            self.rir_voice_samples = self.cut_rir_to_length_samples(self.rir_voice_samples, num_samples_rir)
            self.rir_noise_samples = self.cut_rir_to_length_samples(self.rir_noise_samples, num_samples_rir)

            # normalize volume of impulse responses, but keep relative differences
            max_volume = np.maximum(np.max(np.abs(self.rir_voice_samples)), np.max(np.abs(self.rir_noise_samples)))
            self.rir_voice_samples = 0.95 * self.rir_voice_samples / max_volume
            self.rir_noise_samples = 0.95 * self.rir_noise_samples / max_volume

        self.exp_settings = kwargs

        self.dry_samples = dict()

    @staticmethod
    def cut_rir_to_length_samples(rir_samples, num_nonzero_samples_rir=-1):
        rir_samples = rir_samples[:, :num_nonzero_samples_rir]
        # rir_samples = u.pad_or_trim_to_len(rir_samples, self.nstft)  # in case rir is shorter than nstft
        return rir_samples

    def generate_cosine(self, num_realizations, num_samples):

        amplitude_centered_around_zero = self.exp_settings.get('amplitude_centered_around_zero', True)
        generated_freqs_perc = self.exp_settings.get('frequencies_being_generated', None)

        dry = np.zeros((num_realizations, num_samples))
        amplitude_range = (0.5, 1.0) if not amplitude_centered_around_zero else (-0.5, 0.5)

        time_axes = np.zeros((num_realizations, num_samples))
        time_axis_single = np.arange(num_samples) / g.fs
        time_axes = np.broadcast_to(time_axis_single, time_axes.shape)

        # calculate sinusoidal frequencies
        if generated_freqs_perc is None:
            generated_freqs_perc = self.exp_settings.get('frequencies_being_evaluated', None)
        if generated_freqs_perc is None:
            raise AttributeError(
                "Generating sin signal but no frequency specified under 'frequencies_being_evaluated'")
        generated_freqs_hz = self.get_frequencies_hz_from_percentages(generated_freqs_perc)

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
    def is_silent_sound(sound_samples):
        return sound_samples is None or (np.sum(np.abs(sound_samples)) / sound_samples.shape[-1]) < 0.015

    def get_normalized_sound_from_name(self, sound_name, length_samples, directional_point_source):
        """

        :param sound_name:
        :param length_samples:
        :param directional_point_source:
        :return:
        """

        # interferer
        if sound_name == 'male_voice' or sound_name == 'interferer' or sound_name == 'male':
            if 'male' not in self.dry_samples:
                out_dir_name = dataset_folder / "Anechoic"
                voice_fname = out_dir_name / (f"SI Harvard Word Lists Male_16khz" + ".wav")
                samplerate4, male_dry_samples = scipy.io.wavfile.read(voice_fname)
                self.dry_samples['male'] = male_dry_samples.T

            dry = self.get_mono_clip_random_offset(self.dry_samples['male'], length_samples)

        # non-stationary noise
        elif sound_name == 'shots':
            out_dir_name = dataset_folder / "Anechoic"
            shots_fname = out_dir_name / (f"train_claps_mono" + ".wav")
            samplerate5, shots_dry_samples = scipy.io.wavfile.read(shots_fname)
            shots_dry_samples = u.signed16bitToFloat(shots_dry_samples).T
            shots_dry_samples = u.resample(shots_dry_samples, samplerate5)
            dry = self.get_mono_clip_random_offset(shots_dry_samples, length_samples)

        # babble noise
        elif sound_name == 'babble':
            out_dir_name = dataset_folder
            babble_fname = out_dir_name / f"party-crowd-daniel_simon.wav"
            samplerate5, babble_dry_samples = scipy.io.wavfile.read(babble_fname)
            babble_dry_samples = u.signed16bitToFloat(babble_dry_samples).T
            babble_dry_samples = u.resample(babble_dry_samples, samplerate5)
            dry = self.get_mono_clip_random_offset(babble_dry_samples, length_samples)

        elif sound_name == 'pink' or sound_name == 'brown':
            beta = 1 if sound_name == 'pink' else 2
            size = [length_samples] if directional_point_source else [self.num_mics_max, length_samples]
            dry = cn.powerlaw_psd_gaussian(beta, size=size)

        elif sound_name == 'female_voice' or sound_name == 'female':
            if 'female' not in self.dry_samples:
                out_dir_name = dataset_folder / "Anechoic"
                voice_fname = out_dir_name / (f"SI Harvard Word Lists Female_16khz" + ".wav")
                samplerate3, female_dry_samples = scipy.io.wavfile.read(voice_fname)
                # voice_dry_samples = u.signed16bitToFloat(voice_dry_samples).T
                # voice_dry_samples = resample(voice_dry_samples, samplerate3)
                self.dry_samples['female'] = female_dry_samples.T

            dry = self.get_mono_clip_random_offset(self.dry_samples['female'], length_samples)

        elif sound_name == 'vowel' or sound_name == 'long_vowel':
            out_dir_name = dataset_folder
            file_name = out_dir_name / "long_a.wav"
            current_fs, dry_samples = scipy.io.wavfile.read(file_name)
            dry_samples = u.signed16bitToFloat(dry_samples).T
            dry = u.resample(dry_samples, current_fs, g.fs)
            dry = self.get_mono_clip_random_offset(dry, length_samples)

        elif sound_name == 'wgn' or sound_name == 'white':
            size = [length_samples] if directional_point_source else [self.num_mics_max, length_samples]
            dry = g.rng.standard_normal(size)

        elif sound_name == 'correlated' or sound_name == 'all_sines' or sound_name == 'sinusoid':
            # num_frames = length_samples // self.duration_output_frames
            # dry = self.generate_cosine(num_frames, self.duration_output_frames)
            # assert False  # check that flatten and not flatten('f') is correct
            # dry = dry.flatten()
            dry = self.generate_cosine(1, length_samples)
        else:
            raise TypeError(f"invalid sound_name {sound_name}")

        if np.max(np.abs(dry)) < 1e-6:
            warnings.warn(f"Sound file '{sound_name}' is silent for length {length_samples / g.fs}s. Max value: {np.max(np.abs(dry))}")

        dry = dry - np.mean(dry)
        dry = SignalGenerator.normalize_length(length_samples, dry)
        dry = u.normalize_volume(dry)

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
        for sound_name in signal_names:
            sound_dry = sound_dry + self.get_normalized_sound_from_name(sound_name, self.duration_output_samples,
                                                                        directional_point_source=True)

        # smoothen signal to avoid clicks at the end and at the beginning
        sound_dry = self.fade_avoid_clicks(sound_dry)

        # remove mean
        sound_dry = sound_dry - np.mean(sound_dry)

        _, _, sound_dry_stft = self.stft(sound_dry)
        sound_wet, sound_wet_stft = self.apply_transfer_function(sound_dry, atf_desired,
                                                                 self.rir_voice_samples, sound_dry_stft)

        sound_dry = np.broadcast_to(sound_dry, sound_wet.shape)

        return np.squeeze(sound_wet), np.squeeze(sound_wet_stft), np.squeeze(sound_dry), np.squeeze(sound_dry_stft)

    @staticmethod
    def fade_avoid_clicks(sound_dry):
        if sound_dry is None:
            return sound_dry

        fade_length = int(min(len(sound_dry), (g.fs * 0.010)))
        fading_window = np.ones_like(sound_dry)
        fading_window[:fade_length] = np.linspace(0, 1, fade_length)
        fading_window[-fade_length:] = np.linspace(1, 0, fade_length)
        sound_dry = sound_dry * fading_window
        return sound_dry

    def apply_transfer_function(self, sound_dry, atf_desired, rir=None, sound_dry_stft=None):
        if 'random' in self.exp_settings['rtf_type'] or 'deterministic' in self.exp_settings['rtf_type']:
            sound_wet_stft = sound_dry_stft[np.newaxis] * atf_desired[..., np.newaxis]
            _, sound_wet = self.istft(sound_wet_stft)
            sound_wet = u.pad_or_trim_to_len(sound_wet, sound_dry.shape[-1])
        else:
            sound_dry = np.atleast_2d(sound_dry)
            sound_wet = SignalGenerator.convolve_with_rir(sound_dry, rir)
            sound_wet = u.pad_or_trim_to_len(sound_wet, sound_dry.shape[-1])
            _, _, sound_wet_stft = self.stft(sound_wet)

        return sound_wet, sound_wet_stft

    def get_empty_time_stft_matrices(self):
        if self.duration_output_samples is not None:
            empty_mat_time = np.zeros((self.num_mics_max, self.duration_output_samples))
            _, _, empty_mat_stft = self.stft(empty_mat_time)
            return empty_mat_time, empty_mat_stft
        else:
            return np.zeros((self.num_mics_max, 1)), \
                   np.zeros((self.num_mics_max, self.nstft // 2 + 1, self.duration_output_frames))

    def stft(self, x):
        return u.stft(x, nstft=self.nstft, noverlap=self.noverlap, window=self.window_name)

    def istft(self, x):
        return u.istft(x, nstft=self.nstft, noverlap=self.noverlap, window=self.window_name)

    def load_rir_from_path(self, rir_path):
        samplerate, rir_samples = scipy.io.wavfile.read(rir_path)
        rir_samples = u.signed16bitToFloat(rir_samples).T
        if rir_samples.ndim == 1:
            rir_samples = rir_samples[np.newaxis, :]
        rir_samples = rir_samples[:self.num_mics_max, ...]
        rir_samples = u.resample(rir_samples, samplerate)
        return rir_samples

    def load_and_convolve_noise_samples(self, noise_name, num_samples, dir_point_source=False, same_volume_all_mics=False,
                                        noise_volumes_per_mic=1):

        if len(noise_name) == 0 or (len(noise_name) == 1 and '' in noise_name):
            return np.zeros((self.num_mics_max, num_samples)), 0

        noise_samples = self.get_normalized_sound_from_name(noise_name, num_samples, dir_point_source)

        if dir_point_source and same_volume_all_mics:
            raise ValueError(f"dir_point_source and same_volume_all_mics cannot be both True.")
        elif dir_point_source:
            noise_samples = self.convolve_with_rir(noise_samples[np.newaxis, ...], self.rir_noise_samples)
        elif not same_volume_all_mics:
            # scale different microphones differently
            noise_samples = noise_samples * noise_volumes_per_mic

        noise_samples = u.pad_or_trim_to_len(noise_samples, num_samples)

        return noise_samples

    def compute_ground_truth_rtf(self, rir) -> np.array:

        rir_fft = self.compute_ground_truth_atf(rir)
        rir_fft = rir_fft / (rir_fft[g.idx_ref_mic, ...] + 1e-20)

        return rir_fft

    def compute_ground_truth_atf(self, rir):
        # w = scipy.signal.windows.get_window(g.window_function_name, self.nstft)
        # rir = w[np.newaxis, ...]*rir
        rir = u.pad_or_trim_to_len(rir, self.nstft)
        rir_fft = np.fft.rfft(rir, n=self.nstft, axis=-1)
        rir_fft = np.asarray(rir_fft, dtype=complex)
        return rir_fft

    # def computeGroundTruthRtfs2(self, desired_rir) -> np.array:
    #
    #     use_classical_rtf = True
    #     desired_rir = u.pad_or_trim_to_len(desired_rir, self.nstft)
    #     _, _, rir_stft = self.stft(desired_rir)
    #     rtfs_stft_ = np.zeros_like(rir_stft)
    #
    #     if use_classical_rtf:
    #         rtfs_stft_ = rir_stft / (rir_stft[g.idx_reference_mic, ...] + 1e-20)
    #     else:
    #         # eq 10 "A consolidated perspective"
    #         for kk in range(rir_stft.shape[1]):
    #             for tt in range(rir_stft.shape[2]):
    #                 ref_phase_shift = np.exp(1j * np.angle(rir_stft[g.idx_reference_mic, kk, tt]))
    #                 ref_amplitude = np.linalg.norm(rir_stft[:, kk, tt])
    #                 rtfs_stft_[:, kk, tt] = ref_phase_shift * rir_stft[:, kk, tt] / ref_amplitude
    #
    #     return rtfs_stft_[..., 0]  # return one frame only, is this good practice?

    def generate_atf(self, atf_type='real'):

        atf_shape = (self.num_mics_max, self.nstft // 2 + 1)
        atf, rtf = None, None

        if atf_type == 'deterministic' or atf_type == 'fake':
            num_freqs = self.nstft // 2 + 1
            atf = np.zeros(atf_shape)
            for kk in range(num_freqs):
                atf[:, kk] = np.linspace(1, kk / num_freqs, self.num_mics_max, endpoint=False)

        elif 'random' in atf_type:

            # this strategy gives somewhat weird results, do not use:
            # atf = u.circular_gaussian(atf_shape)
            # atf = atf / np.abs(atf)

            # atf = 1 + 0.1 * u.circular_gaussian(atf_shape)
            # atf = atf / np.sqrt(np.mean(np.abs(atf) ** 2))

            # extract real and imaginary part from uniform distribution in 'amp_range'
            amp_range = (-1, 1) if 'small' in atf_type else (-10, 10)
            atf = g.rng.uniform(amp_range[0], amp_range[1], size=atf_shape) + 1j * g.rng.uniform(amp_range[0], amp_range[1], size=atf_shape)

            # reorder each column so that the first microphone has the largest real part
            for kk in range(atf.shape[1]):
                atf[:, kk] = np.roll(atf[:, kk], -np.argmax(np.abs(atf[:, kk])))

        elif atf_type == 'real':
            atf = self.compute_ground_truth_atf(self.rir_voice_samples)

        elif atf_type == 'debug':
            atf = np.ones(atf_shape)

        atf = u.clip_cpx(atf, a_min=g.rtf_min, a_max=g.rtf_max)

        if np.any(np.isclose(atf, 0)):
            warnings.warn("ATF contains zeros. Might cause instability in CRB calculation. ")

        return atf

    @staticmethod
    def generate_rtf_from_atf(atf):
        """ Generate the relative transfer function from the absolute transfer function. """

        rtf = atf / (atf[g.idx_ref_mic, ...] + g.eps)

        if not np.alltrue(rtf < g.rtf_max) or not np.alltrue(rtf > g.rtf_min):
            warnings.warn("RTF out of bounds. Might cause instability in CRB calculation. ")

        if np.any(np.isclose(rtf, 0)):
            warnings.warn("RTF contains zeros. Might cause instability in CRB calculation. ")

        return rtf

    """
    @staticmethod
    def get_noise_amplitude_from_snr(x, v, desired_snr_db):
        # Return coefficient to be multiplied (sample-wise) by noise signal to obtain desired SNR
        p_x = np.mean(x ** 2)
        p_n = np.mean(v ** 2)

        if p_x == 0 or p_n == 0:
            return 0

        desired_snr_linear = u.db_to_linear(desired_snr_db)
        noise_amp_coeff = np.sqrt(p_x / (desired_snr_linear * p_n))

        return noise_amp_coeff

    def generate_noise_at_snr(self, noises_info_, target_samples, num_samples_target=None):

        if num_samples_target is None:
            num_samples_target = target_samples.shape[-1]
        num_samples_target = int(num_samples_target)

        if not isinstance(noises_info_, list):
            noises_info_ = [noises_info_]

        noises_samples = []
        for noise_info in noises_info_:
            noises_samples.append(self.generate_noise_samples_wrapper(noise_info, num_samples_target))

        noise_tot_samples = np.zeros((self.num_mics_max, num_samples_target))
        for noise_info, noise_samples in zip(noises_info_, noises_samples):
            if np.any(noise_samples > g.eps):
                noise_amp = self.get_noise_amplitude_from_snr(target_samples,
                                                              noise_samples[..., :num_samples_target],
                                                              noise_info['snr'])
                noise_samples *= noise_amp
                noise_tot_samples += noise_samples[..., :num_samples_target]

        _, _, noise_tot_stft = self.stft(noise_tot_samples)
        return noise_tot_stft, noise_tot_samples
"""

    @staticmethod
    def convolve_with_rir(dry, rir_samples):
        if rir_samples is not None:
            # assert dry.ndim == 2
            return scipy.signal.convolve(dry, rir_samples)
        else:
            return dry

    def generate_noise_samples_single_type(self, noise_info, num_samples_target=None):

        noise_name = noise_info['names'][0]
        directional_point_source = noise_info.get('isDirectional', False)
        same_volume_all_mics = noise_info.get('same_volume_all_mics', False)
        noise_samples = self.load_and_convolve_noise_samples(noise_name, num_samples_target,
                                                             directional_point_source, same_volume_all_mics,
                                                             noise_volumes_per_mic=noise_info['noise_volumes_per_mic'])

        return noise_samples

    @staticmethod
    def collect_sounds_in_dict(desired_wet_samples, noise_tot_samples, mix_samples, noises_data):
        return {'desired_wet': desired_wet_samples,
                'mic_noise': noises_data[0]['samples'] if len(noises_data) > 0 else None,
                'dir_noise': noises_data[1]['samples'] if len(noises_data) > 1 else None,
                'sum_noise': noise_tot_samples,
                'mix': mix_samples}

    def generate_signal_samples(self, atf_desired):

        if self.desired_sig_names is None:
            raise ValueError("No desired signal specified. Add \"desired\" to the yaml file. Example: desired: ['female']")

        stft_compact = lambda x: self.stft(x)[2]

        desired_wet_samples, desired_wet_stft, desired_dry_samples, desired_dry_stft = \
            self.generate_desired_signal(self.desired_sig_names, atf_desired)

        # First realization: noise used to estimate noise only covariance
        # Should be long enough to guarantee that the matrix is invertible
        # reference_clean = desired_dry_samples  # use dry signal to calculate SNR (wet signal should be used in general)
        reference_clean = desired_wet_samples  # use wet signal to calculate SNR

        if g.noise_estimation_time > 0:
            print(f"Using {g.noise_estimation_time} seconds of noise to estimate noise covariance matrix")
            min_frames_target = int(self.num_mics_max * (self.nstft // 2 + 1))
            min_samples_target = SignalGenerator.frames_to_samples(min_frames_target, self.nstft, self.noverlap_percentage)
            num_samples_target = int(max(min_samples_target, g.noise_estimation_time * g.fs))
        else:
            num_samples_target = reference_clean.shape[-1]

        noise_samples = self.generate_and_mix_noises_variable_snr(reference_clean, num_samples_target=num_samples_target)

        # Second realization: noise used to make noisy mix
        noise_samples_mix = self.generate_and_mix_noises_variable_snr(reference_clean, num_samples_target=num_samples_target)
        noise_stft_mix = stft_compact(noise_samples_mix)

        num_frames = min(noise_stft_mix.shape[-1], desired_wet_stft.shape[-1])
        mix_stft_ = noise_stft_mix[..., :num_frames] + desired_wet_stft[..., :num_frames]
        _, mix_samples = self.istft(mix_stft_)

        stimuli_samples = {'desired_wet': desired_wet_samples,
                           'mix': mix_samples,
                           'noise': noise_samples,
                           'noise_mix': noise_samples_mix,
                           'desired_dry': desired_dry_samples}

        stimuli_stft = {'desired_wet': desired_wet_stft,
                        'mix': mix_stft_,
                        'noise': stft_compact(noise_samples),
                        'noise_mix': stft_compact(noise_samples_mix) if g.debug_mode else np.zeros_like(mix_stft_),
                        'desired_dry': stft_compact(desired_dry_samples)}

        return stimuli_samples, stimuli_stft

    # Adding noise with a desired signal-to-noise ratio https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    def generate_and_mix_noises_variable_snr(self, reference_samples, num_samples_target=None):

        if num_samples_target is None:
            num_samples_target = self.duration_output_samples
        elif num_samples_target == reference_samples.shape[-1]:
            pass
        else:
            reference_samples = SignalGenerator.normalize_length(num_samples_target, reference_samples)

        noises_samples = [self.generate_noise_samples_single_type(noise_info, num_samples_target)
                          for noise_info in self.noises_info]

        # to ensure that each noise and input signals have the same power
        reference_power = np.var(reference_samples)

        noise_tot_samples = np.zeros((self.num_mics_max, num_samples_target))
        for noise_samples, noise_info in zip(noises_samples, self.noises_info):
            if noise_samples.any() > g.eps:
                # adjust the power of the added noise to obtain the desired signal-to-noise ratio (SNR).
                noise_power_original = np.var(noise_samples)
                snr_linear = u.db_to_linear(float(noise_info['snr']))
                rescaling_coefficient = np.sqrt(reference_power / (snr_linear * noise_power_original))
                noise_tot_samples = noise_tot_samples + rescaling_coefficient * noise_samples

        return noise_tot_samples

    def get_stft_shape(self):
        if self.sig_shape is not None:
            return self.sig_shape
        else:
            _, dummy_stft = self.get_empty_time_stft_matrices()
            self.sig_shape = dummy_stft.shape
            return self.sig_shape

    @staticmethod
    def perturb_noise_stft_with_wgn(amount, noise_stft_):
        print(f"Perturbing noise with WGN of std dev {amount:.6f}")
        noise_stft_ += amount * u.circular_gaussian(noise_stft_.shape)
        return noise_stft_

    def load_rir_from_type(self, rir_type, rir_corpus='pyroom'):

        if rir_corpus == 'pyroom':
            if rir_type == 'noise':
                # rir_path = "E:\\Documents\\TU Delft\\MSc\\Thesis\\code\\01" + self.diffuse_str + ".wav"
                rir_path = dataset_folder / "Pyroom" / "1.wav"
                # rir_path = dataset_folder / "Pyroom" / "1_big.wav"
            elif rir_type == 'target':
                # rir_path = "E:\\Documents\\TU Delft\\MSc\\Thesis\\code\\00" + self.diffuse_str + ".wav"
                rir_path = dataset_folder / "Pyroom" / "0.wav"
                # rir_path = dataset_folder / "Pyroom" / "0_big.wav"
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")
            rir_samples = self.load_rir_from_path(rir_path)

        elif rir_corpus == 'ace':
            # ACE_Corpus_RIRN_Chromebook/Chromebook
            ace_folder = dataset_folder / "ACE_Corpus_RIRN_Chromebook" / "Chromebook" / "Meeting_Room_2"
            # ace_folder = dataset_folder / "ACE_Corpus_RIRN_Chromebook" / "Chromebook" / "Office_1"
            # ace_folder = dataset_folder / "ACE_Corpus_RIRN_Chromebook" / "Chromebook" / "Lecture_Room_1"

            # recursively search in ace_folder for all wav files with "RIR" in name, inside folder with Meeting_Room in name
            rir_paths = list(ace_folder.glob('**/*RIR*.wav'))

            # select randomly one of the wav files
            rir_path = g.rng.choice(rir_paths)
            rir_samples = self.load_rir_from_path(rir_path)

        elif rir_corpus == 'aachen':
            """
            target: air_stairway_1_1_1_30_mls.wav and air_stairway_0_1_1_30_mls.wav
            noise: air_stairway_1_1_1_45_mls.wav and air_stairway_0_1_1_45_mls.wav
            /Users/giovannibologni/Documents/TU Delft/Code/datasets/AIR_1_4/AIR_wav_files
            """
            aachen_folder = dataset_folder / "AIR_1_4" / "AIR_wav_files"

            # random doesn't work because in current implementation a different RIR is picked up for e.g. each SNR, not for each realization
            # angle_list = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
            # noise_angle, target_angle = g.rng.choice(angle_list), g.rng.choice(angle_list)
            # distance_list = [1, 2, 3]
            # noise_distance, target_distance = g.rng.choice(distance_list), g.rng.choice(distance_list)
            # noise_angle, target_angle = 15, 60
            # noise_distance, target_distance = 2, 3
            noise_angle, target_angle = 15, 30
            noise_distance, target_distance = 2, 2
            if rir_type == 'noise':
                rir_path_left = aachen_folder / f"air_stairway_0_1_{noise_distance}_{noise_angle}_mls.wav"
                rir_path_right = aachen_folder / f"air_stairway_1_1_{noise_distance}_{noise_angle}_mls.wav"
            elif rir_type == 'target':
                rir_path_left = aachen_folder / f"air_stairway_0_1_{target_distance}_{target_angle}_mls.wav"
                rir_path_right = aachen_folder / f"air_stairway_1_1_{target_distance}_{target_angle}_mls.wav"
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")

            rir_samples_right = self.load_rir_from_path(rir_path_right)
            rir_samples_left = self.load_rir_from_path(rir_path_left)
            rir_samples = np.vstack((rir_samples_right, rir_samples_left))
        else:
            raise ValueError(f"Unknown corpus {rir_corpus} cannot be loaded.")

        if rir_samples.shape[0] != self.num_mics_max:
            raise ValueError(f"Request num sensors and available measurements in RIR don't match:"
                             f"{rir_samples.shape=}, {self.num_mics_max=}")
        return rir_samples

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

    def get_frequency_bins_from_frequencies(self, freqs_hz):
        if np.alltrue(freqs_hz) and np.alltrue(np.array(freqs_hz) > 0):
            freqs_bins = np.asarray(np.floor(np.array(freqs_hz) / (g.fs / 2.) * self.nstft), dtype='int').flatten()
            freqs_bins = np.unique(freqs_bins)
            return freqs_bins if not np.isscalar(freqs_hz) else freqs_bins[0]
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

        # Generate "white" noise data
        if cpx_data:
            v_white = u.circular_gaussian(sig_shape)
        else:
            v_white = g.rng.standard_normal(size=sig_shape)

        if np.allclose(r_true, np.identity(r_true.shape[0])):
            # If the covariance matrix is the identity matrix, the white noise data is already the correlated signal
            return v_white
        else:
            try:
                # Introduce correlations between variables by multiplying the white data with the square root of the true noise covariance
                rv_true_sqrt = np.linalg.cholesky(r_true)
                v = rv_true_sqrt @ v_white

            except np.linalg.LinAlgError:
                # Alternative procedure to avoid Cholesky decomposition, which is not defined for singular matrices.
                # (https://stats.stackexchange.com/questions/159313/generating-samples-from-singular-gaussian-distribution)
                warnings.warn("Cholesky decomposition failed. Generating correlated signal using SVD instead.")
                U, d, _ = np.linalg.svd(r_true, hermitian=True)
                assert np.alltrue(d >= 0), "Singular values of covariance matrix are negative. Thus this not a valid covariance matrix."
                v = U @ np.diag(np.sqrt(d)) @ v_white

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

        dry = SignalGenerator.generate_correlated_signal_outer(rs_K, uncorrelated_over_space=False, stft_shape=stft_shape)
        transfer_function = np.diag(atf_target.flatten('F'))
        wet = transfer_function @ dry

        noise = SignalGenerator.generate_correlated_signal_outer(rv_true_, uncorrelated_over_space=True, stft_shape=stft_shape)
        noise_for_mix = SignalGenerator.generate_correlated_signal_outer(rv_true_, uncorrelated_over_space=True, stft_shape=stft_shape)

        dry = np.reshape(dry, stft_shape, order='F')
        noise = np.reshape(noise, stft_shape, 'F')
        wet = np.reshape(wet, stft_shape, 'F')
        noise_for_mix = np.reshape(noise_for_mix, stft_shape, 'F')

        assert noise_for_mix.shape[-1] == noise.shape[-1]  # otherwise need to check SNR rescaling

        mix = wet + noise_for_mix

        stimuli_stft = {'desired_wet': wet, 'mix': mix, 'noise': noise, 'desired_dry': dry}

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
            _, stimuli_samples_new[key] = self.istft(stimuli_stft[key])

        return stimuli_samples_new

    def get_mono_clip_random_offset(self, sound_samples, desired_length_samples):

        # select left channel
        if sound_samples.ndim > 1 and sound_samples.shape[0] > 1:
            sound_samples = sound_samples[0]

        if desired_length_samples > sound_samples.shape[-1]:
            raise ValueError(f"Sound file of length {desired_length_samples} ({desired_length_samples // g.fs} s) "
                             f"is too short for desired duration of {sound_samples.shape[-1]} samples ({sound_samples.shape[-1] // g.fs} s).")

        dry = None
        ii = 0
        max_iterations = 100
        while self.is_silent_sound(dry) and ii < max_iterations:
            max_offset = sound_samples.size - desired_length_samples
            starting_offset = g.rng.integers(0, max_offset)
            dry = sound_samples[starting_offset:starting_offset + desired_length_samples]
            ii = ii + 1

        if ii == max_iterations:
            raise ValueError("Could not find a non-silent sample in the sound file.")

        return dry


class SignalHolder:
    def __init__(self, stimuli_stft, sg):
        self.desired_wet_stft = stimuli_stft['desired_wet'],
        self.mix_stft = stimuli_stft['mix']
        self.stimuli_stft = stimuli_stft
        self.signal_generator = sg