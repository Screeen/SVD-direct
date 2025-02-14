import numpy as np
import src.global_constants as g
import src.utils as u
import warnings
import scipy


class RirManager:
    def __init__(self, **rir_sett):

        self.nstft = rir_sett['nstft']
        self.num_mics_max = rir_sett['num_mics_max']

        self.noise_angle, self.target_angle = rir_sett['noise_angle'], rir_sett['target_angle']
        self.noise_distance, self.target_distance = rir_sett['noise_distance'], rir_sett['target_distance']
        self.room_size = rir_sett['room_size']

        self.rir_voice_samples, self.rir_noise_samples = None, None
        self.inter_mic_distance = -1
        if rir_sett['rtf_type'] == 'real':
            self.rir_voice_samples, self.rir_noise_samples = (
                self.load_room_impulse_responses(rir_sett['rir_corpus'],
                                                 rir_sett['num_nonzero_samples_rir_target'],
                                                 rir_sett['num_nonzero_samples_rir_noise']))
            self.inter_mic_distance = RirManager.get_inter_mic_distance(rir_sett['rir_corpus'])

    def compute_ground_truth_rtf(self, rir) -> np.array:

        rir_fft = self.compute_ground_truth_atf(rir)
        rir_fft = rir_fft / (rir_fft[g.idx_ref_mic, ...] + g.eps)

        return rir_fft

    def compute_ground_truth_atf(self, rir):
        # w = scipy.signal.windows.get_window(g.window_function_name, self.nstft)
        # rir = w[np.newaxis, ...] * rir
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

    def compute_or_generate_acoustic_transfer_function(self, atf_type='real'):

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

        if np.any(rtf < g.rtf_max) or np.any(rtf > g.rtf_min):
            warnings.warn(f"RTF out of bounds ({g.rtf_min, g.rtf_max}). Might cause instability in CRB calculation. ")

        if np.any(np.isclose(rtf, 0)):
            warnings.warn("RTF contains zeros. Might cause instability in CRB calculation. ")

        return rtf

    def load_rir_from_type(self, rir_type, rir_corpus='pyroom'):

        if rir_corpus == 'pyroom':
            if rir_type == 'noise':
                # rir_path = "E:\\Documents\\TU Delft\\MSc\\Thesis\\code\\01" + self.diffuse_str + ".wav"
                rir_path = g.dataset_folder / "Pyroom" / "1.wav"
                # rir_path = g.dataset_folder / "Pyroom" / "1_big.wav"
            elif rir_type == 'target':
                # rir_path = "E:\\Documents\\TU Delft\\MSc\\Thesis\\code\\00" + self.diffuse_str + ".wav"
                rir_path = g.dataset_folder / "Pyroom" / "0.wav"
                # rir_path = g.dataset_folder / "Pyroom" / "0_big.wav"
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")
            rir_samples = self.load_rir_from_path(rir_path)

        elif 'ace' in rir_corpus:
            if 'chromebook' in rir_corpus:
                mic_folder = "Chromebook"
            elif 'lin8ch' in rir_corpus:
                mic_folder = "Lin8Ch"
            else:
                raise ValueError(f"Unknown rir_corpus {rir_corpus} cannot be loaded.")

            room_idx = '1'
            if self.room_size == 'small':
                room_folder = "Office"  # RT60 = 0.34/0.39
            elif self.room_size == 'medium':
                room_folder = "Meeting_Room"  # RT60 = 0.64/0.69
            elif self.room_size == 'large':
                room_folder = "Lecture_Room"  # RT60 = 0.64/0.69
            else:
                raise ValueError(f"Unknown {self.room_size = } cannot be loaded.")
            room_folder = '_'.join([room_folder, room_idx])

            ace_folder = g.dataset_folder / "ACE-corpus" / mic_folder / room_folder
            rir_paths = sorted(list(ace_folder.glob('**/*RIR*.wav')))  # recursively search for all wav files with "RIR" in name

            # select randomly one of the wav files
            if len(rir_paths) < 2:
                raise ValueError(f"Only {len(rir_paths)} RIRs found in {ace_folder}, but 2 are needed (target and noise).")
            if rir_type == 'target':
                rir_path = rir_paths[0]
            elif rir_type == 'noise':
                rir_path = rir_paths[1]
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")

            rir_samples = self.load_rir_from_path(rir_path)

        elif rir_corpus == 'aachen':
            """
            target: air_stairway_1_1_1_30_mls.wav and air_stairway_0_1_1_30_mls.wav
            noise: air_stairway_1_1_1_45_mls.wav and air_stairway_0_1_1_45_mls.wav
            ~/Documents/TU Delft/Code/datasets/AIR_1_4/AIR_wav_files
            """
            aachen_folder = g.dataset_folder / "AIR_1_4" / "AIR_wav_files"
            if rir_type == 'noise':
                distance = self.noise_distance
                angle = self.noise_angle
            elif rir_type == 'target':
                distance = self.target_distance
                angle = self.target_angle
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")

            rir_path_left = aachen_folder / f"air_stairway_0_1_{distance}_{angle}_mls.wav"
            rir_path_right = aachen_folder / f"air_stairway_1_1_{distance}_{angle}_mls.wav"

            rir_samples_right = self.load_rir_from_path(rir_path_right)
            rir_samples_left = self.load_rir_from_path(rir_path_left)
            rir_samples = np.vstack((rir_samples_right, rir_samples_left))

        elif rir_corpus == 'hadad':

            if self.room_size == 'small':
                room_rt60 = '0.160'
            elif self.room_size == 'medium':
                room_rt60 = '0.360'
            elif self.room_size == 'large':
                room_rt60 = '0.610'
            else:
                raise ValueError(f"Unknown {self.room_size = } cannot be loaded.")

            boilerplate = "Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
            hadad_sub_folder = f"{boilerplate}_Reverberation_{room_rt60}s__3-3-3-8-3-3-3"
            hadad_folder = (g.dataset_folder / "Hadad_wav" / hadad_sub_folder)
            file_name_start = f"{boilerplate}(Reverberation_{room_rt60}s)_3-3-3-8-3-3-3_"

            if rir_type == 'noise':
                distance = self.noise_distance
                angle = self.noise_angle
            elif rir_type == 'target':
                distance = self.target_distance
                angle = self.target_angle
            else:
                raise ValueError(f"Unknown rir_type {rir_type} cannot be loaded.")
            extension = '.wav'
            file_name = file_name_start + f"{distance}m_{angle:03d}"
            file_path = hadad_folder / (file_name + extension)
            sampling_freq, rir_samples = scipy.io.wavfile.read(file_path)
            rir_samples = rir_samples[:, :self.num_mics_max].T
            # rir_samples = np.flipud(rir_samples)
        else:
            raise ValueError(f"Unknown corpus {rir_corpus} cannot be loaded.")

        if rir_samples.shape[0] != self.num_mics_max:
            raise ValueError(f"Request num sensors and available measurements in RIR don't match:"
                             f"{rir_samples.shape=}, {self.num_mics_max=}")

        return rir_samples

    def load_rir_from_path(self, rir_path):
        samplerate, rir_samples = scipy.io.wavfile.read(rir_path)
        rir_samples = u.signed16bitToFloat(rir_samples).T
        rir_samples = self.rir_resample_and_select_mics(rir_samples, samplerate, self.num_mics_max)
        return rir_samples

    @staticmethod
    def rir_resample_and_select_mics(rir_samples, rir_sample_rate_original, num_mics_max):
        # RIR shape should be (num_mics, num_samples)
        if rir_samples.ndim == 1:
            rir_samples = rir_samples[np.newaxis, :]
        rir_samples = rir_samples[:num_mics_max, ...]
        rir_samples = u.resample(rir_samples, current_fs=rir_sample_rate_original, desired_fs=g.fs)

        return rir_samples

    def load_room_impulse_responses(self, rir_corpus, num_samples_rir_target_arg=-1, num_samples_rir_noise_arg=-1):

        rir_target = self.load_rir_from_type('target', rir_corpus)
        rir_noise = self.load_rir_from_type('noise', rir_corpus)

        # default behaviour (cut late reverberation)
        num_samples_rir_target = self.nstft
        num_samples_rir_noise = self.nstft

        # user manually set this value
        if num_samples_rir_target_arg != -1:
            num_samples_rir_target = num_samples_rir_target_arg
        if num_samples_rir_noise_arg != -1:
            num_samples_rir_noise = num_samples_rir_noise_arg

        rir_target = self.cut_rir_to_length_samples(rir_target, num_samples_rir_target)
        rir_noise = self.cut_rir_to_length_samples(rir_noise, num_samples_rir_noise)

        # normalize volume of impulse responses, but *keep relative differences*
        # max_volume = np.maximum(np.max(np.abs(rir_target)), np.max(np.abs(rir_noise)))
        # rir_target = 0.95 * rir_target / max_volume
        # rir_noise = 0.95 * rir_noise / max_volume

        # Normalize to unit energy
        normalization = np.sum(rir_target ** 2)
        rir_target = rir_target / normalization
        rir_noise = rir_noise / normalization

        self.validate_rir(rir_target)
        self.validate_rir(rir_noise)

        return rir_target, rir_noise

    @staticmethod
    def validate_rir(rir_samples):
        if np.any(np.isnan(rir_samples)):
            raise ValueError("RIR contains NaNs")
        if np.any(np.isinf(rir_samples)):
            raise ValueError("RIR contains Infs")
        # if np.any(rir_samples > 1) or np.any(rir_samples < -1):
        #     raise ValueError("RIR contains values outside [-1, 1]")

    @staticmethod
    def cut_rir_to_length_samples(rir_samples, num_samples_rir=-1):
        """ Cut the RIR to the desired length. If num_samples_rir is not specified, don't cut."""

        if num_samples_rir == -1 or num_samples_rir >= rir_samples.shape[-1]:
            return rir_samples

        rir_samples = rir_samples[:, :num_samples_rir]

        # Apply a window after cutting to avoid clicks
        win_len = min(num_samples_rir // 2, 40)
        w = scipy.signal.windows.get_window('hann', win_len)
        rir_samples[:, -win_len // 2:] *= w[np.newaxis, -win_len // 2:]

        return rir_samples

    @classmethod
    def get_inter_mic_distance(cls, rir_corpus):
        if rir_corpus == 'hadad':
            return 0.08
        elif rir_corpus == 'ace-lin8ch':
            """The ACE challenge — Corpus description and performance evaluation"""
            return 0.06
        elif rir_corpus == 'ace-chromebook':
            """The ACE challenge — Corpus description and performance evaluation"""
            return 0.062
        else:
            raise NotImplementedError(f"Inter-mic distance not implemented for {rir_corpus}")
