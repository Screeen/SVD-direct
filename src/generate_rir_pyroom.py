import Archive.RirGenerator
import numpy as np
import matplotlib.pyplot as plt
import src.utils as u
from pathlib import Path

fs = 16e3

room_number = 2

rg = Archive.RirGenerator.RirGenerator(fs)
rg.enable_ray_tracing = True

if room_number == 0:
    rg.rt60_tgt = 0.3
    mic_spacing = 0.05
    room_size = [3.5, 3, 3]
    mic_center_x = 1
    mic_center_y = 1.6
    mic_center_z = 1.6
elif room_number == 1:
    rg.rt60_tgt = 0.1
    mic_spacing = 0.05
    room_size = [2, 2.1, 2]
    mic_center_x = 1
    mic_center_y = 1
    mic_center_z = 1.6
else:
    rg.rt60_tgt = 0.2
    mic_spacing = 0.05
    room_size = [3, 4.1, 2.5]
    mic_center_x = 1
    mic_center_y = 1
    mic_center_z = 1.6
    rg.enable_ray_tracing = True

rg.setRoomSize(room_size)
rg.micsPos = np.array(
    [[mic_center_x, mic_center_y - 3 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y - 2 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y - 1 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y - 0 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y + 1 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y + 2 * mic_spacing, mic_center_z],
     [mic_center_x, mic_center_y + 3 * mic_spacing, mic_center_z]]).T

rg.createRoom()
rg.room.add_source(position=[mic_center_x + 0.5, mic_center_y + 1, mic_center_z])
rg.room.add_source(position=[mic_center_x - 0.5, mic_center_y - 1, mic_center_z])
rg.room.add_microphone_array(rg.micsPos)
rg.room.compute_rir()

# rg.simulate()
rirs_samples = rg.getRirsAsArray()

src_idx_plot = 0
plt.figure()
plt.grid()
plt.plot(rirs_samples[src_idx_plot, :, :1000].T)
plt.title(f'Room size={room_size[0]}x{room_size[1]}x{room_size[2]}m, '
           f'RT60={rg.rt60_tgt}, sensor spacing={mic_spacing}m')
plt.show()

rirs_samples = u.normalize_volume(rirs_samples)

suffix = "_big" if room_number == 0 else ""
out_dir_name = Path(__file__).parent / "Datasets" / "Pyroom"
for src_idx, rir in enumerate(rirs_samples):
    file_name = out_dir_name / (f"{str(src_idx)}" + suffix + ".wav")
    u.write_audio_file(rir.T, file_name, fs, "wav")

f, ax = rg.room.figure(img_order=0)
ax.set_title(f'Room size={room_size[0]}x{room_size[1]}x{room_size[2]}m, '
           f'RT60={rg.rt60_tgt}, sensor spacing={mic_spacing}m')
f.show()
