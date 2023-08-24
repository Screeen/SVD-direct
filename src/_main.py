import config as cfg
import src.global_constants as g
import src.utils as u
from src.exp_manager import ExperimentManager, print_errors_table_from_settings
import time

from plot_manager import plot_errors

start_time = time.time()

# cfg_name = "_default.yaml"
# cfg_name = "config_stationary_wgn.yaml"
# cfg_name = "config_nonstationary_signals_wgn.yaml"
cfg_name = "config_generate_stft.yaml"
# cfg_name = "config_stationary_sinusoid.yaml"
# cfg_name = "config_nonstationary_signals_sinusoid.yaml"
# cfg_name = "config_speech.yaml"
# cfg_name = "visualize_atf_vectors.yaml"
experiment_settings = cfg.load_configuration(cfg_name)

print("\n----------")

err_mean_std_array, rtf_evaluators, cov_manager, sig_holder, atf = ExperimentManager.run_experiment(experiment_settings)
print_errors_table_from_settings(experiment_settings, err_mean_std_array)

u.play(g.rng.standard_normal((1, int(g.fs / 20))), volume=0.01)
end_time = time.time()
print(f"----------\nConfiguration file: {cfg_name}, time elapsed: {end_time - start_time:.2f} seconds")
u.set_plot_options()
plot_errors(settings=experiment_settings, err_mean_std_array=err_mean_std_array)

# bm = beamforming_manager.BeamformingManager(rtf_evaluator, cov_manager, sig_generator)
# f = bm.run()
