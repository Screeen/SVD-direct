import yaml
import logging
from pathlib import Path
# import platform

# Logger
logger = logging.getLogger(__name__)
config_folder = Path(__file__).parent.parent / "configs"


def load_configuration_from_path(configuration_path):

    # Read settings from configuration file
    # if platform.system() == 'Linux':
    #     config_path = u.convert_to_unix_path(config_path)
    with open(configuration_path, 'r') as f:
        conf_dict = yaml.safe_load(f)
    logger.info(f"Loaded configuration file {configuration_path}")
    return conf_dict


def load_configuration(cfg_name=None):

    print(f"Loading configuration file: {cfg_name}")
    cfg_path = config_folder / cfg_name
    cfg_custom = load_configuration_from_path(cfg_path)
    return cfg_custom

    """
    cfg_default_path = config_folder / "_default.yaml"
    cfg_default = load_configuration_from_path(cfg_default_path)

    if custom_cfg_name is not None:
        cfg_path = config_folder / custom_cfg_name
        cfg_custom = load_configuration_from_path(cfg_path)

        # Merge default and custom configurations, giving priority to the CUSTOM.
        # Do not update parameters whose value is "None"
        cfg_default.update((k, v) for k, v in cfg_custom.items() if v is not None)

    return cfg_default
    """


def write_configuration(config_dict, config_name):

    dest_path = config_folder / config_name
    with open(dest_path, 'w') as outfile:
        yaml.dump(config_dict, outfile)


# def config_to_csv():
#     global conf
#
#     fieldnames = conf.keys()
#
#     # Store results to two different folders
#     for mother_folder in [conf['result_path'], conf['exp_path']]:
#
#         # results_filename = cfg.conf.get('results_filename', 'performance_evaluation')
#         results_filename = 'config_tab'
#         dest_path = os.path.join(mother_folder, results_filename + '.csv')
#         new_eval_file = True if not os.path.exists(dest_path) else False
#
#         with open(dest_path, mode='a') as csv_file:
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#             if new_eval_file:
#                 writer.writeheader()
#             writer.writerow(conf)
#             logger.info('Results added to file {}'.format(dest_path))
