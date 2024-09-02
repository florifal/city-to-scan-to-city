import collections.abc
from pathlib import Path
import experiment.global_vars as glb


class Config:
    """Draft of the as of now unused class Config. Possibly to replace dictionary configs later."""

    def __init__(
            self,
            config_dict: dict | None = None,
            json_filepath: str | Path | None = None,
            settings_snippet: dict | None = None,
            make_default: bool = False
    ):
        # Takes a prepared config dictionary, a path to a json file with a config, a subset of a config as settings
        # snippet, or generates a default config with make_default.
        self.config = {}

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        # Corresponds to update_config_item()
        pass

    def make_default(self):
        # Corresponds to scenario_default_config()
        pass

    def read_json(self, json_filepath: str | Path):
        pass

    def update(self, config_dict: dict):
        # Corresponds to deep_update()
        pass


def deep_update(d: dict, u: dict | collections.abc.Mapping):
    """Recursively update values in a dictionary. Slightly adapted from https://stackoverflow.com/a/3233356"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def scenario_default_config():
    scene_config = {
        "scene_xml_filepath": "",
        "scene_xml_id": "",
        "scene_name": "",
        "scene_parts": []
    }

    survey_generator_config = {
        "survey_template_xml_filepath": str(Path(__file__).parent / "experiment" / "survey_template.xml"),
        "survey_name": "",
        "scene_xml_filepath_with_id": "",
        "platform_id": "sr22",
        "scanner_id": "riegl_vq-1560i",
        "scanner_settings_id": "scanner_settings",
        "scanner_settings_active": True,
        "pulse_freq_hz": 500_000,
        "scan_angle_deg": 15,
        "scan_freq_hz": 300,
        "detector_settings_accuracy": 0.0,  # 0.02
    }

    flight_path_config = {
        "flight_path_xml_filepath": "",
        "bbox": [83043, 446327, 83255, 446496],  # [West, South, East, North]
        "spacing": 40,
        "altitude": 100,
        "velocity": 60,
        "flight_pattern": "parallel",
        "trajectory_time_interval": .05,
        "always_active": False,
        "scanner_settings_id": "scanner_settings"
    }

    survey_executor_config = {
        "las_output": True,
        "zip_output": True,
        "num_threads": 0,
    }

    cloud_merge_config = {
        "clouds_dirpath": "",  # UNUSED
        "output_filepath": ""  # UNUSED
    }

    survey_config = {
        "survey_xml_filepath": "",
        "survey_output_dirpath": "",
        "survey_generator_config": survey_generator_config,
        "flight_path_config": flight_path_config,
        "survey_executor_config": survey_executor_config,
        "cloud_merge_config": cloud_merge_config,
    }

    cloud_processing_config = {
        "cloud_processing_output_dirpath": "",
        "std_horizontal_error": 0.0,
        "std_vertical_error": 0.0
    }

    # Need to add prefix "range_" because all config keys must be unique across all (sub-) dictionaries (at the moment)
    optim_parameter_space = glb.geoflow_optim_parameter_space

    recon_optim_config = {
        "optimization_footprints_filepath": "",
        "optimization_footprints_sql": "",
        "recon_optim_output_dirpath": "",
        "parameter_space": optim_parameter_space,
        "recon_optim_evaluators": ["iou_3d", "hausdorff"],  # list of Evaluators to run
        "recon_optim_metrics": {"iou_3d": ["iou_22_mean"], "hausdorff": ["hausdorff_22_rms", "rms_min_dist_22_mean"]},
        "recon_optim_target_lod": "2.2",  # with point
        # "recon_optim_target_evaluator": "iou_3d",  # name of an Evaluator subclass
        # "recon_optim_target_metric": "iou_22_mean",  # name of a summary statistic key
        # "recon_optim_target_metric_optimum": "max",  # whether to maximize (max) or minimize (min) for optimality
        # "recon_optim_target_evaluator": "hausdorff",  # name of an Evaluator subclass
        # "recon_optim_target_metric": "hausdorff_22_rms",  # name of a summary statistic key
        # "recon_optim_target_metric_optimum": "min",  # whether to maximize (max) or minimize (min) for optimality
        "recon_optim_target_evaluator": "hausdorff",  # name of an Evaluator subclass
        "recon_optim_target_metric": "rms_min_dist_22_mean",  # name of a summary statistic key
        "recon_optim_target_metric_optimum": "min"  # whether to maximize (max) or minimize (min) for optimality
    }

    geoflow_parameters = glb.geoflow_parameters_default

    reconstruction_config = {
        "building_footprints_filepath": "",
        "building_footprints_sql": "",
        "building_identifier": "",
        "reconstruction_output_dirpath": "",
        "geoflow_parameters": geoflow_parameters,
        "geoflow_log_filepath": "",  # UNUSED
        "config_toml_filepath": "",  # UNUSED
        "point_cloud_filepath": "",  # LIKELY UNUSED, but currently an alternative in Reconstruction.__init__()
    }

    evaluation_config = {
        "evaluation_output_dirpath": "",
        "input_cityjson_filepath": "",  # Input building models CityJSON for experiment evaluation
        "input_obj_lod12_filepath": "",
        "input_obj_lod13_filepath": "",
        "input_obj_lod22_filepath": ""
    }

    # Final scenario settings

    scenario_config = {
        "scenario_name": "",
        "crs": "epsg:7415",
        "settings_dirpath": "",
        "scene_config": scene_config,
        "survey_config": survey_config,
        "cloud_processing_config": cloud_processing_config,
        "recon_optim_config": recon_optim_config,
        "reconstruction_config": reconstruction_config,
        "evaluation_config": evaluation_config
    }

    return scenario_config


def update_config_item(
        config: dict,
        key: str,
        value: str | int | float | dict | list,
        not_found_error: bool = True,
        update_all: bool = True
) -> bool | None:
    """Recursively update first / all occurrences of `key` in `config` and all nested dictionaries with `value`.

    :param config: Config dictionary. May contain nested dictionaries.
    :param key: Key to be updated
    :param value: Value to be set
    :param not_found_error: True: If key not found, raise error, else return None. False: Return whether key found.
    :param update_all: Whether to update all occurrences of `key` or only the first occurence
    :return: If not_found_error==False: bool whether key was found; else: None if found, else raise KeyError.
    """
    found = False

    if key in config.keys():
        config[key] = value
        found = True

    # Commented lines: Code that always updates all occurrences recursively
    # Check nested dictionaries
    # for k, v in config.items():
    #     if isinstance(v, dict):
    #         found = update_config_item(v, key, value, not_found_error=False) or found

    if update_all or not found:
        # Check nested dictionaries
        for k, v in config.items():
            if isinstance(v, dict):
                found = update_config_item(v, key, value, not_found_error=False) or found
            if found and not update_all:
                break

    if not_found_error:
        if not found:
            raise KeyError(f"Key '{key}' not found.")
        else:
            return None
    else:
        return found


def get_config_item(config: dict, key: str, not_found_error: bool = True):
    """Recursively finds and returns the value of the 1st occurrence of `key` in `config` or any of its sub-dictionaries

    Does not check for multiple occurrences of `key` in `config` or any of its sub-dictionaries. Raises a KeyError if
    the `key` is not found."""
    if key in config.keys():
        return config[key]
    else:
        for k, v in config.items():
            if isinstance(v, dict):
                value = get_config_item(v, key, not_found_error=False)
                # `value` is None if the key was not found, otherwise it is the key's value.
                if value is not None:
                    return value

    if not_found_error:
        raise KeyError(f"Key '{key}' not found.")
    else:
        return None
