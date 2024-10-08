from decimal import Decimal
from pathlib import Path
import numpy as np

import experiment.global_vars as glb
from experiment.config import scenario_default_config
from experiment.utils import swath_width, scan_freq_from_point_spacing, pulse_freq_from_point_spacing, \
    get_processing_sequence

experiment_name = "utrecht_10-492-594_v2"
experiment_dirpath = Path(r"C:\Users\Florian\Data\city-to-scan-to-city")
input_dirpath = experiment_dirpath / experiment_name / "01_input"

default_config = scenario_default_config()

default_config["crs"] = "epsg:7415"

# ----------------------------------------------------------------------------------------------------------------------
# Survey and scene
# ----------------------------------------------------------------------------------------------------------------------

altitude = 500
scan_angle_deg = 20
velocity = 60

spacing = 0.5 * swath_width(altitude, scan_angle_deg)
scanner_id = "riegl_vq-1560i-single-channel"

default_config["survey_config"]["survey_generator_config"].update(
    {
        "platform_id": "sr22",
        "scanner_id": scanner_id,
        "scanner_settings_id": "scanner_settings",
        "scanner_settings_active": True,
        "scan_angle_deg": scan_angle_deg,
        "pulse_freq_hz": 0,
        "scan_freq_hz": 0,  # 300,
        "detector_settings_accuracy": 0.0
    }
)

extent = [136593.338, 455390.404, 137093.338, 455890.404]  # extent of downloaded tiles

bbox = np.array(extent) + np.array([-50, -50, 50, 50])
bbox = bbox.astype(np.int32)
bbox = [int(c) for c in bbox]  # convert np array to list and int32 to int

default_config["survey_config"]["flight_path_config"].update(
    {
        "bbox": list(bbox),  # [West, South, East, North]
        "spacing": spacing,
        "altitude": altitude,
        "velocity": velocity,
        "flight_pattern": "parallel",
        "trajectory_time_interval": .05,
        "always_active": False,
        "scanner_settings_id": "scanner_settings"
    }
)

scene_parts = [
    {
        "type": "obj",
        "filepath": str(input_dirpath / "OBJ" / "10-492-594_LOD22_mat_python.obj"),
        "up_axis": "z"
    },
    {
        "type": "tif",
        "filepath": str(input_dirpath / "DTM" / "Utrecht_0.5m_31H_Z2+N2_fill_clip_10-492-594_buff50m.TIF"),
        "material_filepath": str(input_dirpath / "DTM" / "Utrecht_0.5m_31H_Z2+N2_fill_clip_10-492-594_buff50m.mtl"),
        "material_name": "ground"
    }
]

# ----------------------------------------------------------------------------------------------------------------------
# Reconstruction optimization and reconstruction
# ----------------------------------------------------------------------------------------------------------------------

optim_parameter_space = glb.geoflow_optim_parameter_space_narrow_2
geoflow_parameters_default = glb.geoflow_parameters_default

default_config["recon_optim_config"].update(
    {
        "optimization_footprints_filepath": str(input_dirpath / "Footprints" / "footprints_10-492-594_subset_optim_01.gpkg"),
        "optimization_cityjson_filepath": str(input_dirpath / "CityJSON" / "10-492-594_subset_optim_01.city.json"),
        "parameter_space": optim_parameter_space,
        "parameter_sets_to_probe": [],

        # List Evaluators to run and which of their metrics to report
        "recon_optim_evaluators": ["hausdorff"],
        "recon_optim_metrics": {"hausdorff": ["hausdorff_22_rms", "rms_min_dist_22_mean"]},

        # Specify optimization targets: LOD, Evaluator, metric, and whether to minimize or maximize
        "recon_optim_target_lod": "2.2",  # provide with point
        "recon_optim_target_evaluator": "hausdorff",  # name of an Evaluator subclass
        "recon_optim_target_metric": "rms_min_dist_22_mean",  # name of a summary statistic key
        "recon_optim_target_metric_optimum": "min",  # whether to maximize (max) or minimize (min) for optimality

        # Experimental settings for a second target metric whose relative magnitude penalizes the first one
        # "recon_optim_evaluators": ["hausdorff", "complexity_diff"],
        # "recon_optim_metrics": {"hausdorff": ["hausdorff_22_rms", "rms_min_dist_22_mean"],
        #                         "complexity_diff": ["n_faces_22_ratio_mean"]},
        # "recon_optim_target_evaluator_2": "complexity_diff",
        # "recon_optim_target_metric_2": "n_faces_22_ratio_mean",
        # "recon_optim_target_metric_2_threshold": 1,
        # "recon_optim_target_metric_2_penalty": 0.1,  # per unit in second target metric's exceedance of threshold
        # "recon_optim_target_metric_2_penalty_mode": "add"
    }
)

default_config["reconstruction_config"].update(
    {
        "building_footprints_filepath": str(input_dirpath / "Footprints" / "footprints_10-492-594.gpkg"),
        "building_identifier": "identificatie",
        "geoflow_parameters": geoflow_parameters_default
    }
)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------

default_config["evaluation_config"]["input_cityjson_filepath"] = str(input_dirpath / "CityJSON" / "10-492-594.city.json")
default_config["evaluation_config"]["input_geopackage_filepath"] = str(input_dirpath / "GPKG" / "10-492-594.gpkg")
default_config["evaluation_config"].update(
    {
        "input_obj_lod12_filepath": str(input_dirpath / "OBJ" / "10-492-594_LOD12_correct_ident.obj"),
        "input_obj_lod13_filepath": str(input_dirpath / "OBJ" / "10-492-594_LOD13_correct_ident.obj"),
        "input_obj_lod22_filepath": str(input_dirpath / "OBJ" / "10-492-594_LOD22_mat_python.obj")
    }
)

# ----------------------------------------------------------------------------------------------------------------------
# Scenario settings: Target densities and error levels, and derived survey parameters (pulse and scan frequencies)
# ----------------------------------------------------------------------------------------------------------------------

target_densities = [2**(n/2) for n in range(2, 15)]
point_spacings = [1 / np.sqrt(d) for d in target_densities]

# pulse_freqs_hz = [int(0.5 + 180_000 * (2**(i/2))) for i in range(0, 10)]

std_horizontal_max = 1/np.sqrt(2)
std_vertical_max = 0.3
error_steps = [i/10 for i in range(11)]  # [0.0]

n_density_levels = len(target_densities)
n_error_levels = len(error_steps)
n_scenarios = n_density_levels * n_error_levels

load_scenario_optimizer_states, next_scenarios, processing_sequence = get_processing_sequence(
    n_scenarios=n_scenarios,
    seed_scenario=55
)

scenario_settings = []
scenario_index = 0

for density, point_spacing in zip(target_densities, point_spacings):
    # Factor at the end of pf computation is to account for HELIOS++ not delivering point spacing as it should according
    # to the equations.
    pf = pulse_freq_from_point_spacing(altitude, scan_angle_deg, velocity, point_spacing, point_spacing) * 2.1615
    sf = scan_freq_from_point_spacing(velocity, point_spacing)
    # round to the closest whole number
    pf = int(pf + 0.5)
    sf = int(sf + 0.5)
    for e in error_steps:
        scenario_settings.append({
            "target_density": density,
            "error_level": float(Decimal(str(e))),
            "pulse_freq_hz": pf,
            "scan_freq_hz": sf,
            "std_horizontal_error": float(Decimal(str(e)) * Decimal(str(std_horizontal_max))),
            "std_vertical_error": float(Decimal(str(e)) * Decimal(str(std_vertical_max))),
            "load_scenario_optimizer_states": load_scenario_optimizer_states[scenario_index]
        })
        scenario_index +=1

# A list of names of the scenarios with zero error levels, for which the ALS simulation is executed. For all other
# scenarios with the same density but different noise settings, the same ALS output point cloud is copied and noise is
# added independently by adding a normally distributed random positional error to all points.
scenarios_unique_surveys = [f"scenario_{(i * 11):03}" for i in range(13)]
