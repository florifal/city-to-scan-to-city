from decimal import Decimal
from pathlib import Path
import numpy as np

from experiment.config import scenario_default_config
from experiment.scenario import Experiment
from experiment.utils import scan_freq_from_pulse_freq_via_point_spacing, swath_width

experiment_name = "experiment_test_case_delft_large_5x5"
experiment_dirpath = r"C:\Users\Florian\Data\city-to-scan-to-city\Experiments"
input_dirpath = Path(experiment_dirpath, experiment_name, "01_input")

altitude = 500
scan_angle_deg = 20
velocity = 60

spacing = 0.8 * swath_width(altitude, scan_angle_deg)
scanner_id = "riegl_vq-1560i-single-channel"

default_config = scenario_default_config()

default_config["crs"] = "epsg:7415"

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

extent = [82585.171875, 445871.25, 83613.390625, 446896.78125]
bbox = np.array(extent) + np.array([-100, -100, 100, 100])
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

default_config["evaluation_config"]["input_cityjson_filepath"] = str(input_dirpath / "9-276-556.city.json")
default_config["evaluation_config"].update(
    {
        "input_obj_lod12_filepath": str(input_dirpath / "9-276-556-LoD12-3D.obj"),
        "input_obj_lod13_filepath": str(input_dirpath / "9-276-556-LoD13-3D.obj"),
        "input_obj_lod22_filepath": str(input_dirpath / "9-276-556-LoD22-3D.obj")
    }
)

scene_parts = [
    {
        "type": "obj",
        "filepath": str(input_dirpath / "9-276-556-LoD22-3D.obj"),
        "up_axis": "z"
    },
    {
        "type": "tif",
        "filepath": str(input_dirpath / "M5_37EN1_5_m_filled.TIF"),
        "material_filepath": str(input_dirpath / "M5_37EN1_5_m_filled.TIF.mtl"),
        "material_name": "ground"
    }
]

default_config["reconstruction_config"].update(
    {
        "building_footprints_filepath": str(input_dirpath / "9-276-556_pand_footprints.gpkg"),
        "building_identifier": "identificatie"
    }
)

pulse_freqs_hz = [2*f for f in [31_250, 62_500, 125_000, 250_000, 500_000]]  # [31_250, 62_500, 125_000]

std_horizontal_max = 1.0
std_vertical_max = 0.3
error_steps = [0.0, 0.25, 0.5, 0.75, 1.0]  # [0.0]

scenario_settings = []

for f in pulse_freqs_hz:
    for e in error_steps:
        scenario_settings.append({
            "pulse_freq_hz": f,
            "scan_freq_hz": scan_freq_from_pulse_freq_via_point_spacing(f, altitude, velocity, scan_angle_deg),
            "std_horizontal_error": float(Decimal(str(e)) * Decimal(str(std_horizontal_max))),
            "std_vertical_error": float(Decimal(str(e)) * Decimal(str(std_vertical_max)))
        })

e = Experiment(experiment_name, experiment_dirpath, default_config, scenario_settings, scene_parts)
