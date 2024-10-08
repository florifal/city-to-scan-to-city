from decimal import Decimal
from pathlib import Path
import numpy as np

from experiment.config import scenario_default_config
from experiment.utils import swath_width, scan_freq_from_point_spacing, pulse_freq_from_point_spacing

experiment_name = "utrecht"
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

extent = [135593.338, 454890.404, 137093.338, 456390.404]  # extent of downloaded tiles
# extent = [135290.046875, 454701.1875, 137116.5625, 456419.6875]  # extent of footprints, including protruding ones
# extent = [135600, 455400, 136400, 455700]  # small bbox for testing

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

default_config["evaluation_config"]["input_cityjson_filepath"] = str(input_dirpath / "CityJSON" / "utrecht_merged.city.json")
default_config["evaluation_config"].update(
    {
        "input_obj_lod12_filepath": str(input_dirpath / "OBJ" / "utrecht_merged_LoD12_correct_ident.obj"),
        "input_obj_lod13_filepath": str(input_dirpath / "OBJ" / "utrecht_merged_LoD13_correct_ident.obj"),
        "input_obj_lod22_filepath": str(input_dirpath / "OBJ" / "utrecht_merged_LoD22_mat_python.obj")
    }
)

scene_parts = [
    {
        "type": "obj",
        "filepath": str(input_dirpath / "OBJ" / "utrecht_merged_LoD22_mat_python.obj"),
        "up_axis": "z"
    },
    {
        "type": "tif",
        "filepath": str(input_dirpath / "DTM" / "Utrecht_5m_31H_Z2+N2_fill_clip.TIF"),
        "material_filepath": str(input_dirpath / "DTM" / "Utrecht_5m_31H_Z2+N2_fill_clip.mtl"),
        "material_name": "ground"
    }
]

default_config["reconstruction_config"].update(
    {
        "building_footprints_filepath": str(input_dirpath / "Footprints" / "utrecht_footprints.gpkg"),  # todo: add
        "building_identifier": "identificatie"
    }
)

target_densities = [2**(n/2) for n in range(4, 15)]
point_spacing = [1/np.sqrt(d) for d in target_densities]

# pulse_freqs_hz = [int(0.5 + 180_000 * (2**(i/2))) for i in range(0, 10)]

std_horizontal_max = 1.0
std_vertical_max = 0.3
error_steps = [i/10 for i in range(11)]  # [0.0]

scenario_settings = []

for s in point_spacing:
    pf = pulse_freq_from_point_spacing(altitude, scan_angle_deg, velocity, s, s) * 2.1615
    sf = scan_freq_from_point_spacing(velocity, s)
    # round to the closest whole number
    pf = int(pf + 0.5)
    sf = int(sf + 0.5)
    for e in error_steps:
        scenario_settings.append({
            "pulse_freq_hz": pf,
            "scan_freq_hz": sf,
            "std_horizontal_error": float(Decimal(str(e)) * Decimal(str(std_horizontal_max))),
            "std_vertical_error": float(Decimal(str(e)) * Decimal(str(std_vertical_max)))
        })

scenarios_unique_surveys = [f"scenario_{(i * 11):03}" for i in range(11)]
