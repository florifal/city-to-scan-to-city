from experiment.scenario import *
from pathlib import Path
import numpy as np

from experiment.scene_part import ScenePartOBJ, ScenePartTIFF


def test_experiment():
    default_config = scenario_default_config()

    input_dirpath = r"C:\Users\Florian\Data\city-to-scan-to-city\Experiments\experiment_test_case\01_input"

    scene_parts = [
        {
            "type": "obj",
            "filepath": str(Path(input_dirpath, "9-276-556-LoD22-3D_subset.obj")),
            "up_axis": "z"
        },
        {
            "type": "tif",
            "filepath": str(Path(input_dirpath, "M5_37EN1_5_m_filled_clip_to_subset.TIF")),
            "material_filepath": str(Path(input_dirpath, "M5_37EN1_5_m_filled_clip_to_subset.TIF.mtl")),
            "material_name": "ground"
        }
    ]

    footprint_config = {
        "building_footprints_filepath": r"C:\Users\Florian\Data\city-to-scan-to-city\Experiments\experiment_test_case\01_input\delft_subset_footprints.gpkg",
        "building_identifier": "fid"
    }

    experiment_name = "experiment_test_case_2"
    experiment_dirpath = r"C:\Users\Florian\Data\city-to-scan-to-city\Experiments"
    scenario_settings = [
        {"pulse_freq_hz": 150_000},
        {"pulse_freq_hz": 250_000},
        {"pulse_freq_hz": 500_000}
    ]

    e = Experiment(experiment_name, experiment_dirpath, default_config, scenario_settings, scene_parts,
                   footprint_config)

    e.setup()

    e.setup_surveys()

    e.run_surveys()


if __name__ == "__main__":
    test_experiment()