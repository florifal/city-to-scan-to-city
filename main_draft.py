import experiment.global_vars as glb
import experiment.scenario as sc
import experiment.experiment as ex
from pathlib import Path
import copy  # use copy.deepcopy() to copy dictionaries: https://stackoverflow.com/a/2465951

# ALL PARAMETERS AND DEFAULT VALUES

default_scene_config = {

}

default_flight_path_config = {

}

default_cloud_merge_config = {
    "input_clouds_dirpath": "",
    "output_cloud_dirpath": "",
    "approach": ""
}

default_survey_config = {
    "survey_xml_filepath": "",
    # In survey XML
    "scannerSettings_id": "",
    "scannerSettings_active": "true",
    "pulseFreq_hz": "",
    "scanAngle_deg": "",
    "scanFreq_hz": "",
    "survey_name": "",
    # Since scene xml path would only be known after scene is generated for experiment,
    # perhaps it can't be a global default?
    "survey_scene": "",
    "survey_platform": "",
    "survey_scanner": "",
    "detectorSettings_accuracy": "",
    # Other
    "output_dirpath": ""
}

default_reconstruction_config = {
    "footprints_filepath": "",
    "point_cloud_filepath": ""
}

# EXPERIMENT

experiment_constants = {
    # INPUT DATA
    # Idea: Scene is generated automatically for experiment level, but not for scenario level
    "building_models_in_filepath": "",
    "dtm_filepath": "",
    "footprints_filepath": "",
    # HELIOS
    "helios": {
        "scene_xml_filepath": "",
        "output_dirpath": ""
    },
    # CLOUDS MERGING
    "clouds": {
        "output_cloud_dirpath"
    },
    # GEOFLOW
    "key": "value",
    # EVALUATION
    "key2": "value"
}

experiment_variables = {
    "pulseFreq_hz": [],
    "accuracy_m": []
}

experiment_param = experiment_constants | experiment_variables

experiment = ex.Experiment(experiment_param)

experiment.run()
experiment.evaluate()


# SCENARIO

# Default config

default_scenario_config = {
    "scene_config": {
        "scene_xml_filepath": "path/to/scene.xml",
        "xml_id": "scene_id",
        "name": "scene_name",
        "scene_parts": [
            {
                "type": "obj",
                "filepath": "path/to/city_model.obj",
                "up_axis": "z"
            },
            {
                "type": "tif",
                "filepath": "path/to/dtm.tif",
                "material_filepath": "path/to/dtm.tif.mat",
                "material_name": "ground"
            }
        ]
    },
    "survey_config": {
        "survey_xml_filepath": "",
        "output_dirpath": "",
        "survey_generator_config": {
            "survey_name": "",
            "scene_xml_filepath": "",  # Since scene xml path would only be known after scene is generated for experiment,
                                       # perhaps it can't be a global default?
            "platform_id": "",
            "scanner_id": "",
            "scanner_settings_id": "",
            "scanner_settings_active": "true",
            "pulse_freq_hz": "",
            "scan_angle_deg": "",
            "scan_freq_hz": "",
            "detector_settings_accuracy": "",
        },
        "flight_path_config": {
            "flight_path_xml_filepath": "",
            "bbox": [0, 0, 0, 0],
            "spacing": .0,
            "altitude": .0,
            "velocity": .0,
            "flight_pattern": "parallel",
            "trajectory_time_interval": .0,
            "always_active": False,
            "scanner_settings_id": ""
        },
        "survey_executor_config": {
            "las_output": True,
            "zip_output": True,
            "num_threads": 0,
        },
        "cloud_merge_config": {
            "clouds_dirpath": "",
            "output_filepath": ""
        }
    },
    "reconstruction_config": {

    }
}

# Scenario-specific parameter values

scene_config = {
    "scene_xml_filepath": "path/to/scene.xml",
    "xml_id": "scene_id",
    "name": "scene_name",
    "scene_parts": [
        {
            "type": "obj",
            "filepath": "path/to/city_model.obj",
            "up_axis": "z"
        },
        {
            "type": "tif",
            "filepath": "path/to/dtm.tif",
            "material_filepath": "path/to/dtm.tif.mat",
            "material_name": "ground"
        }
    ]
}

survey_generator_config = {
    "survey_name": "",
    "scene_xml_filepath": "",  # Since scene xml path would only be known after scene is generated for experiment,
                               # perhaps it can't be a global default?
    "platform_id": "",
    "scanner_id": "",
    "scanner_settings_id": "",
    "scanner_settings_active": "true",
    "pulse_freq_hz": "",
    "scan_angle_deg": "",
    "scan_freq_hz": "",
    "detector_settings_accuracy": "",
}

flight_path_config = {
    "flight_path_xml_filepath": "",
    "bbox": [0, 0, 0, 0],
    "spacing": .0,
    "altitude": .0,
    "velocity": .0,
    "flight_pattern": "parallel",
    "trajectory_time_interval": .0,
    "always_active": False,
    "scanner_settings_id": ""
}

survey_executor_config = {
    "las_output": True,
    "zip_output": True,
    "num_threads": 0,
}

cloud_merge_config = {
    "clouds_dirpath": "",
    "output_filepath": ""
}

survey_config = {
    "survey_xml_filepath": "",
    "output_dirpath": "",
    "survey_generator_config": survey_generator_config,
    "flight_path_config": flight_path_config,
    "survey_executor_config": survey_executor_config,
    "cloud_merge_config": cloud_merge_config
}

reconstruction_config = {
    "point_cloud_filepath": ""
}

# Final scenario settings

scenario_config = {
    "building_models_in_filepath": "",
    "scene_config": scene_config,
    "survey_config": survey_config,
    "reconstruction_config": reconstruction_config,
}

# Scenario execution

scenario = sc.Scenario(scenario_config)

scenario.survey.run()
scenario.survey.merge()
scenario.reconstruction.run()
scenario.reconstruction.evaluate()


