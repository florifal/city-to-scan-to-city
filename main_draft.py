import global_vars as config
import scenario as sc
import experiment as ex
import copy  # use copy.deepcopy() to copy dictionaries: https://stackoverflow.com/a/2465951

# ALL PARAMETERS AND DEFAULT VALUES

helios_param_default = {
    # In survey XML
    "scannerSettings_id": "",
    "scannerSettings_active": "true",
    "pulseFreq_hz": "",
    "scanAngle_deg": "",
    "scanFreq_hz": "",
    "survey_name": "",
    "survey_scene": "",
    "survey_platform": "",
    "survey_scanner": "",
    "detectorSettings_accuracy": "",
    # Other
    "output_dirpath": ""
}

clouds_merge_param_default = {
    "input_clouds_dirpath": "",
    "output_cloud_dirpath": "",
    "approach": ""
}

geoflow_param_default = {
    "footprints_filepath": "",
    "point_cloud_filepath": ""
}

# EXPERIMENT

experiment_constants = {
    # INPUT DATA
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


# SCENARIO

# Scenario-specific parameter values

helios_param_scenario = {
    "pulseFreq_hz": "",
    "accuracy_m": ""
}

clouds_merge_param_scenario = {
    "clouds_dirpath": ""
    "output_filepath": ""
}

geoflow_param_scenario = {
    "point_cloud_filepath": ""
}

# Final scenario settings

scenario_param = {
    "building_models_in_filepath": "",
    "helios_param": helios_param_scenario,
    "geoflow_param": geoflow_param_scenario,
}

# Scenario execution

scenario = sc.Scenario(scenario_param)

scenario.survey.run()
scenario.clouds.merge()
scenario.reconstruction.run()
scenario.models.evaluate()


