import copy
import json
import time
import pyhelios
import bayes_opt as bo
import pandas as pd
from typing_extensions import Self
from typing import Callable, Any
from datetime import timedelta, datetime
from math import isclose, floor
from pathlib import Path

from experiment import global_vars as glb
from experiment.config import deep_update, get_config_item, update_config_item
from experiment.point_cloud_processing import CloudNoiseAdder
from experiment.reconstruction import Reconstruction
from experiment.scene import Scene
from experiment.survey import Survey
from experiment.utils import get_last_line_from_file, deep_replace_in_string_values, point_spacing_along, \
    point_spacing_across
from experiment.evaluator import *

pyhelios.loggingVerbose2()
pyhelios.setDefaultRandomnessGeneratorSeed("42")


class Scenario:

    def __init__(self, name: str = "", config: dict | None = None):
        self.name = name
        self.config = config
        self.scene_config = config["scene_config"]
        self.survey_config = config["survey_config"]
        self.cloud_processing_config = config["cloud_processing_config"]
        self.recon_optim_config = config["recon_optim_config"]
        self.reconstruction_config = config["reconstruction_config"]
        self.evaluation_config = config["evaluation_config"]

        self.scene: Scene | None = None
        self.survey: Survey | None = None
        self.recon_optim: ReconstructionOptimization | None = None
        self.reconstruction: Reconstruction | None = None

        self.evaluators: dict[str, Evaluator] = {}

        self._n_buildings_reconstructed: int | None = None
        self._flag_zero_buildings_reconstructed: bool | None = None

    def setup(self):
        pass

    def save_config(self):
        self.settings_dirpath.mkdir(exist_ok=True)
        with open(self.settings_dirpath / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def setup_scene(self):
        self.scene = Scene(
            filepath=self.scene_config["scene_xml_filepath"],
            xml_id=self.scene_config["scene_xml_id"],
            name=self.scene_config["scene_name"],
            scene_parts=self.scene_config["scene_parts"]
        )
        self.scene.create_scene_xml()

    def setup_survey(self):
        # If self.scene was not initialized by running self.setup_scene(), perhaps because scene XML already existed,
        # try to obtain the scene XML file path with ID from the config. Otherwise, throw error.
        if self.scene is None:
            from_config = self.config["survey_config"]["survey_generator_config"]["scene_xml_filepath_with_id"]
            if from_config == "":
                raise ValueError("Trying to setup survey, but no access to scene XML filepath with scene ID.")
            else:
                scene_xml_filepath_with_id = from_config
        else:
            scene_xml_filepath_with_id = self.scene.filepath_with_id

        self.survey = Survey(
            survey_xml_filepath=self.survey_config["survey_xml_filepath"],
            output_dirpath=self.survey_config["survey_output_dirpath"],
            config=self.survey_config,
            scene_filepath=scene_xml_filepath_with_id,
            crs=self.config["crs"]
        )

    def prepare_survey(self):
        (self.survey.output_base_dirpath_helios / self.name).mkdir(exist_ok=True)
        self.survey.create_flight_path()
        self.survey.create_survey_xml()
        self.survey.setup_executor()

    def run_survey(self):
        self.survey.run()
        self.survey.setup_merger()
        self.survey.merge_clouds()

    def clear_survey(self):
        self.survey.clear()

    def process_point_cloud(self):
        Path(self.cloud_processing_config["cloud_processing_output_dirpath"]).mkdir(exist_ok=True)

        # Only run CloudNoiseAdder if a non-zero horizontal or vertical error is provided in the settings
        if (
                self.cloud_processing_config["std_horizontal_error"] != 0 or
                self.cloud_processing_config["std_vertical_error"] != 0
        ):
            print(f"Scenario `{self.name}`: Preparing to add noise to point cloud ...\n")
            filename = "clouds_merged_noise.laz"
            final_cloud_filepath = Path(self.cloud_processing_config["cloud_processing_output_dirpath"]) / filename

            noise_adder = CloudNoiseAdder(
                input_filepath=self.merged_point_cloud_filepath,
                output_filepath=final_cloud_filepath,
                std_horizontal=self.cloud_processing_config["std_horizontal_error"],
                std_vertical=self.cloud_processing_config["std_vertical_error"],
                crs=self.crs
            )

            noise_adder.run()
        else:
            print(f"Scenario `{self.name}`: Not adding noise to point cloud - zero error set.\n")
            # If no noise is added, the final cloud is the merged cloud from the survey
            final_cloud_filepath = self.merged_point_cloud_filepath

        with open(self.textfile_final_cloud_path_filepath, "a", encoding="utf-8") as f:
            f.write(str(final_cloud_filepath) + "\n")

    def setup_reconstruction_optimization(self):
        self.recon_optim = ReconstructionOptimization(
            crs=self.config["crs"],
            config=self.recon_optim_config,
            scenario_config=self.config
        )
        self.recon_optim.setup()

    def prepare_reconstruction_optimization(self):
        self.recon_optim.output_dirpath.mkdir(exist_ok=True)
        self.recon_optim.prepare()

    def run_reconstruction_optimization(self, init_points: int | None = None, n_iter: int | None = None):
        self.recon_optim.run(init_points, n_iter)
        optim_params = self.recon_optim.result["params"]
        # Update scenario config with optimized Geoflow parameters and save it
        self.reconstruction_config["geoflow_parameters"].update(optim_params)
        self.save_config()

    def clear_reconstruction_optimization(self):
        self.recon_optim.clear()

    def setup_reconstruction(self):
        self.reconstruction = Reconstruction(
            crs=self.config["crs"],
            config=self.reconstruction_config,
            cloud_filepath=self.final_point_cloud_filepath
        )

    def prepare_reconstruction(self):
        self.reconstruction.output_dirpath.mkdir(exist_ok=True)
        self.reconstruction.prepare_config()
        self.reconstruction.setup_executor()

    def run_reconstruction(self):
        self.reconstruction.run()
        self.check_reconstruction_results()
        if self.flag_zero_buildings_reconstructed:
            print(f"WARNING: Reconstruction in {self.name} yielded zero buildings.\n")
        else:
            print(f"Reconstruction in {self.name} yielded {self.n_buildings_reconstructed} buildings.\n")

    def clear_reconstruction(self):
        self.reconstruction.clear()

    def setup_evaluation(self, lods: list[str] | str | None = None):
        if lods is None:
            lods = ["1.2", "1.3", "2.2"]
        elif isinstance(lods, str):
            lods = [lods]

        lods_no_points = [lod.replace(".", "") for lod in lods]

        evaluators = [
            AreaVolumeDifferenceEvaluator(
                output_base_dirpath_1=self.input_evaluation_dirpath,
                output_base_dirpath_2=self.output_evaluation_dirpath,
                input_cityjson_filepath_1=self.cityjson_input_filepath,
                input_cityjson_filepath_2=self.cityjson_output_filepath,
                index_col_name_1="",  # 3DBAG CityJSON file has correct index (loaded with cjio), but index has no name
                index_col_name_2=glb.geoflow_output_cityjson_identifier_name,
                lods=lods,
                crs=self.crs
            ),
            # todo: remove
            # AreaVolumeEvaluator(
            #     output_base_dirpath=self.evaluation_output_dirpath,
            #     input_cityjson_filepath=self.cityjson_output_filepath,
            #     index_col_name=glb.geoflow_output_cityjson_identifier_name,
            #     lods=lods,
            #     crs=self.crs
            # ),
            IOU3DEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                input_cityjson_filepath_1=self.cityjson_input_filepath,
                input_cityjson_filepath_2=self.cityjson_output_filepath,
                index_col_name=glb.geoflow_output_cityjson_identifier_name,
                lods=lods,
                crs=self.crs
            ),
            HausdorffLODSEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                input_obj_filepath_pairs={
                    k: v for k, v in {
                        "1.2": (self.obj_input_lod12_filepath, self.obj_output_lod12_filepath),
                        "1.3": (self.obj_input_lod13_filepath, self.obj_output_lod13_filepath),
                        "2.2": (self.obj_input_lod22_filepath, self.obj_output_lod22_filepath)
                    }.items()
                    if k in lods
                }
            ),
            ComplexityEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                input_filepath=self.geopackage_output_filepath,
                lods=lods,
                index_col_name="OGRLoader.identificatie"
            ),
            # ComplexityEvaluator(
            #     output_base_dirpath=self.output_evaluation_dirpath,
            #     input_filepath=[self.obj_output_lod12_filepath,
            #                     self.obj_output_lod13_filepath,
            #                     self.obj_output_lod22_filepath],
            #     lods=["1.2", "1.3", "2.2"],
            #     ignore_meshes_with_zero_faces=True
            # ),
            HeightEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                input_cityjson_filepath_1=self.cityjson_input_filepath,
                input_cityjson_filepath_2=self.cityjson_output_filepath,
                identifier_name=glb.geoflow_output_cityjson_identifier_name
            ),
            PointDensityDatasetEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                point_cloud_filepath=self.final_point_cloud_filepath,
                bbox=get_config_item(self.survey_config, "bbox"),
                building_footprints_gpkg_filepath=self.building_footprints_filepath,
                crs=self.crs,
                save_filtered_point_clouds=True,
                footprints_density_computation=False,
                radial_density_computation=False
            ),
            PointMeshDistanceEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                point_cloud_filepath=self.final_point_cloud_filepath,
                mesh_filepath=self.obj_input_lod22_filepath,
                crs=self.crs
            ),
            GeoflowOutputEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                gpkg_filepath=self.geopackage_output_filepath,
                gpkg_layers=dict(zip(
                    lods,
                    [self.geoflow_template_json["nodes"][f"OGRWriter-LoD{lod}-3D"]["parameters"]["layername"]
                     for lod in lods_no_points]
                )),
                cityjson_filepath=self.cityjson_output_filepath,
                obj_filepaths={
                    k: v for k, v in
                    dict(zip(
                        ["1.2", "1.3", "2.2"],
                        [self.obj_output_lod12_filepath, self.obj_output_lod13_filepath, self.obj_output_lod22_filepath]
                    )).items()
                    if k in lods
                }
            )
        ]

        self.evaluators = {evaluator.name: evaluator for evaluator in evaluators}

    def run_evaluation(self, evaluator_selection: list[str] | str | None = None):
        if evaluator_selection is None:
            evaluator_selection = list(self.evaluators.keys())
        elif isinstance(evaluator_selection, str):
            evaluator_selection = [evaluator_selection]

        # todo: enable conditional execution of evaluators that do not depend on reconstructed building models
        if self.flag_zero_buildings_reconstructed:
            print(f"WARNING: Reconstruction in {self.name} yielded zero buildings. Skipping evaluation.\n")
        else:
            for evaluator_name in evaluator_selection:
                self.evaluators[evaluator_name].run()

    # todo: remove - area and volume of input data are now computed by AreaVolumeDifferenceEvaluator
    # def setup_input_evaluation(self, output_dirpath: Path | str | None = None):
    #     if output_dirpath is None:
    #         output_dirpath = self.input_evaluation_dirpath
    #     output_dirpath.mkdir(exist_ok=True)
    #
    #     # todo: clean up this mess. either move variable definition to __init__(), or merge this method with
    #     #  setup_evaluation().
    #     input_evaluators = [
    #         AreaVolumeEvaluator(
    #             output_base_dirpath=output_dirpath,
    #             input_cityjson_filepath=self.cityjson_input_filepath,
    #             index_col_name="identificatie",  # todo: un-hardcode
    #             lods=["1.2", "1.3", "2.2"],
    #             crs=self.crs
    #         )
    #     ]
    #
    #     self.input_evaluators = {evaluator.name: evaluator for evaluator in input_evaluators}
    #
    # def run_input_evaluation(self):
    #     for evaluator in self.input_evaluators:
    #         evaluator.run()

    def concat_evaluation_results(self, evaluator_selection: list[str] | str | None = None):
        if evaluator_selection is None:
            evaluator_selection = list(self.evaluators.keys())
        elif isinstance(evaluator_selection, str):
            evaluator_selection = [evaluator_selection]

        results_final = pd.concat(
            [self.evaluators[name].results_final for name in evaluator_selection],
            axis=1
        )
        results_final.to_csv(self.output_evaluation_dirpath / "evaluation_results.csv")

    def get_summary_statistics(self, evaluator_selection: list[str] | str | None = None) -> dict:
        if evaluator_selection is None:
            evaluator_selection = [name for name, evaluator in self.evaluators.items()]
        else:
            if isinstance(evaluator_selection, str):
                evaluator_selection = [evaluator_selection]

        summary_statistics = {}
        for name in evaluator_selection:
            summary_statistics.update(self.evaluators[name].summary_stats)

        return summary_statistics

    def check_reconstruction_results(self):
        gpkg_eval = GeopackageBuildingsEvaluator(
            output_base_dirpath=self.reconstruction_output_dirpath,
            gpkg_filepath=self.geopackage_output_filepath,
            gpkg_layers={"2.2": self.geoflow_template_json["nodes"][f"OGRWriter-LoD22-3D"]["parameters"]["layername"]},
            id_col_name=glb.geoflow_output_cityjson_identifier_name
        )
        gpkg_eval.run()
        self._n_buildings_reconstructed = gpkg_eval.results_df.loc["num_unique", "22"]
        if self._n_buildings_reconstructed == 0:
            self._flag_zero_buildings_reconstructed = True
        else:
            self._flag_zero_buildings_reconstructed = False

    @property
    def flag_zero_buildings_reconstructed(self):
        if self._flag_zero_buildings_reconstructed is None:
            self.check_reconstruction_results()
        return self._flag_zero_buildings_reconstructed

    @property
    def n_buildings_reconstructed(self):
        if self._n_buildings_reconstructed is None:
            self.check_reconstruction_results()
        return self._n_buildings_reconstructed

    @property
    def crs(self):
        return get_config_item(self.config, "crs")

    @property
    def settings_dirpath(self):
        return Path(self.config["settings_dirpath"])

    @property
    def reconstruction_output_dirpath(self):
        return Path(get_config_item(self.config, "reconstruction_output_dirpath"))

    @property
    def base_evaluation_dirpath(self):
        return Path(get_config_item(self.config, "evaluation_output_dirpath"))

    @property
    def output_evaluation_dirpath(self):
        return self.base_evaluation_dirpath / self.name

    @property
    def input_evaluation_dirpath(self):
        return self.base_evaluation_dirpath / "input"

    @property
    def geoflow_template_json(self) -> dict:
        with open(glb.geoflow_reconstruct_template_filepath, "r") as f:
            return json.load(f)

    @property
    def cityjson_output_filepath(self):
        return self.reconstruction_output_dirpath / Path(self.geoflow_template_json["globals"]["output_cityjson"][2])

    @property
    def cityjson_input_filepath(self):
        # Equivalents:
        # return get_config_item(self.config, "input_cityjson_filepath")
        return self.evaluation_config["input_cityjson_filepath"]

    @property
    def geopackage_output_filepath(self):
        return self.reconstruction_output_dirpath / Path(self.geoflow_template_json["globals"]["output_ogr"][2])

    @property
    def obj_input_lod12_filepath(self):
        return Path(self.evaluation_config["input_obj_lod12_filepath"])

    @property
    def obj_input_lod13_filepath(self):
        return Path(self.evaluation_config["input_obj_lod13_filepath"])

    @property
    def obj_input_lod22_filepath(self):
        return Path(self.evaluation_config["input_obj_lod22_filepath"])

    @property
    def obj_output_lod12_filepath(self):
        return self.reconstruction_output_dirpath / Path(self.geoflow_template_json["globals"]["output_obj_lod12"][2])

    @property
    def obj_output_lod13_filepath(self):
        return self.reconstruction_output_dirpath / Path(self.geoflow_template_json["globals"]["output_obj_lod13"][2])

    @property
    def obj_output_lod22_filepath(self):
        return self.reconstruction_output_dirpath / Path(self.geoflow_template_json["globals"]["output_obj_lod22"][2])

    @property
    def textfile_merged_cloud_path_filepath(self):
        return str(Path(
            self.survey_config["survey_output_dirpath"],
            self.survey_config["survey_generator_config"]["survey_name"],
            "merged_cloud_filepath.txt"
        ))

    @property
    def merged_point_cloud_filepath(self):
        # Try to read the path to the merged point cloud file from the textfile that should have been created after
        # merging the clouds if the survey was already run
        error_message = ("Filepath of merged cloud for reconstruction is unknown without prior survey or merger run, "
                         "and text file containing it from prior survey run does not exist.")
        return get_last_line_from_file(self.textfile_merged_cloud_path_filepath, error_message)

    @property
    def textfile_final_cloud_path_filepath(self):
        return str(Path(
            self.cloud_processing_config["cloud_processing_output_dirpath"],
            "final_cloud_filepath.txt"
        ))

    @property
    def final_point_cloud_filepath(self):
        error_message = "Text file containing path to final processed point cloud does not exist."
        return get_last_line_from_file(self.textfile_final_cloud_path_filepath, error_message)

    @property
    def building_footprints_filepath(self):
        return self.reconstruction_config["building_footprints_filepath"]


class Experiment:
    """An experiment consisting of multiple scenarios"""

    def __init__(
            self,
            name: str,
            dirpath: Path | str,
            default_config: dict,
            scenario_settings: list[dict] | None = None,
            scene_parts: list[dict] | None = None,
    ):
        """Experiment

        :param name: Experiment name.
        :param dirpath: Directory in which a subdirectory with the experiment name will be created
        :param default_config: Default values for all scenarios
        :param scenario_settings: Specific settings for each scenario as a list of dictionaries
        :param scene_parts: List of dictionaries, each of which specifies details for an OBJ or TIF scene part
        building_footprints_filepath and building_identifier.
        """
        self.name = name
        self._base_dirpath = Path(dirpath)
        self.default_config = default_config
        self.scenario_settings = scenario_settings
        if scene_parts is not None:
            self.default_config["scene_config"]["scene_parts"] = scene_parts

        self.scene: Scene | None = None

        self.scenarios: dict[str, Scenario] = {}
        self.scenario_configs: dict[str, dict] = {}

        self._summary_stats: pd.DataFrame | None = None

    def __getitem__(self, item):
        return list(self.scenarios.values())[item]

    def __len__(self):
        return len(self.scenarios)

    @property
    def dirpath(self):
        return self._base_dirpath / self.name

    @property
    def settings_dirpath(self):
        return self.dirpath / "02_settings"

    @property
    def scene_dirpath(self):
        return self.dirpath / "03_scene"

    @property
    def survey_dirpath(self):
        return self.dirpath / "04_survey"

    @property
    def cloud_processing_dirpath(self):
        return self.dirpath / "05_point_clouds"

    @property
    def recon_optim_dirpath(self):
        return self.dirpath / "06_reconstruction_optimization"

    @property
    def reconstruction_dirpath(self):
        return self.dirpath / "07_reconstruction"

    @property
    def evaluation_dirpath(self):
        return self.dirpath / "08_evaluation"

    @property
    def scene_xml_filepath(self):
        return self.scene_dirpath / f"{self.name}_scene.xml"

    @property
    def summary_stats(self):
        if self._summary_stats is None:
            try:
                self.load_summary_stats()
            except:
                self.compute_summary_statistics()
        return self._summary_stats

    def rename(self, to_name: str, update_scene: bool = False):
        """

        :param to_name: New experiment name
        :param update_scene: Whether to update the scene. Set to true for experiments with scene, false for
         reconstruction optimization experiments without scene.
        :return: None
        """
        print(f"Renaming experiment from `{self.name}` to `{to_name}` ...")
        from_dirpath = self.dirpath
        to_dirpath = self._base_dirpath / to_name

        # Update filepaths in the experiment's default config for scenarios
        deep_replace_in_string_values(self.default_config, str(from_dirpath), str(to_dirpath))

        # The scene XML is about the only file containing the experiment's name, so it must be renamed. Additionally,
        # if HELIOS++ has already compiled the scene, rename the .scene file as well.
        # These filepaths are updated in the config below by calling self.setup(), which calls self.setup_scene()
        if self.scene_xml_filepath.is_file():
            self.scene_xml_filepath.rename(self.scene_dirpath / f"{to_name}_scene.xml")
        if self.scene_xml_filepath.with_suffix(".scene").is_file():
            self.scene_xml_filepath.with_suffix(".scene").rename(self.scene_dirpath / f"{to_name}_scene.scene")

        from_dirpath.rename(to_dirpath)
        # Critical step: Changing this instance's name. Afterward, all path property return the new path.
        self.name = to_name

        if update_scene:
            # Calling self.setup_scene() updates self.default_config["scene_config"] with the new scene paths and names
            self.setup_scene()

        for n, s in self.scenarios.items():
            # Update filepaths in the scenario's config
            deep_replace_in_string_values(s.config, str(from_dirpath), str(to_dirpath))
            if update_scene:
                # Apply the updated scene paths and names to the scenario's scene config
                s.config["scene_config"] = self.default_config["scene_config"]

        self.save(save_scenarios=True)

    def load_summary_stats(self):
        print("Loading summary statistics from file ...")
        self._summary_stats = pd.read_csv(self.evaluation_dirpath / "summary_statistics.csv")

    def setup(self):
        """Call all setup functions for directories, scene, scenario configs, and scenarios"""
        self.setup_directories()
        self.setup_scene()
        # self.setup_configs()  # todo: remove if no bugs occur; see below.
        self.setup_scenarios()
        self.save()

    def setup_directories(self):
        """Create all main directories for the experiment"""
        # Potentially merge with __init__
        self.dirpath.mkdir(exist_ok=True)
        self.settings_dirpath.mkdir(exist_ok=True)
        self.scene_dirpath.mkdir(exist_ok=True)
        self.survey_dirpath.mkdir(exist_ok=True)
        self.cloud_processing_dirpath.mkdir(exist_ok=True)
        self.reconstruction_dirpath.mkdir(exist_ok=True)
        self.evaluation_dirpath.mkdir(exist_ok=True)

    def setup_scene(self):
        """Create the scene XML for the experiment and update the default config dict accordingly"""
        # Create the scene for the experiment
        scene_xml_id = f"{self.name.lower()}_scene"
        scene_name = f"{self.name}_scene"
        scene_parts = self.default_config["scene_config"]["scene_parts"]
        self.scene = Scene(
            filepath=str(self.scene_xml_filepath),
            xml_id=scene_xml_id,
            name=scene_name,
            scene_parts=scene_parts
        )
        self.scene.create_scene_xml()
        # Update all scene-related settings in the default config
        update_config_item(self.default_config, "scene_xml_filepath", str(self.scene_xml_filepath))
        update_config_item(self.default_config, "scene_xml_id", scene_xml_id)
        update_config_item(self.default_config, "scene_name", scene_name)
        update_config_item(self.default_config, "scene_parts", scene_parts)
        update_config_item(self.default_config, "scene_xml_filepath_with_id", self.scene.filepath_with_id)

    def setup_scenarios(self):
        """Setup individual config dictionaries for each scenario, and save them as JSON for reference"""

        if self.scenario_settings is None:
            raise ValueError("Cannot set up scenarios: No scenario settings provided (scenario_settings is None).")
        else:
            for i, scenario_settings in enumerate(self.scenario_settings):
                self.add_scenario(scenario_settings, i)

    # todo: remove if no bugs occur; setup of both configs and Scenario instances is now done in add_scenario()
    # def setup_scenarios(self):
    #     """Initialize the Scenario objects for each scenario with its corresponding configuration"""
    #     for name, config in self.scenario_configs.items():
    #         self.scenarios[name] = Scenario(name, config)

    def add_scenario(self, scenario_settings: dict, scenario_number: int | None = None):
        """Add a single scenario with new configuration. Required for iterative ReconstructionOptimization.

        :param scenario_settings: Dictionary of settings for the scenario using standard config keys
        :param scenario_number: Number used in the scenario name if no name is specified in the settings dict
        """
        # Scenario name
        scenario_name = scenario_settings.pop(
            "scenario_name",
            f"scenario_{len(self.scenarios):03}" if scenario_number is None else f"scenario_{scenario_number:03}"
        )
        if scenario_name in self.scenarios.keys():
            raise ValueError(f"Duplicate scenario name: '{scenario_name}'")

        # Scenario-specific dir paths
        Path(self.settings_dirpath, scenario_name).mkdir(exist_ok=True)
        # todo: remove if no bugs occur; directories are now created in the respective Scenario methods
        # Path(self.survey_dirpath, scenario_name).mkdir(exist_ok=True)
        # Path(self.cloud_processing_dirpath, scenario_name).mkdir(exist_ok=True)
        # Path(self.recon_optim_dirpath, scenario_name).mkdir(exist_ok=True)
        # Path(self.reconstruction_dirpath, scenario_name).mkdir(exist_ok=True)

        # Scenario-specific file paths
        flight_path_xml_filepath = self.survey_dirpath / scenario_name / "flight_path.xml"
        survey_xml_filepath = self.survey_dirpath / scenario_name / (scenario_name + "_survey.xml")
        survey_output_dirpath = self.survey_dirpath  # HELIOS creates subfolders: /scenario_name/date_time
        cloud_processing_output_dirpath = self.cloud_processing_dirpath / scenario_name
        recon_optim_output_dirpath = self.recon_optim_dirpath / scenario_name
        reconstruction_output_dirpath = self.reconstruction_dirpath / scenario_name  # reconstruct.json has output/
        evaluation_output_dirpath = self.evaluation_dirpath  # / scenario_name  # todo:remove

        # Create a copy of the default config
        config = copy.deepcopy(self.default_config)

        # Update all occurrences of this scenario's settings in the config
        for key, value in scenario_settings.items():
            update_config_item(config, key, value)

        # Update other values in the config
        config["scenario_name"] = scenario_name
        config["settings_dirpath"] = str(self.settings_dirpath / scenario_name)
        update_config_item(config, "survey_name", scenario_name)
        update_config_item(config, "flight_path_xml_filepath", str(flight_path_xml_filepath))
        update_config_item(config, "survey_xml_filepath", str(survey_xml_filepath))
        update_config_item(config, "survey_output_dirpath", str(survey_output_dirpath))
        update_config_item(config, "cloud_processing_output_dirpath", str(cloud_processing_output_dirpath))
        update_config_item(config, "recon_optim_output_dirpath", str(recon_optim_output_dirpath))
        update_config_item(config, "reconstruction_output_dirpath", str(reconstruction_output_dirpath))
        update_config_item(config, "evaluation_output_dirpath", str(evaluation_output_dirpath))

        # Append the finalized name and config to the lists, and create the Scenario instance
        self.scenario_configs[scenario_name] = copy.deepcopy(config)
        self.scenarios[scenario_name] = Scenario(scenario_name, self.scenario_configs[scenario_name])

        # Store the scenario settings snippet and the full config in the settings folder for reference
        with open(self.settings_dirpath / scenario_name / "scenario_settings.json", "w", encoding="utf-8") as f:
            # ensure_ascii=False avoid special characters (such as "ä") being escapes (such as "\u00e4")
            json.dump(scenario_settings, f, indent=4, ensure_ascii=False)
        self.scenarios[scenario_name].save_config()

    def load_scenarios(self):
        """Set up scenarios by loading the config files from the Experiment settings directory"""
        for settings_dirpath in [item for item in self.settings_dirpath.iterdir() if item.is_dir()]:
            scenario_name = settings_dirpath.name
            with open(settings_dirpath / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            self.scenario_configs[scenario_name] = copy.deepcopy(config)
            self.scenarios[scenario_name] = Scenario(scenario_name, self.scenario_configs[scenario_name])

    def save(self, save_scenarios: bool = True):
        print("Saving experiment configuration ...")
        with open(self.settings_dirpath / "default_config.json", "w", encoding="utf-8") as f:
            json.dump(self.default_config, f, indent=4, ensure_ascii=False)
        if self.scenario_settings is not None:
            with open(self.settings_dirpath / "scenario_settings.json", "w", encoding="utf-8") as f:
                json.dump({"scenario_settings": self.scenario_settings}, f, indent=4, ensure_ascii=False)
        if save_scenarios:
            print("Saving scenario configurations ...")
            for name, s in self.scenarios.items():
                s.save_config()

    @classmethod
    def load(cls, dirpath: Path | str, load_scenarios: bool = True) -> Self:
        """
        :param dirpath: Experiment directory, including the folder with the experiment name as last item
        :param load_scenarios: Whether to load scenario configurations and set up scenarios
        :return: An initialized `Experiment` instance
        """
        print("Loading experiment configuration ...")
        dirpath = Path(dirpath)
        settings_dirpath = dirpath / "02_settings"

        with open(settings_dirpath / "default_config.json", "r") as f:
            default_config = json.load(f)

        try:
            with open(settings_dirpath / "scenario_settings.json", "r") as f:
                scenario_settings = json.load(f)["scenario_settings"]
        except FileNotFoundError:
            print("File `scenario_settings.json` not found.")
            scenario_settings = None

        print("Initializing experiment ...")
        e = Experiment(name=dirpath.name, dirpath=dirpath.parent, default_config=default_config,
                       scenario_settings=scenario_settings)
        if load_scenarios:
            print("Loading scenarios ...")
            e.load_scenarios()

        return e

    # todo: update - is outdated
    def run_all(self, by: str = "scenario"):
        """Run all scenarios, either sequentially or in order of the steps"""
        if by not in ["scenario", "step"]:
            raise ValueError("Argument 'by' must be one of ['scenario', 'step'].")

        if by == "scenario":
            for name, s in self.scenarios.items():
                print(f"Executing scenario {name} ...\n")
                t0 = time.time()

                s.setup_survey()
                s.prepare_survey()
                s.run_survey()
                s.setup_reconstruction()
                s.prepare_reconstruction()
                s.run_reconstruction()

                t1 = time.time()
                print(f"Finished scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")

        elif by == "step":
            pass

    def run_scenario(self, name: str | None = None, number: int | None = None, scenario: Scenario | None = None):
        pass

    def run_steps(
            self,
            steps: Callable[[Scenario], Any] | list[Callable[[Scenario], Any]],
            scenarios: list[str] | str | None = None,
            *args,
            **kwargs
    ):
        """Run a Scenario step (a method of the class) for all or a selection of this Experiment's scenarios.

        Additional arguments of the method that is to be run can be given as *args or **kwargs."""

        if scenarios is None:
            scenarios = self.scenarios
        else:
            if isinstance(scenarios, str):
                scenarios = [scenarios]
            scenarios = {name: scenario for name, scenario in self.scenarios.items() if name in scenarios}

        if not isinstance(steps, list):
            steps = [steps]

        for name, s in scenarios.items():
            for step in steps:
                print(f"\nRunning '{step.__name__}' for {name} ...")
                t0 = time.time()
                step(s, *args, **kwargs)
                t1 = time.time()
                print(f"\nFinished '{step.__name__}' for {name} after {str(timedelta(seconds=t1 - t0))}.")

    def compute_summary_statistics(self, evaluator_selection: list[str] | str | None = None):
        print("Computing summary statistics from all scenarios ...")
        rows = []

        for name, s in self.scenarios.items():
            pulse_freq_hz = get_config_item(s.config, "pulse_freq_hz")
            scan_freq_hz = get_config_item(s.config, "scan_freq_hz")
            scan_angle_deg = get_config_item(s.config, "scan_angle_deg")
            altitude = get_config_item(s.config, "altitude")
            velocity = get_config_item(s.config, "velocity")

            cols = {
                "name": s.name,
                "pulse_freq_hz": pulse_freq_hz,
                "point_spacing_along": point_spacing_along(velocity, scan_freq_hz),
                "point_spacing_across": point_spacing_across(altitude, scan_angle_deg, pulse_freq_hz, scan_freq_hz),
                "std_horizontal_error": get_config_item(s.config, "std_horizontal_error"),
                "std_vertical_error": get_config_item(s.config, "std_vertical_error")
            }

            cols.update(s.config["reconstruction_config"]["geoflow_parameters"])

            # Only add evaluation results to the cols dictionary if any buildings were reconstructed
            if not s.flag_zero_buildings_reconstructed:
                cols.update(s.get_summary_statistics(evaluator_selection))

            rows.append(cols)

        self._summary_stats = pd.DataFrame(rows).set_index("name")
        self._summary_stats.to_csv(self.evaluation_dirpath / "summary_statistics.csv")


class ReconstructionOptimization:

    def __init__(
            self,
            crs: str,
            config: dict,
            scenario_config: dict,
            init_points: int = 5,
            n_iter: int = 10
    ):
        self.crs = crs
        self.config = config
        self.optim_config = copy.deepcopy(scenario_config)

        self.output_dirpath = Path(self.config["recon_optim_output_dirpath"])
        # Remove the prefix "range_" from each parameter name in the parameter space
        self.parameter_space = {k.split("_", 1)[1]: v for k, v in self.config["parameter_space"].items()}
        self.evaluators = self.config["recon_optim_evaluators"]
        self.metrics = self.config["recon_optim_metrics"]
        self.target_lod = self.config["recon_optim_target_lod"]
        self.target_evaluator = self.config["recon_optim_target_evaluator"]
        self.target_metric = self.config["recon_optim_target_metric"]
        self.target_metric_optimum = {"max": 1, "min": -1}[self.config["recon_optim_target_metric_optimum"]]

        self.optim_experiment: Experiment | None = None
        self.optimizer: bo.BayesianOptimization | None = None
        self.logger: bo.logger.JSONLogger | None = None
        self.iter_count = 0
        self.init_points = init_points
        self.n_iter = n_iter
        self.past_target_values = []

        self._result: dict | None = None

    def clear(self):
        # todo: Can all of this be cleared safely? What can safely be cleared?
        self.optim_experiment = None
        self.optimizer = None
        self.logger = None

    def setup(self):
        print("Setting up reconstruction optimization ...")
        # The reconstruction optimization experiment will be located in the parent scenario's directory for
        # reconstruction optimization, in a subdirectory with the scenario's name
        experiment_name = self.optim_config["scenario_name"]
        experiment_dirpath = Path(self.optim_config["recon_optim_config"]["recon_optim_output_dirpath"]).parent

        # For the reconstruction optimization experiment, most parameters should be unchanged compared to the parent
        # scenario. However, the reconstruction parameters must be changed, because the reconstruction output must be
        # saved in the parent scenario's directory for reconstruction optimization, and the building footprints used
        # for the optimization may be different from (e.g., a subset of) the parent scenarios footprints.
        self.optim_config["reconstruction_config"]["building_footprints_filepath"] = self.config["optimization_footprints_filepath"]
        self.optim_config["reconstruction_config"]["building_footprints_sql"] = self.config["optimization_footprints_sql"]

        self.optim_experiment = Experiment(
            name=experiment_name,
            dirpath=experiment_dirpath,
            default_config=self.optim_config
        )
        self.optim_experiment.save()

    def prepare(self):
        print("Preparing reconstruction optimization ...")
        # Instead of Experiment.setup_directories(), create only those directories required here
        self.optim_experiment.dirpath.mkdir(exist_ok=True)
        self.optim_experiment.settings_dirpath.mkdir(exist_ok=True)
        self.optim_experiment.reconstruction_dirpath.mkdir(exist_ok=True)
        self.optim_experiment.evaluation_dirpath.mkdir(exist_ok=True)

        date_time = datetime.today().strftime("%y%m%d-%H%M%S")
        with open(self.output_dirpath / f"parameter_space_{date_time}.json", "w", encoding="utf-8") as f:
            json.dump(self.parameter_space, f, indent=4, ensure_ascii=False)

        self.iter_count = 0
        self.past_target_values = []

        self.optimizer = bo.BayesianOptimization(
            f=self.target_function,
            pbounds=self.parameter_space,
            verbose=2,
            random_state=42
        )
        self.logger = bo.logger.JSONLogger(self.output_dirpath / f"optimization_{date_time}.log")
        for event in [bo.Events.OPTIMIZATION_START, bo.Events.OPTIMIZATION_STEP, bo.Events.OPTIMIZATION_END]:
            self.optimizer.subscribe(event, self.logger)

    def load_optimizer_state(self, log_filepaths: list[Path | str], rel_tol: float | None = None, abs_tol: float = 0):
        log_filepaths = [Path(log_filepath) for log_filepath in log_filepaths]
        if rel_tol is None:
            rel_tol = glb.bo_target_value_equality_rel_tolerance

        print(f"Loading optimizer state from log file{'s'*(len(log_filepaths) > 1)} "
              f"{', '.join(log_filepath.name for log_filepath in log_filepaths)} ...")
        bo.util.load_logs(self.optimizer, log_filepaths)

        # Set up list of past target values, which should not include target values of iterations where the
        # reconstruction yielded zero buildings. Such values would be the exact half of another value in the list,
        # with a minimal error due to the float value in the log file being rounded at the last decimal place.
        self.past_target_values = [res["target"] for res in self.optimizer.res]
        self.past_target_values = [
            v for v in self.past_target_values
            if not any([
                isclose(v*2, v2, rel_tol=rel_tol, abs_tol=abs_tol)
                for v2 in self.past_target_values
            ])
        ]

        self.iter_count = len(self.optimizer.space)
        print(f"- Optimizer is now aware of {self.iter_count} observations.")
        print(f"- Of these, {len(self.past_target_values)} are actual target values.")

    def run(self, init_points: int | None = None, n_iter: int | None = None):
        init_points = self.init_points if init_points is None else init_points
        n_iter = self.n_iter if n_iter is None else n_iter

        print("Running reconstruction optimization ...\n")
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Polish results
        self._result = self.optimizer.max
        self._result["params"] = self.make_geoflow_integer_params_integer(self._result["params"])

        self._result["params"][glb.gf_param_plane_k] = self._result["params"][glb.gf_param_plane_min_points]

        print(f"\nFinished reconstruction optimization. Results:")
        print(f"{json.dumps(self._result, indent=2)}\n")

    def target_function(self, **kwargs):
        # BayesianOptimization calling this target_function will choose all parameters as random float values. Ensure
        # Geoflow integer parameters are rounded to integral values.
        scenario_settings = self.make_geoflow_integer_params_integer(kwargs)

        # Add the value of plane_k, which uses the identical value as plane_min_points
        scenario_settings[glb.gf_param_plane_k] = scenario_settings[glb.gf_param_plane_min_points]

        # Add the scenario name to the settings dictionary
        scenario_name = f"optim_{self.iter_count:04}"
        scenario_settings["scenario_name"] = scenario_name

        print(f"\nStarting optimization scenario '{scenario_name}' with the following settings:")
        print(json.dumps(scenario_settings, indent=2) + "\n")

        # Add a new scenario, passing the geoflow parameters as scenario_settings, which ensures they will be included
        # in the scenario's config["reconstruction_config"]["geoflow_parameters"]
        self.optim_experiment.add_scenario(scenario_settings)
        # Set the path containing the textfile that stores the path to the final point cloud identical to that of the
        # parent scenario
        self.optim_experiment.scenarios[scenario_name].config["cloud_processing_config"]["cloud_processing_output_dirpath"] =\
            self.optim_config["cloud_processing_config"]["cloud_processing_output_dirpath"]
        # Save the updated config of the scenario
        self.optim_experiment.scenarios[scenario_name].save_config()

        scenario = self.optim_experiment.scenarios[scenario_name]
        scenario.setup_reconstruction()
        scenario.prepare_reconstruction()
        scenario.run_reconstruction()

        # Handle the undesired case that the reconstruction yielded zero buildings
        if scenario.flag_zero_buildings_reconstructed:
            # Set the target_value to half of the lowest target value found so far, which should indicate to the
            # optimizer that this particular parameter combination is bad. (Not including target values from other cases
            # with zero buildings, which themselves were obtained by halving the lowest target value. Note that,
            # therefore, this target value is not added to the list of past_target_values.)
            # Note that the past_target_values are already computed such that lower values indicate poorer performance,
            # so dividing by two is sufficient and no additional distinction w.r.t. the target metric's optimum (min or
            # max) must be made.
            target_value = min(self.past_target_values) / 2

        else:
            scenario.setup_evaluation(lods=self.target_lod)
            scenario.run_evaluation(evaluator_selection=self.evaluators)

            # Print results of this iteration for all evaluators and metrics requested in the config
            print(f"\nFinished optimization scenario '{scenario_name}'. Results:")
            for evaluator_name, metrics in self.metrics.items():
                print(evaluator_name)
                for metric_name in metrics:
                    print(f"- {metric_name}: {scenario.evaluators[evaluator_name].summary_stats[metric_name]}")
            print()

            target_value = scenario.evaluators[self.target_evaluator].summary_stats[self.target_metric] * self.target_metric_optimum
            self.past_target_values.append(target_value)

        self.iter_count += 1
        return target_value

    @classmethod
    def make_geoflow_integer_params_integer(cls, params):
        # The following statement rounds values for integer parameters to integral values (half values up).
        return {k: (floor(v + 0.5) if k in glb.gf_integer_params else v) for k, v in params.items()}

    @property
    def result(self):
        return self._result


def generate_scenario(name: str, config: dict, default_config: dict) -> Scenario:
    full_config = deep_update(default_config, config)
    return Scenario(name, config)
