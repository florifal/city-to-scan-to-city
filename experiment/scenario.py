import copy
import json
import subprocess
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
from experiment.reconstruction import Reconstruction, ReconstructionError
from experiment.scene import Scene
from experiment.survey import Survey
from experiment.utils import get_last_line_from_file, deep_replace_in_string_values, point_spacing_along, \
    point_spacing_across, plural_s, get_newest_recon_optim_log_filepath, force_duplicate_probe_now
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
        self._flag_no_recon_output: bool | None = None

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

    def merge_clouds(self):
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
            print(f"\nPreparing to add noise to point cloud ...")
            filename = "clouds_merged_noise.laz"
            final_cloud_filepath = Path(self.cloud_processing_config["cloud_processing_output_dirpath"]) / filename

            noise_adder = CloudNoiseAdder(
                input_filepath=self.merged_point_cloud_filepath,
                output_filepath=final_cloud_filepath,
                std_horizontal=self.cloud_processing_config["std_horizontal_error"],
                std_vertical=self.cloud_processing_config["std_vertical_error"],
                crs=self.crs
            )

            print("Running noise adder ...")
            noise_adder.run()
        else:
            print(f"\nNot adding noise to point cloud - zero error set.")
            # If no noise is added, the final cloud is the merged cloud from the survey
            final_cloud_filepath = self.merged_point_cloud_filepath

        print("Writing the path to the processed point cloud to the text file ...")
        with open(self.textfile_final_cloud_path_filepath, "a", encoding="utf-8") as f:
            f.write(str(final_cloud_filepath) + "\n")

    def setup_reconstruction_optimization(self, **kwargs):
        self.recon_optim = ReconstructionOptimization(
            crs=self.config["crs"],
            config=self.recon_optim_config,
            scenario_config=self.config,
            **kwargs
        )
        self.recon_optim.setup()

    def prepare_reconstruction_optimization(self):
        self.recon_optim.output_dirpath.mkdir(exist_ok=True)
        self.recon_optim.prepare()

    def run_reconstruction_optimization(self, init_points: int | None = None, n_iter: int | None = None):
        """Don't use this. Use Experiment.optimize_reconstruction_params() instead."""
        self.recon_optim.run(init_points, n_iter)

        # Update scenario config with optimized Geoflow parameters and save it
        # optim_params = self.recon_optim.result["params"]
        # self.reconstruction_config["geoflow_parameters"].update(optim_params)
        # self.save_config()

    def select_optimal_reconstruction_optimization_scenario(
            self,
            params_filename: str = "best_parameter_set.json",
            update_config: bool = True,
            save_config: bool = True
    ):
        print("\nIdentifying best parameter set from reconstruction optimization ...")
        self.setup_reconstruction_optimization()
        self.recon_optim.load_optim_experiment()
        self.recon_optim.select_optimal_scenario(params_filename=params_filename)

        print(f"\nIdentified optimal scenario: {self.recon_optim.optimal_scenario.name}")

        best_params = self.recon_optim.optimal_scenario.geoflow_parameters

        if update_config:
            self.set_reconstruction_params(best_params, save_config)

    def clear_reconstruction_optimization(self):
        self.recon_optim.clear()

    def load_optimized_reconstruction_params(
            self,
            params_filename: str = "best_parameter_set.json",
            save_config: bool = True
    ):
        params_filepath = self.recon_optim_output_dirpath / params_filename
        print(f"\nReading optimized parameter set from file `{params_filepath.name}` ...")
        with open(params_filepath, "r") as f:
            best_params_json = json.load(f)

        # Handle two different formats in which the data can be stored in this file, the latter being the current one
        if len(best_params_json) == 1:
            best_optim_scenario_name = list(best_params_json.keys())[0]
            best_params = best_params_json[best_optim_scenario_name]
        elif len(best_params_json) == 2:
            best_optim_scenario_name = best_params_json["best_scenario"]
            best_params = best_params_json["best_parameter_set"]
        else:
            raise ValueError(f"Unable to get best reconstruction parameter set from file `{params_filename}`.")

        print(f"- Name of best optimization scenario: {best_optim_scenario_name}")
        print(f"- Best parameter values:\n{json.dumps(best_params, indent=2)}")

        self.set_reconstruction_params(best_params, save_config=save_config)

    def set_reconstruction_params(self, params: dict, save_config: bool = True):
        print("\nUpdating reconstruction config parameters ...")
        self.reconstruction_config["geoflow_parameters"] = params
        if save_config:
            print("- Saving updated config ...")
            self.save_config()

    def setup_reconstruction(self, add_geoflow_params: dict | None = None, geoflow_timeout: int | None = None):
        self.reconstruction = Reconstruction(
            crs=self.config["crs"],
            config=self.reconstruction_config,
            cloud_filepath=self.final_point_cloud_filepath,
            add_geoflow_params=add_geoflow_params,
            geoflow_timeout=geoflow_timeout
        )

    def prepare_reconstruction(self):
        self.reconstruction.output_dirpath.mkdir(exist_ok=True)
        self.reconstruction.prepare_config()
        self.reconstruction.setup_executor()

    def run_reconstruction(self):
        self.reconstruction.run()
        self.check_reconstruction_results()
        if self.flag_zero_buildings_reconstructed:
            print(f"\nWARNING: Reconstruction in {self.name} yielded zero buildings.\n")
        else:
            print(f"\nReconstruction in {self.name} yielded {self.n_buildings_reconstructed} buildings.")
        self.reconstruction_config["recon_execution_time"] = self.reconstruction.executor.execution_time

    def clear_reconstruction(self):
        self.reconstruction.clear()

    def setup_evaluation(self, lods: list[str] | str | None = None):
        if lods is None:
            lods = ["1.2", "1.3", "2.2"]
        elif isinstance(lods, str):
            lods = [lods]

        lods_no_points = [lod.replace(".", "") for lod in lods]

        # Experimentally switiching from GPKG to OBJ Wavefront for complexity evaluation because the 3DBAG GeoPackage
        # geometries are not triangulated, but the Geoflow output ones are, and therefore the complexity diff is wrong.
        # todo: Keep an eye on the results, update here correspondingly

        # input_complexity_evaluator = ComplexityEvaluator(
        #     output_base_dirpath=self.input_evaluation_dirpath,
        #     input_filepath=self.geopackage_input_filepath,
        #     lods=lods,
        #     gpkg_index_col_name="identificatie",
        #     gpkg_lod_layer_names=glb.geoflow_input_gpkg_lod_layer_names
        # )
        #
        # output_complexity_evaluator = ComplexityEvaluator(
        #     output_base_dirpath=self.output_evaluation_dirpath,
        #     input_filepath=self.geopackage_output_filepath,
        #     lods=lods,
        #     gpkg_index_col_name="OGRLoader.identificatie",
        #     gpkg_lod_layer_names=glb.geoflow_output_gpkg_lod_layer_names
        # )

        input_complexity_evaluator = ComplexityEvaluator(
            output_base_dirpath=self.input_evaluation_dirpath,
            input_filepath=[
                filepath for lod, filepath in zip(
                    ["1.2", "1.3", "2.2"],
                    [self.obj_input_lod12_filepath, self.obj_input_lod13_filepath, self.obj_input_lod22_filepath]
                ) if lod in lods],
            lods=lods
        )

        output_complexity_evaluator = ComplexityEvaluator(
            output_base_dirpath=self.output_evaluation_dirpath,
            input_filepath=[
                filepath for lod, filepath in zip(
                    ["1.2", "1.3", "2.2"],
                    [self.obj_output_lod12_filepath, self.obj_output_lod13_filepath, self.obj_output_lod22_filepath]
                ) if lod in lods],
            lods=lods
        )

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
                },
                bidirectional=False
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
                },
                bidirectional=True
            ),
            output_complexity_evaluator,
            ComplexityDifferenceEvaluator(
                output_base_dirpath=self.output_evaluation_dirpath,
                complexity_evaluator_1=input_complexity_evaluator,
                complexity_evaluator_2=output_complexity_evaluator,
                reevaluate_1=False,
                reevaluate_2=False
            ),
            # ComplexityEvaluator(
            #     output_base_dirpath=self.output_evaluation_dirpath,
            #     input_filepath=[self.obj_output_lod12_filepath,
            #                     self.obj_output_lod13_filepath,
            #                     self.obj_output_lod22_filepath],
            #     lods=["1.2", "1.3", "2.2"],
            #     ignore_meshes_with_zero_faces=True
            # ),
            HeightDifferenceEvaluator(
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
                save_filtered_point_clouds=False,
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

    def run_evaluation(self, evaluator_selection: list[str] | str | None = None, reevaluate: bool = True):
        if evaluator_selection is None:
            evaluator_selection = list(self.evaluators.keys())
        elif isinstance(evaluator_selection, str):
            evaluator_selection = [evaluator_selection]

        # Unless only the point density evaluator was requested, check if any buildings were reconstructed by calling
        # the property, which runs the GeopackageEvaluator if necessary.
        if evaluator_selection != [PointDensityDatasetEvaluator.name] and self.flag_zero_buildings_reconstructed:
            print(f"\nWARNING: Reconstruction in {self.name} yielded zero buildings. Skipping all evaluators "
                  f"that require reconstructed building models.")

        for evaluator_name in evaluator_selection:
            evaluate = True
            if not reevaluate:
                try:
                    self.evaluators[evaluator_name].load_results()
                except FileNotFoundError:
                    pass
                else:
                    print(f"\nEvaluation results for evaluator `{evaluator_name}` already exist. Will not reevaluate.")
                    evaluate = False

            if evaluate:
                if evaluator_name == PointDensityDatasetEvaluator.name:
                    self.evaluators[evaluator_name].run()
                else:
                    if not self.flag_zero_buildings_reconstructed:
                        self.evaluators[evaluator_name].run()

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

    def get_summary_statistics(
            self,
            evaluator_selection: list[str] | str | None = None,
            ignore_missing: bool = False
    ) -> dict:
        """

        :param evaluator_selection: Name or names of evaluators whose summary statistics to include
        :param ignore_missing: Ignore missing results of individual evaluators. (This method calls
         Evaluator.summary_stats, which conditionally calls Evaluator.compute_summary_stats(), which calls the property
         Evaluator.results, which attempts to load results if ._results is undefined, which fails if the CSV file with
         the results does not exist.)
        :return: Dictionary with every requested evaluator's summary metrics
        """
        if evaluator_selection is None:
            evaluator_selection = [name for name, evaluator in self.evaluators.items()]
        else:
            if isinstance(evaluator_selection, str):
                evaluator_selection = [evaluator_selection]

        summary_statistics = {}

        # Unless only the point density evaluator was requested, check if any buildings were reconstructed by calling
        # the property, which runs the GeopackageEvaluator if necessary.
        if evaluator_selection != [PointDensityDatasetEvaluator.name] and self.flag_zero_buildings_reconstructed:
            print(f"\nWARNING: Reconstruction in {self.name} yielded zero buildings. Skipping all evaluators "
                  f"that require reconstructed building models.")

        for name in evaluator_selection:
            try:
                if name == PointDensityDatasetEvaluator.name:
                    summary_statistics.update(self.evaluators[name].summary_stats)
                else:
                    if not self.flag_zero_buildings_reconstructed:
                        summary_statistics.update(self.evaluators[name].summary_stats)
            except FileNotFoundError as e:
                if not ignore_missing:
                    raise e
                else:
                    print(f"\nMissing output for evaluator `{name}` in scenario `{self.name}` ignored.")
            except Exception as e:
                raise e

        return summary_statistics

    def check_reconstruction_results(self):
        if not self.geopackage_output_filepath.is_file():
            self._flag_no_recon_output = True
            self._n_buildings_reconstructed = 0
        else:
            gpkg_eval = GeopackageBuildingsEvaluator(
                output_base_dirpath=self.reconstruction_output_dirpath,
                gpkg_filepath=self.geopackage_output_filepath,
                gpkg_layers={"2.2": self.geoflow_template_json["nodes"][f"OGRWriter-LoD22-3D"]["parameters"]["layername"]},
                id_col_name=glb.geoflow_output_cityjson_identifier_name
            )
            try:
                gpkg_eval.load_results()
            except FileNotFoundError:
                gpkg_eval.run()
            self._flag_no_recon_output = False
            self._n_buildings_reconstructed = int(gpkg_eval.results_df.loc["num_unique", "22"])

        self._flag_zero_buildings_reconstructed = (self._n_buildings_reconstructed == 0)

    def has_outlier_vertices(self):
        o = OBJFile(self.obj_output_lod22_filepath)
        return o.has_outlier_vertices(glb.recon_vertices_outlier_threshold)

    @property
    def flag_no_recon_output(self):
        if self._flag_no_recon_output is None:
            self.check_reconstruction_results()
        return self._flag_no_recon_output

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
    def recon_optim_output_dirpath(self):
        return Path(self.recon_optim_config["recon_optim_output_dirpath"])

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
        return Path(self.evaluation_config["input_cityjson_filepath"])

    @property
    def geopackage_input_filepath(self):
        try:
            return Path(self.evaluation_config["input_geopackage_filepath"])
        except KeyError:
            return ""

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
    def obj_output_lod22_filepath(self) -> Path:
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

    @property
    def geoflow_parameters(self):
        return self.reconstruction_config["geoflow_parameters"]

    def get_best_optim_scenario_name(self, params_filename: str = "best_parameter_set.json"):
        params_filepath = self.recon_optim_output_dirpath / params_filename
        try:
            with open(params_filepath, "r") as f:
                best_params_json = json.load(f)
        except FileNotFoundError:
            print(f"\nFile `{params_filename}` for scenario `{self.name}` does not exist")
            return ""
        else:
            if len(best_params_json) == 1:
                best_optim_scenario_name = list(best_params_json.keys())[0]
            elif len(best_params_json) == 2:
                best_optim_scenario_name = best_params_json["best_scenario"]
            else:
                raise ValueError(f"Unable to get best reconstruction parameter set from file `{params_filename}`.")
            return best_optim_scenario_name


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
        self._results: pd.DataFrame | None = None

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
    def visualization_dirpath(self):
        return self.dirpath / "09_visualization"

    @property
    def scene_xml_filepath(self):
        return self.scene_dirpath / f"{self.name}_scene.xml"

    @property
    def summary_stats(self) -> pd.DataFrame:
        if self._summary_stats is None:
            try:
                self.load_summary_stats()
            except FileNotFoundError:
                self.compute_summary_statistics()
        return self._summary_stats

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            try:
                self.load_final_results()
            except FileNotFoundError:
                self.compute_final_results()
        return self._results

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

    def load_summary_stats(self, filename: str = "summary_statistics.csv"):
        print(f"Loading summary statistics from file `{filename}` ...")
        self._summary_stats = pd.read_csv(self.evaluation_dirpath / filename)
        self._summary_stats = self._summary_stats.set_index("scenario")

    def load_final_results(self, filename: str = "evaluation_results.csv"):
        print(f"Loading evaluation results from file `{filename}` ...")
        self._results = pd.read_csv(self.evaluation_dirpath / filename)
        self._results = self._results.set_index(["scenario", "identificatie"])

    def setup(self):
        """Call all setup functions for directories, scene, scenario configs, and scenarios"""
        self.setup_directories()
        self.setup_scene()
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
        self.recon_optim_dirpath.mkdir(exist_ok=True)
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

        # Scenario-specific file paths
        flight_path_xml_filepath = self.survey_dirpath / scenario_name / "flight_path.xml"
        survey_xml_filepath = self.survey_dirpath / scenario_name / (scenario_name + "_survey.xml")
        survey_output_dirpath = self.survey_dirpath  # HELIOS creates subfolders: /scenario_name/date_time
        cloud_processing_output_dirpath = self.cloud_processing_dirpath / scenario_name
        recon_optim_output_dirpath = self.recon_optim_dirpath / scenario_name
        reconstruction_output_dirpath = self.reconstruction_dirpath / scenario_name  # reconstruct.json has output/
        evaluation_output_dirpath = self.evaluation_dirpath  # / scenario_name  # todo:remove
        visualization_dirpath = self.visualization_dirpath / scenario_name

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
        update_config_item(config, "visualization_dirpath", str(visualization_dirpath))

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
        print("\nLoading experiment configuration ...")
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
            print(f"- {len(e)} scenarios")

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

    def parse_scenarios_argument(
            self,
            scenarios: list[str] | str | list[int] | int | list[Scenario] | Scenario | None = None
    ):
        if scenarios is None:
            scenarios = self.scenarios
        else:
            if isinstance(scenarios, str):
                scenarios = [scenarios]
            elif isinstance(scenarios, int):
                scenarios = [scenarios]
            elif isinstance(scenarios, Scenario):
                scenarios = [scenarios]

            if isinstance(scenarios[0], str):
                scenarios = {name: scenario for name, scenario in self.scenarios.items() if name in scenarios}
            elif isinstance(scenarios[0], int):
                scenarios = {self[i].name: self[i] for i in scenarios}
            elif isinstance(scenarios[0], Scenario):
                scenarios = {s.name: s for s in scenarios}
            else:
                raise ValueError("Scenario selection must either be a scenario name (str), index (int), or Scenario "
                                 f"instance, a list of one of these, or None, but received {type(scenarios)}.")
        return scenarios

    def run_steps(
            self,
            steps: Callable[[Scenario], Any] | list[Callable[[Scenario], Any]],
            scenarios: list[str] | str | list[int] | int | list[Scenario] | Scenario | None = None,
            *args,
            **kwargs
    ):
        """Run a Scenario step (a method of the class) for all or a selection of this Experiment's scenarios.

        Additional arguments of the method that is to be run can be given as *args or **kwargs."""
        scenarios = self.parse_scenarios_argument(scenarios)

        if not isinstance(steps, list):
            steps = [steps]

        t00 = time.time()

        for name, s in scenarios.items():
            for step in steps:
                print(f"\nRunning '{step.__name__}' for {name} ...")
                t0 = time.time()
                step(s, *args, **kwargs)
                t1 = time.time()
                print(f"\nFinished '{step.__name__}' for {name} after {str(timedelta(seconds=t1 - t0))}.")

        t01 = time.time()
        print(f"\nFinished running {len(steps)} step{plural_s(steps)} "
              f"for {len(scenarios)} scenario{plural_s(scenarios)} "
              f"after {str(timedelta(seconds=t01 - t00))}.")

    def compute_summary_statistics(
            self,
            evaluator_selection: list[str] | str | None = None,
            save_to_filename: str | None = "summary_statistics.csv",
            scenarios: list[str] | str | list[int] | int | list[Scenario] | Scenario | None = None,
            ignore_missing: bool = False
    ):
        scenarios = self.parse_scenarios_argument(scenarios)

        print(f"\nComputing summary statistics of {len(scenarios)} scenario{plural_s(scenarios)} "
              f"in experiment {self.name} ...")
        rows = []

        for name, s in scenarios.items():
            target_density = get_config_item(s.config, "target_density")
            error_level = get_config_item(s.config, "error_level")
            pulse_freq_hz = get_config_item(s.config, "pulse_freq_hz")
            scan_freq_hz = get_config_item(s.config, "scan_freq_hz")
            scan_angle_deg = get_config_item(s.config, "scan_angle_deg")
            altitude = get_config_item(s.config, "altitude")
            velocity = get_config_item(s.config, "velocity")
            recon_execution_time = get_config_item(s.config, "recon_execution_time", not_found_error=False)

            cols = {
                "scenario": s.name,
                "target_density": target_density,
                "error_level": error_level,
                "pulse_freq_hz": pulse_freq_hz,
                "scan_freq_hz": scan_freq_hz,
                "point_spacing_along": point_spacing_along(velocity, scan_freq_hz),
                "point_spacing_across": point_spacing_across(altitude, scan_angle_deg, pulse_freq_hz, scan_freq_hz),
                "std_horizontal_error": get_config_item(s.config, "std_horizontal_error"),
                "std_vertical_error": get_config_item(s.config, "std_vertical_error"),
                "recon_execution_time": recon_execution_time
            }

            cols.update(s.config["reconstruction_config"]["geoflow_parameters"])
            cols.update(s.get_summary_statistics(evaluator_selection, ignore_missing=ignore_missing))

            rows.append(cols)

        self._summary_stats = pd.DataFrame(rows).set_index("scenario")

        if save_to_filename is not None:
            self._summary_stats.to_csv(self.evaluation_dirpath / save_to_filename)

    def compute_final_results(
            self,
            evaluator_selection: list[str] | str | None = None,
            save_to_filename: str | None = "evaluation_results.csv",
            scenarios: list[str] | str | list[int] | int | list[Scenario] | Scenario | None = None,
    ):
        scenarios = self.parse_scenarios_argument(scenarios)

        print(f"\nConcatenating final evaluation results of {len(scenarios)} scenario{plural_s(scenarios)} "
              f"in experiment {self.name} ...")

        results_scenarios = {}

        for name, s in scenarios.items():
            results_final = pd.concat(
                [s.evaluators[evaluator_name].results[s.evaluators[evaluator_name].final_columns]
                 for evaluator_name in evaluator_selection],
                axis=1
            )
            results_scenarios[name] = results_final

        # Concatenating a dictionary with DataFrames as values creates a multi-index from the dictionary keys and the
        # DataFrames' indices
        self._results = pd.concat(results_scenarios)
        self._results.index.names = ["scenario", "identificatie"]
        if save_to_filename is not None:
            self._results.to_csv(self.evaluation_dirpath / save_to_filename)

    def optimize_reconstruction_params(
            self,
            sequence: list[int],
            init_points: int = 0,
            n_iter: int = 50,
            recon_timeout: int = glb.geoflow_recon_optim_timeout_default,
            probe_parameter_sets: list[dict] | dict | None = None,
            continue_from_previous_run: bool = False,
            previous_run_log_filepaths: list[Path | str] | Path | str | None = None
    ):
        n_probe_parameter_sets = 0
        if probe_parameter_sets is not None:
            if not isinstance(probe_parameter_sets, list):
                probe_parameter_sets = [probe_parameter_sets]
            n_probe_parameter_sets = len(probe_parameter_sets)

        print("\nSequential optimization of reconstruction parameters.")
        print(f"- Sequence: {', '.join([str(i) for i in sequence])}")

        for c, i in enumerate(sequence):
            print(f"\nOptimization of reconstruction parameters for scenario `{self[i].name}`.")

            self[i].setup_reconstruction_optimization(recon_timeout=recon_timeout)

            # Find and load optimizer logs of any scenarios whose indices were provided for this purpose
            load_scenario_optimizer_states = self[i].config["recon_optim_config"]["load_scenario_optimizer_states"]

            if load_scenario_optimizer_states:  # Check if it's not an empty list [], or otherwise False
                if not isinstance(load_scenario_optimizer_states, list):
                    load_scenario_optimizer_states = list(load_scenario_optimizer_states)

                load_optimizer_log_filepaths = []

                print(f"\nFinding optimization logs of scenario{plural_s(load_scenario_optimizer_states)}: "
                      f"{', '.join([self[j].name for j in load_scenario_optimizer_states])}")
                for j in load_scenario_optimizer_states:
                    newest_optim_log_filepath = get_newest_recon_optim_log_filepath(self[j].recon_optim_output_dirpath)
                    load_optimizer_log_filepaths.append(newest_optim_log_filepath)
                    print(f"- {self[j].name}: {newest_optim_log_filepath.name}")

                # Load optimizer logs before calling prepare_reconstruction_optimization(), which sets up the logger, so
                # the loaded optimization results are not written to this scenario's log. Allow duplicate points, so
                # that observations of additional parameter sets from multiple optim experiments are loaded as well.
                self[i].recon_optim.load_optimizer_state(
                    load_optimizer_log_filepaths,
                    update_iter_count=False,
                    load_past_target_values=True,
                    force_allow_duplicate_points=True
                )

            self[i].prepare_reconstruction_optimization()

            this_init_points = init_points
            this_n_iter = n_iter

            # The parameter sets that should be probed additionally, according to the argument `probe_parameter_sets`,
            # should not be probed for the first scenario in the sequence if a previous optimization is continued.
            # For every following scenario (c > 0), it should be probed.
            if c > 0 or not continue_from_previous_run:
                if probe_parameter_sets is not None:
                    print(f"\nProbing {n_probe_parameter_sets} additional parameter "
                          f"set{plural_s(probe_parameter_sets)} before starting default optimization ...")
                    force_duplicate_probe_now(self[i].recon_optim.optimizer, probe_parameter_sets)
                    print("\nFinished probing additional parameter sets before starting default optimization.")

            # The case that this is the first scenario in the sequence and a previous optimization should be continued:
            # We try to load the optimizer state from the log of a previous run. As opposed to the optimizer states
            # loaded from other scenarios, this should be done after calling prepare_reconstruction_optimization(),
            # which sets up the logger, so that the results of the loaded iterations from a previous run are stored in
            # this run's log as well.
            else:
                if previous_run_log_filepaths is None:
                    previous_run_log_filepaths = \
                        [get_newest_recon_optim_log_filepath(self[i].recon_optim_output_dirpath)]
                else:
                    if not isinstance(previous_run_log_filepaths, list):
                        previous_run_log_filepaths = [previous_run_log_filepaths]
                print(f"\nLoading optimization log from previous run: {str(previous_run_log_filepaths[0].name)}")
                # Allow duplicate points is required so that observations of additional parameters sets are loaded, even
                # if such observations were already loaded via load_optimizer_log_filepaths from other optim experiments
                self[i].recon_optim.load_optimizer_state(
                    previous_run_log_filepaths,
                    update_iter_count=True,
                    load_past_target_values=True,
                    force_allow_duplicate_points=True
                )
                n_previous_iterations = self[i].recon_optim.iter_count
                print(f"\nFound {n_previous_iterations} successfully finished previous iterations in the optimization"
                      f" logs.")

                # The rare case that a previous run crashed before all additional parameter sets were probed
                if n_previous_iterations < n_probe_parameter_sets:
                    n_probe_parameter_sets_remaining = n_probe_parameter_sets - n_previous_iterations
                    probe_parameter_sets_remaining = probe_parameter_sets[n_previous_iterations:]
                    print(f"Not all {n_probe_parameter_sets} additional parameter sets were probed in the previous "
                          f"run.")

                    print(f"\nProbing the remaining {n_probe_parameter_sets_remaining} additional parameter "
                          f"set{plural_s(probe_parameter_sets_remaining)} before starting default optimization ...")
                    force_duplicate_probe_now(self[i].recon_optim.optimizer, probe_parameter_sets_remaining)
                    print("\nFinished probing additional parameter sets before starting default optimization.")

                else:
                    # Assuming that for the previous optimization run the same amount of additional parameter sets were
                    # probed, these should not count towards the number of observations that is subtracted from the number
                    # of init_points and n_iter, because these are always probed additionally and independently of the
                    # parameter sets provided in n_probe_parameter_sets.
                    n_observations_loaded = n_previous_iterations - n_probe_parameter_sets
                    print(f"This inludes {n_probe_parameter_sets} additional probe{plural_s(probe_parameter_sets)} "
                          f"and {n_observations_loaded} regular iteration{plural_s(n_observations_loaded)}.")

                    if n_observations_loaded <= this_init_points:
                        this_init_points -= n_observations_loaded
                    else:
                        this_n_iter -= (n_observations_loaded - this_init_points)
                        this_init_points = 0

                    print(f"\nContinuing optimization with adapted iteration numbers:")
                    print(f"- init_points: {this_init_points}")
                    print(f"- n_iter: {this_n_iter}")

            self[i].run_reconstruction_optimization(init_points=this_init_points, n_iter=this_n_iter)

            self[i].clear_reconstruction_optimization()

            # write an independent function to identify the "most optimal" result from all optimization steps
            # since it's custom tailored to the case when optimizing for RMS min dist and then taking IOU and n_faces
            # into account, it likely does not belong into any class in particular
            # NO! I can't use IOU because it takes too long. too bad. so rms min dist and complexity only.


class ReconstructionOptimization:

    def __init__(
            self,
            crs: str,
            config: dict,
            scenario_config: dict,
            init_points: int = 5,
            n_iter: int = 10,
            recon_timeout: int = glb.geoflow_recon_optim_timeout_default
    ):
        """

        :param crs:
        :param config:
        :param scenario_config:
        :param init_points:
        :param n_iter:
        :param recon_timeout: Time (s) before reconstruction is aborted and parameter set penalized. Does not apply to
        the first iteration, because penalty depends on previous observations of the target metric.
        """
        self.crs = crs
        self.config = config
        self.optim_config = copy.deepcopy(scenario_config)

        self.output_dirpath = Path(self.config["recon_optim_output_dirpath"])
        # The reconstruction optimization experiment will be located in the parent scenario's directory for
        # reconstruction optimization, in a subdirectory with the scenario's name
        self.experiment_name = self.optim_config["scenario_name"]
        self.experiment_dirpath = self.output_dirpath.parent

        # Remove the prefix "range_" from each parameter name in the parameter space
        self.parameter_space = {k.split("_", 1)[1]: v for k, v in self.config["parameter_space"].items()}
        self.parameter_sets_to_probe = self.config["parameter_sets_to_probe"]

        self.evaluators = self.config["recon_optim_evaluators"]
        self.metrics = self.config["recon_optim_metrics"]
        self.target_lod = self.config["recon_optim_target_lod"]
        self.target_evaluator = self.config["recon_optim_target_evaluator"]
        self.target_metric = self.config["recon_optim_target_metric"]
        self.target_metric_optimum = {"max": 1, "min": -1}[self.config["recon_optim_target_metric_optimum"]]
        self.target_metric_adaptive_penalty = self.config["recon_optim_target_metric_adaptive_penalty"]
        self.target_metric_penalty_value = self.config["recon_optim_target_metric_penalty_value"]

        self.target_evaluator_2 = self.config["recon_optim_target_evaluator_2"]
        self.target_metric_2 = self.config["recon_optim_target_metric_2"]
        self.target_metric_2_threshold = self.config["recon_optim_target_metric_2_threshold"]
        self.target_metric_2_penalty = self.config["recon_optim_target_metric_2_penalty"]
        self.target_metric_2_penalty_mode = self.config["recon_optim_target_metric_2_penalty_mode"]

        self.optim_experiment: Experiment | None = None
        self.optimizer: bo.BayesianOptimization | None = None
        self.logger: bo.logger.JSONLogger | None = None
        self.iter_count = 0
        self.init_points = init_points
        self.n_iter = n_iter
        self.past_target_values = []

        # Add parameters for the reconstruction to skip unnecessary LODs
        self.add_geoflow_params = {f"skip_lod{lod.replace('.','')}": True
                                   for lod in ["1.2", "1.3", "2.2"] if lod != self.target_lod}
        self.recon_timeout = recon_timeout

        self._result: dict | None = None
        self.optimal_scenario: Scenario | None = None

    def clear(self):
        # todo: Can all of this be cleared safely? What can safely be cleared?
        self.optim_experiment = None
        self.optimizer = None
        self.logger = None

    def load_optim_experiment(self):
        self.optim_experiment = Experiment.load(self.experiment_dirpath / self.experiment_name, load_scenarios=True)

    def setup(self):
        print("\nSetting up reconstruction optimization ...")

        # For the reconstruction optimization experiment, most parameters should be unchanged compared to the parent
        # scenario. However, the reconstruction parameters must be changed, because the reconstruction output must be
        # saved in the parent scenario's directory for reconstruction optimization, and the building footprints used
        # for the optimization may be different from (e.g., a subset of) the parent scenarios footprints.
        self.optim_config["reconstruction_config"]["building_footprints_filepath"] = self.config["optimization_footprints_filepath"]
        self.optim_config["reconstruction_config"]["building_footprints_sql"] = self.config["optimization_footprints_sql"]
        self.optim_config["evaluation_config"]["input_cityjson_filepath"] = self.config["optimization_cityjson_filepath"]

        print("- Setting up experiment ...")
        self.optim_experiment = Experiment(
            name=self.experiment_name,
            dirpath=self.experiment_dirpath,
            default_config=self.optim_config
        )
        print("- Setting up optimizer ...")
        self.optimizer = bo.BayesianOptimization(
            f=self.target_function,
            pbounds=self.parameter_space,
            verbose=2,
            allow_duplicate_points=False,
            # random_state=42
        )
        # `allow_duplicate_points`
        # For optimization based on another scenario's optimization results, where the same default parameter set should
        # be probed again for the new scenario but is already present in the other scenario's optimizer log, this option
        # must be set to True to make the optimizer evaluate the parameter set again. Otherwise,
        # `random_state`:
        # Note that setting a seed for a seeded random state means that the sequence of n = `init_points` random samples
        # (generated by optimizer._space.random_sample()) will always be identical. If loading previous optimization
        # results from a log-file (e.g. with self.load_optimizer_state()), these identical random samples will already
        # have been probed during the previous optimizer run, which means they are going to be skipped now when
        # optimizer._space.probe() is called because, per default, optimizer._space._allow_duplicate_points is False.
        # If `allow_duplicate_points` is True it may be even more important to set no `random_state`, otherwise the same
        # random samples will be drawn and evaluated again in case of an aborted optimization is continued.

    def prepare(self):
        print("\nPreparing reconstruction optimization ...")
        # Instead of Experiment.setup_directories(), create only those directories required here
        self.optim_experiment.dirpath.mkdir(exist_ok=True)
        self.optim_experiment.settings_dirpath.mkdir(exist_ok=True)
        self.optim_experiment.reconstruction_dirpath.mkdir(exist_ok=True)
        self.optim_experiment.evaluation_dirpath.mkdir(exist_ok=True)
        self.optim_experiment.save()

        date_time = datetime.today().strftime("%y%m%d-%H%M%S")
        with open(self.output_dirpath / f"parameter_space_{date_time}.json", "w", encoding="utf-8") as f:
            json.dump(self.parameter_space, f, indent=4, ensure_ascii=False)

        # Disabled resetting iter_count and past_target_values in case load_optimizer_state() was called before
        # prepare() to load logs before the logger is set up, and the variables are already updated.
        # todo: Consider setting up the logger separately, so prepare() can happen before any logs are loaded, then the
        # two variables are definitely reset, and the timing of the logger setup can be decided independently.
        # self.iter_count = 0
        # self.past_target_values = []

        print("- Setting up logger ...")
        self.logger = bo.logger.JSONLogger(self.output_dirpath / f"optimization_{date_time}.log")
        # for event in [bo.Events.OPTIMIZATION_START, bo.Events.OPTIMIZATION_STEP, bo.Events.OPTIMIZATION_END]:
        self.optimizer.subscribe(bo.Events.OPTIMIZATION_STEP, self.logger)

        if len(self.parameter_sets_to_probe) > 0:
            print(f"- Adding {len(self.parameter_sets_to_probe)} parameter sets to be probed by the optimizer ...")
            for params in self.parameter_sets_to_probe:
                self.optimizer.probe(params, lazy=True)

    def load_optimizer_state(
            self,
            log_filepaths: list[Path | str] | Path | str,
            update_iter_count: bool = True,
            load_past_target_values: bool = True,
            force_allow_duplicate_points: bool = False,
            rel_tol: float | None = None,
            abs_tol: float = 0
    ):
        """

        :param log_filepaths:
        :param update_iter_count: Update the iteration counter to consider the loaded scenarios
        :param force_allow_duplicate_points: Allow duplicate points to be loaded temporarily
        :param rel_tol:
        :param abs_tol:
        :return:
        """

        # Process arguments passed
        if not isinstance(log_filepaths, list):
            log_filepaths = [log_filepaths]
        log_filepaths = [Path(log_filepath) for log_filepath in log_filepaths]
        if rel_tol is None:
            rel_tol = glb.bo_target_value_equality_rel_tolerance

        print(f"\nLoading optimizer state from log file{'s'*(len(log_filepaths) > 1)}: "
              f"{', '.join(log_filepath.name for log_filepath in log_filepaths)} ...")

        allow_duplicate_points = None
        if force_allow_duplicate_points:
            allow_duplicate_points = self.optimizer._allow_duplicate_points

            self.optimizer._allow_duplicate_points = True
            self.optimizer._space._allow_duplicate_points = True

        n_observations_before_loading = len(self.optimizer.space)

        # Bug in bayes_opt: Wrong type hint indicating str, bytes, or PathLike, actually needs list of Paths.
        bo.util.load_logs(self.optimizer, log_filepaths)

        n_observations_after_loading = len(self.optimizer.space)
        n_observations_loaded = n_observations_after_loading - n_observations_before_loading

        if force_allow_duplicate_points:
            self.optimizer._allow_duplicate_points = allow_duplicate_points
            self.optimizer._space._allow_duplicate_points = allow_duplicate_points

        if update_iter_count:
            self.iter_count += n_observations_loaded

        if update_iter_count or load_past_target_values:
            # Set up list of past target values, which should not include target values of iterations where the
            # reconstruction yielded zero buildings. Such values would be the exact half of another value in the
            # list, with a minimal error due to the float value in the log file being rounded at the last decimal
            # place.
            # Currently, this is done in all cases, even if the observations do not go into the iter_count, i.e.,
            # there are probably no optimization scenarios for these observations, perhaps if they were loaded
            # from a different optimization. It might make sense to change this and to move this into the `if
            # update_iter_count` clause. In this case, however, only the new entries in self.optimizer.res (which
            # were just loaded, and not any previously loaded entries) should be added to the past_target_values.
            # Since this would take some time to figure out how to do, currently, all loaded target values are added,
            # even those from calls to this function with update_iter_count=False, on a later call with _=True.
            self.past_target_values = [res["target"] for res in self.optimizer.res]
            self.past_target_values = [
                v for v in self.past_target_values
                # exclude values that the exact double (or half) of another value, because they are penalty values
                if not any([
                    isclose(v * 2 ** self.target_metric_optimum, v2, rel_tol=rel_tol, abs_tol=abs_tol)
                    for v2 in self.past_target_values
                ])
                # in case the baseline values which were doubled (or halved) are not present any more, additionally skip
                # any values that occur more than once, because they are also almost certainly penalty values
                and not sum([
                    v == v2 for v2 in self.past_target_values
                ]) > 1  # larger than one because there is always one equality with the value itself
                and not v == self.target_metric_penalty_value
            ]

        print(f"- Optimizer is now aware of {n_observations_loaded} more observations, "
              f"{n_observations_after_loading} total.")
        print(f"- Of these, {len(self.past_target_values)} are actual target values.")

    def run(self, init_points: int | None = None, n_iter: int | None = None):
        init_points = self.init_points if init_points is None else init_points
        n_iter = self.n_iter if n_iter is None else n_iter

        print("\nRunning reconstruction optimization ...")
        t0 = time.time()
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Polish results
        self._result = self.optimizer.max
        self._result["params"] = self.make_geoflow_integer_params_integer(self._result["params"])

        self._result["params"][glb.gf_param_plane_k] = self._result["params"][glb.gf_param_plane_min_points]

        t1 = time.time()
        print(f"\nFinished reconstruction optimization experiment `{self.experiment_name}` "
              f"after {str(timedelta(seconds=t1-t0))}.")
        print(f"Best results in terms of `{self.target_metric}`:")
        print(json.dumps(self._result, indent=2))

    def target_function(self, **kwargs):
        t0 = time.time()
        # BayesianOptimization calling this target_function will choose all parameters as random float values. Ensure
        # Geoflow integer parameters are rounded to integral values.
        scenario_settings = self.make_geoflow_integer_params_integer(kwargs)

        # Add the value of plane_k, which uses the identical value as plane_min_points
        scenario_settings[glb.gf_param_plane_k] = scenario_settings[glb.gf_param_plane_min_points]

        # Add the scenario name to the settings dictionary
        scenario_name = f"optim_{self.iter_count:04}"
        scenario_settings["scenario_name"] = scenario_name

        print(f"\nStarting optimization scenario '{scenario_name}' with the following settings:")
        print(json.dumps(scenario_settings, indent=2))

        # Add a new scenario, passing the geoflow parameters as scenario_settings, which ensures they will be included
        # in the scenario's config["reconstruction_config"]["geoflow_parameters"]
        self.optim_experiment.add_scenario(scenario_settings)
        # Set the path containing the textfile that stores the path to the final point cloud identical to that of the
        # parent scenario
        self.optim_experiment.scenarios[scenario_name].config["cloud_processing_config"]["cloud_processing_output_dirpath"] =\
            self.optim_config["cloud_processing_config"]["cloud_processing_output_dirpath"]
        # Save the updated config of the scenario
        self.optim_experiment.scenarios[scenario_name].save_config()

        # Timeout should not apply to the first iteration, because penalty in case of a timeout depends on previous
        # observations of the target metric.
        # Change of mind: In case an additonal parameter set never finishes, this freezes the program. We can instead
        # activate loading of past_target_values from adjacent experiments and use them in a case of timeout. Should be
        # close enough to what we expect for the running experiment.
        # timeout = None if self.iter_count == 0 else self.recon_timeout

        scenario = self.optim_experiment.scenarios[scenario_name]
        scenario.setup_reconstruction(
            add_geoflow_params=self.add_geoflow_params,  # skip unnecessary LODs
            geoflow_timeout=self.recon_timeout  # changed back from timeout, see above.
        )
        scenario.prepare_reconstruction()

        timeout = False
        try:
            scenario.run_reconstruction()
        except subprocess.TimeoutExpired:
            print(f"\nReconstruction timed out after {self.recon_timeout} seconds. Penalizing parameter set.")
            timeout = True
        else:
            if scenario.flag_zero_buildings_reconstructed:
                print("\nZero building models were reconstructed. Penalizing parameter set.")
            else:
                print("\nChecking for outlier vertices ...")
                if scenario.has_outlier_vertices():
                    raise ReconstructionError("The reconstructed building models have at least one outlier vertex.")
                else:
                    print("- None found.")

        # Handle the undesired case that the reconstruction yielded zero buildings
        if timeout or scenario.flag_zero_buildings_reconstructed:
            if self.target_metric_adaptive_penalty:
                # Set the target_value to half of the lowest target value found so far, which should indicate to the
                # optimizer that this particular parameter combination is bad. (Not including target values from other cases
                # with zero buildings, which themselves were obtained by halving the lowest target value. Note that,
                # therefore, this target value is not added to the list of past_target_values.)
                # The following paragrahp is wrong:
                # Note that the past_target_values are already computed such that lower values indicate poorer performance,
                # so dividing by two is sufficient and no additional distinction w.r.t. the target metric's optimum (min or
                # max) must be made.
                # Correct is: If we are minimizing a function, i.e., maximizing its negative value, we must multiply the
                # (worst past) target value by two instead of dividing it by two to indicate a poor performance.
                # Still only half correct: This only applies for target metrics that are always positive per default, such
                # that if they must be minimized they are turned negative for the optimizer to maximize them.
                target_value = min(self.past_target_values) / 2 ** self.target_metric_optimum
            else:
                target_value = self.target_metric_penalty_value * self.target_metric_optimum

        else:
            scenario.setup_evaluation(lods=self.target_lod)
            scenario.run_evaluation(evaluator_selection=self.evaluators)

            # Print results of this iteration for all evaluators and metrics requested in the config
            print(f"\nResults of optimization scenario `{scenario_name}`:")
            for evaluator_name, metrics in self.metrics.items():
                print(evaluator_name)
                for metric_name in metrics:
                    print(f"- {metric_name}: {scenario.evaluators[evaluator_name].summary_stats[metric_name]}")

            target_value = scenario.evaluators[self.target_evaluator].summary_stats[self.target_metric]

            # Experimental settings for a second target metric whose relative magnitude penalizes the first one
            # todo: Test, potentially remove
            if self.target_metric_2 != "":
                print(f"\nChecking second target metric `{self.target_metric_2}` ...")
                target_value_2 = scenario.evaluators[self.target_evaluator_2].summary_stats[self.target_metric_2]
                if target_value_2 > self.target_metric_2_threshold:
                    penalty = (target_value_2 - self.target_metric_2_threshold) * self.target_metric_2_penalty
                    print(f"- Exceeds threshold of {self.target_metric_2_threshold}.")
                    print(f"- Penalizing target value by {penalty} in mode `{self.target_metric_2_penalty_mode}` ...")
                    if self.target_metric_2_penalty_mode == "add":
                        target_value += penalty
                    elif self.target_metric_2_penalty_mode == "mult":
                        target_value *= penalty
                else:
                    print("- Does not exceed threshold; no penalty applied.")

            target_value *= self.target_metric_optimum

            self.past_target_values.append(target_value)

        t1 = time.time()
        print(f"\nFinished optimization scenario `{scenario_name}` after {str(timedelta(seconds=(t1-t0)))}.")
        print(f"- Target value: {target_value}")

        self.iter_count += 1
        return target_value

    def select_optimal_scenario(
            self,
            range_from_best: float = 0.1,
            n_buildings_min: int = 80,
            params_filename: str = "best_parameter_set.json"
    ):
        additional_evaluators = ["complexity", "geoflow_output"]
        reevaluate = False

        print(f"\nSelecting optimal scenario from all optimization scenarios "
              f"in optimization experiment `{self.experiment_name}`.")
        print("Running additional evaluators for the optimization experiment ...")

        self.optim_experiment.run_steps(Scenario.setup_evaluation, lods=self.target_lod)
        self.optim_experiment.run_steps(
            Scenario.run_evaluation,
            evaluator_selection=additional_evaluators,
            reevaluate=reevaluate
        )
        self.optim_experiment.compute_summary_statistics(
            evaluator_selection=[self.target_evaluator, *additional_evaluators],
            save_to_filename="summary_statistics_for_optim_selection.csv",
        )
        stats = self.optim_experiment.summary_stats

        # Pre-select only those optimization scenarios that yielded a minimum number of reconstructed buildings
        stats = stats[stats[f"gpkg_unique_{self.target_lod.replace('.', '')}"].astype(int) >= n_buildings_min]

        optimal_scenario_name = stats[
            stats.rms_min_dist_22_mean <= stats.rms_min_dist_22_mean.min() * (1+range_from_best)
            ].n_faces_22_mean.idxmin()

        print(f"\nScenario with lowest face number within {range_from_best*100}% range of optimum: {optimal_scenario_name}")

        self.optimal_scenario = self.optim_experiment.scenarios[optimal_scenario_name]

        print(f"- Saving best parameter set as `{params_filename}` ...")
        with open(self.output_dirpath / params_filename, "w", encoding="utf-8") as f:
            json.dump({"best_scenario": self.optimal_scenario.name,
                       "best_parameter_set": self.optimal_scenario.geoflow_parameters},
                      f, indent=4, ensure_ascii=False)

    def select_optimal_scenario_multirank(self, params_filename: str = "best_parameter_set.json"):
        target_metric_optimum = self.config["recon_optim_target_metric_optimum"]
        target_metric_weight = 1

        # Settings for the first selection step:
        # - Evaluators to run additionally (can also include just interesting ones)
        # - Metrics from these evaluators to use for the first selection step
        # - For each of the metrics, if their optimum is the minimum ("min") or maximum ("max")
        # - For each of the metrics, their weight in computing the weighted rank sum
        # - Name of the rank column on which the first selection will be based
        # - Settings for the approach to select the best scenarios. Set one to a value and the others to None.
        additional_evaluators = ["complexity", "complexity_diff", "geoflow_output"]
        additional_metrics = ["n_faces_22_mean"]
        additional_metrics_optima = ["min"]
        additional_metrics_weights = [2]
        metric_to_select_best_step_1 = "rank_sum"

        method_to_select_best = ""  # is later filled automatically by which of the following is not None
        select_best_in_range_threshold = None  # 0.03
        select_best_n = 10
        select_best_quantile = None  # 0.2

        # Settings for the second selection step, like above. Note that the second selection step uses these metrics
        # in addition to the metrics specified for the first selection step.
        additional_evaluators_2 = ["iou_3d"]
        additional_metrics_2 = ["iou_22_mean"]
        additional_metrics_optima_2 = ["max"]
        additional_metrics_weights_2 = [1]
        metric_to_select_best_step_2 = "rank_sum_weighted"
        second_metric_to_select_best_step_2 = "rank_n_faces_22_mean"  # In case there are multiple best scenarios

        reevaluate = False  # Run evaluators again if a CSV file with their results already exists? Tested individually.

        print(f"\nSelecting optimal scenario from all optimization scenarios "
              f"in optimization experiment `{self.experiment_name}`.")
        print("Running additional evaluators for the optimization experiment ...")

        # --------------------------------------------------------------------------------------------------------------
        # EVALUATE 1: Run the additional evaluators for the optimization experiment

        self.optim_experiment.run_steps(Scenario.setup_evaluation, lods=self.target_lod)
        self.optim_experiment.run_steps(
            Scenario.run_evaluation,
            evaluator_selection=additional_evaluators,
            reevaluate=reevaluate
        )
        self.optim_experiment.compute_summary_statistics(
            evaluator_selection=[self.target_evaluator, *additional_evaluators],
            save_to_filename="summary_statistics_all.csv"
        )
        stats = self.optim_experiment.summary_stats

        # --------------------------------------------------------------------------------------------------------------
        # RANK 1: Compute ranks for first set of metrics: Target metric and additonal metrics

        print("\nComputing ranks of target and additional metrics ...")

        columns_to_rank = [self.target_metric, *additional_metrics]
        optima = [target_metric_optimum, *additional_metrics_optima]
        ascending = [True if o == "min" else False for o in optima]
        for c, a in zip(columns_to_rank, ascending):
            stats["rank_" + c] = stats[c].rank(ascending=a)

        for c, a in zip(columns_to_rank, ascending):
            if a:
                stats["rangerank_" + c] = (stats[c] - stats[c].min()) / (stats[c].max() - stats[c].min())
            else:
                stats["rangerank_" + c] = (stats[c].max() - stats[c]) / (stats[c].max() - stats[c].min())

        print("Computing rank sum and weighted rank sum ...")

        # Compute the (weighted) sum of the ranked columns (only if all are not NA per min_count), then compute their
        # rank while keeping any NA as such per na_option.
        ranked_columns = ["rank_" + c for c in columns_to_rank]
        stats["rank_sum"] = (stats[ranked_columns]
                             .sum(axis=1, min_count=len(ranked_columns))
                             .rank(na_option="keep"))
        ranked_columns_weights = [target_metric_weight, *additional_metrics_weights]
        stats["rank_sum_weighted"] = pd.DataFrame.from_dict(
            {c: w * stats[c] for c, w in zip(ranked_columns, ranked_columns_weights)}
        ).sum(axis=1, min_count=len(ranked_columns)).rank(na_option="keep")

        stats.to_csv(self.output_dirpath / "summary_statistics_ranked.csv")

        # --------------------------------------------------------------------------------------------------------------
        # SELECT 1: First selection of the best scenarios: Filter scenarios by the criterion for the best results in
        # terms of the target metric.
        # Here, one could instead also do a first aggregation step and compute a sum rank from the target metric and the
        # first stage additional evaluator(s), and then select the best scenarios according to this sum rank.

        print(f"\nFiltering for best scenarios by selected criterion in terms of `{metric_to_select_best_step_1}` ...")

        tm = self.target_metric
        if select_best_in_range_threshold is not None:
            method_to_select_best = "best_range_" + str(select_best_in_range_threshold)
            stats_best = stats[
                stats[tm] < (
                        stats[tm].min() + select_best_in_range_threshold * (stats[tm].max() - stats[tm].min())
                )
            ]
        elif select_best_n is not None:
            method_to_select_best = "best_n_" + str(select_best_n)
            stats_best = stats[stats[metric_to_select_best_step_1] <= select_best_n]
        elif select_best_quantile is not None:
            method_to_select_best = "best_quantile_" + str(select_best_quantile)
            stats_best = stats[stats[metric_to_select_best_step_1] <= stats[metric_to_select_best_step_1].quantile(q=select_best_quantile)]

        print(f"- Criterion: {method_to_select_best}")
        stats_best_names = list(stats_best.index)

        # --------------------------------------------------------------------------------------------------------------
        # EVALUATE 2: Run the additional evaluator(s) for the second filtering of the best scenarios (likely this is
        # the IOU evaluator)

        print("\nRunning second stage additional evaluators for the pre-selected optimization scenarios ...")
        print(f"- Selected: {', '.join(stats_best_names)}")

        self.optim_experiment.run_steps(
            Scenario.run_evaluation,
            scenarios=stats_best_names,
            evaluator_selection=additional_evaluators_2,
            reevaluate=False
        )
        self.optim_experiment.compute_summary_statistics(
            evaluator_selection=[self.target_evaluator, *additional_evaluators, *additional_evaluators_2],
            save_to_filename="summary_statistics_best.csv",
            scenarios=stats_best_names
        )
        stats_best_eval = self.optim_experiment.summary_stats

        # --------------------------------------------------------------------------------------------------------------
        # RANK 2: Compute ranks of target metric and both first and second stage additional evaluators.
        # Here, one could instead also re-use the ranks computed earlier across all scenarios, which might deliver
        # different results in the final selection step.

        print("\nComputing ranks of target and first and second stage additional metrics ...")

        columns_to_rank.extend(additional_metrics_2)
        optima.extend(additional_metrics_optima_2)
        ascending = [True if o == "min" else False for o in optima]
        for c, a in zip(columns_to_rank, ascending):
            stats_best_eval["rank_" + c] = stats_best_eval[c].rank(ascending=a)

        for c, a in zip(columns_to_rank, ascending):
            if a:
                stats_best_eval["rangerank_" + c] = ((stats_best_eval[c] - stats_best_eval[c].min()) /
                                                     (stats_best_eval[c].max() - stats_best_eval[c].min()))
            else:
                stats_best_eval["rangerank_" + c] = ((stats_best_eval[c].max() - stats_best_eval[c]) /
                                                     (stats_best_eval[c].max() - stats_best_eval[c].min()))

        print("Computing rank sum and weighted rank sum ...")

        # Compute the (weighted) sum of the ranked columns (only if all are not NA per min_count), then compute their
        # rank while keeping any NA as such per na_option.
        ranked_columns = ["rank_" + c for c in columns_to_rank]
        stats_best_eval["rank_sum"] = (stats_best_eval[ranked_columns]
                                       .sum(axis=1, min_count=len(ranked_columns))
                                       .rank(na_option="keep"))
        ranked_columns_weights = [target_metric_weight, *additional_metrics_weights, *additional_metrics_weights_2]
        stats_best_eval["rank_sum_weighted"] = pd.DataFrame.from_dict(
            {c: w * stats_best_eval[c] for c, w in zip(ranked_columns, ranked_columns_weights)}
        ).sum(axis=1, min_count=len(ranked_columns)).rank(na_option="keep")

        stats_best_eval.to_csv(self.output_dirpath / ("summary_statistics_" + method_to_select_best + "_ranked.csv"))

        # --------------------------------------------------------------------------------------------------------------
        # SELECT 2: Select the best scenario from the subset of the best scenarios using the final ranking including all
        # additional evaluators

        print(f"\nSelecting best scenario according to `{metric_to_select_best_step_2}` ...")

        n_best_scenarios = len(
            stats_best_eval[stats_best_eval[metric_to_select_best_step_2] == stats_best_eval[metric_to_select_best_step_2].min()].index
        )

        if n_best_scenarios == 1:
            optimal_scenario_name = stats_best_eval[metric_to_select_best_step_2].idxmin()
        else:
            print(f"- Found multiple according to first metric. Checking `{second_metric_to_select_best_step_2}` ...")
            optimal_scenario_name = stats_best_eval[
                stats_best_eval[metric_to_select_best_step_2] == stats_best_eval[metric_to_select_best_step_2].min()
            ][
                second_metric_to_select_best_step_2
            ].idxmin()

        print(f"- Best scenario: {optimal_scenario_name}")

        self.optimal_scenario = self.optim_experiment.scenarios[optimal_scenario_name]

        print(f"- Saving best parameter set as `{params_filename}` ...")
        with open(self.output_dirpath / params_filename, "w", encoding="utf-8") as f:
            json.dump({"best_scenario": self.optimal_scenario.name,
                       "best_parameter_set": self.optimal_scenario.geoflow_parameters},
                      f, indent=4, ensure_ascii=False)

    @classmethod
    def make_geoflow_integer_params_integer(cls, params):
        # The following statement rounds values for integer parameters to integral values (half values up).
        return {k: (floor(v + 0.5) if k in glb.gf_integer_params else v) for k, v in params.items()}

    @property
    def result(self):
        return self._result

    @property
    def optimal_params(self):
        return self.optimal_scenario.geoflow_parameters


def generate_scenario(name: str, config: dict, default_config: dict) -> Scenario:
    full_config = deep_update(default_config, config)
    return Scenario(name, config)
