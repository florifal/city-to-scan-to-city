# from __future__ import annotations
# from typing_extensions import Self
import json
import time
import datetime
import shutil
import pyhelios
import pdal
import copy
import collections.abc
import numpy as np
from datetime import timedelta
from xml.etree import ElementTree as eT
from scipy.spatial import KDTree

from experiment.scene_part import ScenePartOBJ, ScenePartTIFF
from experiment.utils import execute_subprocess, crs_url_from_epsg
from experiment.xml_generator import SceneGenerator, FlightPathGenerator, SurveyGenerator, parse_xml_with_comments
from experiment.evaluator import *

pyhelios.loggingVerbose2()
pyhelios.setDefaultRandomnessGeneratorSeed("42")


class Scene:

    def __init__(self, filepath: str, xml_id: str, name: str, scene_parts: list[dict]):
        self.filepath = filepath
        self.xml_id = xml_id
        self.name = name
        self.scene_part_dicts = scene_parts

        self.scene_parts = []
        self.scene_generator: SceneGenerator | None = None

    @property
    def filepath_with_id(self):
        """Return the XML filepath including the id of the scene tag."""
        return self.filepath + "#" + self.xml_id

    def create_scene_parts(self):
        """Create SceneParts from the list of dictionaries provided and append them to the list of SceneParts"""
        for d in self.scene_part_dicts:
            if d["type"].lower() == "obj":
                self.scene_parts.append(
                    ScenePartOBJ(filepath=d["filepath"], up_axis=d["up_axis"])
                )
            elif d["type"].lower() in ["tif", "tiff"]:
                self.scene_parts.append(
                    ScenePartTIFF(filepath=d["filepath"], mat_file=d["material_filepath"], mat_name=d["material_name"])
                )
            else:
                raise ValueError(f"Scene part type {d['type']} is not supported.")

    def create_scene_xml(self):
        """Generate the scene XML based on the provided list of scene part dictionaries"""
        self.scene_generator = SceneGenerator(self.filepath, self.xml_id, self.name)
        self.create_scene_parts()
        self.scene_generator.add_scene_parts(self.scene_parts)
        self.scene_generator.write_file()


class FlightPath:

    def __init__(self, filepath: str, config: dict | None = None):
        self.filepath = filepath
        self.config = config

        self.flight_path_generator: FlightPathGenerator | None = None

    def create_flight_path_xml(self):
        self.flight_path_generator = FlightPathGenerator(
            filepath=self.filepath,
            bbox=self.config["bbox"],
            spacing=self.config["spacing"],
            altitude=self.config["altitude"],
            velocity=self.config["velocity"],
            flight_pattern=self.config["flight_pattern"],
            trajectory_time_interval=self.config["trajectory_time_interval"],
            always_active=bool(self.config["always_active"]),
            scanner_settings_id=self.config["scanner_settings_id"]
        )
        self.flight_path_generator.compute_waypoints()
        self.flight_path_generator.write_file()


class Survey:

    def __init__(
            self,
            filepath: str,
            output_dirpath: str,
            crs: str,
            config: dict | None = None,
            scene_filepath: str = ""
    ):
        self.filepath = filepath
        self.output_dirpath = output_dirpath
        self.crs = crs
        self.config = config

        self.survey_generator_config = config["survey_generator_config"]
        self.flight_path_config = config["flight_path_config"]
        self.survey_executor_config = config["survey_executor_config"]
        self.cloud_merge_config = config["cloud_merge_config"]
        self.cloud_error_config = config["cloud_error_config"]

        # If the Survey is created by a Scenario, the scene file path may be passed to this Survey as argument. If the
        # Survey is created independently of a Scenario, the scene file path should be in the survey generator config.
        if scene_filepath == "":
            self.scene_filepath = self.survey_generator_config["scene_xml_filepath_with_id"]
        else:
            self.scene_filepath = scene_filepath

        # Variables defined later
        self.flight_path: FlightPath | None = None
        self.survey_generator: SurveyGenerator | None = None
        self.survey_executor: SurveyExecutor | None = None
        self.output_clouds_filepaths: list[str] | None = None
        self.output_clouds_dirpath: str | None = None
        self.cloud_merger: CloudMerger | None = None
        self.output_cloud_filepath: Path | None = None
        self.noise_adder: CloudNoiseAdder | None = None

        # Paths to write two text documents into, containing (1) a list of the paths to all output point clouds files
        # and (2) a single path to the final merged point cloud file. They should be put into the subdirectory bearing
        # the survey name that is created automatically by HELIOS within the specified output directory.
        self.textfile_output_clouds_list_filepath: str = str(Path(
            self.output_dirpath, self.survey_generator_config["survey_name"], "output_clouds_filepaths.txt"
        ))
        self.textfile_merged_cloud_path_filepath: str = str(Path(
            self.output_dirpath, self.survey_generator_config["survey_name"], "merged_cloud_filepath.txt"
        ))

    def create_flight_path(self):
        """Create the flight path XML for the survey from the values provided in the flight path config

        This method is not called create_flight_path_xml, because only the class FlightPath itself will create a
        FlightPathGenerator object, which is responsible for creating the XML file."""
        self.flight_path = FlightPath(
            filepath=self.flight_path_config["flight_path_xml_filepath"],
            config=self.flight_path_config
        )
        self.flight_path.create_flight_path_xml()

    def create_survey_xml(self):
        """Create the survey XML with all settings as provided in the survey generator config"""
        # Combine the platforms and scanners XML filepaths with the corresponding IDs provided in the config
        platform_filepath_with_id = glb.helios_platforms_filepath + "#" + self.survey_generator_config["platform_id"]
        scanner_filepath_with_id = glb.helios_scanners_filepath + "#" + self.survey_generator_config["scanner_id"]

        self.survey_generator = SurveyGenerator(
            filepath=self.filepath,

            name=self.survey_generator_config["survey_name"],
            scene_filepath=self.scene_filepath,
            platform_filepath=platform_filepath_with_id,
            scanner_filepath=scanner_filepath_with_id,

            scanner_settings_id=self.survey_generator_config["scanner_settings_id"],
            scanner_active=self.survey_generator_config["scanner_settings_active"],
            pulse_freq=self.survey_generator_config["pulse_freq_hz"],
            scan_freq=self.survey_generator_config["scan_freq_hz"],
            scan_angle=self.survey_generator_config["scan_angle_deg"],
            detector_accuracy=self.survey_generator_config["detector_settings_accuracy"],

            flight_path_filepath=self.flight_path_config["flight_path_xml_filepath"]
        )
        # Create the survey ElementTree and write the XML file
        self.survey_generator.write_file()

    def setup_executor(self):
        """Set up the SurveyExecutor for this survey"""
        self.survey_executor = SurveyExecutor(
            survey_filepath=self.filepath,
            output_dirpath=self.output_dirpath,
            config=self.survey_executor_config
        )

    def run(self):
        """Using the SurveyExecutor, build and run the simulation, then write some output"""
        self.survey_executor.build_simulation()
        self.output_clouds_filepaths = self.survey_executor.run()
        self.output_clouds_dirpath = str(Path(self.output_clouds_filepaths[0]).parent)

        # Here, the list of output point clouds files is written to a file in the output directory. Another file with
        # the path to the merged point cloud file is written only after execution of the CloudMerger.
        with open(self.textfile_output_clouds_list_filepath, "w", encoding="utf-8") as f:
            f.writelines([fp + "\n" for fp in self.output_clouds_filepaths])

    def setup_merger(self):
        """Prepare the CloudMerger (UniqueKeeperMerger as of now) for the merging of the output point clouds"""
        # Get the flight path root element from the flight path XMl file that was generated by self.flight_path:
        # FlightPath(), or from the survey XML file as generated by self.survey_generator if the flight path XML does
        # not exist anymore for any reason.
        # Another option would be to use the attribute waypoints from FlightPathGenerator().
        # This is needed for the UniqueKeeperMerger to read each leg's coordinates.
        if Path(self.flight_path_config["flight_path_xml_filepath"]).is_file():
            flight_path_element = parse_xml_with_comments(self.flight_path_config["flight_path_xml_filepath"]).getroot()
        elif Path(self.filepath).is_file():
            flight_path_element = parse_xml_with_comments(self.filepath).getroot().find("survey")
        else:
            raise FileNotFoundError(
                "Neither the flight path nor the survey XML file exist to read the leg coordinates from."
            )

        # If the CloudMerger is being set up without the survey having been run first, this block will attempt to read
        # the filepaths of the clouds that should be merged from a previous survey run from the expected text file
        # location.
        if self.output_clouds_filepaths is None:
            try:
                with open(self.textfile_output_clouds_list_filepath, "r") as f:
                    self.output_clouds_filepaths = [fp[:-1] for fp in f.readlines()]  # remove \n
            except FileNotFoundError as e:
                print("Filepaths of clouds to be merged are unknown without prior survey run, and text file to list "
                      "them from prior survey run does not exist.")
                raise FileNotFoundError(e.errno, e.strerror, self.textfile_output_clouds_list_filepath)
            self.output_clouds_dirpath = str(Path(self.output_clouds_filepaths[0]).parent)

        self.cloud_merger = UniqueKeeperMerger(
            cloud_filepaths=self.output_clouds_filepaths,
            flight_path_element=flight_path_element,
            parallel_dimension="Y",
            output_dirpath=self.output_clouds_dirpath,
            crs=self.crs
        )

    def merge_clouds(self):
        """Run the cloud merger and write a textfile with the output path"""
        # Since we are using the UniqueKeeperMerger, whose run() method calls several of its own functions, the
        # execution of self.cloud_merger.run() calls implicitly:
        #
        # self.cloud_merger.get_parallel_trajectory_coords()
        # self.cloud_merger.compute_separation_coords()
        # self.cloud_merger.filter_clouds()
        # merged_output_cloud_filepath = self.cloud_merger.merge_clouds()
        self.cloud_merger.run()
        self.output_cloud_filepath = self.cloud_merger.merged_output_cloud_filepath

        # Write the path to the merged point cloud file to a text file
        with open(self.textfile_merged_cloud_path_filepath, "w", encoding="utf-8") as f:
            f.write(str(self.output_cloud_filepath) + "\n")

    def add_noise(self):
        # Only run CloudNoiseAdder if a non-zero horizontal or vertical error is provided in the settings
        if self.cloud_error_config["std_horizontal_error"] != 0 or self.cloud_error_config["std_vertical_error"] != 0:
            output_cloud_filename = self.output_cloud_filepath.stem + "_noise" + self.output_cloud_filepath.suffix
            self.output_cloud_filepath = self.output_cloud_filepath.parent / output_cloud_filename

            self.noise_adder = CloudNoiseAdder(
                input_filepath=self.cloud_merger.merged_output_cloud_filepath,
                output_filepath=self.output_cloud_filepath,
                std_horizontal=self.cloud_error_config["std_horizontal_error"],
                std_vertical=self.cloud_error_config["std_vertical_error"],
                crs=self.crs
            )

            self.noise_adder.run()

            with open(self.textfile_merged_cloud_path_filepath, "a", encoding="utf-8") as f:
                f.write(str(self.output_cloud_filepath) + "\n")


class SurveyExecutor:

    def __init__(self, survey_filepath: str, output_dirpath: str = "", config: dict | None = None):
        self.survey_filepath = survey_filepath
        if output_dirpath == "":
            self.output_dirpath = glb.helios_output_dirpath
        else:
            self.output_dirpath = output_dirpath
        self.config = config

        self.sim_builder: pyhelios.SimulationBuilder | None = None
        self.sim_build: pyhelios.SimulationBuild | None = None
        self.output_cloud_filepaths: list[str] | None = None

    def build_simulation(self):
        print("Building survey simulation ...")
        self.sim_builder = pyhelios.SimulationBuilder(
            surveyPath=self.survey_filepath,
            assetsDir=glb.helios_assets_dirpath,
            outputDir=self.output_dirpath
        )
        self.sim_builder.setLasOutput(self.config["las_output"])
        self.sim_builder.setZipOutput(self.config["zip_output"])
        self.sim_builder.setNumThreads(self.config["num_threads"])
        self.sim_builder.setCallbackFrequency(0)  # 0: Run without callback
        self.sim_builder.setFinalOutput(True)    # Output point cloud as numpy array
        self.sim_builder.setExportToFile(True)    # Output point cloud to disk
        self.sim_builder.setRebuildScene(False)   # Do not rebuild scene files if exist
        self.sim_builder.setPlatformNoiseDisabled(True)  # Enable/disable platform noise (check platform XML)
        self.sim_builder.setLegNoiseDisabled(True)  # Enable/disable leg noise (should not do anything)

        self.sim_build = self.sim_builder.build()

    def run(self) -> list[str]:
        print("Starting survey simulation ...")
        time_start = time.time()
        self.sim_build.start()

        while self.sim_build.isRunning():
            time.sleep(1)
            print(f"\rSurvey simulation running. Time elapsed: {int(time.time() - time_start)} s", end="")

        print(f"\nSurvey simulation has finished after {datetime.timedelta(seconds=int(time.time() - time_start))}.")

        sim_output = self.sim_build.join()
        self.output_cloud_filepaths = [str(Path(p)) for p in sim_output.filepaths]

        return self.output_cloud_filepaths


class CloudMerger:

    def __init__(self, cloud_filepaths: list[str]):
        self.cloud_filepaths = cloud_filepaths
        self.merged_output_cloud_filepath: Path | None = None

    def run(self) -> str:
        # Subclass method returns path to merged output cloud file
        return ""


class UniqueKeeperMerger(CloudMerger):

    def __init__(
            self,
            cloud_filepaths: list[str],
            flight_path_element: eT.Element,
            parallel_dimension: str,
            output_dirpath: str,
            crs: str
    ):
        """UniqueKeeperMerger

        :param cloud_filepaths: List of filepaths to all point clouds to be merged.
        :param flight_path_element: The element from the survey XML file that is the parent to all flight path legs
        :param parallel_dimension: Specify the dimension in which the flight path legs are parallel. Can be 'X' or 'Y'.
        :param output_dirpath: Path to the directory in which the filtered and merged point clouds will be stored.
        :param crs: String specifying the CRS as in "epsg:0000"
        """
        super().__init__(cloud_filepaths)
        self.flight_path_element = flight_path_element
        self.parallel_dimension = parallel_dimension
        self.output_dirpath = output_dirpath
        self.crs = crs

        self.parallel_trajectory_coords: list[float] = []
        self.separation_coords: list[float] = []

        self.cloud_filtered_filepaths = [
            Path(self.output_dirpath, Path(filepath).stem + "_filtered.laz").as_posix()
            for filepath in self.cloud_filepaths
        ]

        self.point_cloud_arrays = None
        self.columns_out: list[str] | None = None
        self.merged_output_cloud_filepath = Path(self.output_dirpath, "clouds_merged.laz")

    def run(self):
        # This method is a candidate for deletion because it only adds another layer.
        self.get_parallel_trajectory_coords()
        self.compute_separation_coords()
        self.filter_clouds()
        self.merge_clouds()

    def get_parallel_trajectory_coords(self):
        print("Reading the coordinates in which the trajectories are parallel ...")
        for leg_element in self.flight_path_element.findall("leg"):
            platform_settings_element = leg_element.find("platformSettings")
            scanner_settings_element = leg_element.find("scannerSettings")
            # Currently, only the short legs on which the scanner is not active have an attribute active="false",
            # whereas the legs with active scanner don't have this attribute, because the scanner is active per default
            # as specified in the scanner settings template element. Therefore, the following condition checks for the
            # absence of the attribute "active" to identify legs with active scanner.
            if "active" not in scanner_settings_element.attrib.keys():
                # Read parallel_dimension coordinate attribute from platformSettings element within leg element
                coord = float(platform_settings_element.attrib[self.parallel_dimension.lower()])
                self.parallel_trajectory_coords.append(coord)

    def compute_separation_coords(self):
        print("Computing center coordinates between adjacent parallel trajectory coordinates ...")
        # For each pair of coordinates, compute their distance
        parallel_trajectory_distances = [
            self.parallel_trajectory_coords[i] - self.parallel_trajectory_coords[i - 1]
            for i in range(1, len(self.parallel_trajectory_coords))
        ]

        # The separation between the point clouds of two swaths should be performed exactly along their center line.
        # The separation coordinates are therefore at the middle of the distance between each pair of flight path
        # coordinates.
        self.separation_coords = [
            self.parallel_trajectory_coords[i] + 0.5 * parallel_trajectory_distances[i]
            for i in range(len(parallel_trajectory_distances))
        ]

        # To be able to use the same PDAL pipeline snippet for each swath, add separation coordinates at the
        # beginning and end of the list.
        self.separation_coords.insert(0, 0)
        self.separation_coords.append(2 * self.separation_coords[-1])

    def filter_clouds(self):
        n_clouds = len(self.cloud_filepaths)

        # Pipeline for reading all individual swath's point clouds and computing within-cloud nearest neighbor distances
        readers = [pdal.Reader(filepath, nosrs=True, default_srs=self.crs) for filepath in self.cloud_filepaths]
        # pipelines = [reader | pdal.Filter.nndistance(mode="kth", k=1) for reader in readers]
        pipelines = [reader | pdal.Filter.nndistance(mode="avg", k=4) for reader in readers]  # todo: decide on settings

        print("Reading input point clouds and computing within-cloud nearest-neighbor distance ...")
        for p, pipeline in enumerate(pipelines):
            print(f"Processing swath point cloud {p+1} of {n_clouds} ...")
            n_points = pipeline.execute()
            print(f"- Processed {n_points} points.")

        # Get point clouds as numpy structured arrays
        point_cloud_arrays = [pipeline.arrays[0] for pipeline in pipelines]

        # Prepare list of columns that should be contained in the final output
        undesired_columns = ['GpsTime', 'echo_width', 'fullwaveIndex', 'hitObjectId', 'heliosAmplitude']
        self.columns_out = [col_name for col_name, _ in point_cloud_arrays[0].dtype.descr if
                       col_name not in undesired_columns]

        print("Computing mean within-cloud nearest-neighbor distances ...")
        # Compute mean of within-cloud nearest neighbor distances for each swath
        mean_NN_distances = [np.mean(arr["NNDistance"]) for arr in point_cloud_arrays]
        print(mean_NN_distances)

        print("Computing k-d trees ...")
        # Create a simple XYZ numpy array containing only the coordinates for each swath, and k-d trees.
        numpy_point_clouds = [np.column_stack((arr['X'], arr['Y'], arr['Z'])) for arr in point_cloud_arrays]
        kd_trees = [KDTree(pc) for pc in numpy_point_clouds]

        for p, point_array in enumerate(point_cloud_arrays):
            print(f"Processing swath point cloud {p+1} of {n_clouds} ...")

            adjacent_swath_ids = []
            if p > 0: adjacent_swath_ids.append(p - 1)
            if p < len(pipelines) - 1: adjacent_swath_ids.append(p + 1)
            print(f"- has {len(adjacent_swath_ids)} adjacent swath{(len(adjacent_swath_ids) > 1) * 's'}.")

            print("- Computing between-cloud nearest-neighbor distance to all adjacent swaths ...")
            # Compute between-cloud nearest neighbor distance to all adjacent swaths
            for a in adjacent_swath_ids:
                # Use the adjacent swath's k-d tree to get the distance to the nearest neighbor in the adjacent cloud
                kth_distance, kth_index = kd_trees[a].query(
                    numpy_point_clouds[p], k=1, distance_upper_bound=10, workers=-1
                )

                # Create a new structured array that includes the between-cloud nearest neighbor distance
                new_col_name = f'CC_NNDistance_{a}'
                dtype_new = np.dtype(
                    point_array.dtype.descr + [(new_col_name, '<f8')])  # Append a new column to the dtypes
                point_array_new = np.zeros(point_array.shape, dtype=dtype_new)  # Create an empty array with zeros
                for col_name, _ in point_array.dtype.descr:
                    point_array_new[col_name] = point_array[col_name]  # Copy values from existing array
                point_array_new[new_col_name] = kth_distance  # Insert new values: Between-cloud NN distances

                point_array = point_array_new

            print("- Filtering points in overlapping area ...")
            # Create a filter expression to filter the overlapping area, and a list of new fields to include in the
            # output cloud. First, filter areas lying beyond the separation boundary between adjacent clouds.
            filter_expression = (f"({self.parallel_dimension.upper()} >= {self.separation_coords[p]} && "
                                 f"{self.parallel_dimension.upper()} <= {self.separation_coords[p + 1]})")
            filter_expression += " || ("
            extra_dims = "NNDistance=float64"
            # Second, add a filter to keep points beyond the separation boundary if they do not duplicate (but
            # complement) points in the adjacent cloud
            for c, a in enumerate(adjacent_swath_ids):
                if c > 0: filter_expression += " && "
                filter_expression += f"CC_NNDistance_{a} > {mean_NN_distances[a]}"
                extra_dims += f", CC_NNDistance_{a}=float64"
            filter_expression += ")"
            print(f"- filter_expression: {filter_expression}")
            print(f"- extra_dims: {extra_dims}")

            # Pipeline to apply the filter expression and save the filtered cloud
            pipeline = pdal.Filter.expression(expression=filter_expression).pipeline(point_array)
            pipeline |= pdal.Writer(filename=self.cloud_filtered_filepaths[p], minor_version=4, a_srs=self.crs,
                                    compression=True, extra_dims=extra_dims)
            n_points = pipeline.execute()
            print(f"- Remaining number of points: {n_points}")

            # Update this swath's structured array with the new, filtered one
            point_cloud_arrays[p] = pipeline.arrays[0]

        print("Finished filtering all point clouds.")
        self.point_cloud_arrays = point_cloud_arrays

    def merge_clouds(self):
        print("Concatenating filtered point clouds ...")
        # Concatenate the filtered point clouds into one structured array
        point_cloud_arrays_stacked = np.hstack([arr[self.columns_out] for arr in self.point_cloud_arrays])
        print(f"- Total number of points: {len(point_cloud_arrays_stacked)}")

        print("Writing merged point clouds to output location ...")
        # Pipeline to save the final (filtered and merged) point cloud
        pipeline = pdal.Writer(
            filename=str(self.merged_output_cloud_filepath), minor_version=4, a_srs=self.crs, compression=True
        ).pipeline(point_cloud_arrays_stacked)
        n_points = pipeline.execute()
        print(f"- Written number of points: {n_points}")


class CloudNoiseAdder:

    def __init__(
            self,
            input_filepath: Path | str,
            output_filepath: Path | str,
            std_horizontal: float,
            std_vertical: float,
            crs: str
    ):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.std_horizontal = std_horizontal
        self.std_vertical = std_vertical
        self.crs = crs

        self.n_points: int | None = None

    def run(self):
        print("Adding noise to point cloud ...")
        self.n_points = self.add_noise_to_cloud(
            self.input_filepath,
            self.output_filepath,
            self.std_horizontal,
            self.std_vertical,
            self.crs
        )

    @classmethod
    def add_noise_to_cloud(cls, input_filepath, output_filepath, std_horizontal, std_vertical, crs):
        print("Reading input point cloud ...")
        print(f"- {input_filepath}")
        reader = pdal.Reader(str(input_filepath), nosrs=True, default_srs=crs)
        pipeline = reader.pipeline()
        n_points = pipeline.execute()
        point_array = pipeline.arrays[0]

        print("Adding random error ...")
        print(f"- std horizontal: {std_horizontal}")
        print(f"- std vertical: {std_vertical}")
        # Random number generator
        rng = np.random.default_rng()

        error_x = rng.normal(loc=0.0, scale=std_horizontal, size=n_points)
        error_y = rng.normal(loc=0.0, scale=std_horizontal, size=n_points)
        error_z = rng.normal(loc=0.0, scale=std_vertical, size=n_points)

        axes = ["X", "Y", "Z"]
        errors = dict(zip(axes, [error_x, error_y, error_z]))
        for axis in axes:
            point_array[axis] = point_array[axis] + errors[axis]

        print("Writing output point cloud ...")
        print(f"- {output_filepath}")
        writer = pdal.Writer(output_filepath, minor_version=4, a_srs=crs, compression=True)
        pipeline = writer.pipeline(point_array)
        n_points = pipeline.execute()

        return n_points


class Reconstruction:

    def __init__(self, crs: str, config: dict, cloud_filepath: str = ""):
        self.crs = crs
        self.config = config

        self.output_dirpath = self.config["reconstruction_output_dirpath"]
        self.cloud_filepath = cloud_filepath if cloud_filepath != "" else self.config["point_cloud_filepath"]
        self.geoflow_json_filepath = Path(self.output_dirpath, "reconstruct.json")
        self.config_toml_filepath = Path(self.output_dirpath, "config.toml")
        self.geoflow_log_filepath = Path(self.output_dirpath, "geoflow_log.txt")

        self.executor: ReconstructionExecutor | None = None

    def prepare_config(self):
        if not Path(self.output_dirpath).is_dir():
            Path(self.output_dirpath).mkdir()

        # Copy the Geoflow reconstruction template JSON and rename it to reconstruct.json
        shutil.copy2(glb.geoflow_reconstruct_template_filepath, self.output_dirpath)
        Path(
            self.output_dirpath,
            Path(glb.geoflow_reconstruct_template_filepath).name
        ).rename(self.geoflow_json_filepath)
        # Also copy the nested JSON that it includes to the output directory
        shutil.copy2(glb.geoflow_reconstruct_nested_filepath, self.output_dirpath)

        # Prepare the values for the config.toml file
        config_toml_json = {
            "GF_PROCESS_CRS": self.crs,
            "input_pointcloud_crs_wkt": self.crs,
            "output_crs_wkt": self.crs,
            "output_cj_referenceSystem": crs_url_from_epsg(self.crs),
            "input_footprint": self.config["building_footprints_filepath"],
            "building_identifier": self.config["building_identifier"],
            "input_pointcloud": self.cloud_filepath
        }
        # Write the config.toml
        with open(self.config_toml_filepath, "w", encoding="utf-8") as f:
            for key, value in config_toml_json.items():
                f.write(f"{key}='{value}'\n")

        # copy reconstruct_template.json to output_dir
        # (because if calling geof with argument -w, the output directories will be considered
        # to be relative to the directory that contains the json. this also makes it unnecessary
        # to include all the different output paths into the config.toml)
        #
        # create config.toml to include the following params:
        # - GF_PROCESS_CRS, input_pointcloud_crs_wkt, output_crs_wkt <- all crs
        # - output_cj_referenceSystem <- update URL with epsg code from crs
        # - input_footprint
        # - building_identifier
        # - input_pointcloud
        #
        # or potentially: even simply edit the JSON, should not be much harder, only less convenient
        # to check the settings later on

        pass

    def setup_executor(self):
        # or combine this method with run()

        # Option --workdir makes sure "output" directory is created at the location of the reconstruction.json
        self.executor = ReconstructionExecutor(
            geoflow_cmd=glb.geoflow_cmd,
            geoflow_json_filepath=str(self.geoflow_json_filepath),
            config_toml_filepath=str(self.config_toml_filepath),
            cmd_options=f"--verbose --workdir --config {self.config_toml_filepath.as_posix()}",
            stdout_log_filepath=str(self.geoflow_log_filepath)
        )

    def run(self):
        self.executor.run_geoflow()
        self.fix_obj_material_file_definition()

    def fix_obj_material_file_definition(self):
        """Fix material file definition in Geoflow output OBJ files, which is broken"""
        # Get Geoflow reconstruction JSON
        with open(glb.geoflow_reconstruct_template_filepath, "r") as f:
            geoflow_template_json = json.load(f)
        # Get relative filepaths of OBJ output files
        obj_output_relative_filepaths = [
            geoflow_template_json["globals"][f"output_obj_lod{lod}"][2] for lod in ["12", "13", "22"]
        ]
        # Generate absolute filepaths to OBJ output files
        obj_output_filepaths = [
            Path(self.geoflow_json_filepath).parent /
            obj_rel_path for obj_rel_path in obj_output_relative_filepaths
        ]
        # Read each OBJ output file and correct the material file setting "mtllib"
        for obj_output_filepath in obj_output_filepaths:
            with open(obj_output_filepath, "r") as f:
                lines = f.readlines()
            mtllib_index = next((i for i, line in enumerate(lines) if line.startswith("mtllib ")), -1)
            if mtllib_index == -1:
                raise ValueError(f"Line starting with 'mtllib' not found in file:\n{obj_output_filepath}")
            lines[mtllib_index] = f"mtllib {obj_output_filepath.name}.mtl\n"
            with open(obj_output_filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)


class ReconstructionExecutor:

    def __init__(
            self,
            geoflow_cmd: str,
            geoflow_json_filepath: str,
            stdout_log_filepath: str,
            config_toml_filepath: str = "",
            cmd_options: str | list[str] = "",
            very_verbose: bool = False
    ):
        self.geoflow_cmd = geoflow_cmd
        self.stdout_log_filepath = Path(stdout_log_filepath)
        self.very_verbose = very_verbose
        # Converting backslashes to slashes because not sure if subprocess.run() can handle escaped backslashes
        self.geoflow_json_filepath = Path(geoflow_json_filepath).as_posix()
        self.config_toml_filepath = Path(config_toml_filepath).as_posix()

        # Command line options may be passed as single string or as list of strings. If they are a single string, they
        # must be converted to a list of strings to serve as argument for subprocess.run().
        if cmd_options != "":
            if isinstance(cmd_options, list):
                self.cmd_options = cmd_options
            elif isinstance(cmd_options, str):
                # Split additional command options into list for passing to subprocess.Popen
                self.cmd_options = cmd_options.split(" ")
            else:
                raise TypeError("Command line options must be passed as string or list of strings.")
        else:
            # If no command line options were passed, set the respective variable to None
            self.cmd_options = None

        # Assembling the command for subprocess.run() as list of strings
        self.command = [
            self.geoflow_cmd,
            str(self.geoflow_json_filepath)
        ]
        # If a config TOML file was passed as argument, and if it was not already included in the optional cmd_options,
        # include it into the command now
        if str(self.config_toml_filepath) != "" and "--config" not in self.cmd_options and "-c" not in self.cmd_options:
            self.command.extend(["--config", str(self.config_toml_filepath)])
        # If command line parameters were passed, append the corresponding list of strings to the command list
        if self.cmd_options is not None:
            self.command.extend(self.cmd_options)

    def run_geoflow(self):
        print("Starting 3D building reconstruction ...")
        print(f"- Command: {' '.join(self.command)}")
        print(f"- Output log file: {str(self.stdout_log_filepath)}")
        print("")

        if not self.very_verbose:
            with open(self.stdout_log_filepath, "w", encoding="utf-8") as f:
                for stdout_line in execute_subprocess(self.command):
                    f.write(stdout_line)
        else:
            with open(self.stdout_log_filepath, "w", encoding="utf-8") as f:
                for stdout_line in execute_subprocess(self.command):
                    print(stdout_line, end="")
                    f.write(stdout_line)

        print("")
        print("Finished 3D building reconstruction.")


class Scenario:

    def __init__(self, name: str = "", config: dict | None = None):
        self.name = name
        self.config = config
        self.scene_config = config["scene_config"]
        self.survey_config = config["survey_config"]
        self.reconstruction_config = config["reconstruction_config"]
        self.evaluation_config = config["evaluation_config"]

        self.scene: Scene | None = None
        self.survey: Survey | None = None
        self.reconstruction: Reconstruction | None = None

        self.evaluators: dict[str, Evaluator] = {}

    def setup(self):
        pass

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
            filepath=self.survey_config["survey_xml_filepath"],
            output_dirpath=self.survey_config["survey_output_dirpath"],
            config=self.survey_config,
            scene_filepath=scene_xml_filepath_with_id,
            crs=self.config["crs"]
        )

    def prepare_survey(self):
        self.survey.create_flight_path()
        self.survey.create_survey_xml()
        self.survey.setup_executor()

    def run_survey(self):
        self.survey.run()
        self.survey.setup_merger()
        self.survey.merge_clouds()
        self.survey.add_noise()

    def setup_reconstruction(self):
        self.reconstruction = Reconstruction(
            crs=self.config["crs"],
            config=self.reconstruction_config,
            cloud_filepath=self.merged_point_cloud_filepath
        )

    def prepare_reconstruction(self):
        self.reconstruction.prepare_config()
        self.reconstruction.setup_executor()

    def run_reconstruction(self):
        self.reconstruction.run()

    def setup_evaluation(self):
        evaluators = [
            AreaVolumeEvaluator(
                output_base_dirpath=self.evaluation_output_dirpath,
                input_cityjson_filepath=self.cityjson_output_filepath,
                lods=["1.2", "1.3", "2.2"],
                crs=self.crs
            ),
            IOU3DEvaluator(
                output_base_dirpath=self.evaluation_output_dirpath,
                input_cityjson_filepath_1=self.cityjson_input_filepath,
                input_cityjson_filepath_2=self.cityjson_output_filepath,
                lods=["1.2", "1.3", "2.2"],
                crs=self.crs
            ),
            PointDensityDatasetEvaluator(
                output_base_dirpath=self.evaluation_output_dirpath,
                point_cloud_filepath=self.merged_point_cloud_filepath,
                bbox=get_config_item(self.survey_config, "bbox"),
                building_footprints=gpd.read_file(
                    self.building_footprints_filepath, layer=gpd.list_layers(self.building_footprints_filepath).name.loc[0]
                ).geometry,
                crs=self.crs,
                save_filtered_point_clouds=True
            ),
            HausdorffLODSEvaluator(
                output_base_dirpath=self.evaluation_output_dirpath,
                input_obj_filepath_pairs={
                    "12": (self.obj_input_lod12_filepath, self.obj_output_lod12_filepath),
                    "13": (self.obj_input_lod13_filepath, self.obj_output_lod13_filepath),
                    "22": (self.obj_input_lod22_filepath, self.obj_output_lod22_filepath)
                }
            ),
            PointMeshDistanceEvaluator(
                output_base_dirpath=self.evaluation_output_dirpath,
                point_cloud_filepath=self.merged_point_cloud_filepath,
                mesh_filepath=self.obj_input_lod22_filepath,
                crs=self.crs
            )
        ]

        self.evaluators = {evaluator.name: evaluator for evaluator in evaluators}

    def run_evaluation(self, evaluator_selection: list[str] | str | None = None):
        if evaluator_selection is None:
            evaluator_selection = list(self.evaluators.keys())
        elif isinstance(evaluator_selection, str):
            evaluator_selection = [evaluator_selection]

        for evaluator_name in evaluator_selection:
            self.evaluators[evaluator_name].run()

    @property
    def crs(self):
        return get_config_item(self.config, "crs")

    @property
    def reconstruction_output_dirpath(self):
        return Path(get_config_item(self.config, "reconstruction_output_dirpath"))

    @property
    def evaluation_output_dirpath(self):
        return Path(get_config_item(self.config, "evaluation_output_dirpath"))

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
        textfile_merged_cloud_path_filepath = self.textfile_merged_cloud_path_filepath
        try:
            with open(textfile_merged_cloud_path_filepath, "r") as f:
                filepaths = [fp[:-1] for fp in f.readlines()]  # remove \n
        except FileNotFoundError as e:
            print("Filepath of merged cloud for reconstruction is unknown without prior survey or merger run, "
                  "and text file containing it from prior survey run does not exist.")
            raise FileNotFoundError(e.errno, e.strerror, textfile_merged_cloud_path_filepath)
        # The last line in the text file contains the final processed point cloud. (Usually, the first line is the path
        # to the merged point cloud, and an optional second line is the path to the point cloud with added noise.)
        return filepaths[-1]

    @property
    def building_footprints_filepath(self):
        return self.reconstruction_config["building_footprints_filepath"]


class Experiment:
    """An experiment consisting of multiple scenarios"""

    def __init__(
            self,
            name: str,
            dirpath: str,
            default_config: dict,
            scenario_settings: list[dict],
            scene_parts: list[dict],
            footprint_config: dict | None = None
    ):
        """Experiment

        :param name: Experiment name.
        :param dirpath: Directory in which a subdirectory with the experiment name will be created
        :param default_config: Default values for all scenarios
        :param scenario_settings: Specific settings for each scenario as a list of dictionaries
        :param scene_parts: List of dictionaries, each of which specifies details for an OBJ or TIF scene part
        :param footprint_config: Optional (may already be contained in default_config): Dictionary that specifies
        building_footprints_filepath and building_identifier.
        """
        self.name = name
        self.dirpath = Path(dirpath, name)
        self.default_config = default_config
        self.scenario_settings = scenario_settings
        self.scene_parts = scene_parts
        self.footprint_config = footprint_config

        self.settings_dirpath = self.dirpath / "02_settings"
        self.scene_dirpath = self.dirpath / "03_scene"
        self.survey_dirpath = self.dirpath / "04_survey"
        self.reconstruction_dirpath = self.dirpath / "05_reconstruction"
        self.evaluation_dirpath = self.dirpath / "06_evaluation"

        self.scene: Scene | None = None
        self.scene_xml_filepath = self.scene_dirpath / f"{self.name}_scene.xml"

        self.scenarios: dict[str, Scenario] = {}
        self.scenario_configs: dict[str, dict] = {}

    def setup(self):
        """Call all setup functions for directories, scene, scenario configs, and scenarios"""
        self.setup_directories()
        self.setup_scene()
        self.setup_configs()
        self.setup_scenarios()

    def setup_directories(self):
        """Create all main directories for the experiment"""
        # Potentially merge with __init__
        self.dirpath.mkdir(exist_ok=True)
        self.settings_dirpath.mkdir(exist_ok=True)
        self.scene_dirpath.mkdir(exist_ok=True)
        self.survey_dirpath.mkdir(exist_ok=True)
        self.reconstruction_dirpath.mkdir(exist_ok=True)
        self.evaluation_dirpath.mkdir(exist_ok=True)

    def setup_scene(self):
        """Create the scene XML for the experiment and update the default config dict accordingly"""
        # Create the scene for the experiment
        scene_xml_id = f"{self.name.lower()}_scene"
        scene_name = f"{self.name}_scene"
        self.scene = Scene(
            filepath=str(self.scene_xml_filepath),
            xml_id=scene_xml_id,
            name=scene_name,
            scene_parts=self.scene_parts
        )
        self.scene.create_scene_xml()
        # Update all scene-related settings in the default config
        update_config_item(self.default_config, "scene_xml_filepath", str(self.scene_xml_filepath))
        update_config_item(self.default_config, "scene_xml_id", scene_xml_id)
        update_config_item(self.default_config, "scene_name", scene_name)
        update_config_item(self.default_config, "scene_parts", self.scene_parts)
        update_config_item(self.default_config, "scene_xml_filepath_with_id", self.scene.filepath_with_id)

    def setup_configs(self):
        """Setup individual config dictionaries for each scenario, and save them as JSON for reference"""
        # If information on building footprints was provided as optional argument, integrate them into the config
        if self.footprint_config is not None:
            update_config_item(self.default_config, "building_footprints_filepath", self.footprint_config["building_footprints_filepath"])
            update_config_item(self.default_config, "building_identifier", self.footprint_config["building_identifier"])

        for i, settings in enumerate(self.scenario_settings):
            # Potentially use a new class or function experiment_scenario_generator() here?

            # Scenario name
            if "name" in settings.keys():
                # todo: Ensure no duplicate names, otherwise subfolder problems and scenario_configs dict problems
                scenario_name = settings["name"]
            else:
                scenario_name = f"scenario_{i:03}"

            # Scenario-specific file and dir paths
            Path(self.settings_dirpath, scenario_name).mkdir(exist_ok=True)
            Path(self.survey_dirpath, scenario_name).mkdir(exist_ok=True)
            Path(self.reconstruction_dirpath, scenario_name).mkdir(exist_ok=True)
            flight_path_xml_filepath = self.survey_dirpath / scenario_name / "flight_path.xml"
            survey_xml_filepath = self.survey_dirpath / scenario_name / (scenario_name + "_survey.xml")
            survey_output_dirpath = self.survey_dirpath  # HELIOS creates subfolders: /scenario_name/date_time
            reconstruction_output_dirpath = self.reconstruction_dirpath / scenario_name  # reconstruct.json has output/
            evaluation_output_dirpath = self.evaluation_dirpath / scenario_name

            # Create a copy of the default config
            config = copy.deepcopy(self.default_config)

            # Update all occurrences of this scenario's settings in the config
            for key, value in settings.items():
                update_config_item(config, key, value)

            # Update other values in the config
            update_config_item(config, "survey_name", scenario_name)
            update_config_item(config, "flight_path_xml_filepath", str(flight_path_xml_filepath))
            update_config_item(config, "survey_xml_filepath", str(survey_xml_filepath))
            update_config_item(config, "survey_output_dirpath", str(survey_output_dirpath))
            update_config_item(config, "reconstruction_output_dirpath", str(reconstruction_output_dirpath))
            update_config_item(config, "evaluation_output_dirpath", str(evaluation_output_dirpath))

            # Append the finalized name and config to the lists
            self.scenario_configs[scenario_name] = copy.deepcopy(config)
            # self.scenario_names.append(scenario_name)
            # self.scenario_configs.append(config)

            # Store the scenario settings snippet and the full config in the settings folder for reference
            with open(self.settings_dirpath / scenario_name / "scenario_settings.json", "w", encoding="utf-8") as f:
                # ensure_ascii=False avoid special characters (such as "ä") being escapes (such as "\u00e4")
                json.dump(settings, f, indent=4, ensure_ascii=False)
            with open(self.settings_dirpath / scenario_name / "full_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

    def setup_scenarios(self):
        """Initialize the Scenario objects for each scenario with its corresponding configuration"""
        for name, config in self.scenario_configs.items():
            self.scenarios[name] = Scenario(name, config)

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

    def run_step(self, name: str, function: object):
        pass

    def setup_surveys(self):
        for name, s in self.scenarios.items():
            print(f"Setting up survey for scenario {name} ...\n")
            t0 = time.time()

            s.setup_survey()

            t1 = time.time()
            print(f"Finished setting up survey for scenario {name} after {str(timedelta(seconds=t1 - t0))}.\n")

    def prepare_surveys(self):
        for name, s in self.scenarios.items():
            print(f"Preparing survey for scenario {name} ...\n")
            t0 = time.time()

            s.prepare_survey()

            t1 = time.time()
            print(f"Finished preparing survey for scenario {name} after {str(timedelta(seconds=t1 - t0))}.\n")

    def run_surveys(self):
        for name, s in self.scenarios.items():
            print(f"Simulating survey for scenario {name} ...\n")
            t0 = time.time()

            s.run_survey()

            t1 = time.time()
            print(f"Finished simulating survey for scenario {name} after {str(timedelta(seconds=t1 - t0))}.\n")

    def setup_reconstructions(self):
        for name, s in self.scenarios.items():
            print(f"Setting up building reconstruction for scenario {name} ...\n")
            t0 = time.time()

            s.setup_reconstruction()

            t1 = time.time()
            print(f"Finished setting up building reconstruction for scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")

    def prepare_reconstructions(self):
        for name, s in self.scenarios.items():
            print(f"Preparing building reconstruction for scenario {name} ...\n")
            t0 = time.time()

            s.prepare_reconstruction()

            t1 = time.time()
            print(f"Finished preparing building reconstruction for scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")

    def run_reconstructions(self):
        for name, s in self.scenarios.items():
            print(f"Reconstructing buildings for scenario {name} ...\n")
            t0 = time.time()

            s.run_reconstruction()

            t1 = time.time()
            print(f"Finished building reconstruction for scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")

    def evaluate_input(self):
        # For some standalone metrics, the input (ground truth) building models must be evaluated, too:
        # - AreaVolumeEvaluator
        # - ComplexityEvaluator (if used / implemented)
        # - ...
        # Probably implement for Scenario, and then execute here for only one of the scenarios (all use the same input)
        pass

    def setup_evaluations(self):
        for name, s in self.scenarios.items():
            print(f"Setting up evaluation for scenario {name} ...\n")
            t0 = time.time()

            s.setup_evaluation()

            t1 = time.time()
            print(f"Finished setting up evaluation for scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")

    def run_evaluations(self, evaluator_selection: list[str] | str | None = None):
        for name, s in self.scenarios.items():
            print(f"Evaluating scenario {name} ...\n")
            t0 = time.time()

            s.run_evaluation(evaluator_selection)

            t1 = time.time()
            print(f"Finished evaluating scenario {name} after {str(timedelta(seconds=t1-t0))}.\n")


class Config:

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


def generate_scenario(name: str, config: dict, default_config: dict) -> Scenario:
    full_config = deep_update(default_config, config)
    return Scenario(name, config)


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

    cloud_error_config = {
        "std_horizontal_error": 0.0,
        "std_vertical_error": 0.0
    }

    survey_config = {
        "survey_xml_filepath": "",
        "survey_output_dirpath": "",
        "survey_generator_config": survey_generator_config,
        "flight_path_config": flight_path_config,
        "survey_executor_config": survey_executor_config,
        "cloud_merge_config": cloud_merge_config,
        "cloud_error_config": cloud_error_config
    }

    reconstruction_config = {
        "config_toml_filepath": "",  # UNUSED
        "point_cloud_filepath": "",  # LIKELY UNUSED, but currently an alternative in Reconstruction.__init__()
        "building_footprints_filepath": "",
        "building_identifier": "",
        "reconstruction_output_dirpath": "",
        "geoflow_log_filepath": ""  # UNUSED
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
        "crs": "epsg:7415",
        "scene_config": scene_config,
        "survey_config": survey_config,
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
    """Recursively update all occurrences of 'key' in the config dictionary and all nested dictionaries with 'value'.

    :param config: Config dictionary. May contain nested dictionaries.
    :param key: Key to be updated.
    :param value: Value to be set.
    :param not_found_error: True: If key not found, raise error, else return None. False: Return whether key found.
    :param update_all:
    :return: If not_found_error==False: bool whether key was found; else: None if found, else raise KeyError.
    """
    found = False

    if key in config.keys():
        config[key] = value
        found = True

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


if __name__ == "__main__":
    pass
