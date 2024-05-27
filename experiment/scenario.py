# from __future__ import annotations
# from typing_extensions import Self
import time
import datetime
import shutil
import pyhelios
import pdal
import subprocess
import sys
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as eT
from scipy.spatial import KDTree

from experiment.scene_part import ScenePartOBJ, ScenePartTIFF
from .xml_generator import SceneGenerator, FlightPathGenerator, SurveyGenerator, parse_xml_with_comments
from . import global_vars as glb

pyhelios.loggingVerbose2()
pyhelios.setDefaultRandomnessGeneratorSeed("42")


def execute(cmd):
    """Run a subprocess and yield (return an iterable of) all stdout lines. From: https://stackoverflow.com/a/4417735"""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for stdout_line in iter(process.stdout.readline, ""):
        yield stdout_line
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


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

        # todo: What was the idea behind this? Allow use of a different scene?
        if scene_filepath == "":
            self.scene_filepath = self.survey_generator_config["scene_xml_filepath"]
        else:
            self.scene_filepath = scene_filepath

        self.flight_path: FlightPath | None = None
        self.survey_generator: SurveyGenerator | None = None
        self.survey_executor: SurveyExecutor | None = None
        self.output_cloud_filepaths: list[str] | None = None
        self.cloud_merger: CloudMerger | None = None

    def create_flight_path(self):
        self.flight_path = FlightPath(
            filepath=self.flight_path_config["flight_path_xml_filepath"],
            config=self.flight_path_config
        )
        self.flight_path.create_flight_path_xml()

    def create_survey_xml(self):
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
        self.survey_generator.write_file()

    def setup_executor(self):
        self.survey_executor = SurveyExecutor(
            survey_filepath=self.filepath,
            output_dirpath=self.output_dirpath,
            config=self.survey_executor_config
        )

    def run(self):
        self.survey_executor.build_simulation()
        self.output_cloud_filepaths = self.survey_executor.run()

    def setup_merger(self):
        # Get the flight path root element from the flight path XMl file that was generated by self.flight_path:
        # FlightPath(), or from the survey XML file as generated by self.survey_generator if the flight path XML does
        # not exist anymore for any reason.
        # Another option would be to use the attribute waypoints from FlightPathGenerator().
        if Path(self.flight_path_config["flight_path_xml_filepath"]).is_file():
            flight_path_element = parse_xml_with_comments(self.flight_path_config["flight_path_xml_filepath"]).getroot()
        elif Path(self.filepath).is_file():
            flight_path_element = parse_xml_with_comments(self.filepath).getroot().find("survey")
        else:
            raise FileNotFoundError(
                "Neither the flight path nor the survey XML file exist to read the leg coordinates from."
            )

        merged_output_cloud_dirpath = str(Path(self.output_cloud_filepaths[0]).parent)

        self.cloud_merger = UniqueKeeperMerger(
            cloud_filepaths=self.output_cloud_filepaths,
            flight_path_element=flight_path_element,
            parallel_dimension="Y",
            output_dirpath=merged_output_cloud_dirpath,
            crs=self.crs
        )

    def merge_clouds(self) -> str:
        merged_output_cloud_filepath = self.cloud_merger.run()
        # Since we are using the UniqueKeeperMerger, whose run() method calls several of its own functions, the
        # execution of self.cloud_merger.run() calls implicitly:
        #
        # self.cloud_merger.get_parallel_trajectory_coords()
        # self.cloud_merger.compute_separation_coords()
        # self.cloud_merger.filter_clouds()
        # merged_output_cloud_filepath = self.cloud_merger.merge_clouds()

        return merged_output_cloud_filepath


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
        self.merged_output_cloud_filepath = Path(self.output_dirpath, "clouds_merged.laz").as_posix()

    def run(self) -> str:
        self.get_parallel_trajectory_coords()
        self.compute_separation_coords()
        self.filter_clouds()
        merged_output_cloud_filepath = self.merge_clouds()
        return merged_output_cloud_filepath

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
        readers = [pdal.Reader(filepath, nosrs=True, default_srs="epsg:2056") for filepath in self.cloud_filepaths]
        # pipelines = [reader | pdal.Filter.nndistance(mode="kth", k=1) for reader in readers]
        pipelines = [reader | pdal.Filter.nndistance(mode="avg", k=4) for reader in readers]

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
            filter_expression = (f"({self.parallel_dimension} >= {self.separation_coords[p]} && "
                                 f"{self.parallel_dimension} <= {self.separation_coords[p + 1]})")
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

    def merge_clouds(self) -> str:
        print("Concatenating filtered point clouds ...")
        # Concatenate the filtered point clouds into one structured array
        point_cloud_arrays_stacked = np.hstack([arr[self.columns_out] for arr in self.point_cloud_arrays])
        print(f"- Total number of points: {len(point_cloud_arrays_stacked)}")

        print("Writing merged point clouds to output location ...")
        # Pipeline to save the final (filtered and merged) point cloud
        pipeline = pdal.Writer(
            filename=self.merged_output_cloud_filepath, minor_version=4, a_srs=self.crs, compression=True
        ).pipeline(point_cloud_arrays_stacked)
        n_points = pipeline.execute()
        print(f"- Written number of points: {n_points}")

        return self.merged_output_cloud_filepath


class Reconstruction:

    def __init__(self, crs: str, config: dict, cloud_filepath: str = ""):
        self.crs = crs
        self.config = config

        self.output_dirpath = self.config["output_dirpath"]
        self.cloud_filepath = cloud_filepath if cloud_filepath != "" else self.config["point_cloud_filepath"]
        self.geoflow_json_filepath = Path(self.output_dirpath, "reconstruct.json")
        self.config_toml_filepath = Path(self.output_dirpath, "config.toml")

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

        # Get the EPSG code without "epsg:" prefix
        crs_epsg_code = self.crs.rsplit(":", 1)[1]
        # Prepare the values for the config.toml file
        config_toml_json = {
            "GF_PROCESS_CRS": self.crs,
            "input_pointcloud_crs_wkt": self.crs,
            "output_crs_wkt": self.crs,
            "output_cj_referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/" + crs_epsg_code,
            "input_footprint": self.config["building_footprints_filepath"],
            "building_identifier": self.config["building_identifier"],
            "input_pointcloud": self.cloud_filepath
        }
        # Write the config.toml
        with open(self.config_toml_filepath, "w") as f:
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

        self.executor = ReconstructionExecutor(
            geoflow_cmd=glb.geoflow_cmd,
            geoflow_json_filepath=str(self.geoflow_json_filepath),
            config_toml_filepath=str(self.config_toml_filepath),
            cmd_options=f"--verbose --workdir --config {self.config_toml_filepath.as_posix()}",
            stdout_log_filepath=self.config["geoflow_log_filepath"]
        )

    def run(self):
        self.executor.run_geoflow()


class ReconstructionExecutor:

    def __init__(
            self,
            geoflow_cmd: str,
            geoflow_json_filepath: str,
            config_toml_filepath: str = "",
            cmd_options: str | list[str] = "",
            stdout_log_filepath: str = ""
    ):
        self.geoflow_cmd = geoflow_cmd
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

        # Converting backslashes to slashes just to be sure.
        self.stdout_log_filepath = Path(stdout_log_filepath).as_posix()

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
        print("Starting building reconstruction ...")
        print(f"- Command: {' '.join(self.command)}")
        print(f"- Output log file: {'None' if str(self.stdout_log_filepath) == '' else str(self.stdout_log_filepath)}")
        print("")

        if str(self.stdout_log_filepath) == "":
            for stdout_line in execute(self.command):
                print(stdout_line, end="")
        else:
            with open(self.stdout_log_filepath, "w") as f:
                for stdout_line in execute(self.command):
                    print(stdout_line, end="")
                    f.write(stdout_line)

        print("")
        print("Finished building reconstruction.")


class Scenario:

    def __init__(self, name: str = "", config: dict | None = None):
        self.name = name
        self.config = config
        self.scene_config = config["scene_config"]
        self.survey_config = config["survey_config"]
        self.reconstruction_config = config["reconstruction_config"]

        self.scene: Scene | None = None
        self.survey: Survey | None = None
        self.reconstruction: Reconstruction | None = None

        self.merged_output_cloud_filepath = ""

    def setup(self):
        pass

    def setup_scene(self):
        self.scene = Scene(
            filepath=self.scene_config["scene_xml_filepath"],
            xml_id=self.scene_config["xml_id"],
            name=self.scene_config["name"],
            scene_parts=self.scene_config["scene_parts"]
        )
        self.scene.create_scene_xml()

    def setup_survey(self):
        self.survey = Survey(
            filepath=self.survey_config["survey_xml_filepath"],
            output_dirpath=self.survey_config["output_dirpath"],
            config=self.survey_config,
            scene_filepath=self.scene.filepath_with_id,
            crs=self.config["crs"]
        )
        self.survey.create_flight_path()
        self.survey.create_survey_xml()
        self.survey.setup_executor()

    def run_survey(self):
        self.survey.run()
        self.survey.setup_merger()
        self.merged_output_cloud_filepath = self.survey.merge_clouds()

    def setup_reconstruction(self):
        self.reconstruction = Reconstruction(
            crs=self.config["crs"],
            config=self.reconstruction_config,
            cloud_filepath=self.merged_output_cloud_filepath
        )
        self.reconstruction.prepare_config()
        self.reconstruction.setup_executor()

    def run_reconstruction(self):
        self.reconstruction.run()


if __name__ == "__main__":
    pass
