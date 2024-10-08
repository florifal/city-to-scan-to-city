import time
import pyhelios
from datetime import datetime, timedelta
from pathlib import Path

from experiment import global_vars as glb
from experiment.point_cloud_processing import CloudMerger, UniqueKeeperMerger
from experiment.utils import get_most_recently_created_folder
from experiment.xml_generator import FlightPathGenerator, SurveyGenerator, parse_xml_with_comments


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
            survey_xml_filepath: str,
            output_dirpath: Path | str,
            crs: str,
            config: dict | None = None,
            scene_filepath: str = ""
    ):
        """
        :param survey_xml_filepath: Path to the survey XML file
        :param output_dirpath: Output directory for HELIOS++, in which a directory with the survey name will be created
        :param crs: Coordinate reference system EPSG string
        :param config: Survey config dictionary
        :param scene_filepath: (Optional) Path to the scene XML file including ID of the XML scene tag
        """
        self.survey_xml_filepath = survey_xml_filepath
        self.output_base_dirpath_helios = Path(output_dirpath)
        self.crs = crs
        self.config = config

        self.survey_generator_config = config["survey_generator_config"]
        self.flight_path_config = config["flight_path_config"]
        self.survey_executor_config = config["survey_executor_config"]
        self.cloud_merge_config = config["cloud_merge_config"]

        self.output_dirpath_survey = self.output_base_dirpath_helios / self.survey_generator_config["survey_name"]

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

        # Paths to write two text documents into, containing (1) a list of the paths to all output point clouds files
        # and (2) a single path to the final merged point cloud file. They should be put into the subdirectory bearing
        # the survey name that is created automatically by HELIOS within the specified output directory.
        self.textfile_output_clouds_list_filepath = (self.output_base_dirpath_helios / self.survey_generator_config["survey_name"] /
                                                     "output_clouds_filepaths.txt")
        self.textfile_merged_cloud_path_filepath = (self.output_base_dirpath_helios / self.survey_generator_config["survey_name"] /
                                                    "merged_cloud_filepath.txt")

    def clear(self):
        if self.survey_executor is not None:
            self.survey_executor.clear()
        if self.cloud_merger is not None:
            self.cloud_merger.clear()

        self.flight_path = None
        self.survey_generator = None
        self.survey_executor = None
        self.cloud_merger = None

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
            filepath=self.survey_xml_filepath,

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
            survey_name=self.survey_generator_config["survey_name"],
            survey_xml_filepath=self.survey_xml_filepath,
            output_dirpath=self.output_base_dirpath_helios,
            config=self.survey_executor_config
        )

    def run(self):
        """Using the SurveyExecutor, build and run the simulation, then write some output"""
        self.survey_executor.build_simulation()
        self.survey_executor.run()
        self.survey_executor.clear()  # free up memory
        self.survey_executor = None
        # todo: remove
        # self.output_clouds_filepaths = self.survey_executor.output_cloud_filepaths
        # self.output_clouds_dirpath = str(Path(self.output_clouds_filepaths[0]).parent)
        self.output_clouds_dirpath = get_most_recently_created_folder(self.output_dirpath_survey)
        self.output_clouds_filepaths = self.get_output_point_cloud_filepaths(self.output_clouds_dirpath)

        # Here, the list of output point clouds files is written to a file in the output directory. Another file with
        # the path to the merged point cloud file is written only after execution of the CloudMerger.
        with open(self.textfile_output_clouds_list_filepath, "w", encoding="utf-8") as f:
            f.writelines([str(fp) + "\n" for fp in self.output_clouds_filepaths])

    def setup_merger(self):
        """Prepare the CloudMerger (UniqueKeeperMerger as of now) for the merging of the output point clouds"""
        # Get the flight path root element from the flight path XMl file that was generated by self.flight_path:
        # FlightPath(), or from the survey XML file as generated by self.survey_generator if the flight path XML does
        # not exist anymore for any reason.
        # Another option would be to use the attribute waypoints from FlightPathGenerator().
        # This is needed for the UniqueKeeperMerger to read each leg's coordinates.
        if Path(self.flight_path_config["flight_path_xml_filepath"]).is_file():
            flight_path_element = parse_xml_with_comments(self.flight_path_config["flight_path_xml_filepath"]).getroot()
        elif Path(self.survey_xml_filepath).is_file():
            flight_path_element = parse_xml_with_comments(self.survey_xml_filepath).getroot().find("survey")
        else:
            raise FileNotFoundError(
                "Neither the flight path nor the survey XML file exist to read the leg coordinates from."
            )

        # If the CloudMerger is being set up without the survey having been run first, this block will attempt to read
        # the filepaths of the clouds that should be merged from a previous survey run from the expected text file
        # location.
        if self.output_clouds_filepaths is None:
            print("\nFilepaths of clouds to be merged from prior survey run are unknown.")
            print("Attempting to read from `output_clouds_filepaths.txt` ...")
            try:
                with open(self.textfile_output_clouds_list_filepath, "r") as f:
                    self.output_clouds_filepaths = [Path(fp[:-1]) for fp in f.readlines()]  # remove \n
            except FileNotFoundError as e:
                print(str(e))
                print("Text file listing file paths not found.")
                print("Attempting instead to identify output cloud filepaths in output directory ...")
                self.output_clouds_dirpath = get_most_recently_created_folder(self.output_dirpath_survey)
                self.output_clouds_filepaths = self.get_output_point_cloud_filepaths(self.output_clouds_dirpath)
            else:
                self.output_clouds_dirpath = self.output_clouds_filepaths[0].parent
        print(f"\nIdentified paths of {len(self.output_clouds_filepaths)} clouds to be merged.")
        print(f"- Folder: {self.output_clouds_dirpath.name}")
        for i, fp in enumerate(self.output_clouds_filepaths):
            print(f"- File {i}: {fp.name}")

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

    @property
    def output_point_cloud_file_type(self) -> str:
        if self.survey_executor_config["las_output"]:
            if self.survey_executor_config["zip_output"]:
                return "laz"
            else:
                return "las"
        else:
            return "xyz"

    def get_output_point_cloud_filepaths(self, output_dirpath: Path | str) -> list[Path]:
        output_dirpath = Path(output_dirpath)
        file_ext = self.output_point_cloud_file_type
        return [f for f in output_dirpath.iterdir() if f.is_file() and f.name.endswith(f"_points.{file_ext.lower()}")]


class SurveyExecutor:

    def __init__(
            self,
            survey_name: str,
            survey_xml_filepath: str,
            output_dirpath: Path | str | None = None,
            config: dict | None = None
    ):
        """
        :param survey_name: Name of the survey. Used to identify HELIOS++ output path within `output_dirpath`.
        :param survey_xml_filepath: Path to the survey XML file
        :param output_dirpath:
        :param config:
        """
        self.survey_name = survey_name
        self.survey_filepath = survey_xml_filepath
        if output_dirpath is None:
            self.output_base_dirpath_helios = glb.helios_output_dirpath
        else:
            self.output_base_dirpath_helios = Path(output_dirpath)
        self.config = config

        self.output_dirpath_survey = self.output_base_dirpath_helios / self.survey_name

        self.sim_builder: pyhelios.SimulationBuilder | None = None
        self.sim_build: pyhelios.SimulationBuild | None = None
        # self.output_cloud_filepaths: list[str] | None = None  # todo: remove

    def clear(self):
        self.sim_builder = None
        self.sim_build = None

    def build_simulation(self):
        print("\nBuilding survey simulation ...")
        self.sim_builder = pyhelios.SimulationBuilder(
            surveyPath=self.survey_filepath,
            assetsDir=glb.helios_assets_dirpath,
            outputDir=str(self.output_base_dirpath_helios)
        )
        self.sim_builder.setLasOutput(self.config["las_output"])
        self.sim_builder.setZipOutput(self.config["zip_output"])
        self.sim_builder.setNumThreads(self.config["num_threads"])
        self.sim_builder.setCallbackFrequency(0)  # 0: Run without callback
        self.sim_builder.setFinalOutput(False)  # Output points as NumPy array. Required for sim_build.join() to work.
        self.sim_builder.setExportToFile(True)  # Output point cloud to disk
        self.sim_builder.setRebuildScene(False)  # Do not rebuild scene files if exist
        self.sim_builder.setPlatformNoiseDisabled(True)  # Enable/disable platform noise (check platform XML)
        self.sim_builder.setLegNoiseDisabled(True)  # Enable/disable leg noise (should not do anything)

        self.sim_build = self.sim_builder.build()

    def run(self):
        print(f"\nStarting survey simulation at {datetime.now()}...")
        time_start = time.time()
        self.sim_build.start()

        print("Survey simulation is running.")
        while self.sim_build.isRunning():
            pass
            # todo: remove. Continuously printing every second causes huge CPU load from PyCharm / Jupyter.
            # time.sleep(1)
            # print(f"\rSurvey simulation running. Time elapsed: {int(time.time() - time_start)} s", end="")

        # sim_output = self.sim_build.join()  # todo: remove

        print(f"\nSurvey simulation has finished after {timedelta(seconds=int(time.time() - time_start))}.")

        # self.output_cloud_filepaths = [str(Path(p)) for p in sim_output.filepaths]  # todo: remove
