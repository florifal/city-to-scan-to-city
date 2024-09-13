import json
import shutil
from datetime import datetime
from pathlib import Path

from experiment import global_vars as glb
from experiment.utils import crs_url_from_epsg, is_numeric, execute_subprocess


class Reconstruction:

    def __init__(self, crs: str, config: dict, cloud_filepath: str = ""):
        self.crs = crs
        self.config = config

        self.output_dirpath = Path(self.config["reconstruction_output_dirpath"])
        self.cloud_filepath = cloud_filepath if cloud_filepath != "" else self.config["point_cloud_filepath"]
        self.geoflow_parameters = self.config["geoflow_parameters"]
        self.geoflow_json_filepath = self.output_dirpath / "reconstruct.json"
        self.config_toml_filepath = self.output_dirpath / "config.toml"
        date_time = datetime.today().strftime("%y%m%d-%H%M%S")
        self.geoflow_log_filepath = self.output_dirpath / f"geoflow_log_{date_time}.txt"

        self.executor: ReconstructionExecutor | None = None

    def clear(self):
        self.executor = None

    def prepare_config(self):
        if not self.output_dirpath.is_dir():
            self.output_dirpath.mkdir()

        # Copy the Geoflow reconstruction template JSON and rename it to reconstruct.json
        shutil.copy2(glb.geoflow_reconstruct_template_filepath, self.output_dirpath)
        (self.output_dirpath / Path(glb.geoflow_reconstruct_template_filepath).name).rename(self.geoflow_json_filepath)
        # Also copy the nested JSON that it includes to the output directory
        shutil.copy2(glb.geoflow_reconstruct_nested_filepath, self.output_dirpath)

        # Prepare the values for the config.toml file
        config_toml_json = {
            "GF_PROCESS_CRS": self.crs,
            "input_pointcloud_crs_wkt": self.crs,
            "output_crs_wkt": self.crs,
            "output_cj_referenceSystem": crs_url_from_epsg(self.crs),
            "input_footprint": self.config["building_footprints_filepath"],
            "input_footprint_sql": self.config["building_footprints_sql"],
            "building_identifier": self.config["building_identifier"],
            "input_pointcloud": self.cloud_filepath,
            **{k: v for k, v in self.geoflow_parameters.items() if v is not None}
        }

        # Write the config.toml
        with open(self.config_toml_filepath, "w", encoding="utf-8") as f:
            for key, value in config_toml_json.items():
                # Numeric values (int and float) should not be enclosed in single quotation marks, strings should
                if is_numeric(value):
                    # Make sure integers are written without decimal places
                    if int(value) == float(value):
                        value_str = str(int(value))
                    else:
                        value_str = str(float(value))
                else:
                    value_str = f"'{value}'"
                f.write(f"{key}={value_str}\n")

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

    def setup_executor(self):
        # or combine this method with run()

        # Option --workdir makes sure "output" directory is created at the location of the reconstruction.json
        self.executor = ReconstructionExecutor(
            geoflow_cmd=glb.geoflow_cmd,
            geoflow_json_filepath=self.geoflow_json_filepath,
            config_toml_filepath=self.config_toml_filepath,
            cmd_options=f"--verbose --workdir --config {self.config_toml_filepath.as_posix()}",
            stdout_log_filepath=self.geoflow_log_filepath
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
            geoflow_json_filepath: Path | str,
            stdout_log_filepath: Path | str,
            config_toml_filepath: Path | str = "",
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
        print("Starting 3D building reconstruction ...\n")
        print(f"- Command: {' '.join(self.command)}")
        print(f"- Output log file: {str(self.stdout_log_filepath)}")

        if not self.very_verbose:
            with open(self.stdout_log_filepath, "w", encoding="utf-8") as f:
                for stdout_line in execute_subprocess(self.command):
                    f.write(stdout_line)
        else:
            print("")
            with open(self.stdout_log_filepath, "w", encoding="utf-8") as f:
                for stdout_line in execute_subprocess(self.command):
                    print(stdout_line, end="")
                    f.write(stdout_line)

        print("")
        print("Finished 3D building reconstruction.")
