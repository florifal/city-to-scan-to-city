from pathlib import Path

from experiment import global_vars as glb
from experiment.utils import execute_subprocess, crs_url_from_epsg


def run_fme_workspace(
        fme_workspace_filepath: Path | str,
        user_parameters: dict[str, str],
        stdout_log_filepath: Path | str,
        verbose: bool = True
):
    fme_workspace_filepath = Path(fme_workspace_filepath)
    stdout_log_filepath = Path(stdout_log_filepath)

    command = [
        glb.fme_cmd,
        str(fme_workspace_filepath),
    ]
    for param_name, param_value in user_parameters.items():
        command.append(f"--{param_name}")
        command.append(f"{param_value}")

    if verbose:
        print("Executing FME workspace with command:\n")
        print("\n".join(command))
        print()

    with open(stdout_log_filepath, "a", encoding="utf-8") as f:
        for stdout_line in execute_subprocess(command):
            print(stdout_line, end="")
            f.write(stdout_line)
    print()


class FMEPipeline:

    def __init__(self, output_dirpath: Path | str, crs: str, lod: str):
        self.output_dirpath = Path(output_dirpath)
        self.crs = crs
        self.lod = lod

        self.crs_url = crs_url_from_epsg(self.crs)

        self.output_cityjson_filepath: Path | None = None

        self.fme_workspace_filepath: str | None = None
        self.user_parameters: dict[str, str] | None = None
        self.stdout_log_filepath: Path | str | None = None

    def run(self):
        run_fme_workspace(
            fme_workspace_filepath=self.fme_workspace_filepath,
            user_parameters=self.user_parameters,
            stdout_log_filepath=self.stdout_log_filepath)


class FMEPipelineAreaVolume(FMEPipeline):

    def __init__(
            self,
            input_cityjson_filepath: Path | str,
            output_dirpath: Path | str,
            crs: str,
            lod: str
    ):
        super().__init__(output_dirpath, crs, lod)
        self.input_cityjson_filepath = Path(input_cityjson_filepath)

        self.output_cityjson_filepath = (
            self.output_dirpath /
            (self.input_cityjson_filepath.stem + f"_fme_area_volume_lod{self.lod.replace('.','')}.json")
        )
        self.stdout_log_filepath = (
            self.output_dirpath / (self.output_cityjson_filepath.stem + ".log")
        )

        self.fme_workspace_filepath = glb.fme_workspace_area_volume_filepath
        self.user_parameters = {
            "SourceDataset_CITYJSON": str(self.input_cityjson_filepath),
            "SourceDataset_CITYJSON_2": str(self.input_cityjson_filepath),
            "CITYJSON_IN_LOD_2": self.lod,
            "COORDSYS_1": self.crs_url,
            "COORDSYS_2": self.crs_url,
            "COORDSYS_3": self.crs_url,
            "DestDataset_CITYJSON": str(self.output_cityjson_filepath)
        }


class FMEPipelineIOU3D(FMEPipeline):

    def __init__(
            self,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            output_dirpath: Path | str,
            crs: str,
            lod: str
    ):
        super().__init__(output_dirpath, crs, lod)
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)

        self.output_cityjson_filepath = (
            self.output_dirpath /
            (self.input_cityjson_filepath_2.stem + f"_fme_iou3d_lod{self.lod.replace('.','')}.json")
        )
        self.stdout_log_filepath = (
            self.output_dirpath / (self.output_cityjson_filepath.stem + ".log")
        )

        self.fme_workspace_filepath = glb.fme_workspace_iou3d_filepath
        self.user_parameters = {
            "SourceDataset_CITYJSON": str(self.input_cityjson_filepath_1),
            "SourceDataset_CITYJSON_6": str(self.input_cityjson_filepath_1),
            "SourceDataset_CITYJSON_5": str(self.input_cityjson_filepath_2),
            "SourceDataset_CITYJSON_7": str(self.input_cityjson_filepath_2),
            "COORDSYS": self.crs_url,
            "CITYJSON_IN_LOD_1": self.lod,
            "CITYJSON_IN_LOD_2": self.lod,
            "DestDataset_CITYJSON": str(self.output_cityjson_filepath)
        }
