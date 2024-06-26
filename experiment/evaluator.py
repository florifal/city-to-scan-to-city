from pathlib import Path
from cjio import cityjson
from experiment.scenario import Scenario
from experiment.utils import execute_subprocess, crs_url_from_epsg
import pandas as pd
import experiment.global_vars as glb


class Evaluator:

    def __init__(self, scenario: Scenario, output_base_dirpath: Path | str):
        self.scenario = scenario
        self.output_base_dirpath = Path(output_base_dirpath)

    def run(self):
        pass


class BuildingsEvaluator(Evaluator):

    def __init__(self, scenario: Scenario, output_base_dirpath: Path | str):
        super().__init__(scenario, output_base_dirpath)

    def run(self):
        pass


class FMEEvaluator(BuildingsEvaluator):

    def __init__(
            self,
            scenario: Scenario,
            output_base_dirpath: Path | str,
            name: str,
            lods: list[str],
            index_col_name: str = ""
    ):
        super().__init__(scenario, output_base_dirpath)
        self.name = name
        self.lods = lods
        self.index_col_name = index_col_name

        self.output_dirpath: Path = self.output_base_dirpath / name
        self.output_cityjson_filepath: Path = self.output_dirpath / f"model_{name}.json"
        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{name}.csv"
        self.fme_output_cityjson_filepaths = {}

        self.fme_pipelines_dict: dict[str, FMEPipeline] = {}
        self.results_template_cityjson_filepath: Path | None = None
        self.fme_output_attributes: list[str] = []

    def run_fme_pipeline(self):
        print(f"Running FME pipelines for each LOD ...")
        if not self.output_dirpath.is_dir():
            self.output_dirpath.mkdir()

        self.fme_output_cityjson_filepaths = {}
        for lod, fme_pipeline in self.fme_pipelines_dict:
            fme_pipeline.run()
            self.fme_output_cityjson_filepaths[lod] = fme_pipeline.output_cityjson_filepath

    def process_and_save_results(self):
        print(f"Merging attributes for LODs {self.lods} into a new CityJSON file ...")
        print(f"- Path: {self.output_cityjson_filepath}")

        # Take the input CityJSON (as output from reconstruction) and add all the computed FME attributes
        # from the output CityJSON files (as output from FME) for each LOD.
        input_cityjson = cityjson.load(self.results_template_cityjson_filepath, transform=True)
        input_cityjson_buildings = input_cityjson.get_cityobjects(type="building")

        for lod, output_cityjson_filepath in self.fme_output_cityjson_filepaths.items():
            # Load the FME output CityJSON for this LOD
            fme_output_model = cityjson.load(output_cityjson_filepath)
            fme_output_buildings = fme_output_model.get_cityobjects(type="building")
            for b_id, b in fme_output_buildings.items():
                if b.id in input_cityjson_buildings.keys():
                    # Append all FME output attributes of this LOD to the current building in the input CityJSON (which
                    # is the reconstructed model), adding the LOD as a suffix
                    for attribute_name in self.fme_output_attributes:
                        new_attribute_name = f"{attribute_name}_{lod.replace('.', '')}"
                        input_cityjson_buildings[b.id].attributes[new_attribute_name] = b.attributes[attribute_name]

        # Save the CityJSON including the new attributes to a new file
        cityjson.save(input_cityjson, str(self.output_cityjson_filepath))

        print(f"Preparing results table and saving as CSV file ...")
        print(f"- Path: {self.output_csv_filepath}")

        # Edit the buildings dictionary such that its values only contain the values of the attributes dictionary
        # contained in it
        for key in input_cityjson_buildings.keys():
            input_cityjson_buildings[key] = input_cityjson_buildings[key].attributes
        # Create a dataframe where the dict keys form the index, and set the column specified in self.index_col_name
        # (typically `identificatie`) as index instead of the cryptical FID-derived values (e.g., "1", "1-0", ...)
        cityjson_buildings_df = pd.DataFrame.from_dict(input_cityjson_buildings, orient="index")
        if self.index_col_name != "":
            cityjson_buildings_df = cityjson_buildings_df.set_index(self.index_col_name)
        # Save the dataframe
        cityjson_buildings_df.to_csv(self.output_csv_filepath)

    def run(self):
        self.run_fme_pipeline()
        self.process_and_save_results()
        print()

class AreaVolumeEvaluator(FMEEvaluator):

    def __init__(
            self,
            scenario: Scenario,
            output_base_dirpath: Path | str,
            input_cityjson_filepath: Path | str,
            lods: list[str],
    ):
        super().__init__(scenario, output_base_dirpath, name="area_volume", lods=lods, index_col_name=glb.fme_area_volume_identifier_field_name)
        self.input_cityjson_filepath = Path(input_cityjson_filepath)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath
        self.fme_output_attributes = [glb.fme_area_field_name, glb.fme_volume_field_name]
        # self.lods = lods
        # self.index_col_name = index_col_name
        #
        # self.output_dirpath: Path = self.output_base_dirpath / "area_volume"
        # self.output_cityjson_filepath: Path = self.output_dirpath / "model_area_volume.json"
        # self.output_csv_filepath: Path = self.output_dirpath / "evaluation_area_volume.csv"
        # self.fme_output_cityjson_filepaths = {}

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineAreaVolume(
                input_cityjson_filepath=self.input_cityjson_filepath,
                output_dirpath=self.output_dirpath,
                crs=self.scenario.crs,
                lod=lod
            )

    # section old methods
    # def run_fme_pipeline(self):
        # print("Running area and volume FME pipeline ...")
        # if not self.output_dirpath.is_dir():
        #     self.output_dirpath.mkdir()
        #
        # self.fme_output_cityjson_filepaths = {}
        # for lod in self.lods:
        #     fme_pipeline = FMEPipelineAreaVolume(
        #         input_cityjson_filepath=self.input_cityjson_filepath,
        #         output_dirpath=self.output_dirpath,
        #         crs=self.scenario.crs,
        #         lod=lod
        #     )
        #     fme_pipeline.run()
        #     self.fme_output_cityjson_filepaths[lod] = fme_pipeline.output_cityjson_filepath
        # pass

    # def process_and_save_results(self):
        # print(f"Merging attributes for LODs {self.lods} into a new CityJSON file ...")
        # print(f"- Path: {self.output_cityjson_filepath}")
        #
        # # Take the input CityJSON (as output from reconstruction) and add all the computed volume and area attributes
        # # from the output CityJSON files (as output from FME).
        # input_cityjson = cityjson.load(self.input_cityjson_filepath, transform=True)
        # input_cityjson_buildings = input_cityjson.get_cityobjects(type="building")
        #
        # for lod, output_cityjson_filepath in self.fme_output_cityjson_filepaths.items():
        #     fme_output_model = cityjson.load(output_cityjson_filepath)
        #     fme_output_buildings = fme_output_model.get_cityobjects(type="building")
        #     for b_id, b in fme_output_buildings.items():
        #         if b.id in input_cityjson_buildings.keys():
        #             area_attribute_name = f"{glb.fme_area_field_name}_{lod.replace('.', '')}"
        #             volume_attribute_name = f"{glb.fme_volume_field_name}_{lod.replace('.', '')}"
        #             input_cityjson_buildings[b.id].attributes[area_attribute_name] = b.attributes[glb.fme_area_field_name]
        #             input_cityjson_buildings[b.id].attributes[volume_attribute_name] = b.attributes[glb.fme_volume_field_name]
        #
        # # Save the CityJSON including the new attributes to a new file
        # cityjson.save(input_cityjson, str(self.output_cityjson_filepath))
        #
        # print(f"Preparing results table and saving as CSV file ...")
        # print(f"- Path: {self.output_csv_filepath}")
        #
        # # Edit the buildings dictionary such that its values only contain the values of the attributes dictionary
        # # contained in it
        # for key in input_cityjson_buildings.keys():
        #     input_cityjson_buildings[key] = input_cityjson_buildings[key].attributes
        # # Create a dataframe where the keys form the index, and set the column "OGRLoader.identificatie" as index
        # # instead of the cryptical FID-derived values (e.g., "1", "1-0", ...)
        # cityjson_buildings_df = pd.DataFrame.from_dict(input_cityjson_buildings, orient="index")
        # if self.index_col_name != "":
        #     cityjson_buildings_df = cityjson_buildings_df.set_index(self.index_col_name)
        # # Save the dataframe
        # cityjson_buildings_df.to_csv(self.output_csv_filepath)
        # pass
    # end section

    def run(self):
        print("Starting AreaVolumeEvaluator ...")
        super().run()


class IOU3DEvaluator(FMEEvaluator):

    def __init__(
            self,
            scenario: Scenario,
            output_base_dirpath: Path | str,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            lods: list[str],
    ):
        super().__init__(scenario, output_base_dirpath, name="iou_3d", lods=lods, index_col_name=glb.fme_iou3d_identifier_field_name)
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath_2
        self.fme_output_attributes = [
            glb.fme_iou3d_volume_input_field_name,
            glb.fme_iou3d_volume_output_field_name,
            glb.fme_iou3d_volume_intersection_field_name,
            glb.fme_iou3d_iou_field_name
        ]
        # self.lods = lods
        # self.index_col_name = index_col_name
        #
        # self.output_dirpath: Path = self.output_base_dirpath / "iou_3d"
        # self.output_cityjson_filepath: Path = self.output_dirpath / "model_iou_3d.json"
        # self.output_csv_filepath: Path = self.output_dirpath / "evaluation_iou_3d.csv"
        # self.fme_output_cityjson_filepaths = {}

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineIOU3D(
                input_cityjson_filepath_1=self.input_cityjson_filepath_1,
                input_cityjson_filepath_2=self.input_cityjson_filepath_2,
                output_dirpath=self.output_dirpath,
                crs=self.scenario.crs,
                lod=lod
            )

    # def run_fme_pipeline(self):
        # print("Running IOU 3D FME pipeline ...")
        # if not self.output_dirpath.is_dir():
        #     self.output_dirpath.mkdir()
        #
        # self.fme_output_cityjson_filepaths = {}
        # for lod in self.lods:
        #     fme_pipeline = FMEPipelineIOU3D(
        #         input_cityjson_filepath_1=self.input_cityjson_filepath_1,
        #         input_cityjson_filepath_2=self.input_cityjson_filepath_2,
        #         output_dirpath=self.output_dirpath,
        #         crs=self.scenario.crs,
        #         lod=lod
        #     )
        #     fme_pipeline.run()
        #     self.fme_output_cityjson_filepaths[lod] = fme_pipeline.output_cityjson_filepath
        # pass

    # def process_and_save_results(self):

    def run(self):
        print("Starting IOU3DEvaluator ...")
        super().run()




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
        print("Executing FME workspace with command:")
        print("\n".join(command))
        print()

    with open(stdout_log_filepath, "a", encoding="utf-8") as f:
        for stdout_line in execute_subprocess(command):
            print(stdout_line, end="")
            f.write(stdout_line)


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
            self.output_dirpath / (self.input_cityjson_filepath.stem + f"_fme_area_volume_lod{self.lod.replace('.','')}.json")
        )
        self.stdout_log_filepath = (
            self.output_dirpath / (self.output_cityjson_filepath.stem + "log")
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
            self.output_dirpath / (self.input_cityjson_filepath_2.stem + f"_fme_iou3d_lod{self.lod.replace('.','')}.json")
        )
        self.stdout_log_filepath = (
            self.output_dirpath / (self.output_cityjson_filepath.stem + "log")
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



