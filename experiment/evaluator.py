import numpy as np
import pandas as pd
import geopandas as gpd
import pymeshlab
import pdal
import open3d as o3d
import numpy.lib.recfunctions as rfn
from pathlib import Path
from cjio import cityjson
from shapely import box

import experiment.global_vars as glb
from experiment.fme_pipeline import FMEPipeline, FMEPipelineAreaVolume, FMEPipelineIOU3D
from experiment.obj_file import split_obj_file


def print_starting_message(func):
    def wrapper(self):
        print(f"Starting {self.__class__.__name__} ...\n")
        func(self)
        print()
    return wrapper


def get_bidirectional_hausdorff_distances(
        obj_filepath_1: Path | str,
        obj_filepath_2: Path | str,
        sample_vertices: bool = True,
        sample_edges: bool = True,
        sample_faces: bool = True,
        sample_number: int = 10000
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Computes both directional Hausdorff distances for two OBJ meshes using PyMeshLab"""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj_filepath_1))
    ms.load_new_mesh(str(obj_filepath_2))

    hausdorff_12_dict = ms.get_hausdorff_distance(
        sampledmesh=0,
        targetmesh=1,
        savesample=False,
        samplevert=sample_vertices,
        sampleedge=sample_edges,
        sampleface=sample_faces,
        samplenum=sample_number
    )
    hausdorff_21_dict = ms.get_hausdorff_distance(
        sampledmesh=0,
        targetmesh=1,
        savesample=False,
        samplevert=sample_vertices,
        sampleedge=sample_edges,
        sampleface=sample_faces,
        samplenum=sample_number
    )

    return hausdorff_12_dict, hausdorff_21_dict


def read_building_points(point_cloud_filepath, crs, classification=6, verbose=True):
    reader = pdal.Reader(str(point_cloud_filepath), nosrs=True, default_srs=crs)
    pipeline = reader.pipeline()
    n_points = pipeline.execute()
    point_array = pipeline.arrays[0]
    point_array_subset = point_array[point_array["Classification"] == classification]

    if verbose:
        n_points_on_buildings = point_array_subset.shape[0]
        print("\t\t\tNumber of points")
        print(f"Total\t\t{n_points}")
        print(f"Buildings\t{n_points_on_buildings}\n")

    # Transform the structured into a regular NumPy array. Only keep coordinate columns, and only points on buildings.
    return rfn.structured_to_unstructured(point_array_subset[["X", "Y", "Z"]])


def compute_point_mesh_distance(point_array, mesh_filepath):
    # Transform the NumPy array into an Open3D tensor (Dtype Float32 required for RaycastingScene.compute_distance())
    points_tensor = o3d.core.Tensor(point_array, dtype=o3d.core.Dtype.Float32)

    # Read the mesh and set up a RaycastingScene
    mesh = o3d.t.io.read_triangle_mesh(str(mesh_filepath))  # o3d.t.io instead of o3d.io loads tensor-based geometry
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # Compute the distances from each point to the mesh (any triangle of it).
    distances = scene.compute_distance(points_tensor)
    distances_np = distances.numpy()
    return distances_np


class Evaluator:
    name = "evaluator"

    def __init__(self, output_base_dirpath: Path | str):
        self.output_base_dirpath = Path(output_base_dirpath)
        self.output_dirpath = self.output_base_dirpath / self.name

        self.output_base_dirpath.mkdir(exist_ok=True)
        self.output_dirpath.mkdir(exist_ok=True)

        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{self.name}.csv"
        self.results_df: pd.DataFrame | None = None

    def save_results(self):
        self.results_df.to_csv(self.output_csv_filepath)


class DatasetEvaluator(Evaluator):

    def __init__(self, output_base_dirpath: Path | str):
        super().__init__(output_base_dirpath)


class BuildingsEvaluator(Evaluator):

    def __init__(self, output_base_dirpath: Path | str):
        super().__init__(output_base_dirpath)


class PointMeshDistanceEvaluator(DatasetEvaluator):
    name = "point_mesh_distance"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            point_cloud_filepath: Path | str,
            mesh_filepath: Path | str,
            crs: str
    ):
        super().__init__(output_base_dirpath)
        self.point_cloud_filepath = point_cloud_filepath
        self.mesh_filepath = mesh_filepath
        self.crs = crs

        self.points_and_distances: np.ndarray | None = None
        self.points_and_distances_df: pd.DataFrame | None = None
        self.distances_hist_df: pd.DataFrame | None = None

    def compute_building_point_mesh_distance(self):
        point_array = read_building_points(self.point_cloud_filepath, self.crs)
        distances = compute_point_mesh_distance(point_array, self.mesh_filepath)

        self.results_df = pd.DataFrame({
            "RMS": np.sqrt(np.mean(np.square(distances))),
            "mean": distances.mean(),
            "median": np.median(distances),
            "min": distances.min(),
            "max": distances.max()
        }, index=["buildings"])

        self.points_and_distances = np.concatenate((point_array, np.transpose([distances])), axis=1)
        self.points_and_distances_df = pd.DataFrame(self.points_and_distances, columns=["X", "Y", "Z", "dist"])
        self.points_and_distances_df.index.name = "id"

        counts, bins = np.histogram(distances, bins=20)
        self.distances_hist_df = pd.DataFrame({"greater": bins[:-1], "smaller": bins[1:], "count": counts})
        self.distances_hist_df.index.name = "bin"

    @print_starting_message
    def run(self):
        self.compute_building_point_mesh_distance()
        self.points_and_distances_df.to_csv(self.output_dirpath / f"evaluation_{self.name}_all_points.csv")
        self.distances_hist_df.to_csv(self.output_dirpath / f"evaluation_{self.name}_histogram.csv")
        self.save_results()


class HorizontalAndVerticalPointMeshDistanceEvaluator(DatasetEvaluator):
    pass


class PointMeshDistanceBuildingsEvaluator(BuildingsEvaluator):
    pass


class HausdorffLODSEvaluator(BuildingsEvaluator):
    name = "hausdorff"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_obj_filepath_pairs: dict[str, tuple[Path | str, Path | str]]
    ):
        super().__init__(output_base_dirpath)
        self.input_obj_filepath_pairs = input_obj_filepath_pairs

        self.hausdorff_evaluators: dict[str, HausdorffEvaluator] = {}

    def setup_hausdorff_evaluators(self):
        print("Setting up a HausdorffEvaluator for each LOD ...\n")
        for lod, filepaths in self.input_obj_filepath_pairs.items():
            input_obj_filepath_1, input_obj_filepath_2 = filepaths
            self.hausdorff_evaluators[lod] = HausdorffEvaluator(
                output_base_dirpath=self.output_base_dirpath,
                input_obj_filepath_1=input_obj_filepath_1,
                input_obj_filepath_2=input_obj_filepath_2
            )

    def run_hausdorff_evaluators(self):
        for lod, hausdorff_evaluator in self.hausdorff_evaluators.items():
            print(f"Running HausdorffEvaluator for LOD '{lod}' ...\n")
            hausdorff_evaluator.split_obj_files()
            hausdorff_evaluator.compute_hausdorff_distances()
            print()
        print("Joining results ...")
        self.results_df = pd.concat(
            [evaluator.results_df.add_suffix(f".{lod}") for lod, evaluator in self.hausdorff_evaluators.items()],
            axis=1
        )

    @print_starting_message
    def run(self):
        self.setup_hausdorff_evaluators()
        self.run_hausdorff_evaluators()
        self.save_results()


class HausdorffEvaluator(BuildingsEvaluator):
    name = "hausdorff"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_obj_filepath_1: Path | str,
            input_obj_filepath_2: Path | str,
    ):
        super().__init__(output_base_dirpath)
        self.input_obj_filepath_1 = Path(input_obj_filepath_1)
        self.input_obj_filepath_2 = Path(input_obj_filepath_2)

        self.split_input_obj_dirpath_1 = self.input_obj_filepath_1.parent / self.input_obj_filepath_1.stem
        self.split_input_obj_dirpath_2 = self.input_obj_filepath_2.parent / self.input_obj_filepath_2.stem

    def split_obj_files(self, quick_skip=True):
        """Split both input OBJ files up into one OBJ file for each object group found within the input file
        :param quick_skip: If a subdirectory with the name (without extension) of the respective input OBJ file exists
        already, assume that the input OBJ file was split previously and skip the splitting step
        """
        print("Splitting OBJ files into one file per individual object ...")
        if not quick_skip or not self.split_input_obj_dirpath_1.is_dir():
            split_obj_file(self.input_obj_filepath_1, output_dirpath=self.split_input_obj_dirpath_1, overwrite=False)
        if not quick_skip or not self.split_input_obj_dirpath_2.is_dir():
            split_obj_file(self.input_obj_filepath_2, output_dirpath=self.split_input_obj_dirpath_2, overwrite=False)

    def compute_hausdorff_distances(self, very_verbose=False):
        print("Identifying individual OBJ files present in both input datasets ...")
        print("- Directories:")
        print(f"  Input 1: {self.split_input_obj_dirpath_1}")
        print(f"  Input 2: {self.split_input_obj_dirpath_2}")

        # Get names of all OBJ files in the subdirectory containing the individual building OBJ files
        split_input_obj_filenames_1 = sorted([fn.name for fn in self.split_input_obj_dirpath_1.glob("*.obj")])
        split_input_obj_filenames_2 = sorted([fn.name for fn in self.split_input_obj_dirpath_2.glob("*.obj")])

        # Identify common and different file names between the two datasets to be compared
        common_filenames = sorted(list(set(split_input_obj_filenames_1).intersection(split_input_obj_filenames_2)))
        filenames_only_1 = sorted(list(set(split_input_obj_filenames_1).difference(split_input_obj_filenames_2)))
        filenames_only_2 = sorted(list(set(split_input_obj_filenames_2).difference(split_input_obj_filenames_1)))

        print(f"- File comparison: Found {len(common_filenames)+len(filenames_only_1)+len(filenames_only_2)} unique "
              f"OBJ file names in total, of which")
        print(f"  {len(common_filenames)} file names present both directories")
        print(f"  {len(filenames_only_1)} file names present only in directory 1")
        print(f"  {len(filenames_only_2)} file names present only in directory 2")
        if very_verbose:
            if len(filenames_only_1) > 0:
                print("- Files present only in directory 1:")
                print("\n".join([f"  {fn}" for fn in filenames_only_1]))
            if len(filenames_only_1) > 0:
                print("- Files present only in directory 2:")
                print("\n".join([f"  {fn}" for fn in filenames_only_2]))

        print("Computing Hausdorff distances between split OBJ files ...")
        results = []
        for i, filename in enumerate(common_filenames):
            hausdorff_io_dict, hausdorff_oi_dict = get_bidirectional_hausdorff_distances(
                self.split_input_obj_dirpath_1 / filename,
                self.split_input_obj_dirpath_2 / filename
            )
            results_dict = {"file": filename,  "hausdorff": max(hausdorff_io_dict["max"], hausdorff_oi_dict["max"])}
            results_dict.update({f"{key}.io": value for key, value in hausdorff_io_dict.items()})
            results_dict.update({f"{key}.oi": value for key, value in hausdorff_oi_dict.items()})
            results.append(results_dict)

        self.results_df = pd.DataFrame(results).set_index("file")
        self.results_df.index.name = glb.target_identifier_name

    @print_starting_message
    def run(self):
        self.split_obj_files()
        self.compute_hausdorff_distances()
        self.save_results()


class PointDensityDatasetEvaluator(DatasetEvaluator):
    name = "point_density"
    radial_density_radius = np.sqrt(1/np.pi)  # Radius of disk with area 1 m²

    def __init__(
            self,
            output_base_dirpath: Path | str,
            point_cloud_filepath: Path | str,
            bbox: list[float],
            building_footprints: gpd.GeoSeries,
            crs: str,
            save_filtered_point_clouds: bool = False
    ):
        super().__init__(output_base_dirpath)
        self.point_cloud_filepath = Path(point_cloud_filepath)
        self.bbox = bbox
        self.building_footprints = building_footprints
        self.crs = crs
        self.save_filtered_point_clouds = save_filtered_point_clouds

        self.results_df = pd.DataFrame(columns=["points", "area", "density"])

    def compute_overall_ground_density(self):
        print("Computing overall ground density ...")
        reader = pdal.Reader(str(self.point_cloud_filepath), nosrs=True, default_srs=self.crs)

        filter_expression = f"X >= {self.bbox[0]} && X <= {self.bbox[2]} && Y >= {self.bbox[1]} && Y <= {self.bbox[3]}"
        bbox_filter = pdal.Filter.expression(expression=filter_expression)

        pipeline = reader | bbox_filter
        n_points = pipeline.execute()

        if self.save_filtered_point_clouds:
            self.save_filtered_point_cloud(pipeline, "_bbox")

        bbox_poly = box(*self.bbox)
        area = bbox_poly.area
        density = n_points / area

        self.results_df.loc["overall", :] = [n_points, area, density]

    def compute_buildings_ground_density(self):
        print("Computing ground density within building footprints ...")
        reader = pdal.Reader(str(self.point_cloud_filepath), nosrs=True, default_srs=self.crs)

        polygon_list = list(self.building_footprints.to_wkt())
        cropper = pdal.Filter.crop(polygon=polygon_list)

        pipeline = reader | cropper
        n_points = pipeline.execute()

        if self.save_filtered_point_clouds:
            self.save_filtered_point_cloud(pipeline, "_buildings")

        footprints_area = self.building_footprints.area.sum()
        density = n_points / footprints_area

        self.results_df.loc["buildings", :] = [n_points, footprints_area, density]

    def compute_radial_density(self):
        print("Computing local (radial) volume and surface densities ...")

        reader = pdal.Reader(str(self.point_cloud_filepath), nosrs=True, default_srs=self.crs)

        filter_expression = f"X >= {self.bbox[0]} && X <= {self.bbox[2]} && Y >= {self.bbox[1]} && Y <= {self.bbox[3]}"
        bbox_filter = pdal.Filter.expression(expression=filter_expression)

        radial_density = pdal.Filter.radialdensity(radius=PointDensityDatasetEvaluator.radial_density_radius)

        sphere_volume = 4 / 3 * np.pi * PointDensityDatasetEvaluator.radial_density_radius ** 3
        sphere_cross_section_area = np.pi * PointDensityDatasetEvaluator.radial_density_radius ** 2
        volume_to_surface_density = f"SurfaceDensity = RadialDensity * {sphere_volume} / {sphere_cross_section_area}"
        surface_density = pdal.Filter.assign(value=volume_to_surface_density)

        pipeline = reader | bbox_filter | radial_density | surface_density
        pipeline.execute()
        point_array = pipeline.arrays[0]
        point_df = pd.DataFrame(point_array)[["RadialDensity", "SurfaceDensity"]]

        for name, density_type in {"local_volume": "RadialDensity", "local_surface": "SurfaceDensity"}.items():
            density = point_df[density_type]
            self.results_df.loc[name, ["mean", "median", "std", "min", "max"]] = [
                density.mean(), density.median(), density.std(), density.min(), density.max()
            ]

    def save_filtered_point_cloud(self, input_pipeline: pdal.Pipeline, filename_suffix: str):
        output_filename = self.point_cloud_filepath.stem + filename_suffix + self.point_cloud_filepath.suffix
        output_filepath = self.output_dirpath / output_filename

        writer = pdal.Writer(filename=str(output_filepath), minor_version=4, a_srs=self.crs, compression=True)

        # pdal.Filter.crop returns a list of structured NumPy arrays: One for each of the input crop polygons. To
        # make sure the points for all buildings are written, these arrays must be concatenated, as opposed to the
        # output of many other PDAL filters, where the entire output point cloud is stored in pipeline.arrays[0].
        point_array = np.concatenate(input_pipeline.arrays)
        pipeline = writer.pipeline(point_array)
        pipeline.execute()

    @print_starting_message
    def run(self):
        self.compute_overall_ground_density()
        self.compute_buildings_ground_density()
        self.compute_radial_density()
        self.save_results()


class PointDensityBuildingsEvaluator(BuildingsEvaluator):
    pass


class FMEEvaluator(BuildingsEvaluator):
    name = "fme_evaluator"
    index_col_name = glb.geoflow_output_cityjson_identifier_name

    def __init__(self, output_base_dirpath: Path | str, lods: list[str], crs: str):
        super().__init__(output_base_dirpath)
        self.lods = lods
        self.crs = crs

        self.output_cityjson_filepath: Path = self.output_dirpath / f"model_{self.name}.json"
        self.fme_output_cityjson_filepaths = {}

        self.fme_pipelines_dict: dict[str, FMEPipeline] = {}
        self.results_template_cityjson_filepath: Path | None = None
        self.fme_output_attributes: list[str] = []

    def run_fme_pipeline(self):
        print(f"Running FME pipelines for each LOD ...\n")
        self.output_dirpath.mkdir(exist_ok=True)

        self.fme_output_cityjson_filepaths = {}
        for lod, fme_pipeline in self.fme_pipelines_dict.items():
            fme_pipeline.run()
            self.fme_output_cityjson_filepaths[lod] = fme_pipeline.output_cityjson_filepath

    def merge_lods_results(self):
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

        print(f"Preparing results dataframe ...")
        print(f"- Path: {self.output_csv_filepath}")

        # Edit the buildings dictionary such that its values only contain the values of the attributes dictionary
        # contained in it
        for key in input_cityjson_buildings.keys():
            input_cityjson_buildings[key] = input_cityjson_buildings[key].attributes
        # Create a dataframe where the dict keys form the index, and set the column specified in self.index_col_name
        # (typically `identificatie`) as index instead of the cryptical FID-derived values (e.g., "1", "1-0", ...)
        self.results_df = pd.DataFrame.from_dict(input_cityjson_buildings, orient="index")
        if self.index_col_name != "":
            self.results_df = self.results_df.set_index(self.index_col_name)
            self.results_df.index.name = glb.target_identifier_name

    @print_starting_message
    def run(self):
        self.run_fme_pipeline()
        self.merge_lods_results()
        self.save_results()


class AreaVolumeEvaluator(FMEEvaluator):
    name = "area_volume"
    index_col_name = glb.geoflow_output_cityjson_identifier_name

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_cityjson_filepath: Path | str,
            lods: list[str],
            crs: str
    ):
        super().__init__(output_base_dirpath, lods=lods, crs=crs)
        self.input_cityjson_filepath = Path(input_cityjson_filepath)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath
        self.fme_output_attributes = [glb.fme_area_field_name, glb.fme_volume_field_name]

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineAreaVolume(
                input_cityjson_filepath=self.input_cityjson_filepath,
                output_dirpath=self.output_dirpath,
                crs=self.crs,
                lod=lod
            )


class IOU3DEvaluator(FMEEvaluator):
    name = "iou_3d"
    index_col_name = glb.geoflow_output_cityjson_identifier_name

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            lods: list[str],
            crs: str
    ):
        super().__init__(output_base_dirpath, lods=lods, crs=crs)
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath_2
        self.fme_output_attributes = [
            glb.fme_iou3d_volume_input_field_name,
            glb.fme_iou3d_volume_output_field_name,
            glb.fme_iou3d_volume_intersection_field_name,
            glb.fme_iou3d_iou_field_name
        ]

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineIOU3D(
                input_cityjson_filepath_1=self.input_cityjson_filepath_1,
                input_cityjson_filepath_2=self.input_cityjson_filepath_2,
                output_dirpath=self.output_dirpath,
                crs=self.crs,
                lod=lod
            )

