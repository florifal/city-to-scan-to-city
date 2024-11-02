import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pymeshlab
import pdal
import open3d as o3d
import numpy.lib.recfunctions as rfn
from pathlib import Path
from datetime import timedelta
from cjio import cityjson
from shapely import box

import experiment.global_vars as glb
from experiment.fme_pipeline import FMEPipeline, FMEPipelineAreaVolume, FMEPipelineIOU3D
from experiment.obj_file import split_obj_file, OBJFile, get_face_count_from_obj
from experiment.utils import rms, get_face_count_from_gpkg, describe_value_counts


def print_starting_message(func):
    def wrapper(self):
        print(f"\nStarting {self.__class__.__name__} ...")
        t0 = time.time()
        func(self)
        t1 = time.time()
        print(f"\nFinished {self.__class__.__name__} after {str(timedelta(seconds=t1 - t0))}.")
    return wrapper


def get_bidirectional_hausdorff_distances(
        obj_filepath_1: Path | str,
        obj_filepath_2: Path | str,
        sample_vertices: bool = True,
        sample_edges: bool = True,
        sample_faces: bool = True,
        sample_number: int = 10000
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """
    Computes both unidirectional Hausdorff distances for two OBJ meshes using PyMeshLab
    """
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
        sampledmesh=1,
        targetmesh=0,
        savesample=False,
        samplevert=sample_vertices,
        sampleedge=sample_edges,
        sampleface=sample_faces,
        samplenum=sample_number
    )

    return hausdorff_12_dict, hausdorff_21_dict


def get_unidirectional_hausdorff_distances(
        obj_filepath_1: Path | str,
        obj_filepath_2: Path | str,
        sample_vertices: bool = True,
        sample_edges: bool = True,
        sample_faces: bool = True,
        sample_number: int = 10000
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """
    Computes unidirectional Hausdorff distance for two OBJ meshes using PyMeshLab

    This implementation does not make much sense and its form can be explained by the fact that it is mostly a remnant
    from an earlier application in the HausdorffEvaluator where unintentionally only the unidirectional Hausdorff
    distance was computed.
    """
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
    # Originally, the previous ms.get_hausdorff_distance() would be restated here, with exactly the same arguments,
    # instead of swapping sampledmesh and targetmesh. This delivers slightly different values for mean and RMS, probably
    # due to the random sampling of the mesh's surface. Now, just to keep the function structure identical to the one
    # that computes the actual bidirectional distance, we simply copy the dict.
    hausdorff_21_dict = hausdorff_12_dict.copy()

    return hausdorff_12_dict, hausdorff_21_dict


def read_building_points(point_cloud_filepath: Path | str, crs: str, classification: int = 6, verbose: bool = True):
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

    def __init__(self, output_base_dirpath: Path | str, name: str | None = None):
        # If a name is provided, replace the class attribute `name` for this instance
        if name is not None:
            self.name = name

        self.output_base_dirpath = Path(output_base_dirpath)
        self.output_dirpath = self.output_base_dirpath / self.name

        self.output_base_dirpath.mkdir(exist_ok=True)
        self.output_dirpath.mkdir(exist_ok=True)

        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{self.name}.csv"
        self.results_df: pd.DataFrame | None = None
        self._summary_stats = {}
        self.final_columns = []

    def run(self):
        pass

    def save_results(self):
        self.results_df.to_csv(self.output_csv_filepath)

    def load_results(self):
        self.results_df = pd.read_csv(self.output_csv_filepath, index_col=0)

    def compute_summary_stats(self):
        pass

    @property
    def results(self) -> pd.DataFrame:  # todo: call this "results_df" and change results_df to _results_df, including in subclasses
        if self.results_df is None or self.results_df.empty:
            self.load_results()
        return self.results_df

    @property
    def summary_stats(self) -> dict:
        """A dictionary of descriptive statistics of the relevant results"""
        if self._summary_stats == {}:
            self.compute_summary_stats()
        return self._summary_stats

    @property
    def results_final(self) -> pd.DataFrame:
        """All relevant columns with results for all buildings (BuildingsEvaluator) or the dataset (DatasetEvaluator)"""
        if isinstance(self, BuildingsEvaluator):
            return pd.concat([
                self.results[self.final_columns],
                self.results[self.final_columns].describe(),  # add summary statistics
                pd.DataFrame(self.results[self.final_columns].apply(rms).rename("rms")).transpose()  # add RMS
            ], axis=0)
        else:
            return self.results[self.final_columns]



class DatasetEvaluator(Evaluator):

    def __init__(self, output_base_dirpath: Path | str, name: str | None = None):
        super().__init__(output_base_dirpath, name)


class BuildingsEvaluator(Evaluator):

    def __init__(self, output_base_dirpath: Path | str, name: str | None = None):
        super().__init__(output_base_dirpath, name)


class PointDensityDatasetEvaluator(DatasetEvaluator):
    name = "point_density"
    radial_density_radius = np.sqrt(1/np.pi)  # Radius of disk with area 1 m²

    def __init__(
            self,
            output_base_dirpath: Path | str,
            point_cloud_filepath: Path | str,
            bbox: list[float],
            building_footprints_gpkg_filepath: Path | str,
            crs: str,
            save_filtered_point_clouds: bool = False,
            footprints_density_computation: bool = True,
            radial_density_computation: bool = True
    ):
        super().__init__(output_base_dirpath)
        self.point_cloud_filepath = Path(point_cloud_filepath)
        self.bbox = bbox
        self.building_footprints_gpkg_filepath = Path(building_footprints_gpkg_filepath)
        self.building_footprints: gpd.GeoSeries | None = None
        self.crs = crs
        self.save_filtered_point_clouds = save_filtered_point_clouds
        self.footprints_density_computation = footprints_density_computation
        self.radial_density_computation = radial_density_computation

        self.results_df = pd.DataFrame()
        self.final_columns = ["density_overall"]
        if self.footprints_density_computation:
            self.final_columns.append("density_buildings")
        if self.radial_density_computation:
            self.final_columns.extend(["density_local_volume", "density_local_surface"])

    def compute_overall_ground_density(self):
        print("\nComputing overall ground density ...")
        print("- Reading point cloud and clipping to bbox ...")
        reader = pdal.Reader(str(self.point_cloud_filepath), nosrs=True, default_srs=self.crs)

        filter_expression = f"X >= {self.bbox[0]} && X <= {self.bbox[2]} && Y >= {self.bbox[1]} && Y <= {self.bbox[3]}"
        bbox_filter = pdal.Filter.expression(expression=filter_expression)

        pipeline = reader | bbox_filter
        n_points = pipeline.execute()

        if self.save_filtered_point_clouds:
            print("- Writing clipped point cloud ...")
            self.save_filtered_point_cloud(pipeline, "_bbox")

        bbox_poly = box(*self.bbox)
        area = bbox_poly.area
        density = n_points / area

        return pd.Series(
            name="density_overall",
            data={"points": n_points, "area": area, "mean": density}
        )

    def compute_buildings_ground_density(self):
        print("\nComputing ground density within building footprints ...")
        print("- Reading building footprints ...")

        self.building_footprints = gpd.read_file(
            self.building_footprints_gpkg_filepath,
            layer=gpd.list_layers(self.building_footprints_gpkg_filepath).name.loc[0]
        ).geometry

        print("- Reading point cloud and clipping to footprints ...")
        reader = pdal.Reader(str(self.point_cloud_filepath), nosrs=True, default_srs=self.crs)

        # todo: consider to count points with Classification == 6 instead of within building footprints
        polygon_list = list(self.building_footprints.to_wkt())
        cropper = pdal.Filter.crop(polygon=polygon_list)

        pipeline = reader | cropper
        n_points = pipeline.execute()

        if self.save_filtered_point_clouds:
            print("- Writing clipped point cloud ...")
            self.save_filtered_point_cloud(pipeline, "_buildings")

        footprints_area = self.building_footprints.area.sum()
        density = n_points / footprints_area

        return pd.Series(
            name="density_buildings",
            data={"points": n_points, "area": footprints_area, "mean": density}
        )

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

        density_types = {"density_local_volume": "RadialDensity", "density_local_surface": "SurfaceDensity"}
        row_names = ["mean", "median", "std", "min", "max"]
        df = pd.DataFrame(index=row_names, columns=list(density_types.keys()))
        for col_name, density_type in density_types.items():
            density = point_df[density_type]
            df.loc[row_names, col_name] = [
                density.mean(), density.median(), density.std(), density.min(), density.max()
            ]
        return df

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
        results = []
        results.append(self.compute_overall_ground_density())
        if self.footprints_density_computation:
            results.append(self.compute_buildings_ground_density())
        if self.radial_density_computation:
            results.append(self.compute_radial_density())
        self.results_df = pd.concat(results, axis=1)
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {
            col_name: self.results.loc["mean", col_name] for col_name in self.final_columns
        }

    @property
    def results_final(self):
        # Find available rows. "std", "min", and "max" are only available if compute_radial_density() was used.
        rows = [row for row in ["mean", "std", "min", "max"] if row in self.results.index]
        return self.results.loc[rows, self.final_columns]


# todo
# class PointDensityBuildingsEvaluator(BuildingsEvaluator):
#     pass


class PointMeshDistanceEvaluator(DatasetEvaluator):
    name = "point_mesh_distance"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            point_cloud_filepath: Path | str,
            mesh_filepath: Path | str,
            crs: str,
            save_all_points_with_distances: bool = False
    ):
        super().__init__(output_base_dirpath)
        self.point_cloud_filepath = point_cloud_filepath
        self.mesh_filepath = mesh_filepath
        self.crs = crs
        self.save_all_points_with_distances = save_all_points_with_distances

        self.points_and_distances: np.ndarray | None = None
        self.points_and_distances_df: pd.DataFrame | None = None
        self.distances_hist_df: pd.DataFrame | None = None

        self.final_columns = ["point_mesh_distance"]

    def compute_building_point_mesh_distance(self):
        print("Reading point clouds and filtering for points on buildings ...\n")
        point_array = read_building_points(self.point_cloud_filepath, self.crs)
        print("Computing point-mesh distance ...")
        distances = compute_point_mesh_distance(point_array, self.mesh_filepath)

        self.results_df = pd.DataFrame(
            pd.concat([
                pd.Series(distances).describe(),
                pd.Series({"rms": rms(distances)})
            ]).rename("point_mesh_distance")
        )

        self.points_and_distances = np.concatenate((point_array, np.transpose([distances])), axis=1)
        self.points_and_distances_df = pd.DataFrame(self.points_and_distances, columns=["X", "Y", "Z", "dist"])
        self.points_and_distances_df.index.name = "id"

        counts, bins = np.histogram(distances, bins=20)
        self.distances_hist_df = pd.DataFrame({"greater": bins[:-1], "smaller": bins[1:], "count": counts})
        self.distances_hist_df.index.name = "bin"

    @print_starting_message
    def run(self):
        self.compute_building_point_mesh_distance()
        if self.save_all_points_with_distances:
            self.points_and_distances_df.to_csv(self.output_dirpath / f"evaluation_{self.name}_all_points.csv")
        self.distances_hist_df.to_csv(self.output_dirpath / f"evaluation_{self.name}_histogram.csv")
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {
            f"{self.name}_{metric}": value
            for metric, value in self.results.point_mesh_distance.to_dict().items()
        }


# todo
# class HorizontalAndVerticalPointMeshDistanceEvaluator(DatasetEvaluator):
#     pass


# todo: It could be interesting to know the point-mesh distance for each building independently. The caveat is that this
# would require a 1:1 assignment of points to specific buildings, i.e., a passing-through or use of the hit object ID
# from the ALS simulation to know which point was measured from which building. And since the scene currently consists
# of a single mesh, this would require to separate the building objects in the Wavefront file at first and adding them
# to the scene definition XML individually. Ouch!
# class PointMeshDistanceBuildingsEvaluator(BuildingsEvaluator):
#     pass


class GeoflowOutputEvaluator(DatasetEvaluator):
    """
    Counts the numbers of (unique) buildings, buildings parts, and related values in Geoflow output files
    """
    name = "geoflow_output"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            gpkg_filepath: Path | str,
            gpkg_layers: dict[str, str],
            cityjson_filepath: Path | str,
            obj_filepaths: dict[str, Path | str]
    ):
        super().__init__(output_base_dirpath)

        # self.gpkg_filepath = gpkg_filepath
        # self.cityjson_filepath = cityjson_filepath
        # self.obj_filepaths = {lod_name: Path(fp) for lod_name, fp in obj_filepaths.items()}

        self.gpkg_eval = GeopackageBuildingsEvaluator(
            output_base_dirpath,
            gpkg_filepath,
            gpkg_layers,
            id_col_name=glb.geoflow_output_cityjson_identifier_name,
            name=self.name
        )
        self.cityjson_eval = CityJSONBuildingsEvaluator(
            output_base_dirpath,
            cityjson_filepath,
            name=self.name
        )
        self.wavefront_eval = WavefrontOBJBuildingsEvaluator(
            output_base_dirpath,
            obj_filepaths,
            name=self.name
        )

    @print_starting_message
    def run(self):
        self.gpkg_eval.run()
        self.cityjson_eval.run()
        self.wavefront_eval.run()

        self.results_df = pd.concat(
            [
                self.gpkg_eval.results.add_prefix("gpkg_", axis=0),
                self.cityjson_eval.results.add_prefix("cj_", axis=0),
                self.wavefront_eval.results.add_prefix("obj_", axis=0)
            ],
            axis=0
        )
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {}
        self._summary_stats.update({
            f"{row_name_new}_{lod}": self.results.loc[row_name_results, lod]
            for row_name_new, row_name_results in {"gpkg_total": "gpkg_num_total",
                                                   "gpkg_unique": "gpkg_num_unique",
                                                   "gpkg_multiple": "gpkg_num_multiple"}.items()
            for lod in self.gpkg_eval.gpkg_layers.keys()
        })
        self._summary_stats.update({
            row_name_new: self.results.loc[row_name_results, "cityjson"]
            for row_name_new, row_name_results in {"cj_buildings": "cj_num_buildings",
                                                   "cj_building_parts": "cj_num_building_parts",
                                                   "cj_buildings_with_bp": "cj_num_buildings_with_bp",
                                                   "cj_buildings_zero_bp": "cj_num_buildings_zero_bp",
                                                   "cj_buildings_one_bp": "cj_num_buildings_one_bp",
                                                   "cj_buildings_multiple_bp": "cj_num_buildings_multiple_bp"}.items()
        })

    @property
    def results_final(self) -> pd.DataFrame:
        return pd.DataFrame()


class GeopackageBuildingsEvaluator(DatasetEvaluator):
    name = "gpkg_buildings"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            gpkg_filepath: Path | str,
            gpkg_layers: dict[str, str],
            id_col_name: str,
            name: str | None = None
    ):
        super().__init__(output_base_dirpath, name)
        self.gpkg_filepath = Path(gpkg_filepath)
        self.gpkg_layers = {lod.replace(".", ""): layer_name for lod, layer_name in gpkg_layers.items()}
        self.id_col_name = id_col_name

    def count(self):
        value_counts_descr = {}
        for lod, layer_name in self.gpkg_layers.items():
            gpkg = gpd.read_file(self.gpkg_filepath, layer=layer_name)
            value_counts_descr[lod] = describe_value_counts(gpkg[self.id_col_name])

        self.results_df = pd.DataFrame.from_dict(value_counts_descr, orient="columns")

    @print_starting_message
    def run(self):
        self.count()
        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{GeopackageBuildingsEvaluator.name}.csv"
        self.save_results()


class CityJSONBuildingsEvaluator(DatasetEvaluator):
    """
    Information about buildings and building parts in a CityJSON file

    Currently, this Evaluator does not distinguish between LODs, but reports results across all LODs in the CityJSON.
    """
    name = "cityjson_buildings"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            cityjson_filepath: Path | str,
            name: str | None = None
    ):
        super().__init__(output_base_dirpath, name)
        self.cityjson_filepath = Path(cityjson_filepath)

    @print_starting_message
    def run(self):
        cj = cityjson.load(self.cityjson_filepath)
        b = cj.get_cityobjects(type="building")
        bp = cj.get_cityobjects(type="buildingpart")
        cj_building_stats = {
            "num_buildings": len(b),
            "num_building_parts": len(bp),
            "num_buildings_with_bp": len([bu for buid, bu in b.items() if len(bu.children) > 0]),
            "num_buildings_zero_bp": len([bu for buid, bu in b.items() if len(bu.children) == 0]),
            "num_buildings_one_bp": len([bu for buid, bu in b.items() if len(bu.children) == 1]),
            "num_buildings_multiple_bp": len([bu for buid, bu in b.items() if len(bu.children) > 1]),
            "buildings_no_bp": [buid for buid, bu in b.items() if len(bu.children) == 0],
            # "buildings_one_bp": [buid for buid, bu in b.items() if len(bu.children) == 1],
            "buildings_multiple_bp": [buid for buid, bu in b.items() if len(bu.children) > 1],
        }

        self.results_df = pd.DataFrame.from_dict({"cityjson": cj_building_stats}, orient="columns")
        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{CityJSONBuildingsEvaluator.name}.csv"
        self.save_results()


class WavefrontOBJBuildingsEvaluator(DatasetEvaluator):
    name = "wavefront_obj_buildings"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            obj_filepaths: dict[str, Path | str],
            name: str | None = None
    ):
        super().__init__(output_base_dirpath, name)
        self.obj_filepaths = {lod.replace(".", ""): Path(fp) for lod, fp in obj_filepaths.items()}

    @print_starting_message
    def run(self):
        obj_building_stats = {}
        for lod, obj_fp in self.obj_filepaths.items():
            obj = OBJFile(obj_fp)
            obj_building_stats[lod] = {
                "num_objects": len(obj.objects),
                "num_objects_with_faces": len([o.name for o in obj.objects if len(o.faces) > 0]),
                "num_objects_zero_faces": len([o.name for o in obj.objects if len(o.faces) == 0]),
                "objects_zero_faces": [o.name for o in obj.objects if len(o.faces) == 0],
                # "describe_value_counts": describe_value_counts([o.name for o in obj.objects])
            }

        self.results_df = pd.DataFrame.from_dict(obj_building_stats, orient="columns")
        self.output_csv_filepath: Path = self.output_dirpath / f"evaluation_{WavefrontOBJBuildingsEvaluator.name}.csv"
        self.save_results()


class HeightDifferenceEvaluator(BuildingsEvaluator):
    """
    Obtains building heights from two CityJSON files for corresponding buildings and computes difference metrics
    """
    name = "height_diff"
    height_metrics = ["min", "50p", "70p", "max", "ground"]

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            identifier_name: str,
            height_metrics: list[str] | None = None
    ):
        super().__init__(output_base_dirpath)
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)
        self.identifier_name = identifier_name

        if height_metrics is not None:
            self.height_metrics = height_metrics

        self.cityjson_1: cityjson.CityJSON | None = None
        self.cityjson_2: cityjson.CityJSON | None = None

        self.final_columns = [
            col_name
            for height_metric in self.height_metrics
            for col_name in [
                f"h_{height_metric}_in",
                f"h_{height_metric}_out",
                f"h_{height_metric}_diff",
                f"h_{height_metric}_abs_diff",
                f"h_{height_metric}_ratio",
                f"h_{height_metric}_norm_diff",
                f"h_{height_metric}_norm_abs_diff"
            ]
        ]

    def load_cityjson_files(self):
        print("Loading CityJSON files ...")
        self.cityjson_1 = cityjson.load(self.input_cityjson_filepath_1)
        self.cityjson_2 = cityjson.load(self.input_cityjson_filepath_2)

    def get_building_heights(self):
        buildings_1 = self.cityjson_1.get_cityobjects(type="building")
        buildings_2 = self.cityjson_2.get_cityobjects(type="building")

        heights = {}

        print("Reading building heights from output CityJSON file ...")
        # Get the heights from all buildings in cityjson_2, and use the identifier in the field `identifier_name` as key
        for b in buildings_2.values():
            heights[b.attributes[self.identifier_name]] = {
                f"h_{height_metric}_out": b.attributes[glb.geoflow_output_height_attr_names[height_metric]]
                for height_metric in self.height_metrics
            }

        print("Reading building heights from input CityJSON file ...")
        # Get the heights from the corresponding buildings in cityjson_1 as identified by the keys
        for identifier in heights.keys():
            heights[identifier].update({
                f"h_{height_metric}_in": buildings_1[identifier].attributes[glb.geoflow_input_height_attr_names[height_metric]]
                for height_metric in self.height_metrics
            })

        self.results_df = pd.DataFrame.from_dict(heights, orient="index")

    def compute_height_differences(self):
        print("Computing building height differences ...")
        for height_metric in self.height_metrics:
            prefix = f"h_{height_metric}_"
            metric_in = self.results_df[prefix + "in"]
            metric_out = self.results_df[prefix + "out"]
            self.results_df[prefix + "diff"] = metric_out - metric_in
            self.results_df[prefix + "abs_diff"] = self.results_df[prefix + "diff"].abs()
            self.results_df[prefix + "ratio"] = metric_out / metric_in
            self.results_df[prefix + "norm_diff"] = (self.results_df[prefix + "diff"] / metric_in.abs())
            self.results_df[prefix + "norm_abs_diff"] = (self.results_df[prefix + "abs_diff"] / metric_in.abs())

    @print_starting_message
    def run(self):
        self.load_cityjson_files()
        self.get_building_heights()
        self.compute_height_differences()
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {}
        for height_metric in self.height_metrics:
            prefix = f"h_{height_metric}_"
            results_available = self.results[
                ~self.results[prefix + "out"].isna() &
                ~self.results[prefix + "in"].isna()
            ]
            self._summary_stats.update(
                self.results[prefix + "out"].describe().add_prefix(prefix + "out_").to_dict()
            )
            self._summary_stats.update({
                prefix + "mean_diff": self.results[prefix + "diff"].mean(),
                prefix + "median_diff": self.results[prefix + "diff"].median(),
                prefix + "mean_abs_diff": self.results[prefix + "abs_diff"].mean(),
                prefix + "median_abs_diff": self.results[prefix + "abs_diff"].median(),
                prefix + "rms_diff": rms(self.results[prefix + "diff"]),

                # Mean of the normalized (absolute) difference and normalized mean (absolute) difference. For
                # post-averaging normalization, divide by the absolute value of the reference input metric to retain
                # the sign of the mean. Since number of observations is equal for numerator and denominator, taking the
                # sum instead of the mean is sufficient as the 1/n cancels out.
                prefix + "mean_norm_diff": self.results[prefix + "norm_diff"].mean(),
                prefix + "norm_mean_diff": results_available[prefix + "diff"].sum() /
                                           abs(results_available[prefix + "in"].sum()),
                prefix + "mean_norm_abs_diff": self.results[prefix + "norm_abs_diff"].mean(),
                prefix + "norm_mean_abs_diff": results_available[prefix + "abs_diff"].sum() /
                                               abs(results_available[prefix + "in"].sum())
            })


class ComplexityEvaluator(BuildingsEvaluator):
    """
    Evaluates the complexity of building models provided as GeoPackage or Wavefront OBJ
    """
    name = "complexity"
    possible_extensions = [".gpkg", ".obj"]

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_filepath: Path | str | list[Path] | list[str],
            lods: list[str],
            gpkg_index_col_name: str = "",
            gpkg_lod_layer_names: dict[str, str] | None = None,
            ignore_meshes_with_zero_faces: bool = True
    ):
        """ComplexityEvaluator

        :param output_base_dirpath: Directory for evaluation results
        :param input_filepath: Single filepath for Geopackage (.gpkg), list of filepaths for Wavefront OBJ (.obj)
        :param lods: List of LODs, e.g. ["1.2", "1.3", "2.2"]
        :param gpkg_index_col_name: (required for Geopackage) Name of identifier column, serves as output index
        :param ignore_meshes_with_zero_faces: (required for Wavefront OBJ) Whether to ignore or report zero for building meshes with zero faces
        """
        super().__init__(output_base_dirpath)

        # For Wavefront OBJ, a list of filepaths can be provided
        if isinstance(input_filepath, list):
            if len(input_filepath) != len(lods):
                raise ValueError("Number of input filepaths must be equal to number of LODs, or a single filepath.")
            for fp in input_filepath:
                self.validate_file_extension(fp)
            self.input_filepath = [Path(fp) for fp in input_filepath]
            self.input_file_extension = self.input_filepath[0].suffix.lower()

        # For Geopackage, a single filepath must be provided
        else:
            self.input_filepath = Path(input_filepath)
            self.input_file_extension = self.input_filepath.suffix.lower()

        self.lods = lods
        self.index_col_name = gpkg_index_col_name
        self.gpkg_lod_layer_names = gpkg_lod_layer_names
        self.ignore_meshes_with_zero_faces = ignore_meshes_with_zero_faces

        self.result_col_name = "n_faces"
        self.final_columns = [f"{self.result_col_name}_{lod.replace('.', '')}" for lod in self.lods]

    def validate_file_extension(self, filepath: Path | str):
        filepath = Path(filepath)
        file_extension = filepath.suffix.lower()
        if file_extension not in self.possible_extensions:
            raise TypeError(f"Input file type must be one of {self.possible_extensions}, but is {file_extension}.")

    def get_face_counts(self):
        face_counts = {}
        for i, lod in enumerate(self.lods):
            print(f"\nCounting faces for LOD {lod} ...")

            if self.input_file_extension == ".gpkg":
                face_counts[lod] = get_face_count_from_gpkg(
                    gpkg_filepath=self.input_filepath,
                    layer_name=self.gpkg_lod_layer_names[lod],
                    id_column=self.index_col_name,
                    result_col_name=self.result_col_name,
                    aggregate_by_id=True
                )
            elif self.input_file_extension == ".obj":
                face_counts[lod] = get_face_count_from_obj(
                    obj_filepath=self.input_filepath[i],
                    result_col_name=self.result_col_name,
                    ensure_as_triangle_count=True,
                    aggregate_building_parts=True
                )

            # Add LOD to series / column name
            face_counts[lod].name += f"_{lod.replace('.', '')}"
            if self.ignore_meshes_with_zero_faces:
                # Delete rows where n_faces is zero
                face_counts[lod] = face_counts[lod][face_counts[lod] != 0].copy()

        self.results_df = pd.concat(list(face_counts.values()), axis=1)
        self.results_df.index.name = glb.target_identifier_name

    @print_starting_message
    def run(self):
        self.get_face_counts()
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {
            f"{col_name}_{metric.replace('50%', 'median')}": value
            for col_name, column in self.results.describe().to_dict(orient="dict").items()
            for metric, value in column.items()
            if metric not in ["25%", "75%"]
        }


class ComplexityDifferenceEvaluator(BuildingsEvaluator):
    """
    Computes the difference in face number of the second evaluator's results w.r.t. the first evaluator's results.
    """
    name = "complexity_diff"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            complexity_evaluator_1: ComplexityEvaluator,
            complexity_evaluator_2: ComplexityEvaluator,
            reevaluate_1: bool = False,
            reevaluate_2: bool = False
    ):
        super().__init__(output_base_dirpath)
        self.evaluator_1 = complexity_evaluator_1
        self.evaluator_2 = complexity_evaluator_2
        self.reevaluate_1 = reevaluate_1
        self.reevaluate_2 = reevaluate_2

        self.suffix_1 = "_in"
        self.suffix_2 = "_out"

        # Only the difference metrics go into the final columns, not the columns with the numbers of faces of input and
        # output models (results of the two individual complexity evaluators). This is a design choice to keep the
        # regular ComplexityEvaluator as a default Scenario evaluator as well, which also contributes its final columns
        # to concat_evaluation_results(), to avoid duplicating these columns there. If the regular ComplexityEvaluator
        # gets kicked, the ComplexityDifferenceEvaluator may contribute the numbers of faces instead, too. Then you want
        # to uncomment the first two list entries.
        diff_metrics = ["in", "out", "diff", "abs_diff", "ratio", "norm_diff", "norm_abs_diff"]
        self.final_columns = [
            # *self.evaluator_1.final_columns,
            # *self.evaluator_2.final_columns,
            f"{c}_{metric}" for c in self.evaluator_2.final_columns for metric in diff_metrics
        ]

    def run_evaluators(self):
        for i, (reevaluate, evaluator) in enumerate(
                [(self.reevaluate_1, self.evaluator_1),
                 (self.reevaluate_2, self.evaluator_2)]
        ):
            evaluate = True
            if not reevaluate:
                try:
                    evaluator.load_results()
                except FileNotFoundError:
                    pass
                else:
                    print(f"\nEvaluation results for complexity evaluator {i+1} already exist. Will not reevaluate.")
                    evaluate = False

            if evaluate:
                evaluator.run()

    def compute_difference(self):
        # Do a right join with the results of the second ComplexityEvaluator, which should be the one evaluating the
        # reconstruction output. This ensures that the complexity difference is evaluated only for buildings that were
        # reconstructed.
        self.results_df = self.evaluator_1.results.join(self.evaluator_2.results, how="right",
                                                        lsuffix=self.suffix_1, rsuffix=self.suffix_2)

        for col_name in self.evaluator_2.results.columns:
            face_count_1 = self.results_df[col_name + self.suffix_1]
            face_count_2 = self.results_df[col_name + self.suffix_2]
            self.results_df[col_name + "_diff"] = face_count_2 - face_count_1
            self.results_df[col_name + "_abs_diff"] = (face_count_2 - face_count_1).abs()
            self.results_df[col_name + "_ratio"] = face_count_2 / face_count_1
            self.results_df[col_name + "_norm_diff"] = (face_count_2 - face_count_1) / face_count_1
            self.results_df[col_name + "_norm_abs_diff"] = (face_count_2 - face_count_1).abs() / face_count_1

    @print_starting_message
    def run(self):
        self.run_evaluators()
        self.compute_difference()
        self.save_results()

    def compute_summary_stats(self):
        diff_metric_columns = [c for c in self.results.columns if self.suffix_1 not in c and self.suffix_2 not in c]
        self._summary_stats = {
            f"{col_name}_{metric.replace('50%', 'median')}": value
            for col_name, column in self.results.describe().to_dict(orient="dict").items()
            for metric, value in column.items()
            if col_name in diff_metric_columns and metric not in ["count", "25%", "75%"]
        }


class HausdorffLODSEvaluator(BuildingsEvaluator):
    """
    Compute Hausdorff distances between all objects present in multiple pairs of Wavefront OBJ files
    """
    name = "hausdorff"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_obj_filepath_pairs: dict[str, tuple[Path | str, Path | str]],
            bidirectional: bool = False
    ):
        """

        :param output_base_dirpath: Directory for evaluation results
        :param input_obj_filepath_pairs: Dictionary. Key: Name of LOD. Value: Tuple of two paths to two OBJ files.
        """
        self.bidirectional = bidirectional
        self.col_suffix = "" if self.bidirectional else "_uni"
        self.directionality = "bidirectional" if self.bidirectional else "unidirectional"
        name = "hausdorff" if self.bidirectional else "hausdorff_uni"
        super().__init__(output_base_dirpath, name)

        # In case the LOD strings were passed with a "." (as in "1.2"), remove them
        self.input_obj_filepath_pairs = {lod.replace(".", ""): fp_tuple
                                         for lod, fp_tuple in input_obj_filepath_pairs.items()}
        self.lods = list(self.input_obj_filepath_pairs.keys())

        self.hausdorff_evaluators: dict[str, HausdorffEvaluator] = {}

        self.final_columns = [*[f"hausdorff{self.col_suffix}_{lod}" for lod in self.lods],
                              *[f"rms_min_dist{self.col_suffix}_{lod}" for lod in self.lods]]

    def setup_hausdorff_evaluators(self):
        print(f"\nSetting up a HausdorffEvaluator ({self.directionality}) for each LOD ...")
        for lod, filepaths in self.input_obj_filepath_pairs.items():
            input_obj_filepath_1, input_obj_filepath_2 = filepaths
            self.hausdorff_evaluators[lod] = HausdorffEvaluator(
                output_base_dirpath=self.output_base_dirpath,
                input_obj_filepath_1=input_obj_filepath_1,
                input_obj_filepath_2=input_obj_filepath_2,
                bidirectional=self.bidirectional
            )

    def run_hausdorff_evaluators(self):
        for lod, hausdorff_evaluator in self.hausdorff_evaluators.items():
            print(f"\nRunning HausdorffEvaluator for LOD '{lod}' ...")
            hausdorff_evaluator.split_obj_files()
            hausdorff_evaluator.compute_hausdorff_distances()

        print("\nJoining results ...")
        self.results_df = pd.concat(
            [evaluator.results_df.add_suffix(f"_{lod}") for lod, evaluator in self.hausdorff_evaluators.items()],
            axis=1
        )

    @print_starting_message
    def run(self):
        print(f"\nThis HausdorffLODSEvaluator is {self.directionality}.")
        self.setup_hausdorff_evaluators()
        self.run_hausdorff_evaluators()
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {}

        for lod in self.lods:
            col_names = [f"hausdorff{self.col_suffix}_{lod}", f"rms_min_dist{self.col_suffix}_{lod}"]
            for col_name in col_names:
                column = self.results[col_name]
                key_prefix = col_name + "_"
                self._summary_stats.update({
                    key_prefix + "rms": rms(column),
                    key_prefix + "mean": column.mean(),
                    key_prefix + "median": column.median(),
                    key_prefix + "min": column.min(),
                    key_prefix + "max": column.max()
                })


class HausdorffEvaluator(BuildingsEvaluator):
    """
    Compute Hausdorff distances between all objects present in two Wavefront OBJ files
    """
    name = "hausdorff"

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_obj_filepath_1: Path | str,
            input_obj_filepath_2: Path | str,
            bidirectional: bool = False,
            verbose: bool = False
    ):
        self.bidirectional = bidirectional
        self.col_suffix = "" if self.bidirectional else "_uni"
        self.directionality = "bidirectional" if self.bidirectional else "unidirectional"
        name = "hausdorff" if self.bidirectional else "hausdorff_uni"
        super().__init__(output_base_dirpath, name)

        self.input_obj_filepath_1 = Path(input_obj_filepath_1)
        self.input_obj_filepath_2 = Path(input_obj_filepath_2)

        self.split_input_obj_dirpath_1 = self.input_obj_filepath_1.parent / self.input_obj_filepath_1.stem
        self.split_input_obj_dirpath_2 = self.input_obj_filepath_2.parent / self.input_obj_filepath_2.stem

        self.final_columns = ["hausdorff" + self.col_suffix, "rms_min_dist" + self.col_suffix]

    def split_obj_files(self, quick_skip=True):
        """Split both input OBJ files up into one OBJ file for each object group found within the input file
        :param quick_skip: If a subdirectory with the name (without extension) of the respective input OBJ file exists
        already, assume that the input OBJ file was split previously and skip the splitting step
        """
        print("\nSplitting OBJ files into one file per individual object ...")
        if not quick_skip or not self.split_input_obj_dirpath_1.is_dir():
            split_obj_file(self.input_obj_filepath_1, output_dirpath=self.split_input_obj_dirpath_1, overwrite=False)
        if not quick_skip or not self.split_input_obj_dirpath_2.is_dir():
            split_obj_file(self.input_obj_filepath_2, output_dirpath=self.split_input_obj_dirpath_2, overwrite=False)

    def compute_hausdorff_distances(self, verbose=False, very_verbose=False):
        print("\nIdentifying individual OBJ files present in both input datasets ...")
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

        print(f"Computing {self.directionality} Hausdorff distances between split OBJ files ...")
        results = []

        if self.bidirectional:
            get_hausdorff_distances = get_bidirectional_hausdorff_distances
        else:
            get_hausdorff_distances = get_unidirectional_hausdorff_distances

        if verbose or very_verbose:
            print()
        for i, filename in enumerate(common_filenames):
            if verbose or very_verbose:
                print(f"{filename} ", end="")
            hausdorff_io_dict, hausdorff_oi_dict = get_hausdorff_distances(
                self.split_input_obj_dirpath_1 / filename,
                self.split_input_obj_dirpath_2 / filename
            )
            results_dict = {"file": Path(filename).stem}
            results_dict["hausdorff" + self.col_suffix] = max(hausdorff_io_dict["max"], hausdorff_oi_dict["max"])
            results_dict["rms_min_dist" + self.col_suffix] = np.sqrt(np.mean([hausdorff_io_dict["RMS"] ** 2, hausdorff_oi_dict["RMS"] ** 2]))
            results_dict.update({f"{key}_io{self.col_suffix}": value for key, value in hausdorff_io_dict.items()})
            results_dict.update({f"{key}_oi{self.col_suffix}": value for key, value in hausdorff_oi_dict.items()})
            results.append(results_dict)
        if verbose or very_verbose:
            print()
        print("\n- All done.")

        self.results_df = pd.DataFrame(results).set_index("file")
        self.results_df.index.name = glb.target_identifier_name

    @print_starting_message
    def run(self):
        print(f"\nThis HausdorffEvaluator is {self.directionality}.")
        self.split_obj_files()
        self.compute_hausdorff_distances()
        self.save_results()


class AreaVolumeDifferenceEvaluator(BuildingsEvaluator):
    """
    For two input building models, compute area, volume, and difference in area and volume between the two.

    This Evaluator separately evaluates area and volume of the two input building models in CityJSON format for any
    given LODs using two instances of AreaVolumeEvaluator. Following this, it computes the differences in area and
    volume between corresponding buildings. To enable reuse of the AreaVolumeEvaluators' results, these are stored
    independently according to the class definition.
    """
    name = "area_volume_diff"

    def __init__(
            self,
            output_base_dirpath_1: Path | str,
            output_base_dirpath_2: Path | str,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            index_col_name_1: str,
            index_col_name_2: str,
            lods: list[str],
            crs: str,
            reevaluate_if_exists_1: bool = False,
            reevaluate_if_exists_2: bool = False
    ):
        super().__init__(output_base_dirpath=output_base_dirpath_2)
        self.output_base_dirpath_1 = Path(output_base_dirpath_1)
        self.output_base_dirpath_2 = Path(output_base_dirpath_2)
        self.output_base_dirpath_1.mkdir(exist_ok=True)  # output_base_dirpath_2 is created by parent class via super().__init__()
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)
        self.index_col_name_1 = index_col_name_1
        self.index_col_name_2 = index_col_name_2
        self.lods = lods
        self.lods_short = [lod.replace(".", "") for lod in lods]
        self.crs = crs
        self.reevaluate_if_exists_1 = reevaluate_if_exists_1
        self.reevaluate_if_exists_2 = reevaluate_if_exists_2

        self.area_volume_evaluator_1: AreaVolumeEvaluator | None = None
        self.area_volume_evaluator_2: AreaVolumeEvaluator | None = None

        self.final_columns = [col_name for lod in self.lods_short for col_name in [
            f"area_{lod}_in", f"area_{lod}_out", f"area_{lod}_diff", f"area_{lod}_abs_diff",
            f"area_{lod}_norm_diff", f"area_{lod}_norm_abs_diff", f"area_{lod}_ratio",
            f"volume_{lod}_in", f"volume_{lod}_out", f"volume_{lod}_diff", f"volume_{lod}_abs_diff",
            f"volume_{lod}_norm_diff", f"volume_{lod}_norm_abs_diff", f"volume_{lod}_ratio"
        ]]

    def setup_area_volume_evaluators(self):
        print("Setting up both AreaVolumeEvaluators ...")
        self.area_volume_evaluator_1 = AreaVolumeEvaluator(
            self.output_base_dirpath_1,
            input_cityjson_filepath=self.input_cityjson_filepath_1,
            index_col_name=self.index_col_name_1,
            lods=self.lods,
            crs=self.crs
        )

        self.area_volume_evaluator_2 = AreaVolumeEvaluator(
            self.output_base_dirpath_2,
            input_cityjson_filepath=self.input_cityjson_filepath_2,
            index_col_name=self.index_col_name_2,
            lods=self.lods,
            crs=self.crs
        )

    def run_area_volume_evaluators(self):
        if not self.area_volume_evaluator_1.output_csv_filepath.exists() or self.reevaluate_if_exists_1:
            print(f"Running AreaVolumeEvaluator for {self.input_cityjson_filepath_1.name} ...")
            self.area_volume_evaluator_1.run()
        else:
            print(f"Output of AreaVolumeEvaluator for {self.input_cityjson_filepath_1.name} already exists.")

        if not self.area_volume_evaluator_2.output_csv_filepath.exists() or self.reevaluate_if_exists_2:
            print(f"Running AreaVolumeEvaluator for {self.input_cityjson_filepath_2.name} ...")
            self.area_volume_evaluator_2.run()
        else:
            print(f"Output of AreaVolumeEvaluator for {self.input_cityjson_filepath_2.name} already exists.")

        print("Merging results of both AreaVolumeEvaluators ...")
        # The results dataframe of the AreaVolumeDifferenceEvaluator contains the results of both AreaVolumeEvaluators
        self.results_df = pd.merge(
            self.area_volume_evaluator_1.results,  # .results_df may not exist, therefore call property .results
            self.area_volume_evaluator_2.results,
            how="right",
            left_index=True,
            right_index=True,
            suffixes=("_in", "_out")
        )  # .set_index(glb.target_identifier_name)

        print("Computing differences in area and volume across all LODs ...")
        for lod in self.lods_short:
            # For both the "area" and "volume" fields, compute the differences between input and output
            for field in [
                AreaVolumeEvaluator.fme_output_attributes[glb.fme_area_field_name],
                AreaVolumeEvaluator.fme_output_attributes[glb.fme_volume_field_name]
            ]:
                metric_out = self.results_df[f"{field}_{lod}_out"]
                metric_in = self.results_df[f"{field}_{lod}_in"]

                self.results_df[f"{field}_{lod}_diff"] = metric_out - metric_in
                self.results_df[f"{field}_{lod}_abs_diff"] = (metric_out - metric_in).abs()
                self.results_df[f"{field}_{lod}_ratio"] = metric_out / metric_in
                self.results_df[f"{field}_{lod}_norm_diff"] = (metric_out - metric_in) / metric_in
                self.results_df[f"{field}_{lod}_norm_abs_diff"] = (metric_out - metric_in).abs() / metric_in

    @print_starting_message
    def run(self):
        self.setup_area_volume_evaluators()
        self.run_area_volume_evaluators()
        self.save_results()

    def compute_summary_stats(self):
        self._summary_stats = {}

        for lod in self.lods_short:
            # Get the names of the area and volume columns as output by the AreaVolumeEvaluator for the respective FME
            # attributes
            for field in [
                AreaVolumeEvaluator.fme_output_attributes[glb.fme_area_field_name],
                AreaVolumeEvaluator.fme_output_attributes[glb.fme_volume_field_name]
            ]:
                prefix = f"{field}_{lod}_"
                diff_col = self.results[prefix + "diff"]

                # Filter results for rows where both input and output areas or volumes are available (not NA), so that
                # when calculating a metric based on the means of both columns, only the means of rows are taken into
                # account that are available in both input and output. (That is, if a building is missing in the output,
                # the area or volume of the input building should not be included in the input mean when it is compared
                # in some way to the output mean. As a consequence, the computation can in many cases be done simply
                # using the sum instead of the mean of either column, because the number of observations is equal due to
                # this adjustment and therefore cancels out.)
                # This is only necessary where column statistics are computed here. Results computed in
                # self.run_area_volume_evaluators() can be used immediately, because they are NA in the respective locs.
                results_available = self.results[
                    ~self.results[prefix + "out"].isna() &
                    ~self.results[prefix + "in"].isna()
                ]

                self._summary_stats[prefix + "count"] = diff_col.count()

                self._summary_stats.update({
                    prefix + "in_mean": self.results[prefix + "in"].mean(),
                    prefix + "out_mean": self.results[prefix + "out"].mean(),

                    # Statistics of the actual differences or their absolute values
                    prefix + "mean_diff": diff_col.mean(),
                    prefix + "median_diff": diff_col.median(),
                    prefix + "mean_abs_diff": diff_col.abs().mean(),
                    prefix + "median_abs_diff": diff_col.abs().median(),
                    prefix + "rms_diff": rms(diff_col),

                    # Statistics of the ratios of input and output area or volume
                    prefix + "mean_of_ratios":
                        self.results[prefix + "ratio"].mean(),
                    prefix + "ratio_of_means":
                        results_available[prefix + "out"].sum() /
                        results_available[prefix + "in"].sum(),

                    # Mean of the normalized difference and normalized mean differences (w.r.t. input area or volume)
                    prefix + "mean_norm_diff": self.results[prefix + "norm_diff"].mean(),
                    prefix + "norm_mean_diff": results_available[prefix + "diff"].sum() /
                                               results_available[prefix + "in"].sum(),

                    # RMS of the normalized difference and normalized RMS differences (w.r.t. input area or volume)
                    # todo: decide what to normalize on. currently: mean. something else?
                    prefix + "rms_norm_diff": rms(self.results[prefix + "norm_diff"]),
                    prefix + "norm_rms_diff": rms(results_available[prefix + "diff"]) /
                                              results_available[prefix + "in"].mean(),

                    # Mean of the normalized absolute difference and normalized mean absolute difference
                    prefix + "mean_norm_abs_diff": self.results[prefix + "norm_abs_diff"].mean(),
                    prefix + "norm_mean_abs_diff": results_available[prefix + "abs_diff"].sum() /
                                                   results_available[prefix + "in"].sum(),

                    # RMS of the normalized absolute difference and normalized RMS absolute difference
                    # todo: decide what to normalize on. currently: mean. something else?
                    prefix + "rms_norm_abs_diff": rms(self.results[prefix + "norm_abs_diff"]),
                    prefix + "norm_rms_abs_diff": rms(results_available[prefix + "abs_diff"]) /
                                                  results_available[prefix + "in"].mean()
                })


class FMEEvaluator(BuildingsEvaluator):
    """
    fme_output_attributes: Dictionary that maps attribute names in the FME output CityJSON to evaluation column names.
    """
    name = "fme_evaluator"
    fme_output_attributes: dict[str, str] = {}

    def __init__(self, output_base_dirpath: Path | str, index_col_name: str, lods: list[str], crs: str):
        """

        :param output_base_dirpath: Directory for evaluation results
        :param index_col_name: Name of the attribute in the Geoflow output CityJSON to be used as the index for the
        results table. Ideally, a building identifier
        :param lods: List of levels of detail (LODs) to process. They are used as user parameters for the FME pipeline,
        so they should be formatted as in "2.2".
        :param crs: Coordinate reference system EPSG string.
        """
        super().__init__(output_base_dirpath)
        self.index_col_name = index_col_name
        self.lods = lods
        self.lods_short = [lod.replace(".", "") for lod in lods]
        self.crs = crs

        self.output_cityjson_filepath: Path = self.output_dirpath / f"model_{self.name}.json"
        self.fme_output_cityjson_filepaths = {}

        self.fme_pipelines_dict: dict[str, FMEPipeline] = {}
        self.results_template_cityjson_filepath: Path | None = None

        self.final_columns = [
            f"{col_prefix}_{lod}" for lod in self.lods_short for col_prefix in list(self.fme_output_attributes.values())
        ]

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

        csv_output_attributes = []
        for lod, output_cityjson_filepath in self.fme_output_cityjson_filepaths.items():
            lod_short = lod.replace(".", "")
            # Load the FME output CityJSON for this LOD
            fme_output_model = cityjson.load(output_cityjson_filepath)
            fme_output_buildings = fme_output_model.get_cityobjects(type="building")
            for fme_attribute_name, new_attribute_name in self.fme_output_attributes.items():
                new_attribute_name_lod = f"{new_attribute_name}_{lod_short}"
                csv_output_attributes.append(new_attribute_name_lod)
                # Append all FME output attributes of this LOD to the current building in the input CityJSON (which
                # is the reconstructed model), adding the LOD as a suffix
                for b_id, b in fme_output_buildings.items():
                    if b.id in input_cityjson_buildings.keys():
                        input_cityjson_buildings[b.id].attributes[new_attribute_name_lod] = b.attributes[fme_attribute_name]

        # Save the CityJSON including the new attributes to a new file
        cityjson.save(input_cityjson, str(self.output_cityjson_filepath))

        print(f"Preparing results dataframe ...")
        print(f"- Path: {self.output_csv_filepath}")

        # Modify the buildings dictionary such that its values only contain the values of the attributes dictionary
        # contained in it, i.e., drop the other nested dictionaries
        for key in input_cityjson_buildings.keys():
            input_cityjson_buildings[key] = input_cityjson_buildings[key].attributes
        # Create a dataframe where the dict keys form the index
        self.results_df = pd.DataFrame.from_dict(input_cityjson_buildings, orient="index")
        # 1. Set the column specified in self.index_col_name (e.g., `OGRLoader.identificatie`) as index instead of
        #    the cryptical FID-derived values (e.g., "1", "1-0", ...).
        #    Note: The input 3DBAG CityJSON files have the correct index, but the index has no name. In this case, this
        #    evaluator can be set up with index_col_name_1 = "", and then the set_index() operation is skipped, and the
        #    index is merely renamed.
        # 2. Rename the index column to the target identifier name (e.g., `identificatie`, to strip off the
        #    `OGRLoader` prefix resulting from Geoflow output
        if self.index_col_name != "":
            self.results_df = self.results_df.set_index(self.index_col_name)
        self.results_df.index.name = glb.target_identifier_name
        # Remove all attributes present in the reconstructed CityJSON irrelevant to this evaluator
        self.results_df = self.results_df[csv_output_attributes]

    @print_starting_message
    def run(self):
        self.run_fme_pipeline()
        self.merge_lods_results()
        self.save_results()


class AreaVolumeEvaluator(FMEEvaluator):
    name = "area_volume"
    fme_output_attributes = {
        glb.fme_area_field_name: "area",
        glb.fme_volume_field_name: "volume"
    }

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_cityjson_filepath: Path | str,
            index_col_name: str,
            lods: list[str],
            crs: str
    ):
        super().__init__(output_base_dirpath, index_col_name, lods=lods, crs=crs)
        self.input_cityjson_filepath = Path(input_cityjson_filepath)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineAreaVolume(
                input_cityjson_filepath=self.input_cityjson_filepath,
                output_dirpath=self.output_dirpath,
                crs=self.crs,
                lod=lod
            )


class IOU3DEvaluator(FMEEvaluator):
    name = "iou_3d"
    fme_output_attributes = {
        glb.fme_iou3d_volume_input_field_name: "volume_input",
        glb.fme_iou3d_volume_output_field_name: "volume_output",
        glb.fme_iou3d_volume_intersection_field_name: "volume_intersection",
        glb.fme_iou3d_iou_field_name: "iou"
    }

    def __init__(
            self,
            output_base_dirpath: Path | str,
            input_cityjson_filepath_1: Path | str,
            input_cityjson_filepath_2: Path | str,
            index_col_name: str,
            lods: list[str],
            crs: str
    ):
        super().__init__(output_base_dirpath, index_col_name, lods=lods, crs=crs)
        self.input_cityjson_filepath_1 = Path(input_cityjson_filepath_1)
        self.input_cityjson_filepath_2 = Path(input_cityjson_filepath_2)
        self.results_template_cityjson_filepath = self.input_cityjson_filepath_2

        for lod in self.lods:
            self.fme_pipelines_dict[lod] = FMEPipelineIOU3D(
                input_cityjson_filepath_1=self.input_cityjson_filepath_1,
                input_cityjson_filepath_2=self.input_cityjson_filepath_2,
                output_dirpath=self.output_dirpath,
                crs=self.crs,
                lod=lod
            )

    def compute_summary_stats(self):
        self._summary_stats = {}
        for lod in self.lods:
            lod = lod.replace(".", "")
            key_prefix = f"iou_{lod}_"
            iou_col = self.results[f"iou_{lod}"]
            self._summary_stats.update({
                key_prefix + "count": iou_col.count(),
                key_prefix + "rms": rms(iou_col),
                key_prefix + "mean": iou_col.mean(),
                key_prefix + "median": iou_col.median(),
                key_prefix + "min": iou_col.min(),
                key_prefix + "max": iou_col.max()
            })
