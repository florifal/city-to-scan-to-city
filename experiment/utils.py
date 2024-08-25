import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from experiment.obj_file import OBJFile


def execute_subprocess(cmd):
    """Run a subprocess and yield (return an iterable of) all stdout lines. From: https://stackoverflow.com/a/4417735"""
    # text=True: stdout as string not bytes
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for stdout_line in iter(process.stdout.readline, ""):
        yield stdout_line
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def crs_url_from_epsg(epsg_str: str) -> str:
    # Get the EPSG code without "epsg:" prefix
    crs_epsg_code = epsg_str.rsplit(":", 1)[1]
    return "https://www.opengis.net/def/crs/EPSG/0/" + crs_epsg_code


def get_wkt_from_gpkg():
    pass


def get_wkt_from_cityjson():
    pass


def get_last_line_from_file(filepath: Path | str, error_message: str = "") -> str:
    try:
        with open(filepath, "r") as f:
            filepaths = [fp[:-1] for fp in f.readlines()]  # remove \n
    except FileNotFoundError as e:
        if error_message != "":
            print(error_message)
        raise FileNotFoundError(e.errno, e.strerror, filepath)
    return filepaths[-1]


def scan_freq_from_pulse_freq_via_point_spacing(
        pulse_freq_hz: float,
        altitude: float,
        velocity: float,
        scan_angle_deg: float
):
    # Adjustment factors to decrease scan frequency / relatively increase pulse frequency
    # -> increase along-track point spacing
    # -> make it more similar to across-track point spacing
    # If the actual formula would work, then this factor should be unnecessary.
    adjustment_factors = [1, 1/3, 1/np.pi]
    adjustment_factor = adjustment_factors[1]
    return np.sqrt(
        adjustment_factor * 0.5 * pulse_freq_hz * velocity
        / altitude / np.tan(1.0 * scan_angle_deg * np.pi / 180)
    )


def point_spacing_along(velocity, scan_freq_hz):
    # horizontal_point_spacing = velocity / scan_freq_hz
    return velocity / scan_freq_hz


def point_spacing_across(altitude: float, scan_angle_deg: float, pulse_freq_hz: float, scan_freq_hz: float):
    # vertical_point_spacing = (2 * altitude * np.tan(scan_angle_deg / 2) * scan_freq_hz) / pulse_freq_hz
    return 2 * altitude * np.tan(scan_angle_deg * np.pi / 180) * scan_freq_hz / pulse_freq_hz


def swath_width(altitude: int, scan_angle_deg: int) -> int:
    return int(0.5 + 2 * altitude * np.tan(scan_angle_deg/180*np.pi))  # int() cuts off decimal places, thus +0.5


def rms(x: list | np.ndarray | pd.Series):
    return np.sqrt(np.mean(np.square(x)))


def get_face_count_from_gpkg(
        gpkg_filepath: Path | str,
        layer_name: str,
        id_column: str,
        result_col_name: str = "n_faces",
        aggregate_by_id: bool = True
) -> pd.Series:
    """

    :param gpkg_filepath: Path to Geopackage file
    :param layer_name: Name of layer in Geopackage
    :param id_column: Column that serves as identifier. Is used as index for returned Series.
    :param result_col_name: Name of the column containing the face counts.
    :param aggregate_by_id: Whether to sum up faces for rows (features) with identical values in `id_column`
    :return: A pandas.Series with the numbers of faces and the `id_column` as index.
    """
    # Note that it would, in theory, also be possible to dissolve the geometries by the id_column before counting the
    # number of faces / polygons in each geometry. However, this approach has two problems. First, it not only combines
    # multiple geometries with identical IDs, but it also turns them from a MultiPolygon Z to a single Polygon Z, which
    # means that a single polygon is returned instead of a multipolygon consisting of triangles. This could perhaps be
    # undone by triangulating it again. Second and worse, however, because GeoPandas and Shapely only operate in 2D, it
    # seems that the 3rd dimension is dropped, and the single Polygon Z that is returned actually only consists of the
    # building footprint. Thus, the dissolve is not viable. Instead, aggregate the number of faces after counting them.
    gpkg = gpd.read_file(gpkg_filepath, layer=layer_name)
    gpkg[result_col_name] = gpkg.apply(lambda x: len(x.geometry.geoms), axis=1)
    if aggregate_by_id:
        gpkg = gpkg[[id_column, result_col_name]].groupby(id_column)[result_col_name].sum()  # returns a Series
    else:
        gpkg = gpkg.set_index(id_column)
        gpkg = gpkg[result_col_name].copy()  # make a Series
    return gpkg


def get_face_count_from_obj(
        obj_filepath: Path | str,
        result_col_name: str = "n_faces",
        ensure_as_triangle_count: bool = False
) -> pd.Series:
    obj = OBJFile(obj_filepath)
    n_faces = obj.num_triangles if ensure_as_triangle_count else obj.num_faces
    return pd.Series(data=list(n_faces.values()), index=list(n_faces.keys()), name=result_col_name)


def describe_value_counts(series: list | np.ndarray | pd.Series):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    vc: pd.Series = series.value_counts()
    return {
        "num_total": len(series),
        "num_unique": series.nunique(),
        "num_multiple": sum(vc > 1),
        "val_multiple": vc[vc > 1].to_dict()
    }
