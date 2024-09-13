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


def get_wkt_from_gpkg() -> pd.Series:
    pass


def get_wkt_from_cityjson() -> pd.Series:
    pass


def get_last_line_from_file(filepath: Path | str, error_message: str = "") -> str:
    try:
        with open(filepath, "r") as f:
            filepaths = [fp[:-1] for fp in f.readlines()]  # remove \n
    except FileNotFoundError as e:
        error_message = (error_message != "") * f"{error_message} " + e.strerror
        raise FileNotFoundError(e.errno, error_message, filepath)
    else:
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
    adjustment_factors = {
        "should_be": 1,
        "could_be_but_is_not": 1/np.pi,
        "works_for_exp_random_error": 1/3,
        "works_for_exp_utrecht": 1/3 * 1.326**2
    }
    adjustment_factor = adjustment_factors["works_for_exp_utrecht"]
    return np.sqrt(
        adjustment_factor * 0.5 * pulse_freq_hz * velocity
        / altitude / np.tan(1.0 * scan_angle_deg * np.pi / 180)
    )


def scan_freq_from_point_spacing(velocity: float, spacing_along: float) -> float:
    return velocity / spacing_along


def pulse_freq_from_point_spacing(
        altitude: float,
        scan_angle_deg: float,
        velocity: float,
        spacing_along: float,
        spacing_across: float
) -> float:
    return 2 * altitude * np.tan(scan_angle_deg / 180 * np.pi) * velocity / spacing_along / spacing_across


def point_spacing_along(velocity: float, scan_freq_hz: float):
    # horizontal_point_spacing = velocity / scan_freq_hz
    return velocity / scan_freq_hz


def point_spacing_across(altitude: float, scan_angle_deg: float, pulse_freq_hz: float, scan_freq_hz: float):
    """Compute across-track point spacing

    Note that scan_angle_deg here is defined as the half-swath angle, i.e. measured from nadir."""

    # vertical_point_spacing = (2 * altitude * np.tan(scan_angle_deg / 2) * scan_freq_hz) / pulse_freq_hz
    return 2 * altitude * np.tan(scan_angle_deg * np.pi / 180) * scan_freq_hz / pulse_freq_hz


def point_density_theoretical(pulse_freq_hz: float, velocity: float, altitude: float, scan_angle_deg: float):
    """Compute the theoretical point density

    Note that scan_angle_deg here is defined as the half-swath angle, i.e. measured from nadir.
    Currently, the actually observed point density as evaluated by PointDensityDatasetEvaluator is lower by a factor of
    about 2.07."""

    return pulse_freq_hz / 2 / velocity / altitude / np.tan(scan_angle_deg * np.pi / 180)


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


def is_numeric(o: object) -> bool:
    try:
        float(o)
    except ValueError:
        return False
    else:
        return True


def deep_replace_in_string_values(d: dict, search: str, replace: str):
    for k, v in d.items():
        if isinstance(v, dict):
            deep_replace_in_string_values(v, search, replace)
        elif isinstance(v, str):
            d[k] = v.replace(search, replace,)
        else:
            pass


def add_material_to_wavefront_objects(obj_in_filepath: str | Path, obj_out_filepath: str | Path, mat_name: str):
    obj_in_filepath, obj_out_filepath = Path(obj_in_filepath), Path(obj_out_filepath)
    with open(obj_in_filepath, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.startswith("o "):
            new_lines.append(f"usemtl {mat_name}\n")
    with open(obj_out_filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def get_most_recently_created_folder(dirpath: Path | str):
    folders = [f for f in Path(dirpath).iterdir() if f.is_dir()]
    return max(folders, key=lambda f: f.stat().st_ctime)
