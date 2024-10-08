from __future__ import annotations

import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import bayes_opt as bo
from pathlib import Path
from cjio import cityjson as cj


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


def get_newest_recon_optim_log_filepath(dirpath: Path | str):
    optim_log_filepaths = [f for f in Path(dirpath).iterdir()
                           if f.is_file() and f.name.startswith("optimization_") and f.suffix == ".log"]
    return max(optim_log_filepaths, key=lambda f: f.stat().st_ctime)


def subset_cityjson_to_gpkg_contents(gpkg_fp: Path | str, cj_in_fp: Path | str, cj_out_fp: Path | str):
    gpkg_fp, cj_in_fp, cj_out_fp = Path(gpkg_fp), Path(cj_in_fp), Path(cj_out_fp)

    gpkg = gpd.read_file(gpkg_fp, layer=gpd.list_layers(gpkg_fp)["name"][0])
    ids = list(gpkg.identificatie)

    cj_in = cj.load(cj_in_fp)
    cj_subset = cjio_subset_properly(cj_in, ids)

    # cj_subset = cj_in.get_subset_ids(ids)
    # # After taking the subset, it appears to be necessary to call .load_from_j() to copy the CityObjects from the JSON
    # # representation to the CityJSON API structure. Otherwise, .save() will write zero CityObjects, and while before
    # # saving .get_info() reports the expected number of CityObjects, after saving it also reports zero CityObjects.
    # cj_subset.load_from_j()

    cj.save(cj_subset, cj_out_fp, indent=True)


def cjio_subset_properly(cm: cj.CityJSON, ids: list[str], exclude: bool = False):
    """An attempt to fix the CityJSON.subset() method"""

    cm_subset = cm.get_subset_ids(ids, exclude=exclude)

    # After taking the subset, it appears to be necessary to call .load_from_j() to copy the CityObjects from the JSON
    # representation to the CityJSON API structure. Otherwise, .save() will write zero CityObjects, and while before
    # saving .get_info() reports the expected number of CityObjects, after saving it also reports zero CityObjects.
    cm_subset.load_from_j()

    # Then, the transform must be copied manually. It would be copied if it were still in the original CityJSON's `j`
    # dictionary, but it cannot be, because when the original CityJSON is loaded from the disk, .load() calls
    # .load_from_j(), and the latter .pop()s the transform from the `j` dictionary to transfer it to the .transform
    # variable. Therefore, it is not copied from the original `j["transform"]`, even though .subset() tries to.
    cm_subset.transform = cm.transform
    cm_subset.j["transform"] = cm_subset.transform

    return cm_subset


def plural_s(things: list | dict | int | float | bool):
    try:
        things = len(things)
    except TypeError:
        try:
            things = abs(things)
        except TypeError:
            raise TypeError(f"Type {type(things)} does not have a length or an absolute value.")
    return "s" * (things != 1)


def get_processing_sequence(n_scenarios = 143, seed_scenario = 55):
    """
    This one is still very much hard-coded for the number of density and error scenarios. To be generalized if required.
    Would have to introduce: n_error_levels, n_density_levels
    :param n_scenarios:
    :param seed_scenario:
    :return:
    """

    # The error cascade is simple: Each scenario depends on the previous one in terms of numbering (and, in terms of
    # error, the next-lower one with identical density), except those that are multiples of 11, because they already
    # have zero error, so they cannot rely on a scenario with lower error.
    # Note that a number -1 indicates that no adjacent scenario exists / should be relied on.
    error_cascade = [i - 1 if i not in [j * 11 for j in range(13)] else -1 for i in range(n_scenarios)]

    # The density cascade is a bit more tricky: Because the seed scenario is 55, scenarios with lower density are
    # supposed to load optimizer logs from those with the next-higher density and identical error, while scenarios
    # with higher density than 55 (i.e., starting at 66) are supposed to load optimizer logs from those with the
    # next-lower density.
    # Note that a number -1 indicates that no adjacent scenario exists / should be relied on.
    density_cascade = []

    for i in range(n_scenarios):
        if i < seed_scenario:
            density_cascade.append(i + 11)
        elif seed_scenario <= i < seed_scenario + 11:
            density_cascade.append(-1)
        elif i >= seed_scenario + 11:
            density_cascade.append(i - 11)
        else:
            print("Crap. This shouldn't have happened.")

    # The error and density cascades are merged into a single list, which identifies for each scenario up to two scenarios from which it should load the optimizer logs.
    load_scenario_optimizer_states = []
    for i, (e, d) in enumerate(zip(error_cascade, density_cascade)):
        adjacent_scenarios = []
        if e != -1:
            adjacent_scenarios.append(e)
        if d != -1:
            adjacent_scenarios.append(d)
        load_scenario_optimizer_states.append(adjacent_scenarios)

    # For each scenario, identify which other scenarios depend on it and can therefore be executed afterwards.
    next_scenarios = [[] for i in range(n_scenarios)]

    for current_scenario, adjacent_scenarios in enumerate(load_scenario_optimizer_states):
        for adjacent_scenario in adjacent_scenarios:
            next_scenarios[adjacent_scenario].append(current_scenario)

    # Establish the processing sequence such that all dependencies are fulfilled at each step, i.e., for each scenario once it is run
    def append_next_to_sequence_after(current_scenario, sequence):
        # Check the scenarios that depend on this scenario
        for next_scenario in next_scenarios[current_scenario]:
            # Only consider adding the following scenario if it's not already in the sequence
            if next_scenario not in sequence:
                # Make sure not only the current scenario, but also all other scenarios the following scenario may rely on are already in the sequence before adding the following scenario. Otherwise, do nothing, and it will be added later on.
                if all([adjacent_scenario in sequence for adjacent_scenario in
                        load_scenario_optimizer_states[next_scenario]]):
                    sequence.append(next_scenario)
                    # Follow the cascade: Check whether to add the scenarios following the scenario just added.
                    append_next_to_sequence_after(next_scenario, sequence)

    sequence = [seed_scenario]
    append_next_to_sequence_after(seed_scenario, sequence)

    # Check if calculated sequence is valid: For each scenario occurring, all scenarios it relies on must be present in
    # the sequence before it.
    order_correct = []
    for i, s in enumerate(sequence):
        all_dependencies_fulfilled = [scenario in sequence[:i] for scenario in load_scenario_optimizer_states[s]]
        order_correct.extend(all_dependencies_fulfilled)

    if not all(order_correct):
        raise Exception("Algorithm to compute processing sequence failed: Order of final sequence is not correct.")

    return load_scenario_optimizer_states, next_scenarios, sequence


def force_duplicate_probe_now(optimizer: bo.BayesianOptimization, probe_parameter_sets: list[dict] | dict):
    if not isinstance(probe_parameter_sets, list):
        probe_parameter_sets = [probe_parameter_sets]

    allow_duplicate_points = optimizer._allow_duplicate_points

    optimizer._allow_duplicate_points = True
    optimizer._space._allow_duplicate_points = True

    for parameter_set in probe_parameter_sets:
        optimizer.probe(parameter_set, lazy=False)

    optimizer._allow_duplicate_points = allow_duplicate_points
    optimizer._space._allow_duplicate_points = allow_duplicate_points


def has_outliers(data: np.ndarray, threshold: float = 100) -> bool:
    """Check for outliers in data

    Outliers are values whose absolute distance from the median is larger than the median absolute distance from the
    median by a factor that exceeds the threshold value. Adapted from https://stackoverflow.com/a/16562028
    """
    distance_from_median = np.abs(data - np.median(data))
    median_distance_from_median = np.median(distance_from_median)
    ratio = distance_from_median / median_distance_from_median if median_distance_from_median else np.zeros(len(d))
    return (ratio > threshold).any()
