import subprocess
import numpy as np
from pathlib import Path


def execute_subprocess(cmd):
    """Run a subprocess and yield (return an iterable of) all stdout lines. From: https://stackoverflow.com/a/4417735"""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)  # text=True: stdout as string not bytes
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


def scan_freq_from_pulse_freq_via_point_spacing(pulse_freq_hz, altitude, velocity, scan_angle_deg):
    # Adjustment factors to decrease scan frequency / relatively increase pulse frequency
    # -> increase along-track point spacing
    # -> make it more similar to across-track point spacing
    # If the actual formula would work, then this factor should be unnecessary.
    adjustment_factors = [1, 1/3, 1/np.pi]
    adjustment_factor = adjustment_factors[1]
    return np.sqrt(adjustment_factor * 0.5 * pulse_freq_hz * velocity / altitude / np.tan(1.0 * scan_angle_deg * np.pi / 180))


def point_spacing_along(velocity, scan_freq_hz):
    # horizontal_point_spacing = velocity / scan_freq_hz
    return velocity / scan_freq_hz


def point_spacing_across(altitude, scan_angle_deg, pulse_freq_hz, scan_freq_hz):
    # vertical_point_spacing = (2 * altitude * np.tan(scan_angle_deg / 2) * scan_freq_hz) / pulse_freq_hz
    return 2 * altitude * np.tan(scan_angle_deg * np.pi / 180) * scan_freq_hz / pulse_freq_hz