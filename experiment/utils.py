import subprocess


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
    crs_epsg_code = epsg_str.rsplit(":", 1)[1]
    return "https://www.opengis.net/def/crs/EPSG/0/" + crs_epsg_code

