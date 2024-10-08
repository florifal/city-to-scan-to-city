import numpy as np
import pdal
from pathlib import Path
from xml.etree import ElementTree as eT
from scipy.spatial import KDTree


class CloudMerger:

    def __init__(self, cloud_filepaths: list[Path] | list[str]):
        self.cloud_filepaths = [Path(p) for p in cloud_filepaths]
        self.merged_output_cloud_filepath: Path | None = None

    def clear(self):
        pass

    def run(self):
        pass


class UniqueKeeperMerger(CloudMerger):

    def __init__(
            self,
            cloud_filepaths: list[Path] | list[str],
            flight_path_element: eT.Element,
            parallel_dimension: str,
            output_dirpath: Path | str,
            crs: str
    ):
        """UniqueKeeperMerger

        :param cloud_filepaths: List of filepaths to all point clouds to be merged.
        :param flight_path_element: The element from the survey XML file that is the parent to all flight path legs
        :param parallel_dimension: Specify the dimension in which the flight path legs are parallel. Can be 'X' or 'Y'.
        :param output_dirpath: Path to the directory in which the filtered and merged point clouds will be stored.
        :param crs: String specifying the CRS as in "epsg:0000"
        """
        super().__init__(cloud_filepaths)
        self.flight_path_element = flight_path_element
        self.parallel_dimension = parallel_dimension
        self.output_dirpath = Path(output_dirpath)
        self.crs = crs

        self.parallel_trajectory_coords: list[float] = []
        self.separation_coords: list[float] = []

        self.cloud_filtered_filepaths = [
            self.output_dirpath / (filepath.stem + "_filtered.laz") for filepath in self.cloud_filepaths
        ]

        self.point_cloud_arrays = None
        self.columns_out: list[str] | None = None
        self.merged_output_cloud_filepath = self.output_dirpath / "clouds_merged.laz"

    def clear(self):
        self.point_cloud_arrays = None

    def run(self):
        # This method is a candidate for deletion because it only adds another layer.
        self.get_parallel_trajectory_coords()
        self.compute_separation_coords()
        self.filter_clouds()
        self.merge_clouds()

    def get_parallel_trajectory_coords(self):
        print("\nReading the coordinates in which the trajectories are parallel ...")
        for leg_element in self.flight_path_element.findall("leg"):
            platform_settings_element = leg_element.find("platformSettings")
            scanner_settings_element = leg_element.find("scannerSettings")
            # Currently, only the short legs on which the scanner is not active have an attribute active="false",
            # whereas the legs with active scanner don't have this attribute, because the scanner is active per default
            # as specified in the scanner settings template element. Therefore, the following condition checks for the
            # absence of the attribute "active" to identify legs with active scanner.
            if "active" not in scanner_settings_element.attrib.keys():
                # Read parallel_dimension coordinate attribute from platformSettings element within leg element
                coord = float(platform_settings_element.attrib[self.parallel_dimension.lower()])
                self.parallel_trajectory_coords.append(coord)

    def compute_separation_coords(self):
        print("Computing center coordinates between adjacent parallel trajectory coordinates ...")
        # For each pair of coordinates, compute their distance
        parallel_trajectory_distances = [
            self.parallel_trajectory_coords[i] - self.parallel_trajectory_coords[i - 1]
            for i in range(1, len(self.parallel_trajectory_coords))
        ]

        # The separation between the point clouds of two swaths should be performed exactly along their center line.
        # The separation coordinates are therefore at the middle of the distance between each pair of flight path
        # coordinates.
        self.separation_coords = [
            self.parallel_trajectory_coords[i] + 0.5 * parallel_trajectory_distances[i]
            for i in range(len(parallel_trajectory_distances))
        ]

        # To be able to use the same PDAL pipeline snippet for each swath, add separation coordinates at the
        # beginning and end of the list.
        self.separation_coords.insert(0, 0)
        self.separation_coords.append(2 * self.separation_coords[-1])

    def filter_clouds(self):
        n_clouds = len(self.cloud_filepaths)

        # Pipeline for reading all individual swath's point clouds and computing within-cloud nearest neighbor distances
        readers = [pdal.Reader(str(filepath), nosrs=True, default_srs=self.crs) for filepath in self.cloud_filepaths]
        # pipelines = [reader | pdal.Filter.nndistance(mode="kth", k=1) for reader in readers]
        pipelines = [reader | pdal.Filter.nndistance(mode="avg", k=4) for reader in readers]  # todo: decide on settings

        print("\nReading input point clouds and computing within-cloud nearest-neighbor distance ...")
        for p, pipeline in enumerate(pipelines):
            print(f"Processing swath point cloud {p+1} of {n_clouds} ...")
            n_points = pipeline.execute()
            print(f"- Processed {n_points} points.")

        # Get point clouds as numpy structured arrays
        point_cloud_arrays = [pipeline.arrays[0] for pipeline in pipelines]
        del pipelines

        print("\nComputing mean within-cloud nearest-neighbor distances ...")
        # Compute mean of within-cloud nearest neighbor distances for each swath
        mean_NN_distances = [np.mean(arr["NNDistance"]) for arr in point_cloud_arrays]
        print(mean_NN_distances)

        print("Computing k-d trees ...")
        # Create a simple XYZ numpy array containing only the coordinates for each swath, and k-d trees.
        numpy_point_clouds = [np.column_stack((arr['X'], arr['Y'], arr['Z'])) for arr in point_cloud_arrays]
        kd_trees = [KDTree(pc) for pc in numpy_point_clouds]

        for p, point_array in enumerate(point_cloud_arrays):
            print(f"Processing swath point cloud {p+1} of {n_clouds} ...")

            adjacent_swath_ids = []
            if p > 0: adjacent_swath_ids.append(p - 1)
            if p < n_clouds - 1: adjacent_swath_ids.append(p + 1)
            print(f"- has {len(adjacent_swath_ids)} adjacent swath{(len(adjacent_swath_ids) > 1) * 's'}.")

            print("- Computing between-cloud nearest-neighbor distance to all adjacent swaths ...")
            # Compute between-cloud nearest neighbor distance to all adjacent swaths
            for a in adjacent_swath_ids:
                # Use the adjacent swath's k-d tree to get the distance to the nearest neighbor in the adjacent cloud
                kth_distance, kth_index = kd_trees[a].query(
                    numpy_point_clouds[p], k=1, distance_upper_bound=10, workers=-1
                )

                # todo: generalize this into a function
                # Create a new structured array that includes the between-cloud nearest neighbor distance
                new_col_name = f'CC_NNDistance_{a}'
                dtype_new = np.dtype(
                    point_array.dtype.descr + [(new_col_name, '<f8')])  # Append a new column to the dtypes
                point_array_new = np.zeros(point_array.shape, dtype=dtype_new)  # Create an empty array with zeros
                for col_name, _ in point_array.dtype.descr:
                    point_array_new[col_name] = point_array[col_name]  # Copy values from existing array
                point_array_new[new_col_name] = kth_distance  # Insert new values: Between-cloud NN distances

                point_array = point_array_new

            print("- Filtering points in overlapping area ...")
            # Create a filter expression to filter the overlapping area, and a list of new fields to include in the
            # output cloud. First, filter areas lying beyond the separation boundary between adjacent clouds.
            filter_expression = (f"({self.parallel_dimension.upper()} >= {self.separation_coords[p]} && "
                                 f"{self.parallel_dimension.upper()} <= {self.separation_coords[p + 1]})")
            filter_expression += " || ("
            extra_dims = "NNDistance=float64"
            # Second, add a filter to keep points beyond the separation boundary if they do not duplicate (but
            # complement) points in the adjacent cloud
            for c, a in enumerate(adjacent_swath_ids):
                if c > 0: filter_expression += " && "
                filter_expression += f"CC_NNDistance_{a} > {mean_NN_distances[a]}"
                extra_dims += f", CC_NNDistance_{a}=float64"
            filter_expression += ")"
            print(f"- filter_expression: {filter_expression}")
            print(f"- extra_dims: {extra_dims}")

            # Pipeline to apply the filter expression and save the filtered cloud
            pipeline = pdal.Filter.expression(expression=filter_expression).pipeline(point_array)
            pipeline |= pdal.Writer(filename=str(self.cloud_filtered_filepaths[p]), minor_version=4,
                                    a_srs=self.crs, compression=True, extra_dims=extra_dims)
            n_points = pipeline.execute()
            print(f"- Remaining number of points: {n_points}")

            # Update this swath's structured array with the new, filtered one
            point_cloud_arrays[p] = pipeline.arrays[0]

        print("Finished filtering all point clouds.")
        self.point_cloud_arrays = point_cloud_arrays

    def filter_clouds_sequentially(self):
        pass

    def merge_clouds(self):
        # If merging previously filtered and saved clouds (i.e., without running filter_clouds() again), load them
        if self.point_cloud_arrays is None:
            self.point_cloud_arrays = []
            print("\nReading previously filtered point clouds from disk ...")

            for filepath in self.cloud_filtered_filepaths:
                print(f"- Reading {filepath.name} ...", end="\r")
                pipeline = pdal.Reader(str(filepath), nosrs=True, default_srs=self.crs).pipeline()
                n_points = pipeline.execute()
                print(f"- Reading {filepath.name} ... {n_points} points read.")
                self.point_cloud_arrays.append(pipeline.arrays[0])
            del pipeline

        # Add a new field to each point cloud that will indicate the source point cloud after merging
        print("\nAdding a source field to each filtered point cloud ...")
        source_col_name = "Source"
        source_col_dtype = "uint8"
        extra_dims = f"{source_col_name}={source_col_dtype}"
        for p, point_array in enumerate(self.point_cloud_arrays):
            # todo: generalize this into a function
            # Create a new structured array that includes a new source field
            # Append a new unsigned int column to the dtypes
            dtype_new = np.dtype(
                point_array.dtype.descr + [(source_col_name, source_col_dtype)]
            )
            point_array_new = np.zeros(point_array.shape, dtype=dtype_new)  # Create an empty array with zeros
            for col_name, _ in point_array.dtype.descr:
                point_array_new[col_name] = point_array[col_name]  # Copy values from existing array
            point_array_new[source_col_name] = p  # Insert new values: Index of the point cloud array

            self.point_cloud_arrays[p] = point_array_new  # Replace the point cloud array with the new one

        # Prepare list of columns that should be contained in the final output
        undesired_columns = ["GpsTime", "Gps Time", "echo_width", "fullwaveIndex", "hitObjectId", "heliosAmplitude",
                             "NNDistance"]
        self.columns_out = [
            col_name for col_name, _ in self.point_cloud_arrays[0].dtype.descr
            if col_name not in undesired_columns and not col_name.startswith("CC_NNDistance")
        ]

        print("\nConcatenating filtered point clouds ...")
        # Concatenate the filtered point clouds into one structured array
        point_cloud_arrays_stacked = np.hstack([arr[self.columns_out] for arr in self.point_cloud_arrays])
        print(f"- Total number of points: {len(point_cloud_arrays_stacked)}")

        print("Writing merged point clouds to output location ...")
        # Pipeline to save the final (filtered and merged) point cloud
        pipeline = pdal.Writer(
            filename=str(self.merged_output_cloud_filepath),
            minor_version=4,
            a_srs=self.crs,
            compression=True,
            extra_dims=extra_dims
        ).pipeline(point_cloud_arrays_stacked)
        n_points = pipeline.execute()
        print(f"- Written number of points: {n_points}")


class CloudNoiseAdder:

    def __init__(
            self,
            input_filepath: Path | str,
            output_filepath: Path | str,
            std_horizontal: float,
            std_vertical: float,
            crs: str
    ):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.std_horizontal = std_horizontal
        self.std_vertical = std_vertical
        self.crs = crs

        self.n_points: int | None = None

    def run(self):
        self.n_points = self.add_noise_to_cloud(
            self.input_filepath,
            self.output_filepath,
            self.std_horizontal,
            self.std_vertical,
            self.crs
        )

    @classmethod
    def add_noise_to_cloud(cls, input_filepath, output_filepath, std_horizontal, std_vertical, crs):
        print("Reading input point cloud ...")
        print(f"- {input_filepath}")
        reader = pdal.Reader(str(input_filepath), nosrs=True, default_srs=crs)
        pipeline = reader.pipeline()
        n_points = pipeline.execute()
        point_array = pipeline.arrays[0]

        print("Adding random error ...")
        print(f"- std horizontal: {std_horizontal}")
        print(f"- std vertical: {std_vertical}")
        # Random number generator
        rng = np.random.default_rng()

        error_x = rng.normal(loc=0.0, scale=std_horizontal, size=n_points)
        error_y = rng.normal(loc=0.0, scale=std_horizontal, size=n_points)
        error_z = rng.normal(loc=0.0, scale=std_vertical, size=n_points)

        axes = ["X", "Y", "Z"]
        errors = dict(zip(axes, [error_x, error_y, error_z]))
        for axis in axes:
            point_array[axis] = point_array[axis] + errors[axis]

        # Define extra dimension for source field (currently added by UniqueKeeperMerger) todo: generalize for mergers
        source_col_name = "Source"
        source_col_dtype = "uint8"
        extra_dims = f"{source_col_name}={source_col_dtype}"

        print("Writing output point cloud ...")
        print(f"- {output_filepath}")
        writer = pdal.Writer(str(output_filepath), minor_version=4, a_srs=crs, compression=True, extra_dims=extra_dims)
        pipeline = writer.pipeline(point_array)
        n_points = pipeline.execute()

        return n_points
