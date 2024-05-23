# from __future__ import annotations
# from typing_extensions import Self

import xml.etree.ElementTree as eT
from pyhelios.util import scene_writer, flight_planner
from global_vars import *


def parse_xml_with_comments(xml_filepath: str) -> eT.ElementTree:
    """Parse an XML file using ElementTree including the comments and return an ElementTree object."""
    parser = eT.XMLParser(target=eT.TreeBuilder(insert_comments=True))  # Retain XML comments (req. Python 3.8)
    return eT.parse(xml_filepath, parser)


class ScenePart:
    """Create XML code as string for a scene part."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._xml_string = ""

    @property
    def xml_string(self):
        return self._xml_string

    def __str__(self):
        return self.xml_string


class ScenePartOBJ(ScenePart):
    """Create XML code as string for a scene part consisting of a Wavefront OBJ file."""

    def __init__(self, filepath: str, up_axis: str = "z"):
        super().__init__(filepath)
        self.up_axis = up_axis

        self._xml_string = ScenePartOBJ.create_xml_string(self.filepath, self.up_axis)

    @classmethod
    def create_xml_string(cls, filepath: str, up_axis: str = "z"):
        # Call the respective function from pyhelios.utils.scene_writer.
        return scene_writer.create_scenepart_obj(filepath, up_axis)


class ScenePartTIFF(ScenePart):
    """Create XML code as string for a scene part consisting of a GeoTIFF file."""

    def __init__(self, filepath: str, mat_file: str, mat_name: str):
        super().__init__(filepath)
        self.mat_file = mat_file
        self.mat_name = mat_name

        self._xml_string = ScenePartTIFF.create_xml_string(self.filepath, self.mat_file, self.mat_name)

    @classmethod
    def create_xml_string(cls, filepath: str, mat_file: str, mat_name: str):
        # Call the respective function from pyhelios.utils.scene_writer.
        return scene_writer.create_scenepart_tiff(filepath, matfile=mat_file, matname=mat_name)


class Scene:
    """Create an XML file for a pyhelios scene from scene parts."""

    def __init__(self, xml_id: str, name: str = "", filepath: str = ""):
        """Parameters

        :param xml_id: Attribute id of tag scene.
        :param name: Attribute name of tag scene.
        :param filepath: Destination path for the scene XML file.
        """
        self.xml_id = xml_id
        self.name = name
        self.filepath = filepath
        self._scene_parts = []
        self._xml_string = None

    @property
    def xml_string(self):
        return self._xml_string

    @property
    def filepath_with_id(self):
        """Return the XML filepath including the id of the scene tag."""
        return self.filepath + "#" + self.xml_id

    def add_scene_parts(self, scene_parts: list[ScenePart] | ScenePart):
        """Add scene parts to the scene.

        :param scene_parts: A single ScenePart object or a list of ScenePart objects.
        :return: None
        """
        if (
                not isinstance(scene_parts, ScenePart) and
                not (isinstance(scene_parts, list) and all(isinstance(p, ScenePart) for p in scene_parts))
        ):
            raise TypeError("Argument scene_parts must be an instance of ScenePart or a list of ScenePart instances.")
        if isinstance(scene_parts, ScenePart):
            self._scene_parts.append(scene_parts)
        else:
            self._scene_parts.extend(scene_parts)

    def create_xml_string(self):
        if self.name is None:
            raise ValueError("Name of scene must be defined before creating XML string.")

        # Generate the XML as string using build_scene from pyhelios.utils.scene_writer
        self._xml_string = scene_writer.build_scene(scene_id=self.xml_id, name=self.name, sceneparts=[str(p) for p in self._scene_parts])

    def write_scene_file(self):
        """Write the scene XML string to the destination filepath."""
        if self.xml_string is None:
            self.create_xml_string()
        if self.filepath is None:
            raise ValueError("Scene filepath must be defined before writing scene file.")

        with open(self.filepath, "w") as f:
            f.write(self.xml_string)


class FlightPath:
    """Create an XML file containing the flight path, i.e. all legs, for a HELIOS++ survey."""
    # Flight patterns allowed by pyhelios.utils.flight_planner
    flight_pattern_options = ["parallel", "criss-cross"]

    def __init__(
            self,
            filepath: str,
            bbox: list[float],
            spacing: float,
            altitude: float,
            velocity: float,
            flight_pattern: str = "parallel",
            trajectory_time_interval: float = .05,
            always_active: bool = False,
            scanner_settings_id: str = "scanner_settings"
    ):
        self.filepath = filepath
        self.bbox = bbox
        self.spacing = spacing
        self.altitude = altitude
        self.velocity = velocity

        if flight_pattern not in FlightPath.flight_pattern_options:
            raise ValueError(f"Argument flight_pattern must be one of {FlightPath.flight_pattern_options}.")
        self.flight_pattern = flight_pattern

        self.trajectory_time_interval = trajectory_time_interval
        self.always_active = always_active
        self.scanner_settings_id = scanner_settings_id

        self.rotate_deg = 0.0
        self.waypoints = None
        self.tree = None
        self._xml_string = ""

    @property
    def xml_string(self):
        return self._xml_string

    def compute_waypoints(self):
        """Compute flight path waypoints using pyhelios.utils.flight_planner."""
        self.waypoints, _, _ = flight_planner.compute_flight_lines(
            self.bbox, self.spacing, rotate_deg=self.rotate_deg, flight_pattern=self.flight_pattern
        )

    def create_element_tree(self):
        """Create XML element tree from waypoints using ElementTree."""
        root = eT.Element("survey")
        self.tree = eT.ElementTree(root)

        # Determine if scanner should be active on the short connecting legs.
        short_leg_active = "true" if self.always_active else "false"

        for i, leg in enumerate(self.waypoints):
            # Add comment with leg number.
            root.append(eT.Comment(f"leg {i:03}"))

            # Add leg element
            leg_element = eT.Element("leg")
            root.append(leg_element)

            # Create platformSettings element and append it to the leg element.
            platform_settings_element = eT.Element(
                "platformSettings",
                attrib={
                    "x": str(leg[0]),
                    "y": str(leg[1]),
                    "z": str(self.altitude),
                    "movePerSec_m": str(self.velocity)
                }
            )
            leg_element.append(platform_settings_element)

            # Create scannerSettings element and append it ot the legs element.
            scanner_settings_element = eT.Element(
                "scannerSettings",
                attrib={
                    "template": str(self.scanner_settings_id),
                    "trajectoryTimeInterval_s": str(self.trajectory_time_interval)
                }
            )
            # For odd legs, specify if the scanner should be active. (Default: false)
            if i % 2 != 0:
                scanner_settings_element.attrib["active"] = str(short_leg_active)
            leg_element.append(scanner_settings_element)

        # Apply correct indentation and line breaks.
        eT.indent(self.tree, "    ")

    def create_xml_string(self, using_helios_flight_planner=False):
        """Create a string representation of flight path XML if necessary."""
        if self.tree is None:
            self.create_element_tree()
        self._xml_string = eT.tostring(self.tree.getroot(), encoding="unicode", xml_declaration=False)

    def write_flight_path_file(self):
        """Write flight path XML to destination filepath."""
        if self.filepath is None:
            raise ValueError("Scene filepath must be defined before writing scene file.")

        if self.tree is None:
            self.create_element_tree()
        self.tree.write(self.filepath, encoding="UTF-8", xml_declaration=False)


class Survey:

    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath

        self.config = None
        self.generator = None
        self.executor = None

    def create_config:
        pass


class SurveyConfig:

    def __init__(
            self,
            scene: str | Scene = "",
            platform: str = "",
            scanner: str = "",
            flight_path: str | FlightPath = "",
            scanner_settings_id: str = "scsanner_settings",
            pulse_freq: float = .0,
            scan_freq: float = .0,
            scan_angle: int = 0,
            velocity: int = 0,
            accuracy: float = .0
    ):
        pass


class SurveyGenerator:
    """Create a HELIOS++ survey XML file based on a template file by populating it with parameter values,
    scene information, and flight path, i.e., legs."""

    def __init__(
            self,
            name: str,
            filepath: str,
            scene: Scene,
            platform: str = "",
            scanner: str = "",
            flight_path: str | FlightPath = ""
    ):
        self.name = name
        self.filepath = filepath
        self.scene = scene
        self.platform = platform
        self.scanner = scanner
        if not (isinstance(flight_path, str) or isinstance(flight_path, FlightPath)):
            raise TypeError("Argument flight_path must be string or FlightPath object.")
        self.flight_path = flight_path

        self.scanner_settings_id = "scanner_settings"
        self.pulse_freq = None
        self.scan_freq = None
        self.scan_angle = None
        self.velocity = None
        self.accuracy = None

        self.tree = None

    @classmethod
    def read_template(cls):
        return parse_xml_with_comments(helios_survey_template_filepath)

    def populate_template(self):
        self.tree = SurveyGenerator.read_template()
        root = self.tree.getroot()

        # Apply attribute values to scannerSettings element
        scanner_settings = root.find("scannerSettings").attrib
        scanner_settings["id"] = str(self.scanner_settings_id)
        scanner_settings["active"] = "true"
        scanner_settings["pulseFreq_hz"] = str(self.pulse_freq)
        scanner_settings["scanAngle_deg"] = str(self.scan_angle)
        scanner_settings["scanFreq_hz"] = str(self.scan_freq)

        # Apply attribute values to survey element
        survey = root.find("survey").attrib
        survey["name"] = str(self.name)
        survey["scene"] = str(self.scene.filepath_with_id)
        survey["platform"] = str(self.platform)
        survey["scanner"] = str(self.scanner)

        # Apply attribute values to detectorSettings elemenet
        detector_settings = root.find("survey/detectorSettings").attrib
        detector_settings["accuracy_m"] = str(self.accuracy)

        # Insert legs into survey element from flight path XML
        survey_element = root.find("survey")

        if isinstance(self.flight_path, str):
            flight_path_tree = parse_xml_with_comments(self.flight_path)
        else:  # is FlightPath object
            flight_path_tree = self.flight_path.tree
        flight_path_root_element = flight_path_tree.getroot()

        for child in flight_path_root_element:
            survey_element.append(child)

        # Correct indentation after insertion of leg elements with different indentation
        eT.indent(self.tree, "    ")

    def write_survey_file(self):
        if not isinstance(self.tree, eT.ElementTree):
            raise TypeError("Populate XML element tree first, e.g. using populate_template().")
        else:
            self.tree.write(self.filepath, encoding="UTF-8", xml_declaration=True)


class SurveyExecutor:

    def __init__(self):
        pass


class Scenario:

    def __init__(self):
        pass


if __name__ == "__main__":
    pass
