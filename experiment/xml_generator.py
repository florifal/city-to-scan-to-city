from xml.etree import ElementTree as eT
from pyhelios.util import flight_planner

from .global_vars import helios_survey_template_filepath


def parse_xml_with_comments(xml_filepath: str) -> eT.ElementTree:
    """Parse an XML file using ElementTree, including the comments, and return an ElementTree object."""
    parser = eT.XMLParser(target=eT.TreeBuilder(insert_comments=True))  # Retain XML comments (req. Python 3.8)
    return eT.parse(xml_filepath, parser)


class XMLGenerator:

    def __init__(self, filepath: str, xml_declaration: bool = True):
        """Superclass for classes generating HElIOS++ XML configuration files

        :param filepath: Destination file path for the output XML file.
        :param xml_declaration: Option to add XML declaration tag at the top of the file.
        """
        self.filepath: str = filepath
        self.xml_declaration: bool = xml_declaration

        self.tree: eT.ElementTree | None = None
        self.xml_string: str = ""

    @classmethod
    def parser(cls) -> eT.XMLParser:
        """Return an ElementTree parser that includes comments in the tree structure

        A new parser must be generated for each ElementTree. For this reason, this is implemented as a class method
        instead of a class attribute. See https://stackoverflow.com/a/28127427
        """
        return eT.XMLParser(target=eT.TreeBuilder(insert_comments=True))  # Retain XML comments (req. Python 3.8)

    def create_element_tree(self):
        """Create an ElementTree, either from an already existing XML string or an empty tree"""
        if self.xml_string != "":
            root_element = eT.fromstring(self.xml_string, XMLGenerator.parser())
            self.tree = eT.ElementTree(root_element)
        else:
            self.tree = eT.ElementTree()

        # Apply correct indentation and line breaks.
        eT.indent(self.tree, "    ")

    def create_xml_string(self):
        """Create a string representation of XML element tree if necessary."""
        # The following either creates an empty ElementTree, which it does not make sense to turn into a string, or it
        # creates a string from an ElementTree, only to then turn it back to a string using this method, which does not
        # make much sense either. The idea may have been to obtain an XML string with nice formatting from one with
        # ugly formatting. For now, I decided against using it.
        # if self.tree is None:
        #     self.create_element_tree()
        if not isinstance(self.tree, eT.ElementTree):
            raise ValueError("Cannot create XML string: ElementTree undefined.")
        self.xml_string = eT.tostring(self.tree.getroot(), encoding="unicode", xml_declaration=self.xml_declaration)

    def write_file(self):
        """Write the ElementTree to an XML file."""
        if self.filepath is None or self.filepath == "":
            raise ValueError("Argument filepath must be defined before writing XML file.")
        if self.tree is None:
            self.create_element_tree()
        self.tree.write(self.filepath, encoding="UTF-8", xml_declaration=self.xml_declaration)


class FlightPathGenerator(XMLGenerator):
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
        # No XML declaration tag for flight path XML, because it only serves to be embedded into survey XML
        super().__init__(filepath, xml_declaration=False)

        self.bbox = bbox
        self.spacing = spacing
        self.altitude = altitude
        self.velocity = velocity

        if flight_pattern not in FlightPathGenerator.flight_pattern_options:
            raise ValueError(f"Argument flight_pattern must be one of {FlightPathGenerator.flight_pattern_options}.")
        self.flight_pattern = flight_pattern

        self.trajectory_time_interval = trajectory_time_interval
        self.always_active = always_active
        self.scanner_settings_id = scanner_settings_id

        self.rotate_deg = 0.0
        self.waypoints = None

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


class SurveyGenerator(XMLGenerator):
    """Create a HELIOS++ survey XML file based on a template file by populating it with parameter values,
    scene information, and flight path, i.e., legs."""

    def __init__(
            self,
            filepath: str,
            # <survey>
            name: str,
            scene_filepath: str,
            platform_filepath: str,
            scanner_filepath: str,
            # <scannerSettings>
            scanner_settings_id: str,
            scanner_active: bool,
            pulse_freq: float,
            scan_freq: float,
            scan_angle: float,
            # <detectorSettings>
            detector_accuracy: float,
            # Flight path
            flight_path_filepath: str
    ):
        super().__init__(filepath, xml_declaration=True)

        self.name = name
        self.scene_filepath = scene_filepath
        self.platform_filepath = platform_filepath
        self.scanner_filepath = scanner_filepath

        self.scanner_settings_id = scanner_settings_id
        self.scanner_active = scanner_active
        self.pulse_freq = pulse_freq
        self.scan_freq = scan_freq
        self.scan_angle = scan_angle
        self.detector_accuracy = detector_accuracy

        self.flight_path_filepath = flight_path_filepath

    @classmethod
    def read_template(cls):
        return parse_xml_with_comments(helios_survey_template_filepath)

    def create_element_tree(self):
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
        survey["scene"] = str(self.scene_filepath)
        survey["platform"] = str(self.platform_filepath)
        survey["scanner"] = str(self.scanner_filepath)

        # Apply attribute values to detectorSettings element
        detector_settings = root.find("survey/detectorSettings").attrib
        detector_settings["accuracy_m"] = str(self.detector_accuracy)

        # Insert legs into survey element from flight path XML
        survey_element = root.find("survey")
        flight_path_tree = parse_xml_with_comments(self.flight_path_filepath)
        flight_path_root_element = flight_path_tree.getroot()
        for child in flight_path_root_element:
            survey_element.append(child)

        # Correct indentation after insertion of leg elements with different indentation
        eT.indent(self.tree, "    ")
