from pyhelios.util import scene_writer
from experiment.xml_generator import XMLGenerator


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

    def __init__(self, filepath: str, xml_id: str, name: str, scene_parts: list[dict]):
        self.filepath = filepath
        self.xml_id = xml_id
        self.name = name
        self.scene_part_dicts = scene_parts

        self.scene_parts = []
        self.scene_generator: SceneGenerator | None = None

    @property
    def filepath_with_id(self):
        """Return the XML filepath including the id of the scene tag."""
        return self.filepath + "#" + self.xml_id

    def create_scene_parts(self):
        """Create SceneParts from the list of dictionaries provided and append them to the list of SceneParts"""
        for d in self.scene_part_dicts:
            if d["type"].lower() == "obj":
                self.scene_parts.append(
                    ScenePartOBJ(filepath=d["filepath"], up_axis=d["up_axis"])
                )
            elif d["type"].lower() in ["tif", "tiff"]:
                self.scene_parts.append(
                    ScenePartTIFF(filepath=d["filepath"], mat_file=d["material_filepath"], mat_name=d["material_name"])
                )
            else:
                raise ValueError(f"Scene part type {d['type']} is not supported.")

    def create_scene_xml(self):
        """Generate the scene XML based on the provided list of scene part dictionaries"""
        self.scene_generator = SceneGenerator(self.filepath, self.xml_id, self.name)
        self.create_scene_parts()
        self.scene_generator.add_scene_parts(self.scene_parts)
        self.scene_generator.write_file()


class SceneGenerator(XMLGenerator):
    """Create an XML file for a pyhelios scene from scene parts."""

    def __init__(self, filepath: str, xml_id: str, name: str):
        """Parameters

        :param filepath: Destination path for the scene XML file.
        :param xml_id: Attribute id of tag scene.
        :param name: Attribute name of tag scene.
        """
        super().__init__(filepath, xml_declaration=True)

        self.xml_id = xml_id
        self.name = name

        self._scene_parts = []
        self.xml_string = None

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
        self.xml_string = scene_writer.build_scene(
            scene_id=self.xml_id,
            name=self.name,
            sceneparts=[str(p) for p in self._scene_parts]
        )

    def write_file(self):
        """Write the scene XML string to the destination filepath."""
        if self.xml_string is None:
            self.create_xml_string()
        super().write_file()
