from pyhelios.util import scene_writer


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
