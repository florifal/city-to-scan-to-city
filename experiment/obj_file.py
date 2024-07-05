from __future__ import annotations
from pathlib import Path
from shutil import copy2


def split_obj_file(obj_filepath: Path | str, output_dirpath: Path | str, overwrite: bool = False):
    file = OBJFile(obj_filepath)
    file.write_individual_objects(output_dirpath, overwrite)


class OBJFile:

    def __init__(self, filepath: Path | str):
        """Parse input OBJ file"""
        self.filepath = Path(filepath)

        self.mtllib: str | None = None
        self.vertices: list[list[str]] = []
        self.faces: list[OBJFace] = []
        self.objects: list[OBJObject] = []

        current_object: OBJObject | None = None
        current_material: str | None = None

        with open(self.filepath, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            # Remove \n at the end
            line = line[:-1]

            # Faces
            if line.startswith("f "):
                # Append a new face to the list
                vertex_ids = line.split(" ")[1:]
                vertex_ids = [vertex_id.split("//")[0] for vertex_id in vertex_ids]
                face = OBJFace(vertex_ids, current_material, current_object)
                self.faces.append(face)

                # Add the face to an object group
                if current_object is not None:
                    current_object.add_face(face)

            # New material setting for following lines
            elif line.startswith("usemtl "):
                current_material = line.split(" ", 1)[1]

            # Vertices
            elif line.startswith("v "):
                coords = line.split(" ")[1:]
                self.vertices.append(coords)

            # New object definition
            elif line.startswith("o "):
                # Before starting a new object, add the current object to self's object list
                self.add_object(current_object)

                name = line.split(" ", 1)[1]
                current_object = OBJObject(name)

            # Material file definition (should occur only once)
            elif line.startswith("mtllib "):
                self.mtllib = line.split(" ", 1)[1]

            # # Pass over comments
            # elif line.startswith("#"):
            #     pass
            #
            # else:
            #     raise Exception(f"OBJ parser encountered unparsable line (no. {i+1}):\n{line}\nIn file:\n{filepath}")

        self.add_object(current_object)

    def add_object(self, o: OBJObject | None):
        if o is not None:
            self.objects.append(o)

    # todo: remove
    # def write_individual_object(self, o: Object, filepath: Path | str):
    #     filepath = Path(filepath)
    #
    #     vertex_ids = o.get_vertex_ids()
    #
    #     lines = []
    #
    #     if self.mtllib is not None and self.mtllib != "":
    #         lines.append(f"mtllib {self.mtllib}\n")
    #
    #     # Create vertex lines for all vertices that are used by any face of the object
    #     for vertex_id in vertex_ids:
    #         # OBJ vertex indexing starts at 1, list indexing starts at 0
    #         coord_string = " ".join(self.vertices[vertex_id-1])
    #         lines.append(f"v {coord_string}\n")
    #
    #     # Object definition line
    #     lines.append(f"o {o.name}\n")
    #
    #     # Create face lines
    #     for face in o.faces:
    #         # Add line to set the material for the face, if defined
    #         if face.material is not None and face.material != "":
    #             lines.append(f"usemtl {face.material}\n")
    #
    #         # Find the indices of the vertices used for the face: It corresponds to the position of the original vertex
    #         # index in the new list `vertex_ids`, which defines the order in which the vertices are written to the new
    #         # file. (+1 because OBJ vertex indexing starts at 1 as opposed to list indexing.)
    #         new_vertex_ids = [str(vertex_ids.index(vertex_id)+1) for vertex_id in face.vertex_ids]
    #         vertex_id_string = " ".join(new_vertex_ids)
    #         lines.append(f"f {vertex_id_string}\n")
    #
    #     with open(filepath, "w") as file:
    #         file.writelines(lines)

    def write_individual_objects(self, dirpath: Path | str, overwrite: bool = False):
        dirpath = Path(dirpath)
        dirpath.mkdir(exist_ok=True)

        for o in self.objects:
            output_filepath = dirpath / f"{o.name}.obj"
            if overwrite or not output_filepath.is_file():
                o.write(output_filepath, vertices=self.vertices, mtllib=self.mtllib)
                # self.write_individual_object(o, dirpath / f"{o.name}.obj")  # todo: remove

        if self.mtllib is not None and self.mtllib != "":
            if overwrite or not (dirpath / self.mtllib).is_file():
                copy2(self.filepath.parent / self.mtllib, dirpath / self.mtllib)


class OBJFace:

    def __init__(self, vertex_ids: list, material: str | None = None, o: OBJObject | None = None):
        """Define a new face

        :param vertex_ids: List of indices of the face's vertices
        :param material: Name of a material for the face, if specified
        :param o: Object group the face belongs to, if specified
        """
        self.vertex_ids = [int(vertex_id) for vertex_id in vertex_ids]
        self.material = material
        self.object = o


class OBJObject:

    def __init__(self, name: str):
        self.name = name
        self.faces = []

    def add_face(self, face: OBJFace):
        self.faces.append(face)

    def get_vertex_ids(self):
        """Get the unique IDs of all vertices of all faces of this object."""
        vertex_ids = []
        for face in self.faces:
            for vertex_id in face.vertex_ids:
                if vertex_id not in vertex_ids:
                    vertex_ids.append(vertex_id)
        return sorted(vertex_ids)

    def write(self, filepath: Path | str, vertices: list[list[str]], mtllib: str | None = None):
        # Problem: Only OBJ has vertices. They are not a property of Face. Face has only vertex_ids.
        # Could give Face the vertex coordinates, but then would have to make sure that there are no duplicate vertices
        # when writing the object to an individual file.
        # For now: Solved by passing the list of vertices from OBJFile to OBJObject

        filepath = Path(filepath)

        vertex_ids = self.get_vertex_ids()

        lines = []

        if mtllib is not None and mtllib != "":
            lines.append(f"mtllib {mtllib}\n")

        # Create vertex lines for all vertices that are used by any face of the object
        for vertex_id in vertex_ids:
            # OBJ vertex indexing starts at 1, list indexing starts at 0
            coord_string = " ".join(vertices[vertex_id-1])
            lines.append(f"v {coord_string}\n")

        # Object definition line
        lines.append(f"o {self.name}\n")

        # Create face lines
        for face in self.faces:
            # Add line to set the material for the face, if defined
            if face.material is not None and face.material != "":
                lines.append(f"usemtl {face.material}\n")

            # Find the indices of the vertices used for the face: It corresponds to the position of the original vertex
            # index in the new list `vertex_ids`, which defines the order in which the vertices are written to the new
            # file. (+1 because OBJ vertex indexing starts at 1 as opposed to list indexing.)
            new_vertex_ids = [str(vertex_ids.index(vertex_id)+1) for vertex_id in face.vertex_ids]
            vertex_id_string = " ".join(new_vertex_ids)
            lines.append(f"f {vertex_id_string}\n")

        with open(filepath, "w", encoding="utf-8") as file:
            file.writelines(lines)
