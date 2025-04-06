from typing_extensions import Self


class ColorNode:
    """
    Atomic component used to build ColorHierarchyTree instance.

    Parameters
    ----------
    color : int
        Color designation

    Attributes
    ----------
    children : list[ColorNode]
        List containing children of type ColorNode

    associated_vertices : list[int]
        List of integers containing the associated vertices assigned the color of the ColorNode after refinement

    """

    def __init__(self, color: int = None) -> None:
        self.color = color
        self.children: list[Self] = list()
        self.associated_vertices = None

    def __eq__(self, other: Self):
        return self.color == other.color

    def __repr__(self):
        return "NA" if self.color is None else str(self.color)

    def __str__(self):
        return (f"{{ColorNode with -> Color: {self.color}, Children: {self.children}, "
                f"associated vertices: {self.associated_vertices}}}")

    def add_child(self, child: Self) -> None:
        """
        Adds a child to the ColorNode instance.

        Parameters
        ----------
        child : ColorNode
            An instance of ColorNode type

        """

        self.children.append(child)

    def update_associated_vertices(self, vertices: list[int]) -> None:
        """
        Updates the list of associated vertices.

        Parameters
        ----------
        vertices : list[int]
            List containing the vertices with the color of the ColorNode instance after refinement

        """

        self.associated_vertices = vertices
