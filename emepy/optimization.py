from emepy.eme import EME
from emepy.geometries import Geometry, DynamicPolygon
import numpy as np


class Optimization(object):
    """Optimizatoin objects store geometries and can manipulate design regions. Essentially, they form the methods needed for running shape optimizations"""

    def __init__(self, eme: "EME", geometries: list = []):
        """Creates an instance of Optimization for running shape optimization"""
        self.eme = eme
        self.geometries = geometries
        self.start()

    def add_geometry(self, geometry: "Geometry"):
        """Adds a Geometry object to the optimization"""
        self.geometries.append(geometry)

    def add_geometries(self, *geometries):
        """Adds any number of Geometry objects as arguments to the optimization"""
        for geometry in geometries:
            self.geometries.append(geometry)

    def get_design(self):
        """Returns the design region associated with all design geometries in the optimization"""
        design = []
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                design += geometry.get_design()

        return design

    def get_design_readable(self, dimensions: int = 2):
        """Returns the design region associated with all design geometries in the optimization in a readable form (a tuple in the form (x,z) or (x,y,z)"""
        design = self.get_design()
        if not len(design):
            return [], [] if dimensions == 2 else [], [], []
        elif dimensions == 2:
            x, z = (design[::2], design[1::2])
            return x, z
        elif dimensions == 3:
            x, y, z = (design[::3], design[1::3], design[2::3])
            return x, y, z
        else:
            return []

    def set_design(self, design: list):
        """Sets the design region provided for the entire system of design regions"""
        remaining_design = design[:]
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                length = len(geometry)
                if length:
                    geometry.set_design(remaining_design[:length])
                    remaining_design = remaining_design[length:] if length < len(remaining_design) else []

    def set_design_readable(
        self,
        design_x: list = [],
        design_y: list = [],
        design_z: list = [],
        dimensions: int = 2,
    ):
        """Sets the design region provided for the entire system of design regions using readable coordinates"""
        design = []
        if dimensions == 2:
            for x, z in zip(design_x, design_z):
                design.append(x)
                design.append(z)
        elif dimensions == 3:
            for x, y, z in zip(design_x, design_y, design_z):
                design.append(x)
                design.append(y)
                design.append(z)
        if len(design):
            self.set_design(design)

    def start(self):
        """Initializes the EME"""
        layers = [layer for geometry in self.geometries for layer in geometry]
        self.eme.reset()
        self.eme.add_layers(*layers)

    def update_eme(self):
        """Updades the eme object with geometric changes"""
        self.start()

    def optimize(self, design: list):
        """Runs a single step of shape optimization"""

        # Update the design region

        # Compute the forward run

        # Compute the partial gradient of the objective function f_x

        # Calculate the adjoint sources

        # Compute the adjoint run

        # Compute the gradient of the constraint A_u

        # Calcualte the full gradient of the objective function f_u

        # Return the gradient
        return None
