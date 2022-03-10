from sympy import jacobi_normalized
from emepy.eme import EME
from emepy.geometries import Geometry, DynamicPolygon
from emepy.source import Source
from emepy.monitors import Monitor
import numpy as np


class Optimization(object):
    """Optimizatoin objects store geometries and can manipulate design regions. Essentially, they form the methods needed for running shape optimizations"""

    def __init__(self, eme: "EME", geometries: list = [], mesh_z: int = 100) -> None:
        """Creates an instance of Optimization for running shape optimization"""
        self.eme = eme
        self.geometries = geometries
        self.mesh_z = mesh_z
        self.start()

    def add_geometry(self, geometry: "Geometry") -> None:
        """Adds a Geometry object to the optimization"""
        self.geometries.append(geometry)

    def add_geometries(self, *geometries) -> None:
        """Adds any number of Geometry objects as arguments to the optimization"""
        for geometry in geometries:
            self.geometries.append(geometry)

    def get_design(self) -> list:
        """Returns the design region associated with all design geometries in the optimization"""
        design = []
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                design += geometry.get_design()

        return design

    def get_design_readable(self, dimensions: int = 2) -> list:
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

    def set_design(self, design: list) -> None:
        """Sets the design region provided for the entire system of design regions"""
        remaining_design = design[:]
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                length = len(geometry)
                if length:
                    geometry.set_design(remaining_design[:length])
                    remaining_design = (
                        remaining_design[length:]
                        if length < len(remaining_design)
                        else []
                    )

    def set_design_readable(
        self,
        design_x: list = [],
        design_y: list = [],
        design_z: list = [],
        dimensions: int = 2,
    ) -> None:
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

    def start(self) -> None:
        """Initializes the EME"""
        layers = [layer for geometry in self.geometries for layer in geometry]
        self.eme.reset()
        self.eme.add_layers(*layers)

    def update_eme(self) -> None:
        """Updades the eme object with geometric changes"""
        self.start()

    def get_n(self, grid_x: "np.ndarray", grid_z: "np.ndarray") -> "np.ndarray":
        """Currently returns the n for the first design region"""
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                return geometry.get_n(grid_x, grid_z)

    def gradient(
        self, grid_x: "np.ndarray", grid_z: "np.ndarray", dp=1e-14
    ) -> "np.ndarray":
        """Computes the gradient A_u using a finite difference"""

        # Get initial design
        design = self.get_design()

        # Final jacobian setup
        jacobian = np.zeros(grid_x.shape, grid_z.shape, len(design))

        # Get initial A
        A_ii = self.get_n(grid_x, grid_z)

        # Get gradients
        for i, d in enumerate(design):

            # Step
            design[i] = d + dp
            self.set_design(design)

            # Compute new A
            A_new = self.get_n(grid_x, grid_z)

            # Revert step
            design[i] = d + dp
            self.set_design(design)

            # Compute gradient
            gradient = (A_new - A_ii) / dp

            # Assign gradient
            jacobian[:, :, i] = gradient

        return jacobian

    def forward_run(self) -> tuple:
        """Computes the forward run for the adjoint formulation"""

        # Clear the eme and ensure design is inside
        self.start()

        # Find where monitor should be in range of only the design region
        z_start, z_end = (0, 0)
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                z_end += geometry.length
                break
            else:
                z_start += geometry.length

        # Create source and monitor
        source = Source(z=0.25e-6, mode_coeffs=[1], k=1)  ### Hard coded
        monitor = self.eme.add_monitor(mesh_z=self.mesh_z, sources=[source])

        # Run eme
        self.eme.propagate()

        # Get results
        grid_x, grid_z, field = monitor.get_array("Ex", z_range=(z_start, z_end))
        return grid_x, grid_z, field, monitor

    def objective_gradient(self, monitor: "Monitor"):
        """Computes the objective function gradient to the sources for the adjoint formulation"""

        #### HARD CODED FOR NOW

        # Compute power in end
        _, _, power = monitor.get_array(z_range=(2.74e-6, 2.76e-6))
        power = np.sum(power[:, 0])

        # Compute power in source

        # Compute autogradient
        f_x = 0.0

        return f_x, power

    def set_adjoint_sources(self, f_x: float = 0.0, overlap: float = 1.0):
        """Computes and places the adjoint sources for use in the adjoint formulation"""

        # Create source and monitor
        scale = 2 * np.pi * self.eme.wavelength * overlap
        source = Source(z=2.75e-6, mode_coeffs=[scale], k=-1)  ### Hard coded
        return [source]
        

    def adjoint_run(self, sources:list):
        """Performs the adjoint run for use in the adjoint formulation"""
        # Clear the eme and ensure design is inside
        self.start()

        # Find where monitor should be in range of only the design region
        z_start, z_end = (0, 0)
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                z_end += geometry.length
                break
            else:
                z_start += geometry.length

        # Set monitor
        monitor = self.eme.add_monitor(
            mesh_z=self.mesh_z, z_range=(z_start, z_end), sources=sources
        )

        # Run eme
        self.eme.propagate()

        # Get results
        grid_x, grid_z, field = monitor.get_array("Ex")
        return grid_x, grid_z, field

    def optimize(self, design: list) -> "np.ndarray":
        """Runs a single step of shape optimization"""

        # Update the design region
        self.set_design(design)

        # Compute the forward run
        grid_x, grid_z, X, monitor = self.forward_run()

        # Compute the partial gradient of the objective function f_x
        f_x, overlap = self.objective_gradient(monitor)

        # Calculate the adjoint sources
        sources = self.set_adjoint_sources(overlap)

        # Compute the adjoint run
        lamdagger, grid_x, grid_z = self.adjoint_run(sources)

        # Compute the gradient of the constraint A_u
        A_u = self.gradient(grid_x, grid_z)

        # Calculate the full gradient of the objective function f_u
        f_u = np.transpose(np.conj(lamdagger)) @ A_u @ X

        # Return the gradient
        return f_u
