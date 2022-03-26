from emepy.eme import EME
from emepy.geometries import Geometry, DynamicPolygon
from emepy.source import Source
from emepy.monitors import Monitor
import emepy
import numpy as np
from matplotlib import pyplot as plt


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
                    remaining_design = remaining_design[length:] if length < len(remaining_design) else []

    def set_design_readable(
        self, design_x: list = [], design_y: list = [], design_z: list = [], dimensions: int = 2
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
        self.eme.reset(parallel=self.eme.parallel, configure_parallel=False)
        self.eme.add_layers(*layers)

    def update_eme(self) -> None:
        """Updades the eme object with geometric changes"""
        self.start()

    def get_n(self, grid_x: "np.ndarray", grid_y: "np.ndarray", grid_z: "np.ndarray") -> "np.ndarray":
        """Currently returns the n for the first design region"""
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                return geometry.get_n(grid_x, grid_y, grid_z-grid_z[0])

    def gradient(self, grid_x: "np.ndarray", grid_y, grid_z: "np.ndarray", dp=1e-10) -> "np.ndarray":
        """Computes the gradient A_u using a finite difference"""

        # Get initial design
        design = self.get_design()

        # Final jacobian setup
        jacobian = np.zeros((3, 3, grid_x.shape[0] - 1, grid_y.shape[0] - 1, grid_z.shape[0] - 1, len(design)), dtype=complex)

        # Get initial A
        A_ii = self.get_n(grid_x, grid_y, grid_z)
        

        # Get gradients
        for i, d in enumerate(design):

            # Step
            design[i] = d + dp
            self.set_design(design)

            # Compute new A
            A_new = self.get_n(grid_x, grid_y, grid_z)

            # Revert step
            design[i] = d
            self.set_design(design)

            # Compute gradient
            gradient = (A_new - A_ii) / dp

            # Assign gradient
            jacobian[0, 0, :, :, :, i] = gradient
            jacobian[1, 1, :, :, :, i] = gradient
            jacobian[2, 2, :, :, :, i] = gradient

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
                z_end += geometry.length

        # Create source and monitor
        source = Source(z=0.25, mode_coeffs=[1], k=1)  # Hard coded
        forward_monitor = self.eme.add_monitor(axes="xyz",mesh_z=self.mesh_z, sources=[source])

        a_source = Source(z=2.75, mode_coeffs=[1], k=-1)  # Hard coded
        adjoint_monitor = self.eme.add_monitor(axes="xyz",mesh_z=self.mesh_z, sources=[a_source])

        # Run eme
        self.eme.propagate()

        # Get near results
        grid_x, grid_y, grid_z, field_x = forward_monitor.get_array("Ex", z_range=(z_start, z_end))
        field_x = 0.125 * (field_x[1:, 1:, 1:] + field_x[1:, :-1, 1:] + field_x[:-1, 1:, 1:] + field_x[:-1, :-1, 1:] + field_x[1:, 1:,:-1] + field_x[1:, :-1,:-1] + field_x[:-1, 1:,:-1] + field_x[:-1, :-1,:-1])
        field_y = forward_monitor.get_array("Ey", z_range=(z_start, z_end))[3]
        field_y = 0.125 * (field_y[1:, 1:, 1:] + field_y[1:, :-1, 1:] + field_y[:-1, 1:, 1:] + field_y[:-1, :-1, 1:] + field_y[1:, 1:,:-1] + field_y[1:, :-1,:-1] + field_y[:-1, 1:,:-1] + field_y[:-1, :-1,:-1])
        field_z = forward_monitor.get_array("Ez", z_range=(z_start, z_end))[3]
        field_z = 0.125 * (field_z[1:, 1:, 1:] + field_z[1:, :-1, 1:] + field_z[:-1, 1:, 1:] + field_z[:-1, :-1, 1:] + field_z[1:, 1:,:-1] + field_z[1:, :-1,:-1] + field_z[:-1, 1:,:-1] + field_z[:-1, :-1,:-1])
        field = np.array([field_x, field_y, field_z])
        results = (grid_x, grid_y, grid_z, field, forward_monitor)

        # Save adjoint results
        a_grid_x, a_grid_y, a_grid_z, a_field_x = adjoint_monitor.get_array("Ex", z_range=(z_start, z_end))
        a_field_x = 0.125 * (a_field_x[1:, 1:, 1:] + a_field_x[1:, :-1, 1:] + a_field_x[:-1, 1:, 1:] + a_field_x[:-1, :-1, 1:] + a_field_x[1:, 1:,:-1] + a_field_x[1:, :-1,:-1] + a_field_x[:-1, 1:,:-1] + a_field_x[:-1, :-1,:-1])
        a_field_y = adjoint_monitor.get_array("Ey", z_range=(z_start, z_end))[3]
        a_field_y = 0.125 * (a_field_y[1:, 1:, 1:] + a_field_y[1:, :-1, 1:] + a_field_y[:-1, 1:, 1:] + a_field_y[:-1, :-1, 1:] + a_field_y[1:, 1:,:-1] + a_field_y[1:, :-1,:-1] + a_field_y[:-1, 1:,:-1] + a_field_y[:-1, :-1,:-1])
        a_field_z = adjoint_monitor.get_array("Ez", z_range=(z_start, z_end))[3]
        a_field_z = 0.125 * (a_field_z[1:, 1:, 1:] + a_field_z[1:, :-1, 1:] + a_field_z[:-1, 1:, 1:] + a_field_z[:-1, :-1, 1:] + a_field_z[1:, 1:,:-1] + a_field_z[1:, :-1,:-1] + a_field_z[:-1, 1:,:-1] + a_field_z[:-1, :-1,:-1])
        a_field = np.array([a_field_x, a_field_y, a_field_z])
        self.adjoint_results = (a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor)

        return results

    def objective_gradient(self, monitor: "Monitor"):
        """Computes the objective function gradient to the sources for the adjoint formulation"""

        #### HARD CODED FOR NOW

        # # Compute power in end
        # x, _, Ex = monitor.get_array(component="Ex", z_range=(2.746, 2.754))
        # Ex = Ex[:, 0]
        # Ey = monitor.get_array(component="Ey", z_range=(2.746, 2.754))[2][:, 0]
        # Hx = monitor.get_array(component="Hx", z_range=(2.746, 2.754))[2][:, 0]
        # Hy = monitor.get_array(component="Hy", z_range=(2.746, 2.754))[2][:, 0]
        # exp_Ex = self.eme.activated_layers[0][-1].modes[0].Ex
        # exp_Ey = self.eme.activated_layers[0][-1].modes[0].Ey
        # exp_Hx = self.eme.activated_layers[0][-1].modes[0].Hx
        # exp_Hy = self.eme.activated_layers[0][-1].modes[0].Hy
        # exp_Ex = exp_Ex[:, exp_Ex.shape[1] // 2]
        # exp_Ey = exp_Ey[:, exp_Ey.shape[1] // 2]
        # exp_Hx = exp_Hx[:, exp_Hx.shape[1] // 2]
        # exp_Hy = exp_Hy[:, exp_Hy.shape[1] // 2]

        # # Compute power in source
        # def overlap(Ex, Ey, Hx, Hy, grid):
        #     return np.trapz(Ex * np.conj(Hy) - Ey * np.conj(Hx), grid)

        # norm = overlap(Ex, Ey, Hx, Hy, x)
        # exp_norm = overlap(exp_Ex, exp_Ey, exp_Hx, exp_Hy, x)
        # power = overlap(Ex, Ey, exp_Hx, exp_Hy, x) / np.sqrt(norm) / np.sqrt(exp_norm)
        # power = np.abs(power)
        network = self.eme.network
        pins = dict(zip([pin.name for pin in network.pins], [0.0 for pin in network.pins]))
        pins["left0"] = 1
        power = np.abs(emepy.ModelTools.compute(network, pins, 0)["right0"])

        # Compute autogradient
        f_x = 0.0

        return f_x, power

    def set_adjoint_sources(self, f_x: float = 0.0, overlap: float = 1.0):
        """Computes and places the adjoint sources for use in the adjoint formulation"""

        # Create source and monitor
        scale = 2 * np.pi * self.eme.wavelength * overlap
        source = Source(z=2.75, mode_coeffs=[scale], k=-1)  # Hard coded
        return [source]

    def adjoint_run(self, sources: list):
        """Performs the adjoint run for use in the adjoint formulation"""
        # # Clear the eme and ensure design is inside
        # self.start()

        # # Find where monitor should be in range of only the design region
        # z_start, z_end = (0.5, 2.5)
        # # for geometry in self.geometries:
        # #     if isinstance(geometry, DynamicPolygon):
        # #         z_end += geometry.length
        # #         break
        # #     else:
        # #         z_start += geometry.length

        # # Set monitor
        # monitor = self.eme.add_monitor(mesh_z=self.mesh_z, sources=sources)

        # # Run eme
        # self.eme.propagate()

        # # Get results
        # grid_x, grid_z, field_x = monitor.get_array("Ex", z_range=(z_start, z_end))
        # field_x = 0.25 * (field_x[1:, 1:] + field_x[1:, :-1] + field_x[:-1, 1:] + field_x[:-1, :-1])
        # field_y = monitor.get_array("Ey", z_range=(z_start, z_end))[2]
        # field_y = 0.25 * (field_y[1:, 1:] + field_y[1:, :-1] + field_y[:-1, 1:] + field_y[:-1, :-1])
        # field_z = monitor.get_array("Ez", z_range=(z_start, z_end))[2]
        # field_z = 0.25 * (field_z[1:, 1:] + field_z[1:, :-1] + field_z[:-1, 1:] + field_z[:-1, :-1])
        # field = np.array([field_x, field_y, field_z])
        # return grid_x, grid_z, field, monitor

        a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor = self.adjoint_results
        a_field *= sources[0].mode_coeffs[0]
        return a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor

    def optimize(self, design: list, dp=1e-10) -> "np.ndarray":
        """Runs a single step of shape optimization"""

        if self.eme.am_master():
            print("Now performing an optimization...")

        # Update the design region
        design = design if not isinstance(design, np.ndarray) else design.tolist()
        self.set_design(design)

        # Compute the forward run
        grid_x, grid_y, grid_z, X, monitor_forward = self.forward_run()

        plt.figure()
        monitor_forward.visualize(axes="xz")
        plt.savefig('forward')

        # Compute the partial gradient of the objective function f_x
        f_x, overlap = self.objective_gradient(monitor_forward)

        # Calculate the adjoint sources
        sources = self.set_adjoint_sources(overlap)

        # Compute the adjoint run
        grid_x, grid_y, grid_z, lamdagger, monitor_adjoint = self.adjoint_run(sources)

        plt.figure()
        monitor_adjoint.visualize(axes="xz")
        plt.savefig('adjoint')

        # Compute the gradient of the constraint A_u
        A_u = self.gradient(
            grid_x, grid_y, grid_z, dp=dp
        )  # A will need to be 3x3xthe rest to incorporate the right dimensions and cross dielectric etc.

        # plt.figure()
        # plt.imshow(np.real(np.sum(A_u[0,0], axis=2)*dp))
        # plt.show()

        # Calculate the full gradient of the objective function f_u
        f_u = self.compute_final_gradient(lamdagger, A_u, X)
    
        # Return the gradient
        return overlap, f_u, monitor_forward

    def compute_final_gradient(self, lamdagger: "np.ndarray", A_u: "np.ndarray", X: "np.ndarray"):
        """Computes the final gradient using the adjoint formulation and loops to conserve memory"""

        # Initialize final result
        f_u = np.zeros(A_u.shape[-1], dtype=float)
        f_u_grid = np.zeros([3] + list(A_u.shape[2:-1]) + [A_u.shape[-1]], dtype=complex)

        # Reshape
        lamdagger = np.transpose(np.conj(lamdagger))
        A_u = A_u
        X = X

        # Loop through all params
        for p in range(len(f_u)):
            A_u_temp = A_u[..., p]

            # Compute all 9 components of the matrix
            A_u_x = np.zeros([3] + list(A_u.shape[2:-1]), dtype=complex)
            for i, mi in enumerate(A_u_temp):
                for j, mij in enumerate(mi):
                    A_u_x[i] += mij * X[j]

            # Compute lambda * A_u_x
            for i in range(3):
                f_u[p] += np.real(np.sum(A_u_x[i] * lamdagger[..., i].T))
                f_u_grid[...,p] += A_u_x[i] * lamdagger[..., i].T

        ppp = np.sum(f_u_grid, axis=0)
        ppp = ppp[:,ppp.shape[1]//2,:,:]

        return f_u, ppp

    def draw(self) -> None:
        self.start()
        self.eme.draw()
