from emepy.eme import EME
from emepy.geometries import Geometry, DynamicPolygon
from emepy.source import Source
from emepy.monitors import Monitor
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
                d = (
                    geometry.get_design()
                    if not isinstance(geometry.get_design(), np.ndarray)
                    else geometry.get_design().tolist()
                )
                design += d

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
                return geometry.get_n(grid_x, grid_y, grid_z - grid_z[0])

    def gradient(self, grid_x: "np.ndarray", grid_y, grid_z: "np.ndarray", dp=1e-10) -> "np.ndarray":

        """Computes the gradient A_u using a finite difference"""

        # Get initial design
        design = self.get_design()

        # Final jacobian setup
        jacobian = np.zeros(
            (3, 3, grid_x.shape[0] - 1, grid_y.shape[0] - 1, grid_z.shape[0] - 1, len(design)), dtype=complex
        )

        # Get gradients
        for i, d in enumerate(design):

            # Step up
            design[i] = d + dp
            self.set_design(design)

            # Compute up A
            A_up = self.get_n(grid_x, grid_y, grid_z) ** 2

            # Step down
            design[i] = d - 2 * dp
            self.set_design(design)

            # Compute down A
            A_down = self.get_n(grid_x, grid_y, grid_z) ** 2

            # Compute gradient
            gradient = (A_up - A_down) / (2 * dp)

            # Revert step
            design[i] = d + dp
            self.set_design(design)

            # Assign gradient
            jacobian[0, 0, :, :, :, i] = gradient
            jacobian[1, 1, :, :, :, i] = gradient
            jacobian[2, 2, :, :, :, i] = gradient

        return jacobian

    def get_design_region(self) -> tuple:
        z_start, z_end = (0, 0)
        for geometry in self.geometries:
            if isinstance(geometry, DynamicPolygon):
                z_end += geometry.length
                break
            else:
                z_start += geometry.length
                z_end += geometry.length
        return z_start, z_end

    def forward_run(self) -> tuple:
        """Computes the forward run for the adjoint formulation"""

        # Clear the eme and ensure design is inside
        self.start()

        # Find where monitor should be in range of only the design region
        z_start, z_end = self.get_design_region()

        # Create forward source and monitor
        source = Source(z=0.2500001, mode_coeffs=[1], k=1)  # Hard coded
        forward_monitor = self.eme.add_monitor(axes="xyz", mesh_z=self.mesh_z, sources=[source])

        # FOM monitor
        source = Source(z=0.25, mode_coeffs=[1], k=1)  # Hard coded
        fom_monitor = self.eme.add_monitor(axes="xy", location=2.75, sources=[source])

        # Adjoint source and monitor
        source = Source(z=2.75, mode_coeffs=[1], k=-1)
        adjoint_monitor = self.eme.add_monitor(axes="xyz", mesh_z=self.mesh_z, sources=[source])

        # Run eme
        self.eme.propagate()

        # Get near results
        grid_x, grid_y, grid_z, field_x = forward_monitor.get_array("Ex", z_range=(z_start, z_end))
        field_x = 0.125 * (
            field_x[1:, 1:, 1:]
            + field_x[1:, :-1, 1:]
            + field_x[:-1, 1:, 1:]
            + field_x[:-1, :-1, 1:]
            + field_x[1:, 1:, :-1]
            + field_x[1:, :-1, :-1]
            + field_x[:-1, 1:, :-1]
            + field_x[:-1, :-1, :-1]
        )
        field_y = forward_monitor.get_array("Ey", z_range=(z_start, z_end))[3]
        field_y = 0.125 * (
            field_y[1:, 1:, 1:]
            + field_y[1:, :-1, 1:]
            + field_y[:-1, 1:, 1:]
            + field_y[:-1, :-1, 1:]
            + field_y[1:, 1:, :-1]
            + field_y[1:, :-1, :-1]
            + field_y[:-1, 1:, :-1]
            + field_y[:-1, :-1, :-1]
        )
        field_z = forward_monitor.get_array("Ez", z_range=(z_start, z_end))[3]
        field_z = 0.125 * (
            field_z[1:, 1:, 1:]
            + field_z[1:, :-1, 1:]
            + field_z[:-1, 1:, 1:]
            + field_z[:-1, :-1, 1:]
            + field_z[1:, 1:, :-1]
            + field_z[1:, :-1, :-1]
            + field_z[:-1, 1:, :-1]
            + field_z[:-1, :-1, :-1]
        )
        field = np.array([field_x, field_y, field_z])

        forward_results = (grid_x, grid_y, grid_z, field, forward_monitor, fom_monitor)

        # Save adjoint results
        a_grid_x, a_grid_y, a_grid_z, a_field_x = adjoint_monitor.get_array("Ex", z_range=(z_start, z_end))
        a_field_x = 0.125 * (
            a_field_x[1:, 1:, 1:]
            + a_field_x[1:, :-1, 1:]
            + a_field_x[:-1, 1:, 1:]
            + a_field_x[:-1, :-1, 1:]
            + a_field_x[1:, 1:, :-1]
            + a_field_x[1:, :-1, :-1]
            + a_field_x[:-1, 1:, :-1]
            + a_field_x[:-1, :-1, :-1]
        )
        a_field_y = adjoint_monitor.get_array("Ey", z_range=(z_start, z_end))[3]
        a_field_y = 0.125 * (
            a_field_y[1:, 1:, 1:]
            + a_field_y[1:, :-1, 1:]
            + a_field_y[:-1, 1:, 1:]
            + a_field_y[:-1, :-1, 1:]
            + a_field_y[1:, 1:, :-1]
            + a_field_y[1:, :-1, :-1]
            + a_field_y[:-1, 1:, :-1]
            + a_field_y[:-1, :-1, :-1]
        )
        a_field_z = adjoint_monitor.get_array("Ez", z_range=(z_start, z_end))[3]
        a_field_z = 0.125 * (
            a_field_z[1:, 1:, 1:]
            + a_field_z[1:, :-1, 1:]
            + a_field_z[:-1, 1:, 1:]
            + a_field_z[:-1, :-1, 1:]
            + a_field_z[1:, 1:, :-1]
            + a_field_z[1:, :-1, :-1]
            + a_field_z[:-1, 1:, :-1]
            + a_field_z[:-1, :-1, :-1]
        )
        a_field = np.array([a_field_x, a_field_y, a_field_z])

        self.adjoint_results = (a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor)

        return forward_results

    def overlap(self, E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        term1 = np.conj(E1x) * H2y - np.conj(E1y) * H2x
        term2 = E2x * np.conj(H1y) - E2y * np.conj(H1x)
        return np.trapz(np.trapz((term1 + term2), x), y)

    def objective_gradient(self, monitor: "Monitor"):
        """Computes the objective function gradient to the sources for the adjoint formulation"""

        # Compute power in end
        x, y, Ex = monitor.get_array(component="Ex")
        _, _, Ey = monitor.get_array(component="Ey")
        _, _, Hx = monitor.get_array(component="Hx")
        _, _, Hy = monitor.get_array(component="Hy")

        # Reference mode
        if self.eme.am_master():
            reference_mode = self.eme.activated_layers[0][-1].modes[0]
            r_Ex = reference_mode.Ex
            r_Ey = reference_mode.Ey
            r_Hx = reference_mode.Hx
            r_Hy = reference_mode.Hy
            norm = np.sqrt(np.abs(self.overlap(r_Ex, r_Ey, r_Hx, r_Hy, r_Ex, r_Ey, r_Hx, r_Hy, x, y)))
            r_Ex = r_Ex / norm
            r_Ey = r_Ey / norm
            r_Hx = r_Hx / norm
            r_Hy = r_Hy / norm

            # Compute overlap
            overlap = self.overlap(Ex, Ey, Hx, Hy, r_Ex, r_Ey, r_Hx, r_Hy, x, y)
        else:
            overlap = 0

        # Compute autogradient
        f_x = 2 * np.pi * self.eme.wavelength * (np.conj(overlap))
        power = np.abs(overlap) ** 2

        return f_x, power

    def set_adjoint_sources(self, f_x: complex = 0 + 0j):
        """Computes and places the adjoint sources for use in the adjoint formulation"""
        return [Source(z=2.75, mode_coeffs=[f_x], k=-1)]

    def adjoint_run(self, sources: list):
        """Performs the adjoint run for use in the adjoint formulation"""

        a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor = self.adjoint_results
        a_field *= sources[0].mode_coeffs[0]
        adjoint_monitor.field[:-1] *= sources[0].mode_coeffs[0]
        return a_grid_x, a_grid_y, a_grid_z, a_field, adjoint_monitor

    def optimize(self, design: list, dp=1e-10) -> "np.ndarray":
        """Runs a single step of shape optimization"""

        if self.eme.am_master():
            print("Now performing an optimization...")

        # Update the design region
        design = design if not isinstance(design, np.ndarray) else design.tolist()
        self.set_design(design)

        # Compute the forward run
        grid_x, grid_y, grid_z, X, monitor_forward, monitor_fom = self.forward_run()

        if self.eme.am_master():
            plt.figure()
            monitor_forward.visualize(axes="xz")
            plt.savefig("forward")

        # Compute the partial gradient of the objective function f_x
        f_x, fom = self.objective_gradient(monitor_fom)

        # Calculate the adjoint sources
        sources = self.set_adjoint_sources(f_x)

        # Compute the adjoint run
        grid_x, grid_y, grid_z, lamdagger, monitor_adjoint = self.adjoint_run(sources)

        if self.eme.am_master():
            plt.figure()
            monitor_adjoint.visualize(axes="xz")
            plt.savefig("adjoint")

        # Compute the gradient of the constraint A_u
        A_u = self.gradient(
            grid_x, grid_y, grid_z, dp=dp
        )  # A will need to be 3x3xthe rest to incorporate the right dimensions and cross dielectric etc.

        # Calculate the full gradient of the objective function f_u
        f_u = self.compute_final_gradient(lamdagger, A_u, X)

        plt.figure()
        self.plot_gradients(f_u, monitor_adjoint)
        if self.eme.am_master():
            plt.savefig("gradients")

        # Return the gradient
        return fom, f_u, monitor_forward

    def compute_final_gradient(self, lamdagger: "np.ndarray", A_u: "np.ndarray", X: "np.ndarray"):
        """Computes the final gradient using the adjoint formulation and loops to conserve memory"""

        # Initialize final result
        f_u = np.zeros(A_u.shape[-1], dtype=float)

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
                f_u[p] += -2 * np.real(np.sum(A_u_x[i] * lamdagger[..., i].T))

        return f_u

    def draw(self) -> None:
        self.start()
        self.eme.draw()

    def plot_gradients(self, gradients, monitor) -> None:

        # Get the gradients
        design_x, design_z = self.get_design_readable()
        z_start, z_end = self.get_design_region()
        vertices_gradients = np.array([[z, x] for z, x in zip(gradients[1::2], gradients[::2])])
        vertices_origins = np.array([[z + z_start, x] for z, x in zip(design_z, design_x)]).T

        # Get n
        grid_x, grid_z, n_original = monitor.get_array(axes="xz", component="n")
        n_grid_z = np.array([i for i in grid_z if i >= z_start and i <= z_end])
        n = self.get_n(grid_x, None, n_grid_z)

        # Plot gradients
        plt.imshow(np.real(n_original[::-1]), extent=[grid_z[0], grid_z[-1], grid_x[0], grid_x[-1]], cmap="Greys")
        plt.imshow(np.real(n[::-1]), extent=[z_start, z_end, grid_x[0], grid_x[-1]], cmap="Greys")
        plt.quiver(*vertices_origins, vertices_gradients[:, 0], vertices_gradients[:, 1], color="r")

    def calculate_fd_gradient(
        self,
        num_gradients: int = 1,
        dp: float = 1e-4,
        rand: "np.random.RandomState" = None,
        idx: list = None,
        design: list = None,
    ):
        """
        Estimate central difference gradients.

        Parameters
        ----------
        num_gradients : int
            number of gradients to estimate. Randomly sampled from parameters.
        du : float
            finite difference step size

        Returns
        -----------
        fd_gradient : lists
            [number of objective functions][number of gradients]

        """

        # Get the design
        design = self.get_design() if design is None else design
        if num_gradients > len(design):
            raise ValueError(
                "The requested number of gradients must be less than or equal to the total number of design parameters."
            )

        # cleanup
        self.start()

        # preallocate result vector
        fd_gradient = []

        # randomly choose indices to loop estimate
        if idx is None:
            fd_gradient_idx = np.random.choice(
                len(design) // 2,
                num_gradients,
                replace=False,
            ) if rand is None else rand.choice(
                len(design) // 2,
                num_gradients,
                replace=False,
            )
        else:
            fd_gradient_idx = idx[:]

        # loop over indices
        for k in fd_gradient_idx:

            # get current design region
            b0 = np.ones(len(design))
            b0[:] = design[:]

            # assign new design vector
            b0[k * 2] -= dp
            self.set_design(b0)
            self.start()

            # FOM monitor
            source = Source(z=0.25, mode_coeffs=[1], k=1)  # Hard coded
            fom_monitor = self.eme.add_monitor(axes="xy", location=2.75, sources=[source])

            # Adjoint source and monitor
            self.eme.propagate()

            # record final objective function value
            _, fm = self.objective_gradient(fom_monitor)

            # assign new design vector
            b0[k * 2] += 2 * dp
            self.set_design(b0)
            self.start()

            # propagate
            source = Source(z=0.25, mode_coeffs=[1], k=1)  # Hard coded
            fom_monitor = self.eme.add_monitor(axes="xy", location=2.75, sources=[source])
            self.eme.propagate()

            # record final objective function value
            _, fp = self.objective_gradient(fom_monitor)

            # revert design
            b0[k * 2] -= dp
            self.set_design(b0)

            # derivative
            # if self.eme.am_master():
            #     print("end",fp, fm, dp, k, b0[k*2])
            fd_gradient.append((fp - fm) / (2 * dp))

        return fd_gradient, fd_gradient_idx, (fp + fm) / 2, fom_monitor
