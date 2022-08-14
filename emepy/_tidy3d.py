import numpy as np
from matplotlib import pyplot as plt

from emepy.fd import ModeSolver
from emepy.mode import Mode, Mode1D, EigenMode
from emepy.tools import interp, interp1d, Si, SiO2, get_epsfunc, rectangle_to_n
from emepy.materials import Si, SiO2

from tidy3d.plugins import ModeSolver as tidy3d_ModeSolver
from tidy3d import PolySlab
import tidy3d as td


class MSTidy3D(ModeSolver):
    def __init__(
        self,
        wavelength=1.55,
        xz_vertices=None,
        thickness=0.22,
        z_loc=None,
        num_modes=1,
        n_core=None,
        n_cladding=None,
        cladding_width=3.0,
        cladding_thickness=3.0,
        mesh=128,
    ):

        # Fix params
        self.wavelength = wavelength
        self.xz_vertices = (
            xz_vertices if xz_vertices is not None else [(0, 0), (0, 1), (1, 1), (1, 0)]
        )
        self.thickness = thickness
        self.z_loc = (
            z_loc
            if z_loc is not None
            else 0.5 * (self.xz_vertices[0][1] + self.xz_vertices[-1][1])
        )
        self.n_core = n_core if n_core is not None else Si(self.wavelength)
        self.n_cladding = (
            n_cladding if n_cladding is not None else SiO2(self.wavelength)
        )
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness

        # Other params
        max_z = max([v[1] for v in self.xz_vertices])
        min_z = min([v[1] for v in self.xz_vertices])
        max_x = max([v[0] for v in self.xz_vertices])
        min_x = min([v[0] for v in self.xz_vertices])
        self.delta_z = max_z - min_z
        self.delta_x = max_x - min_x
        self.mesh = mesh

    def solve(self):

        # Set up geometry
        waveguide = td.Structure(
            geometry=PolySlab(
                vertices=self.xz_vertices,
                axis=2,
                slab_bounds=(-self.thickness / 2, self.thickness / 2),
            ),
            medium=td.Medium(permittivity=self.n_core**2),
        )

        # Set up simulation
        Lx, Ly, Lz = self.cladding_width, self.cladding_thickness, self.delta_z + 2.0

        grid_x = td.UniformGrid(dl=Lx / (self.mesh - 3))
        grid_y = td.UniformGrid(dl=Ly / (self.mesh - 3))
        grid_z = td.UniformGrid(dl=Lz / (self.mesh - 3))
        grid_spec = td.GridSpec(
            grid_x=grid_x, grid_y=grid_z, grid_z=grid_y, wavelength=self.wavelength
        )  # Note the difference in convention for yz

        sim = td.Simulation(
            size=(Lx, Lz, Ly),
            grid_spec=grid_spec,
            structures=[waveguide],
            run_time=1e-12,
            medium=td.Medium(permittivity=self.n_cladding**2),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=20),
                y=td.Boundary.pml(num_layers=20),
                z=td.Boundary.pml(num_layers=20),
            ),
        )

        # Define mode solver
        mode_spec = td.ModeSpec(num_modes=self.num_modes, target_neff=2.0)

        plane = td.Box(center=(0, self.z_loc, 0), size=(Lx, 0, Ly))

        mode_solver = tidy3d_ModeSolver(
            simulation=sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[td.constants.C_0 / self.wavelength],
        )

        # Run simulation
        mode_data = mode_solver.solve()
        index = np.zeros((self.mesh, self.mesh), dtype=complex)
        index[:, :] = mode_solver._solver_eps(td.constants.C_0 / self.wavelength)[
            0, :, :
        ]

        # Save modes
        self.modes = []
        for m in range(self.num_modes):
            self.modes.append(
                Mode(
                    x=mode_data.Ex.x.to_numpy(),
                    y=mode_data.Ex.z.to_numpy(),
                    Ex=mode_data.Ex.to_numpy()[:, 0, :, 0, m],
                    Ey=mode_data.Ez.to_numpy()[:, 0, :, 0, m],
                    Ez=mode_data.Ey.to_numpy()[:, 0, :, 0, m],
                    Hx=mode_data.Hx.to_numpy()[:, 0, :, 0, m],
                    Hy=mode_data.Hz.to_numpy()[:, 0, :, 0, m],
                    Hz=mode_data.Hy.to_numpy()[:, 0, :, 0, m],
                    n=index,
                )
            )

    def get_mode(self, index):
        return self.modes[index]


if __name__ == "__main__":

    # Define polygon
    xz_vertices = [
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.75, 0.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.75, 0.0),
    ]
    thickness = 0.22
    wavelength = 1.55
    z_loc = 0.4
    n_core = Si(wavelength)
    n_cladding = SiO2(wavelength)
    num_modes = 3

    # Define solver
    solver = MSTidy3D(
        wavelength, xz_vertices, thickness, z_loc, num_modes, n_core, n_cladding
    )

    # Solve
    solver.solve()

    # Get modes
    mode = solver.get_mode(0)
    mode.plot_material()
    plt.show()
