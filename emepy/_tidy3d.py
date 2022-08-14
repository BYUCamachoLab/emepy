from dataclasses import asdict
import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Sequence

from emepy.fd import ModeSolver
from emepy.mode import Mode, Mode1D, EigenMode
from emepy.tools import interp, interp1d, get_epsfunc, rectangle_to_n
from emepy.materials import Si, SiO2
from emepy.geometries import DynamicPolygon
from emepy.eme import Layer

from tidy3d.plugins import ModeSolver as tidy3d_ModeSolver
from tidy3d import PolySlab
import tidy3d as td


class _MSTidy3D(ModeSolver):
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
        x=None,
        y=None,
        target_neff=3.0,
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
        self.target_neff = target_neff

        # Other params
        self.max_z = max([v[1] for v in self.xz_vertices])
        self.min_z = min([v[1] for v in self.xz_vertices])
        self.max_x = max([v[0] for v in self.xz_vertices])
        self.min_x = min([v[0] for v in self.xz_vertices])
        self.delta_z = self.max_z - self.min_z
        self.delta_x = self.max_x - self.min_x
        self.mesh = mesh

        # Get x and y and z
        Lx, Ly, Lz = self.cladding_width, self.cladding_thickness, self.delta_z + 2.0
        self.x = x if x is not None else np.linspace(-Lx / 2, Lx / 2, mesh)
        self.y = y if y is not None else np.linspace(-Ly / 2, Ly / 2, mesh)
        self.z = np.linspace(-Lz / 2, Lz / 2, mesh)
        self.after_x = self.x
        self.after_y = self.y
        self._old_x = np.linspace(-Lx / 2, Lx / 2, mesh + 2)
        self._old_y = np.linspace(-Ly / 2, Ly / 2, mesh + 2)

        self.setup()
        self.n = self.get_any_n(self.x, self.y, self.z_loc)

    def setup(self):
        # Set up geometry
        waveguide = td.Structure(
            geometry=PolySlab(
                vertices=[(z, x) for x, z, in self.xz_vertices],
                axis=2,
                slab_bounds=(-self.thickness / 2, self.thickness / 2),
            ),
            medium=td.Medium(permittivity=self.n_core**2),
        )

        # Set up simulation
        Lx, Ly, Lz = self.delta_z + 2.0, self.cladding_width, self.cladding_thickness

        grid_x = td.CustomGrid(dl=np.diff(self.z).tolist())
        grid_y = td.CustomGrid(dl=np.diff(self.x).tolist())
        grid_z = td.CustomGrid(dl=np.diff(self.y).tolist())
        grid_spec = td.GridSpec(
            grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, wavelength=self.wavelength
        )  # Note the difference in convention for

        self.sim = td.Simulation(
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
            subpixel=True,
        )

        # Define mode solver
        mode_spec = td.ModeSpec(num_modes=self.num_modes, target_neff=self.target_neff)

        plane = td.Box(center=(self.z_loc, 0, 0), size=(0, Ly, Lz))

        self.mode_solver = tidy3d_ModeSolver(
            simulation=self.sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[td.constants.C_0 / self.wavelength],
        )

    def solve(self):

        # Run simulation
        mode_data = self.mode_solver.solve()
        self._old_x = mode_data.sel_mode_index(0).field_components["Ey"].y
        self._old_y = mode_data.sel_mode_index(0).field_components["Ey"].z
        index = self.get_n()
        self.n = index

        # Save modes
        self.modes = []
        for m in range(self.num_modes):
            # Get mode
            mode = mode_data.sel_mode_index(m)

            # Place all fields onto a shared grid
            Ex = mode.field_components["Ey"]
            Ex = interp(
                self.x,
                self.y,
                Ex.y.to_numpy(),
                Ex.z.to_numpy(),
                Ex.to_numpy()[0, :, :, 0],
                centered=False,
            )

            Ey = mode.field_components["Ez"]
            Ey = interp(
                self.x,
                self.y,
                Ey.y.to_numpy(),
                Ey.z.to_numpy(),
                Ey.to_numpy()[0, :, :, 0],
                centered=False,
            )

            Ez = mode.field_components["Ex"]
            Ez = interp(
                self.x,
                self.y,
                Ez.y.to_numpy(),
                Ez.z.to_numpy(),
                Ez.to_numpy()[0, :, :, 0],
                centered=False,
            )

            Hx = mode.field_components["Hy"]
            Hx = interp(
                self.x,
                self.y,
                Hx.y.to_numpy(),
                Hx.z.to_numpy(),
                Hx.to_numpy()[0, :, :, 0],
                centered=False,
            )

            Hy = mode.field_components["Hz"]
            Hy = interp(
                self.x,
                self.y,
                Hy.y.to_numpy(),
                Hy.z.to_numpy(),
                Hy.to_numpy()[0, :, :, 0],
                centered=False,
            )

            Hz = mode.field_components["Hx"]
            Hz = interp(
                self.x,
                self.y,
                Hz.y.to_numpy(),
                Hz.z.to_numpy(),
                Hz.to_numpy()[0, :, :, 0],
                centered=False,
            )

            self.modes.append(
                Mode(
                    x=self.x,
                    y=self.y,
                    neff=mode_data.n_eff.to_numpy()[0][m],
                    Ex=Ex,
                    Ey=Ey,
                    Ez=Ez,
                    Hx=Hx,
                    Hy=Hy,
                    Hz=Hz,
                    n=index,
                )
            )

    def get_mode(self, index):
        return self.modes[index]

    def get_n(self):
        index = np.zeros((self.mesh + 2, self.mesh + 2), dtype=complex)
        index[:, :] = self.mode_solver._solver_eps(td.constants.C_0 / self.wavelength)[
            0, :, :
        ]
        index = interp(self.x, self.y, self._old_x, self._old_y, index, centered=False)
        return np.sqrt(index)

    def get_any_n(self, x=None, y=None, z=None):
        x_flag, y_flag, z_flag = False, False, False
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        z = z if z is not None else self.z
        x = x if isinstance(x, Sequence) or isinstance(x, np.ndarray) else np.array([x])
        y = y if isinstance(y, Sequence) or isinstance(y, np.ndarray) else np.array([y])
        z = z if isinstance(z, Sequence) or isinstance(z, np.ndarray) else np.array([z])

        if len(x.flatten()) == 1:
            x = np.array([x.item(0), x.item(0), x.item(0)])
            x_flag = True
        if len(y.flatten()) == 1:
            y = np.array([y.item(0), y.item(0), y.item(0)])
            y_flag = True
        if len(z.flatten()) == 1:
            z = np.array([z.item(0), z.item(0), z.item(0)])
            z_flag = True

        bound_x = 0.5 * (z[1:] + z[:-1])
        bound_x = np.insert(bound_x, 0, bound_x[0] - np.diff(bound_x)[0])
        bound_x = np.insert(bound_x, len(bound_x), bound_x[-1] + np.diff(bound_x)[-1])

        bound_y = 0.5 * (x[1:] + x[:-1])
        bound_y = np.insert(bound_y, 0, bound_y[0] - np.diff(bound_y)[0])
        bound_y = np.insert(bound_y, len(bound_y), bound_y[-1] + np.diff(bound_y)[-1])

        bound_z = 0.5 * (y[1:] + y[:-1])
        bound_z = np.insert(bound_z, 0, bound_z[0] - np.diff(bound_z)[0])
        bound_z = np.insert(bound_z, len(bound_z), bound_z[-1] + np.diff(bound_z)[-1])

        coords = td.Coords(x=bound_x, y=bound_y, z=bound_z)
        grid = td.Grid(boundaries=coords)
        n = self.sim.epsilon_on_grid(grid).to_numpy().transpose((1, 2, 0))

        if y_flag:
            n = n[:, 0, :]
        if x_flag:
            n = n[0, ...]
        if z_flag:
            n = n[..., 0]

        return np.sqrt(n)


class Tidy3DPoly(DynamicPolygon):
    def __init__(
        self,
        wavelength=1.55,
        xz_vertices=None,
        thickness=0.22,
        num_modes=1,
        n_core=None,
        n_cladding=None,
        cladding_width=3.0,
        cladding_thickness=3.0,
        mesh=128,
        num_layers=1,
    ):
        self.design = xz_vertices
        self.wavelength = wavelength
        self.thickness = thickness
        self.num_modes = num_modes
        self.n_core = n_core
        self.n_cladding = n_cladding
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.mesh = mesh
        self.num_layers = num_layers
        self.update()

    def update(self):
        self.mode_solver = _MSTidy3D(
            wavelength=self.wavelength,
            xz_vertices=self.design,
            thickness=self.thickness,
            z_loc=0,
            num_modes=self.num_modes,
            n_core=self.n_core,
            n_cladding=self.n_cladding,
            cladding_width=self.cladding_width,
            cladding_thickness=self.cladding_thickness,
            mesh=self.mesh,
        )

        layer_length = self.mode_solver.delta_z / self.num_layers
        z_loc = lambda i: (i + 0.5) * layer_length + self.mode_solver.min_z
        self.layers = [
            Layer(
                _MSTidy3D(
                    wavelength=self.wavelength,
                    xz_vertices=self.design,
                    thickness=self.thickness,
                    z_loc=z_loc(i),
                    num_modes=self.num_modes,
                    n_core=self.n_core,
                    n_cladding=self.n_cladding,
                    cladding_width=self.cladding_width,
                    cladding_thickness=self.cladding_thickness,
                    mesh=self.mesh,
                ),
                self.num_modes,
                self.wavelength,
                layer_length,
            )
            for i in range(self.num_layers)
        ]

    def get_design(self) -> list:
        """Returns the design region as a list of parameters. Each parameter represents a spacial location in a single direction for a single vertex of the polygon. Note in 2D this would look like [x0,z0,x1,z1,...xn,zn] and in 3D [x0,y0,z0,x1,y1,z1,...xn,yn,zn]"""
        return self.design

    def set_design(self, design) -> list:
        """Sets the design region"""
        self.design = design
        self.update()

    def get_n(self, grid_x, grid_y, grid_z) -> "np.ndarray":
        """Returns the index profile for the grid provided"""
        return self.mode_solver.get_any_n(grid_x, grid_y, grid_z)


if __name__ == "__main__":

    # Define polygon
    xz_vertices = [
        (-0.4, -0.5),
        (0.4, -0.5),
        (0.45, 0.0),
        (0.4, 0.5),
        (-0.4, 0.5),
        (-0.45, 0.0),
    ]
    thickness = 0.22
    wavelength = 1.55
    z_loc = 0.4
    n_core = Si(wavelength)
    n_cladding = SiO2(wavelength)
    num_modes = 4

    # Define solver
    polygon = Tidy3DPoly(
        wavelength,
        xz_vertices,
        thickness,
        num_modes,
        n_core,
        n_cladding,
        mesh=50,
        num_layers=1,
    )

    # xz_vertices = [
    #     (-0.4, -0.5),
    #     (0.4, -0.5),
    #     (0.445, 0.0),
    #     (0.4, 0.5),
    #     (-0.4, 0.5),
    #     (-0.445, 0.0),
    # ]
    # polygon2 = Tidy3DPoly(
    #     wavelength, xz_vertices, thickness, num_modes, n_core, n_cladding, mesh=50, num_layers=1
    # )
    # polygon.layers[0].mode_solver.solve()
    # # polygon2.layers[0].mode_solver.solve()
    # n1 = polygon.layers[0].mode_solver.n.real
    # n2 = polygon2.layers[0].mode_solver.n.real

    # plt.figure()
    # plt.imshow(np.rot90(n2), cmap="Greys")
    # plt.show()

    # from emepy.eme import EME

    # eme = EME(layers=[*polygon])
    # # eme.draw()
    # # plt.show()

    from emepy.fd import MSTidy3D

    solver = MSTidy3D(1.55, 0.5, 0.22, num_modes=5, mesh=100, PML=False)

    solver.solve()

    # solver.get_mode(0).plot_material()
    # plt.show()
    for i in range(5):
        solver.get_mode(i).plot()
        print(solver.get_mode(i).neff)
        plt.show()
