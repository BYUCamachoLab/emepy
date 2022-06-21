"""
    This file interfaces with the Lumerical API for modesolving
"""

from emepy.eme import EME
from emepy.fd import ModeSolver, ModeSolver1D
import importlib

if importlib.util.find_spec("lumapi") is not None:
    import lumapi as lm
import numpy as np
import os

from emepy.mode import Mode, Mode1D
from emepy import tools


class LumEME(EME):
    """
    This class is a wrapper for EME, it performs the same operations but uses Lumerical MODE to solve for the modes at the interfaces
    """

    def __init__(self, layers=[], num_periods=1):
        super().__init__(layers, num_periods)

        # open file
        if os.path.isfile("api.lms"):
            os.remove("api.lms")
        self.mode = lm.MODE(hide=True)
        self.mode.save("api.lms")

    def close(self):
        if os.path.isfile("api.lms"):
            os.remove("api.lms")

    def __del__(self):
        if os.path.isfile("api.lms"):
            os.remove("api.lms")

        if os.path.isfile("api_p0.log"):
            os.remove("api_p0.log")


class MSLumerical(ModeSolver):
    """Outdated Lumerical Modesolver. Uses the lumapi Lumerical API. See Modesolver. Parameterizes the cross section as a rectangular waveguide."""

    def __init__(
        self,
        wl=1.55e-6,
        width=0.5e-6,
        thickness=0.22e-6,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        core_index=None,
        cladding_index=None,
        mesh=300,
        mode=None,
        eme_modes=False,
        polygons=[],
        PML=False,
        **kwargs
    ):
        """MSLumerical class constructor

        Parameters
        ----------
        wl : number
            wavelength of the eigenmodes
        width : number
            width of the core in the cross section
        thickness : number
            thickness of the core in the cross section
        num_modes : int
            number of modes to solve for (default:1)
        cladding_width : number
            width of the cladding in the cross section (default:5e-6)
        cladding_thickness : number
            thickness of the cladding in the cross section (default:5e-6)
        core_index : number
            refractive index of the core (default:Si)
        cladding_index : number
            refractive index of the cladding (default:SiO2)
        mesh : int
            number of mesh points in each direction (xy)
        mode : lumapi.MODE
            MODE object that contains the file information
        eme_modes : boolean
            if true, will utilize the lumerical eme wrapped fde solver which is not normalized to one. Produces slightly different results purely due to roundoff error during normalization.
        PML : boolean
            if true, will enable PML boundary conditions, note: this will increase the mesh and grid space
        """

        # Create global params
        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.mesh = mesh
        self.mode = mode
        self.eme_modes = eme_modes
        self.PML = PML
        self.polygons = polygons

        # Check indexes
        self.core_index = tools.Si(wl * 1e6) if core_index is None else core_index
        self.cladding_index = tools.SiO2(wl * 1e6) if cladding_index is None else cladding_index
        self.close_after = False

        # Setup PML
        if self.PML:
            self.setup_PML()
        else:
            self.setup_no_PML()

        # Setup file
        if self.mode is None:
            self.setup_file()

    def setup_PML(self):
        self.num_pml_layers = int(self.mesh / 8.0)
        self.after_x = np.linspace(
            -0.5 * self.cladding_width * (1 + 1 / 4),
            0.5 * self.cladding_width * (1 + 1 / 4),
            self.mesh + 2 * self.num_pml_layers - 1,
        )
        self.after_y = np.linspace(
            -0.5 * self.cladding_width * (1 + 1 / 4),
            0.5 * self.cladding_width * (1 + 1 / 4),
            self.mesh + 2 * self.num_pml_layers - 1,
        )
        self.x = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)
        self.y = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)

    def setup_file(self):
        self.close_after = True

        # open file
        if os.path.isfile("api.lms"):
            os.remove("api.lms")

        self.mode = lm.MODE(hide=True)
        self.mode.save("api.lms")

    def setup_no_PML(self):
        self.after_x = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)
        self.after_y = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)
        self.x = self.after_x
        self.y = self.after_y

    def setup_cladding(self):
        cladding = self.mode.addrect()
        cladding.name = "cladding"
        cladding.x = 0
        cladding.x_span = 10.0e-6
        cladding.y = 0
        cladding.y_span = self.cladding_width * 2.0
        cladding.z = 0
        cladding.z_span = self.cladding_thickness * 2.0
        cladding.index = self.cladding_index
        return cladding

    def setup_waveguide_core(self):
        core = self.mode.addrect()
        core.name = "core"
        core.x = 0
        core.x_span = 10.0e-6
        core.y = 0
        core.y_span = self.width
        core.z = 0
        core.z_span = self.thickness
        core.index = self.core_index
        return core

    def setup_polygons_core(self):
        for i, p in enumerate(self.polygons):
            core = self.mode.addpoly()
            self.mode.set("name", str("poly{}".format(i)))
            self.mode.set("vertices", p[0])
            self.mode.set("z span", self.thickness)
            self.mode.set("first axis", "y")
            self.mode.set("rotation 1", 90)
            self.mode.set("y", p[1])
            self.mode.set("z", p[2])
            self.mode.set("index", self.core_index)
        return core

    def setup_eme(self):
        eme = self.mode.addeme()
        self.mode.set("wavelength", self.wl)
        self.mode.set("mesh cells y", self.mesh)
        self.mode.set("mesh cells z", self.mesh)
        self.mode.set("x min", -1e-6)
        self.mode.set("y", 0)
        self.mode.set("y span", self.cladding_width)
        self.mode.set("z", 0)
        self.mode.set("z span", self.cladding_thickness)
        self.mode.set("allow custom eigensolver settings", 1)
        self.mode.set("cells", 1)
        self.mode.set("group spans", 2e-6)
        self.mode.set("modes", self.num_modes)
        if self.PML:
            self.mode.set("y min bc", "PML")
            self.mode.set("y max bc", "PML")
            self.mode.set("z min bc", "PML")
            self.mode.set("z max bc", "PML")
            self.mode.set("pml layers", self.num_pml_layers)
        return eme

    def get_eme_results(self):
        results = self.mode.getresult("EME::Cells::cell_1", "mode fields")
        neffs = []
        gridx, gridy = [results["y"], results["z"]]
        gridx, gridy = [gridx.reshape(gridx.shape[0]), gridy.reshape(gridy.shape[0])]
        mesh = len(gridx)
        fields = []
        for mode_num in range(1, self.num_modes + 1):
            neffs.append(self.mode.getresult("EME::Cells::cell_1", "neff")["neff"].flatten()[mode_num - 1])
            E = results["E" + str(mode_num)].reshape(mesh, mesh, 3)
            H = results["H" + str(mode_num)].reshape(mesh, mesh, 3)
            Ex, Ey, Ez, Hx, Hy, Hz = [E[:, :, 1], E[:, :, 2], E[:, :, 0], H[:, :, 1], H[:, :, 2], H[:, :, 0]]
            fields.append([Hx, Hy, Hz, Ex, Ey, Ez])
        n = results["index"].reshape(mesh, mesh, 3)[:, :, 0]
        return fields, neffs, gridx, gridy, mesh, n

    def setup_fde(self):
        fde = self.mode.addfde()
        fde.y = 0
        fde.y_span = self.cladding_width  # clad_width / 2
        fde.solver_type = "2D X normal"
        fde.x = 0
        fde.z = 0
        fde.z_span = self.cladding_thickness  # clad_thickness / 2
        fde.mesh_cells_y = self.mesh - 1
        fde.mesh_cells_z = self.mesh - 1
        self.mode.set("number of trial modes", self.num_modes)
        self.mode.set("wavelength", self.wl)
        return fde

    def get_fde_results(self):
        self.mode.findmodes()
        gridx, gridy = [self.mode.getresult("FDE::data::mode1", "y"), self.mode.getresult("FDE::data::mode1", "z")]
        gridx, gridy = [gridx.reshape(gridx.shape[0]), gridy.reshape(gridy.shape[0])]
        fields, neffs = [[], []]
        m = self.mesh
        for mode_num in range(1, self.num_modes + 1):
            mode_field = []
            neffs.append(self.mode.getdata("FDE::data::mode" + str(mode_num), "neff")[0][0])
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hy").reshape(m, m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hz").reshape(m, m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hx").reshape(m, m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ey").reshape(m, m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ez").reshape(m, m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ex").reshape(m, m))
            fields.append(mode_field)
        n = self.mode.getdata("FDE::data::material", "index_x").reshape(m, m)
        return fields, neffs, gridx, gridy, n

    def solve(self):
        """Solves for the eigenmodes"""

        # Restart everything in case this isn't the first go
        self.mode.switchtolayout()
        self.mode.deleteall()

        # Create cladding
        self.cladding = self.setup_cladding()

        # add core x=prop y=width z=thickness
        self.core = self.setup_waveguide_core() if not len(self.polygons) else self.setup_polygons_core()

        # Solve using EME modesolver
        if self.eme_modes or self.PML:

            # set up EME for FDE extraction
            self.eme = self.setup_eme()

            # run
            self.mode.run()
            self.mode.emepropagate()

            # get modes
            fields, neffs, gridx, gridy, mesh, n = self.get_eme_results()
            self.mesh = mesh

        # Solve using FDE modesolver
        else:

            # set up FDE
            self.fde = self.setup_fde()

            # get fde results
            fields, neffs, gridx, gridy, n = self.get_fde_results()

        # Set field, grid, and effective index
        self.x, self.y = [gridx, gridy][:]
        self.neffs, self.fields = [neffs, fields][:]
        self.n = n

        # Check for gui close
        if self.close_after and os.path.isfile("api.lms"):
            os.remove("api.lms")

    def clear(self):
        """Clears the modesolver's eigenmodes to make memory"""

        self.x, self.y, self.neffs, self.fields = [None] * 4

    def get_mode(self, mode_num=0):
        """Get the indexed mode number

        Parameters
        ----------
        mode_num : int
            index of the mode of choice

        Returns
        -------
        Mode
            the eigenmode of index mode_num
        """

        field = self.fields[mode_num]
        Hx, Hy, Hz, Ex, Ey, Ez = field[0:6]
        neff = self.neffs[mode_num]
        mode = Mode(x=self.x, y=self.y, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez, n=self.n)

        return mode


class MSLumerical1D(MSLumerical, ModeSolver1D):
    """
    Exact same as MSLumerical except solves on a 1D grid.
    Create geometries in 2D as before, but only the middle y value will be used.
    This declaration will only overload the methods that require changing code for
    """

    def setup_eme(self):
        eme = self.mode.addeme()
        self.mode.set("solver type", "2D XY plane: X prop")
        self.mode.set("wavelength", self.wl)
        self.mode.set("mesh cells y", self.mesh)
        self.mode.set("x min", -1e-6)
        self.mode.set("y", 0)
        self.mode.set("y span", self.cladding_width)
        self.mode.set("z", 0)
        self.mode.set("allow custom eigensolver settings", 1)
        self.mode.set("cells", 1)
        self.mode.set("group spans", 2e-6)
        self.mode.set("modes", self.num_modes)
        if self.PML:
            self.mode.set("y min bc", "PML")
            self.mode.set("y max bc", "PML")
            self.mode.set("pml layers", self.num_pml_layers)
        return eme

    def get_eme_results(self):
        results = self.mode.getresult("EME::Cells::cell_1", "mode fields")
        neffs = []
        gridx = results["y"]
        gridx, gridy = [gridx.reshape(gridx.shape[0]), np.zeros(1)]
        mesh = len(gridx)
        fields = []
        for mode_num in range(1, self.num_modes + 1):
            neffs.append(self.mode.getresult("EME::Cells::cell_1", "neff")["neff"].flatten()[mode_num - 1])
            E = results["E" + str(mode_num)].reshape(mesh, 3)
            H = results["H" + str(mode_num)].reshape(mesh, 3)
            Ex, Ey, Ez, Hx, Hy, Hz = [E[:, 1], E[:, 2], E[:, 0], H[:, 1], H[:, 2], H[:, 0]]
            fields.append([Hx, Hy, Hz, Ex, Ey, Ez])
        n = results["index"].reshape(mesh, 3)[:, 0]
        return fields, neffs, gridx, gridy, mesh, n

    def setup_fde(self):
        fde = self.mode.addfde()
        fde.y = 0
        fde.y_span = self.cladding_width  # clad_width / 2
        fde.solver_type = "1D Y:X prop"
        fde.x = 0
        fde.z = 0
        fde.mesh_cells_y = self.mesh - 1
        self.mode.set("number of trial modes", self.num_modes)
        self.mode.set("wavelength", self.wl)
        return fde

    def get_fde_results(self):
        self.mode.findmodes()
        gridx = self.mode.getresult("FDE::data::mode1", "y")
        gridx = gridx.reshape(gridx.shape[0])
        gridy = np.zeros(1)
        fields, neffs = [[], []]
        m = self.mesh
        for mode_num in range(1, self.num_modes + 1):
            mode_field = []
            neffs.append(self.mode.getdata("FDE::data::mode" + str(mode_num), "neff")[0][0])
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hy").reshape(m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hz").reshape(m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Hx").reshape(m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ey").reshape(m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ez").reshape(m))
            mode_field.append(self.mode.getdata("FDE::data::mode{}".format(mode_num), "Ex").reshape(m))
            fields.append(mode_field)
        n = self.mode.getdata("FDE::data::material", "index_x").reshape(m)
        return fields, neffs, gridx, gridy, n

    def get_mode(self, mode_num=0):
        field = self.fields[mode_num]
        Hx, Hy, Hz, Ex, Ey, Ez = field[0:6]
        neff = self.neffs[mode_num]
        mode = Mode1D(x=self.x, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez, n=self.n)
        return mode

    def setup_PML(self):
        self.num_pml_layers = int(self.mesh / 8.0)
        self.after_x = np.linspace(
            -0.5 * self.cladding_width * (1 + 1 / 4),
            0.5 * self.cladding_width * (1 + 1 / 4),
            self.mesh + 2 * self.num_pml_layers - 1,
        )
        self.after_y = np.zeros(1)
        self.x = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)
        self.y = np.zeros(1)

    def setup_no_PML(self):
        self.after_x = np.linspace(-0.5 * self.cladding_width, 0.5 * self.cladding_width, self.mesh)
        self.after_y = np.zeros(1)
        self.x = self.after_x
        self.y = self.after_y
