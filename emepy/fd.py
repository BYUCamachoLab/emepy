import numpy as np

from emepy.mode import Mode
from emepy import tools
import EMpy_gpu
from EMpy_gpu.modesolvers.FD import stretchmesh
from emepy.tools import interp
import pickle


class ModeSolver(object):
    """The ModeSolver object is the heart of finding eigenmodes for use in eigenmode expansion or simple examination. This parent class should be inherited and used as a wrapper for certain modules such as EMpy, Lumerical, Pickled data, Neural Networks, etc."""

    def __init__(self, **kwargs):
        """ModeSolver class constructor"""
        raise NotImplementedError

    def solve(self):
        """Solves the eigenmode solver for the specific eigenmodes of desire"""
        raise NotImplementedError

    def clear(self):
        """Clears the modesolver's eigenmodes to make memory"""
        raise NotImplementedError

    def get_mode(self, mode_num):
        """Must extract the mode of choice

        Parameters
        ----------
        mode_num : int
            index of the mode of choice
        """
        raise NotImplementedError


class MSEMpy(ModeSolver):
    """Electromagnetic Python Modesolver. Uses the EMpy library See Modesolver. Parameterizes the cross section as a rectangular waveguide."""

    def __init__(
        self,
        wl,
        width=None,
        thickness=None,
        num_modes=1,
        cladding_width=2.5e-6,
        cladding_thickness=2.5e-6,
        core_index=None,
        cladding_index=None,
        x=None,
        y=None,
        mesh=128,
        accuracy=1e-8,
        boundary="0000",
        epsfunc=None,
        n=None,
        PML=False,
        **kwargs
    ):
        """MSEMpy class constructor

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
        x : numpy array
            the cross section grid in the x direction (z propagation) (default:None)
        y : numpy array
            the cross section grid in the y direction (z propagation) (default:None)
        mesh : int
            the number of mesh points in each xy direction
        accuracy : number
            the minimum accuracy of the finite difference solution (default:1e-8)
        boundary : string
            the boundaries according to the EMpy library (default:"0000")
        epsfunc : function
            the function which defines the permittivity based on a grid (see EMpy library) (default:"0000")
        n : numpy array
            2D profile of the refractive index
        PML : boolean
            if True, will use PML boundaries. Default : False, PEC
        """

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.core_index = core_index
        self.cladding_index = cladding_index
        self.x = x
        self.y = y
        self.mesh = mesh
        self.accuracy = accuracy
        self.boundary = boundary
        self.epsfunc = epsfunc
        self.n = n
        self.PML = PML

        if core_index is None:
            self.core_index = tools.Si(wl * 1e6)
        if cladding_index is None:
            self.cladding_index = tools.SiO2(wl * 1e6)
        if x is None:
            self.x = np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
        if y is None:
            self.y = np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
        if self.PML:  # Create a PML at least half a wavelength long
            dx = np.diff(self.x)
            dy = np.diff(self.y)
            layer_xp = int(np.abs(0.5 * self.wl / dx[-1]))
            layer_xn = int(np.abs(0.5 * self.wl / dx[0]))
            layer_yp = int(np.abs(0.5 * self.wl / dy[-1]))
            layer_yn = int(np.abs(0.5 * self.wl / dy[0]))
            self.nlayers = [layer_yp, layer_yn, layer_xp, layer_xn]
            factor = 1 + 2j
            self.x, self.y, _, _, _, _ = stretchmesh(self.x, self.y, self.nlayers, factor)
        if epsfunc is None:
            self.epsfunc = tools.get_epsfunc(
                self.width,
                self.thickness,
                self.cladding_width,
                self.cladding_thickness,
                self.core_index,
                self.cladding_index,
                profile=self.n,
                nx=self.x,
                ny=self.y,
            )

        self.after_x = self.x
        self.after_y = self.y
        self.n = self.epsfunc(self.x, self.y)

    def solve(self):
        """Solves for the eigenmodes"""
        self.solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(self.wl, self.x, self.y, self.epsfunc, self.boundary).solve(
            self.num_modes, self.accuracy
        )
        return self

    def clear(self):
        """Clears the modesolver's eigenmodes to make memory"""

        self.solver = None
        return self

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

        x = self.solver.modes[mode_num].get_x()
        y = self.solver.modes[mode_num].get_y()
        x0, y0 = [np.real(x), np.real(y)]
        diffx, diffy = [np.diff(x0), np.diff(y0)]
        x0_new, y0_new = [np.ones(len(x) + 1), np.ones(len(y) + 1)]
        x0_new[0:-1], y0_new[0:-1] = [x0, y0]
        x0_new[-1], y0_new[-1] = [x0[-1] + diffx[-1], y0[-1] + diffy[-1]]

        if not self.PML:
            self.nlayers = [1, 0, 1, 0]
        Ex = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ex"), True)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        Ey = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ey"), True)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        Ez = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ez"), True)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        Hx = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hx"), False)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        Hy = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hy"), False)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        Hz = interp(x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hz"), False)[
            self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
        ]
        self.x = x0_new[self.nlayers[1] : -self.nlayers[0]]
        self.y = y0_new[self.nlayers[3] : -self.nlayers[2]]

        neff = self.solver.modes[mode_num].neff
        n = self.epsfunc(self.x, self.y)

        return Mode(
            x=self.x,
            y=self.y,
            wl=self.wl,
            neff=neff,
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            width=self.width,
            thickness=self.thickness,
            n=n,
        )


class MSPickle(object):
    """Pickle Modesolver. See Modesolver. Pickle should serialize a list of Mode objects that can be opened here."""

    def __init__(self, filename, index=None, width=None, thickness=None, **kwargs):
        """MSPickle class constructor

        Parameters
        ----------
        filename : string
            the name of the pickled file where the eigenmode is stored
        index : int
            the index of the mode in the pickle file if the data stored is an array of Modes (default:None)
        width : number
            width of the core in the cross section, used for drawing (default:None)
        thickness : number
            thickness of the core in the cross section, used for drawing  (default:None)
        """

        self.filename = filename
        self.index = index
        self.width = width
        self.thickness = thickness
        self.PML = False

    def solve(self):
        """Solves for the eigenmodes by loading them from the pickle file"""

        with open(self.filename, "rb") as f:
            self.mode = pickle.load(f)[self.index] if self.index else pickle.load(f)

        self.x = self.mode.x
        self.y = self.mode.y
        return self

    def clear(self):
        """Clears the modesolver's eigenmodes to make memory"""

        self.x = None
        self.y = None
        self.mode = None
        return self

    def get_mode(self, mode_num=0):
        """Get the stored mode

        Returns
        -------
        Mode
            the eigenmode of index mode_num
        """

        return self.mode
