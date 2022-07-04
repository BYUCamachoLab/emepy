import numpy as np
from matplotlib import pyplot as plt
import EMpy_gpu
from EMpy_gpu.modesolvers.FD import stretchmesh
from typing import Callable

from emepy.mode import Mode, EigenMode
from emepy.tools import interp, Si, SiO2, get_epsfunc, rectangle_to_n


class ModeSolver(object):
    """The ModeSolver object is the heart of finding eigenmodes for use in eigenmode expansion or simple examination. This parent class should be inherited and used as a wrapper for certain modules such as EMpy, Lumerical, Pickled data, Neural Networks, etc."""

    def __init__(self, wl: list = [1.5], **kwargs) -> None:
        """ModeSolver class constructor"""
        raise NotImplementedError

    def solve(self) -> any:
        """Solves the eigenmode solver for the specific eigenmodes of desire"""
        raise NotImplementedError

    def clear(self) -> None:
        """Clears the modesolver's eigenmodes to make memory"""
        raise NotImplementedError

    def get_mode(self, mode_num: int) -> EigenMode:
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
        wl: list = [1.55],
        width: float = None,
        thickness: float = None,
        num_modes: int = 1,
        cladding_width: float = 2.5,
        cladding_thickness: float = 2.5,
        core_index: float = None,
        cladding_index: float = None,
        x: "np.ndarray" = None,
        y: "np.ndarray" = None,
        mesh: int = 128,
        accuracy: float = 1e-8,
        boundary: str = "0000",
        epsfunc: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"] = None,
        n: "np.ndarray" = None,
        PML: bool = False,
        subpixel: bool = True,
        center: tuple = (0, 0),
        **kwargs
    ) -> None:
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
            width of the cladding in the cross section (default:5)
        cladding_thickness : number
            thickness of the cladding in the cross section (default:5)
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
        PML : bool
            if True, will use PML boundaries. Default : False, PEC
        subpixel : bool
            if true, will use subpixel smoothing, assuming asking for a waveguide cross section and not providing an index map (recommended)
        """

        # Get wl array
        self.wl = wl

        # Get parameterized arguments
        self.width = width
        self.thickness = thickness

        # Number of modes
        self.num_modes = num_modes

        # Solver params
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.core_index = core_index if core_index is not None else Si(wl)
        self.cladding_index = cladding_index if cladding_index is not None else SiO2(wl)
        self.accuracy = accuracy
        self.boundary = boundary
        self.PML = PML # This is in development

        # Optionally can use a function for the permittivity
        self.epsfunc = epsfunc

        # Geometric params
        self.mesh = mesh
        self.x = x if x is not None else np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
        self.y = y if y is not None else np.linspace(-0.5 * cladding_thickness, 0.5 * cladding_thickness, mesh)

        # Optionally can set n directly
        self.n = n

        # PML Initialization
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

        # Parse into a useable epsfunction
        if epsfunc is None and (not subpixel) or (self.width is None):
            self.epsfunc = get_epsfunc(
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
        elif epsfunc is None and subpixel and (self.width is not None):
            n = rectangle_to_n(
                center, self.width, self.thickness, self.x, self.y, subpixel, self.core_index, self.cladding_index
            )
            self.x = ((self.x)[1:] + (self.x)[:-1]) / 2
            self.y = ((self.y)[1:] + (self.y)[:-1]) / 2
            self.epsfunc = get_epsfunc(
                None,
                None,
                self.cladding_width,
                self.cladding_thickness,
                self.core_index,
                self.cladding_index,
                profile=n,
                nx=self.x,
                ny=self.y,
            )

        # These params are used for drawing if accessed before the modesolving
        self.after_x = self.x
        self.after_y = self.y
        self.n = np.sqrt(self.epsfunc(self.x, self.y))

    def solve(self) -> ModeSolver:
        """Solves for the eigenmodes"""

        # Iterate through all frequencies for now
        self.modes = dict(zip(self.wl, [None] * self.num_modes))
        for wl in self.wl:
            # Solve for the modes
            solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(wl, self.x, self.y, self.epsfunc, self.boundary).solve(
                self.num_modes, self.accuracy
            )

            # Save the modes
            for mode_num in range(self.num_modes):
                x = solver.modes[mode_num].get_x()
                y = solver.modes[mode_num].get_y()
                x0, y0 = [np.real(x), np.real(y)]
                diffx, diffy = [np.diff(x0), np.diff(y0)]
                x0_new, y0_new = [np.ones(len(x) + 1), np.ones(len(y) + 1)]
                x0_new[0:-1], y0_new[0:-1] = [x0, y0]
                x0_new[-1], y0_new[-1] = [x0[-1] + diffx[-1], y0[-1] + diffy[-1]]

                # Interpolate the mode
                if not self.PML:
                    self.nlayers = [1, 0, 1, 0]
                Ex = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Ex"), True)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                Ey = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Ey"), True)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                Ez = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Ez"), True)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                Hx = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Hx"), False)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                Hy = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Hy"), False)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                Hz = interp(x0_new, y0_new, x0, y0, solver.modes[mode_num].get_field("Hz"), False)[
                    self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]
                ]
                self.x = x0_new[self.nlayers[1] : -self.nlayers[0]]
                self.y = y0_new[self.nlayers[3] : -self.nlayers[2]]

                # Create an EMEPy mode object
                neff = solver.modes[mode_num].neff
                n = np.sqrt(self.epsfunc(self.x, self.y))
                mode = Mode(x=self.x, y=self.y, wl=wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez, n=n)

                # Add the mode to the list
                self.modes[wl].append(mode)


        return self

    def clear(self) -> ModeSolver:
        """Clears the modesolver's eigenmodes to make memory"""

        self.solver = None
        return self

    def get_mode(self, freq: float, mode_num: int = 0) -> EigenMode:
        """Get the indexed mode number

        Parameters
        ----------
        freq: float
            the frequency of the mode. Should match perfectly to a value in self.wl
        mode_num : int
            index of the mode of choice

        Returns
        -------
        Mode
            the eigenmode of index mode_num
        """
    
        return self.modes[freq][mode_num]

    def plot_material(self) -> None:
        """Plots the index of refraction profile"""
        plt.imshow(np.sqrt(np.real(self.n)).T, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], cmap="Greys")
        plt.colorbar()
        plt.title("Index of Refraction")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
