import numpy as np
import pickle
from matplotlib import pyplot as plt
import EMpy_gpu
from EMpy_gpu.modesolvers.FD import stretchmesh
from typing import Callable

from emepy.mode import Mode, Mode1D, EigenMode
from emepy.tools import interp, interp1d, Si, SiO2, get_epsfunc, rectangle_to_n


class ModeSolver(object):
    """The ModeSolver object is the heart of finding eigenmodes for use in eigenmode expansion or simple examination. This parent class should be inherited and used as a wrapper for certain modules such as EMpy, Lumerical, Pickled data, Neural Networks, etc."""

    def __init__(self, **kwargs) -> None:
        """ModeSolver class constructor"""
        raise NotImplementedError

    def solve(self) -> None:
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


class ModeSolver1D(ModeSolver):
    pass


class MSEMpy(ModeSolver):
    """Electromagnetic Python Modesolver. Uses the EMpy library See Modesolver. Parameterizes the cross section as a rectangular waveguide."""

    def __init__(
        self,
        wl: float = 1.55,
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
            if True, will use PML boundaries. Only works for Tidy3D, not EMpy. Default : False, PEC
        subpixel : bool
            if true, will use subpixel smoothing, assuming asking for a waveguide cross section and not providing an index map (recommended)
        """

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.core_index = core_index if core_index is not None else Si(wl)
        self.cladding_index = cladding_index if cladding_index is not None else SiO2(wl)
        self.x = (
            x
            if x is not None
            else np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
        )
        self.y = (
            y
            if y is not None
            else np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
        )
        self.mesh = mesh
        self.accuracy = accuracy
        self.boundary = boundary
        self.epsfunc = epsfunc
        self.n = n
        self.PML = PML

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
                center,
                self.width,
                self.thickness,
                self.x,
                self.y,
                subpixel,
                self.core_index,
                self.cladding_index,
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

        self.after_x = self.x
        self.after_y = self.y
        self.n = np.sqrt(self.epsfunc(self.x, self.y))

    def solve(self) -> ModeSolver:
        """Solves for the eigenmodes"""
        self.solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(
            self.wl, self.x, self.y, self.epsfunc, self.boundary
        ).solve(self.num_modes, self.accuracy)
        return self

    def clear(self) -> ModeSolver:
        """Clears the modesolver's eigenmodes to make memory"""

        self.solver = None
        return self

    def get_mode(self, mode_num: int = 0) -> EigenMode:
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

        # if not self.PML:
        self.nlayers = [1, 0, 1, 0]
        Ex = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ex"), True
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        Ey = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ey"), True
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        Ez = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Ez"), True
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        Hx = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hx"), False
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        Hy = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hy"), False
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        Hz = interp(
            x0_new, y0_new, x0, y0, self.solver.modes[mode_num].get_field("Hz"), False
        )[self.nlayers[1] : -self.nlayers[0], self.nlayers[3] : -self.nlayers[2]]
        self.x = x0_new[self.nlayers[1] : -self.nlayers[0]]
        self.y = y0_new[self.nlayers[3] : -self.nlayers[2]]

        neff = self.solver.modes[mode_num].neff
        n = np.sqrt(self.epsfunc(self.x, self.y))

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
            n=n,
        )

    def plot_material(self) -> None:
        """Plots the index of refraction profile"""
        plt.imshow(
            np.sqrt(np.real(self.n)).T,
            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
            cmap="Greys",
        )
        plt.colorbar()
        plt.title("Index of Refraction")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")


class MSTidy3D(MSEMpy):
    def solve(self) -> ModeSolver:
        """Solves for the eigenmodes"""
        from tidy3d.plugins.mode.solver import compute_modes
        import tidy3d as td
        from types import SimpleNamespace

        bound_x = 0.5 * (self.x[1:] + self.x[:-1])
        bound_x = np.insert(bound_x, 0, bound_x[0] - np.diff(bound_x)[0])
        bound_x = np.insert(bound_x, len(bound_x), bound_x[-1] + np.diff(bound_x)[-1])

        bound_y = 0.5 * (self.y[1:] + self.y[:-1])
        bound_y = np.insert(bound_y, 0, bound_y[0] - np.diff(bound_y)[0])
        bound_y = np.insert(bound_y, len(bound_y), bound_y[-1] + np.diff(bound_y)[-1])

        (E, H), neff = compute_modes(
            eps_cross=np.array([self.n**2, self.n**2, self.n**2]),
            coords=(bound_x, bound_y),
            freq=td.constants.C_0 / (self.wl),
            mode_spec=SimpleNamespace(
                num_modes=int(self.num_modes),
                bend_radius=None,
                bend_axis=1,
                angle_theta=0.0,
                angle_phi=0.0,
                num_pml=(0, 0) if not self.PML else (20, 20),
                target_neff=self.core_index,
                precision="single",
                filter_pol=None,
            ),
        )

        self.modes = [
            Mode(
                self.x,
                self.y,
                self.wl,
                neff[i],
                H[0, :, :, 0, i],
                H[1, :, :, 0, i],
                H[2, :, :, 0, i],
                E[0, :, :, 0, i],
                E[1, :, :, 0, i],
                E[2, :, :, 0, i],
                self.n,
            )
            for i in range(self.num_modes)
        ]

        return self

    def clear(self) -> ModeSolver:
        """Clears the modesolver's eigenmodes to make memory"""

        self.solver = None
        return self

    def get_mode(self, mode_num: int = 0) -> EigenMode:
        return self.modes[mode_num]


# class MSEMpy1D(ModeSolver):
#     """NOTICE: DOES NOT CURRENTLY WORK!! Electromagnetic Python Modesolver. Uses the EMpy library See Modesolver. Parameterizes the cross section as a rectangular waveguide."""

#     def __init__(
#         self,
#         wl: float,
#         width: float = None,
#         num_modes: int = 1,
#         cladding_width: float = 2.5,
#         core_index: float = None,
#         cladding_index: float = None,
#         x: "np.ndarray" = None,
#         mesh: int = 128,
#         accuracy: float = 1e-8,
#         boundary: str = "0000",
#         epsfunc: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"] = None,
#         n: "np.ndarray" = None,
#         PML: bool = False,
#         **kwargs
#     ):
#         """MSEMpy class constructor

#         Parameters
#         ----------
#         wl : number
#             wavelength of the eigenmodes
#         width : number
#             width of the core in the cross section
#         num_modes : int
#             number of modes to solve for (default:1)
#         cladding_width : number
#             width of the cladding in the cross section (default:5)
#         core_index : number
#             refractive index of the core (default:Si)
#         cladding_index : number
#             refractive index of the cladding (default:SiO2)
#         mesh : int
#             number of mesh points in each direction (xy)
#         x : numpy array
#             the cross section grid in the x direction (z propagation) (default:None)
#         mesh : int
#             the number of mesh points in each xy direction
#         accuracy : number
#             the minimum accuracy of the finite difference solution (default:1e-8)
#         boundary : string
#             the boundaries according to the EMpy library (default:"0000")
#         epsfunc : function
#             the function which defines the permittivity based on a grid (see EMpy library) (default:"0000")
#         n : numpy array
#             2D profile of the refractive index
#         PML : boolean
#             if True, will use PML boundaries. Default : False, PEC
#         """

#         self.wl = wl
#         self.width = width
#         self.num_modes = num_modes
#         self.cladding_width = cladding_width
#         self.core_index = core_index
#         self.cladding_index = cladding_index
#         self.x = x
#         self.mesh = mesh
#         self.accuracy = accuracy
#         self.boundary = boundary
#         self.epsfunc = epsfunc
#         self.n = n
#         self.PML = PML

#         if core_index is None:
#             self.core_index = Si(wl)
#         if cladding_index is None:
#             self.cladding_index = SiO2(wl)
#         if x is None:
#             self.x = np.linspace(-0.5 * cladding_width, 0.5 * cladding_width, mesh)
#         if self.PML:  # Create a PML at least half a wavelength long
#             dx = np.diff(self.x)
#             layer_xp = int(np.abs(0.5 * self.wl / dx[-1]))
#             layer_xn = int(np.abs(0.5 * self.wl / dx[0]))
#             self.nlayers = [layer_xp, layer_xn, 0, 0]
#             factor = 1 + 2j
#             self.x, _, _, _, _, _ = stretchmesh(self.x, np.zeros(1), self.nlayers, factor)
#         if epsfunc is None:
#             self.epsfunc = get_epsfunc(
#                 self.width,
#                 None,
#                 self.cladding_width,
#                 None,
#                 self.core_index,
#                 self.cladding_index,
#                 profile=self.n,
#                 nx=self.x,
#             )

#         self.after_x = self.x
#         self.n = self.epsfunc(self.x, np.zeros(1))

#     def solve(self) -> ModeSolver:
#         """Solves for the eigenmodes"""
#         self.solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(
#             self.wl, self.x, np.zeros(1), self.epsfunc, self.boundary
#         ).solve(self.num_modes, self.accuracy)
#         return self

#     def clear(self) -> ModeSolver:
#         """Clears the modesolver's eigenmodes to make memory"""

#         self.solver = None
#         return self

#     def get_mode(self, mode_num: int = 0) -> EigenMode:
#         """Get the indexed mode number

#         Parameters
#         ----------
#         mode_num : int
#             index of the mode of choice

#         Returns
#         -------
#         Mode
#             the eigenmode of index mode_num
#         """

#         x = self.solver.modes[mode_num].get_x()
#         x0, y0 = [np.real(x), np.real(x)]
#         diffx, diffy = [np.diff(x0), np.diff(y0)]
#         x0_new, y0_new = [np.ones(len(x) + 1), np.ones(len(x) + 1)]
#         x0_new[0:-1], y0_new[0:-1] = [x0, y0]
#         x0_new[-1], y0_new[-1] = [x0[-1] + diffx[-1], y0[-1] + diffy[-1]]

#         if not self.PML:
#             self.nlayers = [1, 0, 1, 0]
#         Ex = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Ex"), True)[self.nlayers[1] : -self.nlayers[0]]
#         Ey = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Ey"), True)[self.nlayers[1] : -self.nlayers[0]]
#         Ez = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Ez"), True)[self.nlayers[1] : -self.nlayers[0]]
#         Hx = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Hx"), False)[self.nlayers[1] : -self.nlayers[0]]
#         Hy = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Hy"), False)[self.nlayers[1] : -self.nlayers[0]]
#         Hz = interp1d(x0_new, x0.self.solver.modes[mode_num].get_field("Hz"), False)[self.nlayers[1] : -self.nlayers[0]]
#         self.x = x0_new[self.nlayers[1] : -self.nlayers[0]]

#         neff = self.solver.modes[mode_num].neff
#         n = np.sqrt(self.epsfunc(self.x, np.zeros(0)))

#         return Mode1D(x=self.x, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez, n=n)

#     def plot_material(self) -> None:
#         """Plots the index of refraction profile"""
#         plt.plot(self.x, self.n)
#         plt.title("Index of Refraction")
#         plt.xlabel("x (µm)")
#         plt.ylabel("y (µm)")


# class MSPickle(object):
#     """Pickle Modesolver. See Modesolver. Pickle should serialize a list of Mode objects that can be opened here."""

#     def __init__(
#         self, filename: str, index: int = None, width: float = None, thickness: float = None, **kwargs
#     ) -> None:
#         """MSPickle class constructor

#         Parameters
#         ----------
#         filename : string
#             the name of the pickled file where the eigenmode is stored
#         index : int
#             the index of the mode in the pickle file if the data stored is an array of Modes (default:None)
#         width : number
#             width of the core in the cross section, used for drawing (default:None)
#         thickness : number
#             thickness of the core in the cross section, used for drawing  (default:None)
#         """

#         self.filename = filename
#         self.index = index
#         self.width = width
#         self.thickness = thickness
#         self.PML = False

#     def solve(self) -> ModeSolver:
#         """Solves for the eigenmodes by loading them from the pickle file"""

#         with open(self.filename, "rb") as f:
#             self.mode = pickle.load(f)[self.index] if self.index else pickle.load(f)

#         self.x = self.mode.x
#         self.y = self.mode.y
#         return self

#     def clear(self) -> ModeSolver:
#         """Clears the modesolver's eigenmodes to make memory"""

#         self.x = None
#         self.y = None
#         self.mode = None
#         return self

#     def get_mode(self, mode_num: int = 0) -> EigenMode:
#         """Get the stored mode

#         Returns
#         -------
#         Mode
#             the eigenmode of index mode_num
#         """

#         return self.mode
