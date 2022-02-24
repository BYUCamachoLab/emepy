import numpy as np
from emepy import *


class Geometry(object):
    """Geoemtries are not required for users, however they do allow for easier creation of complex structures"""

    def __init__(self, layers: list) -> None:
        """Constructors should take in parameters from the user and build the layers"""
        self.layers = layers

    def __iter__(self):
        return iter(self.layers)


class Waveguide(Geometry):
    """Block forms the simplest geometry in emepy, a single layer with a single waveguide defined"""

    def __init__(
        self,
        wl: float = 1.55e-6,
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 1e-6,
        num_modes: int = 1,
        cladding_width: float = 2.5e-6,
        cladding_thickness: float = 2.5e-6,
        core_index: float = None,
        cladding_index: float = None,
        x: "np.ndarray" = None,
        y: "np.ndarray" = None,
        mesh: int = 128,
        accuracy: float = 1e-8,
        boundary: str = "0000",
    ) -> None:
        """Creates an instance of block which can be called to access the required layers for solving

        Parameters
        ----------
        wl : number
            wavelength of the eigenmodes
        width : number
            width of the core in the cross section
        thickness : number
            thickness of the core in the cross section
        length : number
            length of the structure
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
        """

        mode_solver = MSEMpy(
            wl,
            width,
            thickness,
            num_modes,
            cladding_width,
            cladding_thickness,
            core_index,
            cladding_index,
            x,
            y,
            mesh,
            accuracy,
            boundary,
        )
        layers = [Layer(mode_solver, num_modes, wl, length)]
        super().__init__(layers)

class WaveguideChannels(Geometry):

    def __init__(
        self,
        wl: float = 1.55e-6,
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 1e-6,
        gap : float=0.1e-6,
        num_channels: int= 2,
        num_modes: int = 1,
        cladding_width: float = 2.5e-6,
        cladding_thickness: float = 2.5e-6,
        core_index: float = None,
        cladding_index: float = None,
        x: "np.ndarray" = None,
        y: "np.ndarray" = None,
        mesh: int = 128,
        accuracy: float = 1e-8,
        boundary: str = "0000",
    ) -> None:

        return

class BraggGrating(Geometry):
    pass

class DirectionalCoupler(Geometry):
    pass

class MMI(Geometry):
    pass

class TaperedMMI(Geometry):
    pass

class Taper(Geometry):
    pass