import numpy as np
from emepy.fd import MSEMpy
from emepy.models import Layer


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
        wavelength: float = 1.55e-6,
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
        wavelength : number
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
            wavelength,
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
        layers = [Layer(mode_solver, num_modes, wavelength, length)]
        super().__init__(layers)


class WaveguideChannels(Geometry):
    def __init__(
        self,
        wavelength: float = 1.55e-6,
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 1e-6,
        gap: float = 0.1e-6,
        num_channels: int = 2,
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
        exclude_indices: list = [],
    ) -> None:

        # Create n
        starting_center = -0.5 * (num_channels - 1) * (gap + width)
        n_output = np.ones(mesh) * cladding_index
        for out in range(num_channels):
            if not out in exclude_indices:
                center = starting_center + out * (gap + width)
                left_edge = center - 0.5 * width
                right_edge = center + 0.5 * width
                n_output = np.where((left_edge <= x) * (x <= right_edge), core_index, n_output)

        # Create modesolver
        output_channel = MSEMpy(
            wl=wavelength,
            width=width,
            thickness=thickness,
            cladding_index=cladding_index,
            cladding_width=cladding_width,
            cladding_thickness=cladding_thickness,
            num_modes=num_modes,
            mesh=mesh,
            x=x,
            y=x,
            boundary=boundary,
            accuracy=accuracy,
        )

        # Create layers
        self.layers = [Layer(output_channel, num_modes, wavelength, length)]
        super().__init__(self.layers)


class BraggGrating(Geometry):
    def __init__(
        self,
        wavelength: float = 1.55e-6,
        width_left: float = 0.4e-6,
        thickness_left: float = 0.22e-6,
        length_left: float = 1e-6,
        width_right: float = 0.6e-6,
        thickness_right: float = 0.22e-6,
        length_right: float = 1e-6,
        num_modes: int = 1,
        cladding_width: float = 2.5e-6,
        cladding_thickness: float = 2.5e-6,
        core_left_index: float = None,
        cladding_left_index: float = None,
        core_right_index: float = None,
        cladding_right_index: float = None,
        x: "np.ndarray" = None,
        y: "np.ndarray" = None,
        mesh: int = 128,
        accuracy: float = 1e-8,
        boundary: str = "0000",
    ) -> None:

        # Create waveguides
        waveguide_left = Waveguide(
            wavelength=wavelength,
            width=width_left,
            thickness=thickness_left,
            length=length_left,
            num_modes=num_modes,
            cladding_width=cladding_width,
            cladding_thickness=cladding_thickness,
            core_index=core_left_index,
            cladding_index=cladding_left_index,
            x=x,
            y=y,
            mesh=mesh,
            accuracy=accuracy,
            boundary=boundary,
        )

        waveguide_right = Waveguide(
            wavelength=wavelength,
            width=width_right,
            thickness=thickness_right,
            length=length_right,
            num_modes=num_modes,
            cladding_width=cladding_width,
            cladding_thickness=cladding_thickness,
            core_index=core_right_index,
            cladding_index=cladding_right_index,
            x=x,
            y=y,
            mesh=mesh,
            accuracy=accuracy,
            boundary=boundary,
        )

        # Create layers
        self.layers = [*waveguide_left, *waveguide_right]
        super().__init__(self.layers)


class DirectionalCoupler(Geometry):
    def __init__(
        self,
        wavelength: float = 1.55e-6,
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 25e-6,
        gap: float = 0.2e-6,
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

        # Create input waveguide channel
        input = WaveguideChannels(
            wavelength=wavelength,
            width=width,
            thickness=thickness,
            length=length,
            gap=gap,
            num_channels=2,
            num_modes=num_modes,
            cladding_width=cladding_width,
            cladding_thickness=cladding_thickness,
            core_index=core_index,
            cladding_index=cladding_index,
            x=x,
            y=y,
            mesh=mesh,
            accuracy=accuracy,
            boundary=boundary,
            exclude_indices=[1],
        )

        # Create main directional coupler
        coupler = WaveguideChannels(
            wavelength=wavelength,
            width=width,
            thickness=thickness,
            length=length,
            gap=gap,
            num_channels=2,
            num_modes=num_modes,
            cladding_width=cladding_width,
            cladding_thickness=cladding_thickness,
            core_index=core_index,
            cladding_index=cladding_index,
            x=x,
            y=y,
            mesh=mesh,
            accuracy=accuracy,
            boundary=boundary,
            exclude_indices=[],
        )

        # Create layers
        self.layers = [*input, *coupler]
        super().__init__(self.layers)
