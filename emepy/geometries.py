import numpy as np
from emepy.fd import MSEMpy, ModeSolver
from emepy.models import Layer


class Geometry(object):
    """Geoemtries are not required for users, however they do allow for easier creation of complex structures"""

    def __init__(self, layers: list) -> None:
        """Constructors should take in parameters from the user and build the layers"""
        self.layers = layers

    def __iter__(self):
        return iter(self.layers)


class Params(object):
    def __init__(self) -> None:
        return

    def get_solver_rect(self) -> ModeSolver:
        return

    def get_solver_index(self) -> ModeSolver:
        return


class EMpyGeometryParameters(Params):
    def __init__(
        self,
        wavelength: float = 1.55e-6,
        cladding_width: float = 2.5e-6,
        cladding_thickness: float = 2.5e-6,
        core_index: float = None,
        cladding_index: float = None,
        x: "np.ndarray" = None,
        y: "np.ndarray" = None,
        mesh: int = 128,
        accuracy: float = 1e-8,
        boundary: str = "0000",
        PML: bool = False,
        **kwargs,
    ):
        """Creates an instance of EMpyGeometryParameters which is used for abstract geometries that use EMpy as the solver
        """

        self.wavelength = wavelength
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.core_index = core_index
        self.cladding_index = cladding_index
        self.x = x
        self.y = y
        self.mesh = mesh
        self.accuracy = accuracy
        self.boundary = boundary
        self.PML = PML
        for key, val in kwargs.items():
            setattr(key, val)

    def get_solver_rect(self, width: float = 0.5e-6, thickness: float = 0.22e-6, num_modes: int = 1) -> "MSEMpy":
        """Returns an EMPy solver that represents a simple rectangle
        """

        return MSEMpy(
            wavelength=self.wavelength,
            width=width,
            thickness=thickness,
            num_modes=num_modes,
            cladding_width=self.cladding_width,
            cladding_thickness=self.cladding_thickness,
            core_index=self.core_index,
            cladding_index=self.cladding_index,
            x=self.x,
            y=self.y,
            mesh=self.mesh,
            accuracy=self.accuracy,
            boundary=self.boundary,
            PML=self.PML,
        )

    def get_solver_index(self, thickness: float = None, num_modes: int = None, n: "np.ndarray" = None) -> "MSEMpy":
        """Returns an EMPy solver that represents the provided index profile
        """

        return MSEMpy(
            wavelength=self.wavelength,
            width=None,
            thickness=thickness,
            num_modes=num_modes,
            cladding_width=self.cladding_width,
            cladding_thickness=self.cladding_thickness,
            core_index=self.core_index,
            cladding_index=self.cladding_index,
            x=self.x,
            y=self.y,
            mesh=self.mesh,
            accuracy=self.accuracy,
            boundary=self.boundary,
            n=n,
            PML=self.PML,
        )


class Waveguide(Geometry):
    """Block forms the simplest geometry in emepy, a single layer with a single waveguide defined"""

    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 1e-6,
        num_modes: int = 1,
    ) -> None:
        """Creates an instance of block which can be called to access the required layers for solving

        Parameters
        ----------
        params : Params
            Geometry Parameters object containing large scale parameters
        width : number
            width of the core in the cross section
        thickness : number
            thickness of the core in the cross section
        length : number
            length of the structure
        num_modes : int
            number of modes to solve for (default:1)
        """

        mode_solver = params.get_solver_rect(width, thickness, num_modes)
        layers = [Layer(mode_solver, num_modes, params.wavelength, length)]
        super().__init__(layers)


class WaveguideChannels(Geometry):
    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 1e-6,
        num_modes: int = 1,
        gap: float = 0.1e-6,
        num_channels: int = 2,
        exclude_indices: list = [],
    ) -> None:

        # Create n
        starting_center = -0.5 * (num_channels - 1) * (gap + width)
        n_output = np.ones(params.mesh) * params.cladding_index
        for out in range(num_channels):
            if out not in exclude_indices:
                center = starting_center + out * (gap + width)
                left_edge = center - 0.5 * width
                right_edge = center + 0.5 * width
                n_output = np.where((left_edge <= params.x) * (params.x <= right_edge), params.core_index, n_output)

        # Create modesolver
        output_channel = params.get_solver_index(thickness, num_modes, n_output)

        # Create layers
        self.layers = [Layer(output_channel, num_modes, params.wavelength, length)]
        super().__init__(self.layers)


class BraggGrating(Geometry):
    def __init__(
        self,
        params_left: Params = EMpyGeometryParameters(),
        params_right: Params = EMpyGeometryParameters(),
        width_left: float = 0.4e-6,
        thickness_left: float = 0.22e-6,
        length_left: float = 1e-6,
        width_right: float = 0.6e-6,
        thickness_right: float = 0.22e-6,
        length_right: float = 1e-6,
        num_modes: int = 1,
    ) -> None:

        # Create waveguides
        waveguide_left = Waveguide(params_left, width_left, thickness_left, length_left, num_modes)
        waveguide_right = Waveguide(params_right, width_right, thickness_right, length_right, num_modes)

        # Create layers
        self.layers = [*waveguide_left, *waveguide_right]
        super().__init__(self.layers)


class DirectionalCoupler(Geometry):
    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5e-6,
        thickness: float = 0.22e-6,
        length: float = 25e-6,
        gap: float = 0.2e-6,
        num_modes: int = 1,
    ) -> None:

        # Create input waveguide channel
        input = WaveguideChannels(params, width, thickness, length, num_modes, gap, 2, exclude_indices=[1])

        # Create main directional coupler
        coupler = WaveguideChannels(params, width, thickness, length, num_modes, gap, 2, exclude_indices=[])

        # Create layers
        self.layers = [*input, *coupler]
        super().__init__(self.layers)
