import numpy as np
from emepy.fd import MSEMpy, ModeSolver, MSTidy3D
from emepy.models import Layer
from emepy.tools import vertices_to_n
from copy import deepcopy

"""
Write polygon that utilizes tidy3d
"""


class Geometry(object):
    """Geoemtries are not required for users, however they do allow for easier creation of complex structures"""

    def __init__(self, layers: list) -> None:
        """Constructors should take in parameters from the user and build the layers"""
        self.layers = layers

    def __iter__(self):
        return deepcopy(iter(self.layers))


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
        wavelength: float = 1.55,
        cladding_width: float = 2.5,
        cladding_thickness: float = 2.5,
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
        """Creates an instance of EMpyGeometryParameters which is used for abstract geometries that use EMpy as the solver"""

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

    def get_solver_rect(
        self,
        width: float = 0.5,
        thickness: float = 0.22,
        num_modes: int = 1,
        center: bool = (0, 0),
    ) -> "MSEMpy":
        """Returns an EMPy solver that represents a simple rectangle"""

        return MSEMpy(
            wl=self.wavelength,
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
            center=center,
        )

    def get_solver_index(
        self, thickness: float = None, num_modes: int = None, n: "np.ndarray" = None
    ) -> "MSEMpy":
        """Returns an EMPy solver that represents the provided index profile"""

        return MSEMpy(
            wl=self.wavelength,
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


class MSTidy3DGeometryParameters(Params):
    def __init__(
        self,
        wavelength: float = 1.55,
        cladding_width: float = 2.5,
        cladding_thickness: float = 2.5,
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
        """Creates an instance of Tidy3DGeometryParameters which is used for abstract geometries that use Tidy3D as the solver"""

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

    def get_solver_rect(
        self,
        width: float = 0.5,
        thickness: float = 0.22,
        num_modes: int = 1,
        center: bool = (0, 0),
    ) -> "MSTidy3D":
        """Returns an MSTidy3D solver that represents a simple rectangle"""

        return MSTidy3D(
            wl=self.wavelength,
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
            center=center,
        )

    def get_solver_index(
        self, thickness: float = None, num_modes: int = None, n: "np.ndarray" = None
    ) -> "MSTidy3D":
        """Returns an MSTidy3D solver that represents the provided index profile"""

        return MSTidy3D(
            wl=self.wavelength,
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


class DynamicPolygon(Geometry):
    """Creates a polygon in EMEPy given a list of solid vertices and changeable vertices and can be changed for shape optimization"""

    def get_design(self) -> list:
        """Returns the design region as a list of parameters. Each parameter represents a spacial location in a single direction for a single vertex of the polygon. Note in 2D this would look like [x0,z0,x1,z1,...xn,zn] and in 3D [x0,y0,z0,x1,y1,z1,...xn,yn,zn]"""
        return self.design

    def set_design(self) -> list:
        """Sets the design region"""
        return

    def get_n(self, grid_x, grid_y, grid_z) -> "np.ndarray":
        """Returns the index profile for the grid provided"""
        return

    def __len__(self):
        return len(self.get_design())


class DynamicRect2D(DynamicPolygon):
    def __init__(
        self,
        params: Params = MSTidy3DGeometryParameters(),
        width: float = 0.5,
        thickness: float = 0.22,
        length: float = 1,
        num_modes: int = 1,
        num_params: int = 10,
        symmetry: bool = False,
        subpixel: bool = True,
        mesh_z: int = 10,
        input_width: float = None,
        output_width: float = None,
    ) -> None:
        """Creates an instance of DynamicPolygon2D"""
        input_width = input_width if input_width is not None else width
        output_width = output_width if output_width is not None else width
        self.num_modes = num_modes
        self.params = deepcopy(params)
        self.symmetry = symmetry
        self.subpixel = subpixel
        self.width, self.thickness, self.length = (width, thickness, length)
        self.num_params = num_params
        self.grid_x = (
            params.x
            if params.x is not None
            else np.linspace(
                -params.cladding_width / 2, params.cladding_width / 2, params.mesh
            )
        )
        self.grid_z = np.linspace(0, length, mesh_z)

        # Set left side static vertices
        x = [-input_width / 2, input_width / 2]
        z = [0, 0]
        self.static_vertices_left = list(zip(x, z))

        # Set top dynamic vertices
        x = np.linspace(
            input_width / 2, output_width / 2, num_params
        ).tolist()  # [width / 2] * num_params
        z = np.linspace(0, length, num_params + 2)[1:-1].tolist()
        dynamic_vertices_top = list(zip(x, z))

        # Set right side static vertices
        x = [output_width / 2, -output_width / 2]
        z = [length, length]
        self.static_vertices_right = list(zip(x, z))

        # Set bottom dynamic vertices
        x = np.linspace(
            -output_width / 2, -input_width / 2, num_params
        ).tolist()  # [-width / 2] * num_params
        z = np.linspace(0, length, num_params + 2)[1:-1][::-1].tolist()
        dynamic_vertices_bottom = list(zip(x, z))

        # Establish design
        design = (
            dynamic_vertices_top[:]
            if symmetry
            else dynamic_vertices_top + dynamic_vertices_bottom
        )
        design = [i for j in design for i in j]

        # Fix params
        self.params.x = (
            0.5 * (self.params.x[1:] + self.params.x[:-1])
            if self.params.x is not None
            else self.params.x
        )
        self.params.y = (
            0.5 * (self.params.y[1:] + self.params.y[:-1])
            if self.params.y is not None
            else self.params.y
        )
        self.params.mesh -= 1

        # Set design
        self.set_design(design)

    def set_design(self, design: list):
        """Sets the design region parameters"""
        self.design = design
        self.set_layers()

    def get_n(self, grid_x, grid_y, grid_z):
        """Will form the refractive index map given the current parameters"""
        # Create vertices
        vertices = []

        # Add static left vertices
        vertices += self.static_vertices_left

        # Add top design
        top = self.design if self.symmetry else self.design[: len(self.design) // 2]
        vertices += [(x, z) for x, z in zip(top[:-1:2], top[1::2])]

        # Add static right vertices
        vertices += self.static_vertices_right

        # Add bottom design
        bottom = self.design if self.symmetry else self.design[len(self.design) // 2 :]
        if self.symmetry:
            vertices += [
                (-x, z) for x, z in list(zip(bottom[:-1:2], bottom[1::2]))[::-1]
            ]
        else:
            vertices += [(x, z) for x, z in list(zip(bottom[:-1:2], bottom[1::2]))]

        # Form polygon
        polygon = vertices_to_n(
            vertices,
            grid_x,
            grid_z,
            self.subpixel,
            self.params.core_index,
            self.params.cladding_index,
        )

        # Extend in 3D
        if grid_y is not None:
            polygon = np.stack([polygon] * (len(grid_y) - 1), axis=1)
            x, y, z = (
                0.5 * (grid_x[1:] + grid_x[:-1]),
                0.5 * (grid_y[1:] + grid_y[:-1]),
                0.5 * (grid_z[1:] + grid_z[:-1]),
            )
            x_, y_, z_ = np.meshgrid(x, y, z, indexing="ij")
            n = np.ones(x_.shape) * self.params.cladding_index
            n = np.where(np.abs(y_) < self.thickness / 2, polygon, n)
            return n

        # Return polygon
        return polygon

    def set_layers(self):
        """Creates the layers needed for the geometry"""

        # Get n
        n = self.get_n(self.grid_x, None, self.grid_z)
        from matplotlib import pyplot as plt

        # Iterate through n and create layers
        self.layers = []
        diff_z = np.diff(self.grid_z)
        for i, dz in enumerate(diff_z):
            mode_solver = self.params.get_solver_index(0.22, self.num_modes, n[:, i])
            layer = Layer(mode_solver, self.num_modes, self.params.wavelength, dz)
            self.layers.append(layer)


class Waveguide(Geometry):
    """Block forms the simplest geometry in emepy, a single layer with a single waveguide defined"""

    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5,
        thickness: float = 0.22,
        length: float = 1,
        num_modes: int = 1,
        center: tuple = (0, 0),
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

        self.length = length
        self.width = width
        mode_solver = params.get_solver_rect(width, thickness, num_modes, center=center)
        layers = [Layer(mode_solver, num_modes, params.wavelength, length)]
        super().__init__(layers)


class WaveguideChannels(Geometry):
    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5,
        thickness: float = 0.22,
        length: float = 1,
        num_modes: int = 1,
        gap: float = 0.1,
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
                n_output = np.where(
                    (left_edge <= params.x) * (params.x <= right_edge),
                    params.core_index,
                    n_output,
                )

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
        width_left: float = 0.4,
        thickness_left: float = 0.22,
        length_left: float = 1,
        width_right: float = 0.6,
        thickness_right: float = 0.22,
        length_right: float = 1,
        num_modes: int = 1,
    ) -> None:

        # Create waveguides
        waveguide_left = Waveguide(
            params_left, width_left, thickness_left, length_left, num_modes
        )
        waveguide_right = Waveguide(
            params_right, width_right, thickness_right, length_right, num_modes
        )

        # Create layers
        self.layers = [*waveguide_left, *waveguide_right]
        super().__init__(self.layers)


class DirectionalCoupler(Geometry):
    def __init__(
        self,
        params: Params = EMpyGeometryParameters(),
        width: float = 0.5,
        thickness: float = 0.22,
        length: float = 25,
        gap: float = 0.2,
        num_modes: int = 1,
    ) -> None:

        # Create input waveguide channel
        input = WaveguideChannels(
            params, width, thickness, length, num_modes, gap, 2, exclude_indices=[1]
        )

        # Create main directional coupler
        coupler = WaveguideChannels(
            params, width, thickness, length, num_modes, gap, 2, exclude_indices=[]
        )

        # Create layers
        self.layers = [*input, *coupler]
        super().__init__(self.layers)
