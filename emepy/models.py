import numpy as np
import numpy
from simphony import Model
from simphony.pins import Pin
from simphony.models import Subcircuit
from emepy.mode import EigenMode
from emepy.fd import ModeSolver
from copy import deepcopy


class Layer(object):
    """Layer objects form the building blocks inside of an EME or PeriodicEME. These represent geometric layers of rectangular waveguides that approximate continuous structures."""

    def __init__(self, mode_solver: ModeSolver, num_modes: int, wavelength: float, length: float) -> None:
        """Layer class constructor

        Parameters
        ----------
        mode_solver : Modesolver
            ModeSolver object used to solve for the modes
        num_modes : int
            Number of total modes for the layer.
        wavelength : number
            Wavelength of eigenmode to solve for (m).
        length : number
            Geometric length of the Layer (m). The length affects the phase of the eigenmodes inside the layer via the complex phasor $e^(jβz)$.
        """

        self.num_modes = num_modes
        self.mode_solver = mode_solver
        self.wavelength = wavelength
        self.length = length
        self.activated_layers = []

    def begin_activate(self):
        return ModelTools._solve_modes_wrapper, self.mode_solver

    def finish_activate(self, sources: list = [], start: float = 0.0, period_length: float = 0.0, mode_solver=None):
        self.mode_solver = mode_solver
        return self.activate_layer(sources, start, period_length, False)

    def activate_layer(
        self, sources: list = [], start: float = 0.0, period_length: float = 0.0, compute_modes=True
    ) -> dict:
        """Solves for the modes in the layer and creates an ActivatedLayer object

        Parameters
        ----------
        sources : list[Source]
            the Sources used to indicate where periodic layers are needed
        start : number
            the starting z value
        periodic_length : number
            the length of a single period

        Returns
        -------
        dict
            a dictionary that maps the period number to the activated layers. If there is no source in a period, it will be None instead at that index

        """

        modes = []

        # Solve for modes
        if compute_modes:
            self.mode_solver.solve()
        for mode in range(self.num_modes):
            modes.append(self.mode_solver.get_mode(mode))

        # Purge spurious mode
        modes = ModelTools.purge_spurious(modes)

        # Create activated layers
        self.activated_layers = dict(zip(sources.keys(), [[] for _ in range(len(sources.keys()))]))

        # Loop through all periods
        for per, srcs in sources.items():

            # Only care about sources between the ends
            start_ = start + per * period_length
            custom_sources = ModelTools.get_sources(srcs, start_, start_ + self.length)

            # First period
            if not per:

                # If no custom sources
                if not len(custom_sources):
                    self.activated_layers[per] += [ActivatedLayer(modes, self.wavelength, self.length)]

                # Other sources
                else:
                    self.activated_layers[per] += ModelTools.get_source_system(
                        modes, self.wavelength, self.length, custom_sources, start_
                    )

            # Any other period
            else:

                # If no custom sources
                if not len(custom_sources):
                    self.activated_layers[per] += [None]

                # Other sources
                else:
                    self.activated_layers[per] += ModelTools.get_source_system(
                        modes, self.wavelength, self.length, custom_sources, start_
                    )

        return self.activated_layers

    def get_activated_layer(self, sources: list = [], start: float = 0.0) -> dict:
        """Gets the activated layer if it exists or calls activate_layer first

        Parameters
        ----------
        sources : list[Source]
            a list of Source objects for this layer

        Returns
        -------
        dict
            a dictionary that maps the period number to the activated layers. If there is no source in a period, it will be None instead at that index
        """

        if not len(self.activated_layers):
            self.activate_layer(sources=sources, start=start)

        return self.activated_layers

    def clear(self) -> "numpy.ndarray":
        """Empties the modes in the ModeSolver to clear memory

        Returns
        -------
        numpy array
            the edited image
        """

        self.mode_solver.clear()


class Duplicator(Model):
    """Duplicator is used for observing scattering parameters in the middle of an arbitrary network"""

    def __init__(self, wavelength: float, num_modes: int, label: str = "", **kwargs) -> None:
        """Creates an instance of Duplicator which is used only for finding the scattering values inside of an arbitrary network after cascaded

        Parameters
        ----------
        wavelength : float
            the wavelength of the simulation
        num_modes : int
            number of modes in the layer being duplicated
        label : str
            The label indicating where this specific duplicator looks
        """
        self.num_modes = num_modes
        self.wavelength = wavelength

        self.left_pins = ["left" + str(i) for i in range(self.num_modes)] + [
            "left_dup{}{}".format(str(i), label) for i in range(self.num_modes)
        ]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)] + [
            "right_dup{}{}".format(str(i), label) for i in range(self.num_modes)
        ]
        self.S0 = None

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            if "dup" in name:
                pins.append(Pin(self, name))
        for name in self.right_pins:
            if "dup" not in name:
                pins.append(Pin(self, name))
        self.pins = pins
        super().__init__(**kwargs, pins=pins)
        self.s_params = self.calculate_s_params()

    def s_parameters(self, freqs: "np.array" = None) -> "np.ndarray":

        return self.s_params

    def calculate_s_params(self) -> "np.ndarray":
        """Calculates the scattering parameters for the duplicator model"""

        # Create template for final s matrix
        m = self.num_modes

        # Create propagation diagonal matrix
        propagation_matrix1 = np.diag(np.exp((0j) * np.ones(self.num_modes * 4)))

        # Create sub matrix
        s_matrix = np.zeros((2 * m, 2 * m), dtype=complex)
        s_matrix[0:m, 0:m] = propagation_matrix1[m : 2 * m, 0:m]
        s_matrix[m : 2 * m, 0:m] = propagation_matrix1[0:m, 0:m]
        s_matrix[0:m, m : 2 * m] = propagation_matrix1[m : 2 * m, m : 2 * m]
        s_matrix[m : 2 * m, m : 2 * m] = propagation_matrix1[0:m, m : 2 * m]

        # Join all
        s_matrix_new = np.zeros((4 * m, 4 * m), dtype=complex)
        s_matrix_new[:m, 3 * m :] = s_matrix[:m, m:]
        s_matrix_new[m : 2 * m, 3 * m :] = s_matrix[:m, m:]
        s_matrix_new[3 * m :, m : 2 * m] = s_matrix[:m, m:]
        s_matrix_new[:m, 2 * m : 3 * m] = s_matrix[m:, :m]
        s_matrix_new[2 * m : 3 * m, :m] = s_matrix[m:, :m]
        s_matrix_new[3 * m :, :m] = s_matrix[m:, :m]
        s_matrix = s_matrix_new

        # Assign number of ports
        self.right_ports = m  # 2 * m - self.which_s * m
        self.left_ports = m  # 2 * m - (1 - self.which_s) * m
        self.num_ports = 2 * m  # 3 * m
        s_matrix = s_matrix.reshape(1, 4 * m, 4 * m)

        return s_matrix


class Current(Model):
    """The object that the EME uses to track the s_parameters and cascade them as they come along to save memory"""

    def __init__(self, wavelength: float, s: "np.ndarray", **kwargs) -> None:
        """Current class constructor

        Parameters
        ----------
        wavelength : number
            the wavelength of the simulation
        s : numpy array
            the starting scattering matrix
        """
        self.left_ports = s.left_ports
        self.left_pins = s.left_pins
        self.s_params = s.s_params
        self.right_ports = s.right_ports
        self.num_ports = self.right_ports + self.left_ports
        self.right_pins = s.right_pins
        self.wavelength = wavelength

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def update_s(self, s: "np.ndarray", layer: "Layer") -> None:
        """Updates the scattering matrix of the object

        Parameters
        ----------
        s : numpy array
            scattering matrix to use as the update
        layer : Layer
            the layer object whos ports to match
        """

        self.s_params = s
        self.right_ports = layer.right_ports
        self.num_ports = self.right_ports + self.left_ports
        self.right_pins = layer.right_pins

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(pins=pins)

    def s_parameters(self, freqs: "np.ndarray" = None) -> "np.ndarray":
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """
        return self.s_params


class ActivatedLayer(Model):
    """ActivatedLayer is produced by the Layer class after the ModeSolvers calculate eigenmodes. This is used to create interfaces. This inherits from Simphony's Model class."""

    def __init__(self, modes: list, wavelength: float, length: float, n_only: bool = False, **kwargs) -> None:
        """ActivatedLayer class constructor

        Parameters
        ----------
        modes : list [Mode]
            list of solved eigenmodes in Mode class form
        wavelength : number
            the wavelength of the eigenmodes
        length : number
            the length of the layer object that produced the eigenmodes. This number is used for phase propagation.
        n_only : bool
            if true, will only use the refractive index profile (default: False)
        """

        self.num_modes = len(modes)
        self.modes = modes
        self.wavelength = wavelength
        self.length = length
        self.left_pins = ["left" + str(i) for i in range(self.num_modes)]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)]
        self.S0 = None
        self.nk = []
        self.pk = []
        if not n_only:
            self.normalize_fields()
            self.s_params = self.calculate_s_params()

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        self.pins = pins
        super().__init__(**kwargs, pins=pins)

    def normalize_fields(self) -> None:
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1."""

        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def calculate_s_params(self) -> "np.ndarray":
        """Calculates the s params for the phase propagation and returns it.

        Returns
        -------
        numpy array
            the scattering matrix for phase propagation.
        """

        # Create template for final s matrix
        m = self.num_modes
        s_matrix = np.zeros((1, 2 * m, 2 * m), dtype=complex)

        # Create eigenvalue vector
        eigenvalues1 = (2 * np.pi) * np.array([mode.neff for mode in self.modes * 2]) / (self.wavelength)

        # Create propagation diagonal matrix
        propagation_matrix1 = np.diag(np.exp(self.length * 1j * eigenvalues1))

        # Assign prop matrix to final s params (moving corners to right spots)
        s_matrix[0, 0:m, 0:m] = propagation_matrix1[m : 2 * m, 0:m]
        s_matrix[0, m : 2 * m, 0:m] = propagation_matrix1[0:m, 0:m]
        s_matrix[0, 0:m, m : 2 * m] = propagation_matrix1[m : 2 * m, m : 2 * m]
        s_matrix[0, m : 2 * m, m : 2 * m] = propagation_matrix1[0:m, m : 2 * m]

        # Assign ports
        self.right_ports = m
        self.left_ports = m
        self.num_ports = 2 * m

        return s_matrix

    def s_parameters(self, freqs: "np.ndarray" = None) -> "np.ndarray":

        return self.s_params


class InterfaceSingleMode(Model):
    """The InterfaceSingleMode class represents the interface between two different layers. This class is an approximation to speed up the process and can ONLY be used during single mode EME."""

    def __init__(self, layer1: "Layer", layer2: "Layer", **kwargs) -> None:
        """InterfaceSingleMode class constructor

        Parameters
        ----------
        layer1 : Layer
            the left Layer object of the interface
        layer2 : Layer
            the right Layer object of the interface
        """

        self.layer1 = layer1
        self.layer2 = layer2
        self.num_modes = 1
        self.left_ports = 1
        self.right_ports = 1
        self.num_ports = self.left_ports + self.right_ports
        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))
        super().__init__(**kwargs, pins=pins)
        self.solve()

    def s_parameters(self, freqs: "np.ndarray" = None) -> "np.ndarray":
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """

        return self.s_params

    def solve(self) -> None:
        """Solves for the scattering matrix based on transmission and reflection"""

        s = np.zeros((2 * self.num_modes, 2 * self.num_modes), dtype=complex)

        for inp in range(len(self.layer1.modes)):
            for outp in range(len(self.layer2.modes)):

                left_mode = self.layer1.modes[inp]
                right_mode = self.layer2.modes[outp]

                r, t = self.get_values(left_mode, right_mode)

                s[outp, inp] = r
                s[outp + self.num_modes, inp] = t

        for inp in range(len(self.layer2.modes)):
            for outp in range(len(self.layer1.modes)):

                left_mode = self.layer1.modes[outp]
                right_mode = self.layer2.modes[inp]

                r, t = self.get_values(right_mode, left_mode)

                s[outp, inp + self.num_modes] = t
                s[outp + self.num_modes, inp + self.num_modes] = r

        self.s_params = s.reshape((1, 2 * self.num_modes, 2 * self.num_modes))

    def get_values(self, left: EigenMode, right: EigenMode) -> tuple:
        """Returns the reflection and transmission coefficient based on the two modes

        Parameters
        ----------
        left : EigenMode
            leftside eigenmode
        right : EigenMode
            rightside eigenmode

        Returns
        -------
        r : number
            reflection coefficient
        t : number
            transmission coefficient
        """

        a = 0.5 * left.inner_product(right) + 0.5 * right.inner_product(left)
        b = 0.5 * left.inner_product(right) - 0.5 * right.inner_product(left)

        t = (a ** 2 - b ** 2) / a
        r = 1 - t / (a + b)

        return -r, t

    def clear(self) -> None:
        """Clears the scattering matrix in the object"""

        self.s_params = None


class InterfaceMultiMode(Model):
    """The InterfaceMultiMode class represents the interface between two different layers."""

    def __init__(self, layer1: "Layer", layer2: "Layer", **kwargs) -> None:
        """InterfaceMultiMode class constructor

        Parameters
        ----------
        layer1 : Layer
            the left Layer object of the interface
        layer2 : Layer
            the right Layer object of the interface
        """

        self.layer1 = layer1
        self.layer2 = layer2
        self.left_ports = layer1.right_ports
        self.right_ports = layer2.left_ports
        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)
        self.num_ports = layer1.right_ports + layer2.left_ports
        self.solve()

    def s_parameters(self, freqs: "np.ndarray" = None) -> "np.ndarray":

        return self.s_params

    def solve(self) -> None:
        """Solves for the scattering matrix based on transmission and reflection"""

        s = np.zeros((self.num_ports, self.num_ports), dtype=complex)

        # Forward values
        for p in range(self.left_ports):

            ts = self.get_t(p, self.layer1, self.layer2, self.left_ports)
            rs = self.get_r(p, ts, self.layer1, self.layer2, self.left_ports)

            for t in range(len(ts)):
                s[self.left_ports + t][p] = ts[t]
            for r in range(len(rs)):
                s[r][p] = rs[r]

        # Reverse values
        for p in range(self.right_ports):

            ts = self.get_t(p, self.layer2, self.layer1, self.right_ports)
            rs = self.get_r(p, ts, self.layer2, self.layer1, self.right_ports)

            for t in range(len(ts)):
                s[t][self.left_ports + p] = ts[t]
            for r in range(len(rs)):
                s[self.left_ports + r][self.left_ports + p] = rs[r]

        # Keep s params and clear the layers
        self.s_params = s.reshape((1, self.num_ports, self.num_ports))
        self.layer1 = None
        self.layer2 = None

    def get_t(self, p: int, left: "EigenMode", right: "EigenMode", curr_ports: int) -> "np.ndarray":
        """Returns the transmission coefficient based on the two modes

        Parameters
        ----------
        p : int
            port number to look at
        left : Mode
            leftside eigenmode
        right : Mode
            rightside eigenmode
        curr_ports : int
            total number of ports

        Returns
        -------
        np.ndarray
            transmission coefficients
        """

        # Ax = b
        A = np.array(
            [
                [
                    right.modes[k].inner_product(left.modes[i]) + left.modes[i].inner_product(right.modes[k])
                    for k in range(self.num_ports - curr_ports)
                ]
                for i in range(curr_ports)
            ]
        )
        b = np.array([0 if i != p else 2 * left.modes[p].inner_product(left.modes[p]) for i in range(curr_ports)])
        x = np.matmul(np.linalg.pinv(A), b)

        return x

    def get_r(self, p: int, x: "np.ndarray", left: EigenMode, right: EigenMode, curr_ports: int) -> "np.ndarray":
        """Returns the transmission coefficient based on the two modes

        Parameters
        ----------
        p : int
            port number to look at
        x : np.ndarray
            transmission coefficients
        left : Mode
            leftside eigenmode
        right : Mode
            rightside eigenmode
        curr_ports : int
            total number of ports

        Returns
        -------
        r : number
            reflection coefficient
        """

        rs = np.array(
            [
                np.sum(
                    [
                        (right.modes[k].inner_product(left.modes[i]) - left.modes[i].inner_product(right.modes[k]))
                        * x[k]
                        for k in range(self.num_ports - curr_ports)
                    ]
                )
                / (2 * left.modes[i].inner_product(left.modes[i]))
                for i in range(curr_ports)
            ]
        )

        return rs

    def clear(self) -> None:
        """Clears the scattering matrix in the object"""

        self.s_params = None


class SourceDuplicator(Model):
    """SourceDuplicator is used for custom sources at an arbitrary location inside of the network"""

    def __init__(
        self,
        wavelength: float,
        modes: list,
        length: float,
        pk: list = [],
        nk: list = [],
        label: str = "",
        special_left: list = [],
        special_right: list = [],
        **kwargs,
    ) -> None:
        """Like Duplicator, but is used as a custom input rather than peaker. Optimized for minimum ports necessary

        Parameters
        ----------
        wavelength : number
            wavelength of the simulation
        modes : list[EigenMode]
            list of the EigenModes at the layer being duplicated
        length : float
            the length of the duplicated layer
        pk : list[float]
            a list of the mode coefficients propagating in the positive direction
        nk : list[float]
            a list of the mode coefficients propagating in the negative direction
        label : str
            the label indicating the location of the duplicator
        special_left : list
            the left coefficients to keep that are not included in pk
        special_right : list
            the right coefficients to keep taht are not included in nk
        """

        self.num_modes = len(modes)
        self.wavelength = wavelength
        self.modes = modes
        self.length = length
        self.pk = pk
        self.nk = nk
        self.normalize_fields()

        self.left_pins = (
            ["left" + str(i) for i in range(self.num_modes)]
            + ["left_dup{}{}".format(str(i), label) for i in range(len(pk) * (not len(special_left)))]
            + special_left
        )
        self.right_pins = (
            ["right" + str(i) for i in range(self.num_modes)]
            + ["right_dup{}{}".format(str(i), label) for i in range(len(nk) * (not len(special_right)))]
            + special_right
        )
        self.S0 = None
        self.S1 = None

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            if "dup" in name:
                pins.append(Pin(self, name))
        for name in self.right_pins:
            if "dup" not in name:
                pins.append(Pin(self, name))
        self.pins = pins
        super().__init__(**kwargs, pins=pins)
        self.s_params = self.calculate_s_params()

    def s_parameters(self, freqs: "np.array" = None) -> "np.ndarray":

        return self.s_params

    def normalize_fields(self) -> None:
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1."""

        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def calculate_s_params(self) -> "np.ndarray":
        """Calculates the scattering parameters for the system"""

        # Create template for final s matrix
        m = self.num_modes

        # Create eigenvalue vector
        eigenvalues1 = (2 * np.pi) * np.array([mode.neff for mode in self.modes * 4]) / (self.wavelength)

        # Create propagation diagonal matrix
        propagation_matrix1 = np.diag(np.exp(self.length * 1j * eigenvalues1))

        # Create sub matrix
        s_matrix = np.zeros((2 * m, 2 * m), dtype=complex)
        s_matrix[0:m, 0:m] = propagation_matrix1[m : 2 * m, 0:m]
        s_matrix[m : 2 * m, 0:m] = propagation_matrix1[0:m, 0:m]
        s_matrix[0:m, m : 2 * m] = propagation_matrix1[m : 2 * m, m : 2 * m]
        s_matrix[m : 2 * m, m : 2 * m] = propagation_matrix1[0:m, m : 2 * m]

        # Join all
        s_matrix_new = np.zeros((4 * m, 4 * m), dtype=complex)
        s_matrix_new[:m, 3 * m :] = s_matrix[:m, m:]
        s_matrix_new[m : 2 * m, 3 * m :] = s_matrix[:m, m:]
        s_matrix_new[3 * m :, m : 2 * m] = s_matrix[:m, m:]
        s_matrix_new[:m, 2 * m : 3 * m] = s_matrix[m:, :m]
        s_matrix_new[2 * m : 3 * m, :m] = s_matrix[m:, :m]
        s_matrix_new[3 * m :, :m] = s_matrix[m:, :m]
        # s_matrix_new[m:3*m,m:3*m] = s_matrix[:,:]
        s_matrix = s_matrix_new

        # Delete rows and cols
        pk_remove = m - len(self.pk)
        nk_remove = m - len(self.nk)
        for _ in range(nk_remove):
            s_matrix = np.delete(s_matrix, -m - 1, axis=0)
            s_matrix = np.delete(s_matrix, -m - 1, axis=1)
        for _ in range(pk_remove):
            s_matrix = np.delete(s_matrix, -m - len(self.nk) - 1, axis=0)
            s_matrix = np.delete(s_matrix, -m - len(self.nk) - 1, axis=1)

        # Assign number of ports
        self.right_ports = m  # 2 * m - self.which_s * m
        self.left_ports = m  # 2 * m - (1 - self.which_s) * m
        self.num_ports = 2 * m  # 3 * m
        s_matrix = s_matrix.reshape(1, 2 * m + len(self.pk) + len(self.nk), 2 * m + len(self.pk) + len(self.nk))

        return s_matrix


class CopyModel(Model):
    """A simple Model that can be used to deep copy any of EMEPy's Models"""

    def __init__(self, model: "Model", keep_modes: bool = True, **kwargs) -> None:
        """Creates an instance of CopyModel by deepcopying all the attributes of model

        Parameters
        ----------
        model : Model
            the model ot deepcopy
        """

        self.num_modes = model.num_modes if hasattr(model, "num_modes") else None
        self.modes = model.modes if hasattr(model, "modes") and keep_modes else []
        self.wavelength = model.wavelength if hasattr(model, "wavelength") else None
        self.length = model.length if hasattr(model, "length") else None
        self.left_ports = model.left_ports if hasattr(model, "left_ports") else 0
        self.right_ports = model.right_ports if hasattr(model, "right_ports") else 0

        self.S0 = ModelTools.make_copy_model(model.S0, keep_modes=False) if hasattr(model, "S0") else None
        self.nk = model.nk if hasattr(model, "nk") else []
        self.pk = model.pk if hasattr(model, "pk") else []
        self.pins = self.copy_pins(model.pins) if hasattr(model, "pins") else []
        self.left_pins = (
            model.left_pins if hasattr(model, "left_pins") else []
        )  # [self.find_pin(pin) for pin in model.left_pins] if hasattr(model,"left_pins") else []
        self.right_pins = (
            model.right_pins if hasattr(model, "right_pins") else []
        )  # [self.find_pin(pin) for pin in model.right_pins] if hasattr(model,"right_pins") else []
        self.s_params = model.s_parameters([0])

        super().__init__(**kwargs, pins=self.pins)
        return

    def s_parameters(self, freqs: "np.array" = None) -> "np.ndarray":

        return self.s_params

    def copy_pins(self, pins: list) -> list:
        """Copies the pins by creating new pins that are no longer attached to the original Model's connections

        Parameters
        ----------
        pins : list[Pin]
            the list of simphony pins to deepcopy

        Returns
        -------
        list[Pin]
            returns a list of copied pins
        """

        return [Pin(self, p.name) for p in pins]

    def find_pin(self, pin: "Pin") -> "Pin":
        """Searches the instance's pins for the provided pin based on names

        Parameters
        ----------
        pin : Pin
            the pin whos name to search for in the pinlist

        Returns
        -------
        Pin
            pin object corresponding to the searched pin
        """

        for p in self.pins:
            if p.name == pin.name:
                return p
        return Pin(self, pin.name)


class ModelTools(object):
    @staticmethod
    def purge_spurious(modes: list) -> list:
        """Purges all spurious modes in the dataset to prevent EME failure for high mode simulations

        Parameters
        ----------
        modes : list[EigenMode]
            list of EigenModes to remove spurious from

        Returns
        -------
        list[EigenMode]
            returns a list of the eigenmodes that are not spurious
        """

        mm = deepcopy(modes)
        for i, mode in enumerate(mm[::-1]):
            if mode.check_spurious():
                mm.remove(mode)

        return mm

    @staticmethod
    def get_sources(sources: list = [], start: float = 0.0, end: float = 0.0) -> list:
        """Given a list of sources, returns a new list of sources that can be found within the given range

        Parameters
        ----------
        sources : list[Source]
            list of Source objects
        start : number
            the starting point of the range
        end : number
            the ending point of the range

        Returns
        -------
        list[Source]
            returns a list of sources within the provided range
        """

        return [
            i
            for i in sources
            if i.z is not None and (((start <= i.z < end) and i.k) or ((start < i.z <= end) and not i.k))
        ]

    @staticmethod
    def get_source_system(
        modes: list, wavelength: float, length: float, custom_sources: list, start: float = 0.0
    ) -> list:
        """Creates SourceDuplicators matching the provided source locations and modes provided. This is a replacement for creating ActivatedLayers which contain no information about sources.

        Parameters
        ----------
        modes : list[EigenMode]
            the solved modes in the system
        wavelength : float
            the wavelength of the simulation
        length : float
            the length of the layer
        custom_sources : list[Source]
            the custom sources used to create SourceDuplicator objects
        start : float
            the starting z value for these layers

        Returns
        -------
        list[SourceDuplicator]
            returns the newly created source duplicators whom when cascaded form an equivalent ActivatedLayer with bonus source locations
        """

        # Get lengths between components
        lengths = np.diff([start] + [i.z for i in custom_sources] + [start + length])
        dups = []

        # Enumerate through all lengths
        length_tracker = start
        for i, length in enumerate(lengths):

            # Create coefficents indexes to keep in the models
            pk, nk = [[], []]
            if (i - 1) > -1 and custom_sources[i - 1].k:
                pk = custom_sources[i - 1].mode_coeffs
            if (i) < len(custom_sources) and not custom_sources[i].k:
                nk = custom_sources[i].mode_coeffs

            # Create label
            left = custom_sources[i - 1].get_label() if (i - 1) > -1 else "n" + str(length_tracker)
            right = custom_sources[i].get_label() if (i) < len(custom_sources) else "n" + str(length_tracker + length)
            label = "_{}_to_{}".format(left, right)

            # Create duplicators
            dups.append(SourceDuplicator(wavelength, modes, length, pk=pk, nk=nk, label=label))
            length_tracker += length

        return dups

    @staticmethod
    def _prop_all(*args) -> "Model":
        """Given an arbitrary amount of simphony models as inputs, cascades them and returns the cascaded network"""

        layers = [ModelTools.make_copy_model(a, keep_modes=False) for a in args if a is not None]

        temp_s = layers[0]
        for s in layers[1:]:
            Subcircuit.clear_scache()

            # make sure the components are completely disconnected
            temp_s.disconnect()
            s.disconnect()

            # # connect the components
            right_pins = [i for i in temp_s.pins if "dup" not in i.name and "left" not in i.name]
            for port in range(len(right_pins)):
                temp_s[f"right{port}"].connect(s[f"left{port}"])

            temp_s = temp_s.circuit.to_subcircuit()
            temp_s.s_params = temp_s.s_parameters([0])

        return ModelTools.make_copy_model(temp_s, keep_modes=False)

    @staticmethod
    def periodic_duplicate_format(model: "Model", start: float, end: float) -> "Model":
        """Checks if the model that has custom sources installed actually needs the installed sources between start and end. If not, removes those pins and rows/columns from the s_matrices

        Parameters
        ----------
        model : Model
            the simphony model to check
        start : float
            the starting point of the range of concern
        end : float
            the ending point of the range of concern

        Returns
        -------
        Model
            the simplified simphony model
        """

        model = ModelTools.make_copy_model(model)
        wanted = [i for i, pin in enumerate(model.pins) if "dup" in pin.name]
        indices = [i for i, pin in enumerate(model.pins) if "dup" in pin.name]
        model.s_params = model.s_parameters([0])

        # Find wanted indices
        for i in wanted[::-1]:
            if "_to_" not in model.pins[i].name:
                pass
            elif "left" in model.pins[i].name:
                n = model.pins[i].name
                n_ = n.split("_")
                l, _ = (float(n_[2][1:]), float(n_[4][1:]))
                if start <= l < end or np.isclose(start, l, 1e-5):
                    continue
            else:
                n = model.pins[i].name
                n_ = n.split("_")
                _, r = (float(n_[2][1:]), float(n_[4][1:]))
                if start < r <= end or np.isclose(r, end, 1e-5):
                    continue
            wanted.remove(i)

        # Remove unwanted rows and columns
        for i in indices:
            if i not in wanted:
                model.s_params = np.delete(model.s_params, i, 1)
                model.s_params = np.delete(model.s_params, i, 2)

        # Remove unwanted pins
        model.pins = [i for j, i in enumerate(model.pins) if j in wanted or "dup" not in i.name]
        pin_names = [i.name for i in model.pins]
        model.left_pins = [i for i in model.left_pins if i in pin_names]
        model.right_pins = [i for i in model.right_pins if i in pin_names]

        return model

    @staticmethod
    def make_copy_model(model: Model, keep_modes=True) -> CopyModel:
        """Takes an EMEPy model (inheriting a simphony model) and deepcopies it

        Parameters
        ----------
        model: Model
            the EMEPy model to copy

        Returns
        -------
        CopyModel
            the deepcopied model
        """
        new_model = CopyModel(model, keep_modes) if model is not None else None
        return new_model

    @staticmethod
    def compute(model, pin_values: "dict", freq: "float" = 0) -> dict:
        """Takes a dictionary mapping each pin name to a coefficent and multiplies by the S matrix

        Parameters
        ----------
        pin_values : dict
            a mapping of the pin names and the corresponding weights (mode coefficients)
        freq : float
            the frequency to compute at

        Returns
        -------
        dict
           a mapping of the output pins to the resulting weights (mode coefficients)
        """
        cfs = np.zeros(len(model.pins), dtype=complex)
        pin_names = [i.name for i in model.pins]
        key_index = dict(zip(pin_names, [pin_names.index(i) for i in pin_names]))
        for key, value in pin_values.items():
            if key in key_index:
                cfs[key_index[key]] = value
        matrix = model.s_parameters(np.array([freq]))
        output = np.matmul(matrix, cfs)[0]
        return dict(zip(pin_names, output))

    # Define tasks
    @staticmethod
    def layers_task(l, r, interface_type):
        return l, ModelTools._prop_all(l, interface_type(l, r))

    @staticmethod
    def _prop_all_wrapper(arg_list, result_list):
        return ModelTools._prop_all(*arg_list), result_list

    @staticmethod
    def _solve_modes_wrapper(mode_solver):
        mode_solver.solve()
        return mode_solver
