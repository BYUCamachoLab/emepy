import numpy as np
from simphony import Model
from simphony.pins import Pin
from simphony.models import Subcircuit
from emepy.mode import Mode, Mode1D
from emepy.fd import ModeSolver1D
from emepy.materials import *
from copy import deepcopy


class Layer(object):
    """Layer objects form the building blocks inside of an EME or PeriodicEME. These represent geometric layers of rectangular waveguides that approximate continuous structures."""

    def __init__(self, mode_solvers, num_modes, wavelength, length):
        """Layer class constructor

        Parameters
        ----------
        mode_solvers : list [tuple (ModeSolver, int)], Modesolver
            List of tuples that contain ModeSolver objects and the number of modes that corresponds to each. Should be in order from fundamental mode to least significant mode. If only one ModeSolver is needed, can simply be that object instead of a list.
        num_modes : int
            Number of total modes for the layer.
        wavelength : number
            Wavelength of eigenmode to solve for (m).
        length : number
            Geometric length of the Layer (m). The length affects the phase of the eigenmodes inside the layer via the complex phasor $e^(jβz)$.
        """

        self.num_modes = num_modes
        self.mode_solvers = mode_solvers
        self.wavelength = wavelength
        self.length = length
        self.activated_layers = []


    def activate_layer(self, sources=[], start=0.0, period_length=0.0):
        """Solves for the modes in the layer and creates an ActivatedLayer object"""

        modes = []

        # Solve for modes 
        if type(self.mode_solvers) != list:
            self.mode_solvers.solve()
            for mode in range(self.num_modes):
                modes.append(self.mode_solvers.get_mode(mode))

        else:
            for index in range(len(self.mode_solvers)):
                self.mode_solvers[index][0].solve()
                for mode in range(self.mode_solvers[index][1]):
                    modes.append(self.mode_solvers[index][0].get_mode(mode))

        # Purge spurious mode
        modes = ModelTools.purge_spurious(modes)

        # Create activated layers
        self.activated_layers = dict(zip(sources.keys(), [[] for _ in range(len(sources.keys()))]))

        # Loop through all periods
        for per, srcs in sources.items():

            # Only care about sources between the ends
            start_ = start + per * period_length
            custom_sources = ModelTools.get_sources(srcs, start_, start_+self.length)

            # First period
            if not per:

                # If no custom sources
                if not len(custom_sources):
                    self.activated_layers[per] += [ActivatedLayer(modes, self.wavelength, self.length)]

                # Other sources
                else:
                    self.activated_layers[per] += ModelTools.get_source_system(modes, self.wavelength, self.length, custom_sources, start_)

            # Any other period
            else:

                # If no custom sources
                if not len(custom_sources):
                    self.activated_layers[per] += [None]

                # Other sources
                else:
                    self.activated_layers[per] += ModelTools.get_source_system(modes, self.wavelength, self.length, custom_sources, start_)

        return self.activated_layers


    def get_activated_layer(self, sources=[], start=0.0):
        """Gets the activated layer if it exists or calls activate_layer first

        Returns
        -------
        ActivatedLayer
            the object that stores the eigenmodes
        """

        if not len(self.activated_layers):
            self.activate_layer(sources=sources, start=start)

        return self.activated_layers

    def get_n_only(self):
        """Creates a psuedo layer for accessing the material only no fields
        """

        modes = []

        if type(self.mode_solvers) != list:
            for mode in range(self.mode_solvers.num_modes):
                if not isinstance(self.mode_solvers, ModeSolver1D):
                    modes.append(
                        Mode(self.mode_solvers.x, self.mode_solvers.y, self.mode_solvers.wl, n=self.mode_solvers.n)
                    )
                else:
                    modes.append(Mode1D(self.mode_solvers.x, self.mode_solvers.wl, n=self.mode_solvers.n))

        else:
            for index in range(len(self.mode_solvers)):
                for mode in range(self.mode_solvers[index][1]):
                    if not isinstance(self.mode_solvers[index][0], ModeSolver1D):
                        modes.append(
                            Mode(
                                self.mode_solvers[index][0].x,
                                self.mode_solvers[index][0].y,
                                self.mode_solvers[index][0].wl,
                                n=self.mode_solvers[index][0].n,
                            )
                        )
                    else:
                        modes.append(
                            Mode1D(
                                self.mode_solvers[index][0].x,
                                self.mode_solvers[index][0].wl,
                                n=self.mode_solvers[index][0].n,
                            )
                        )

        return ActivatedLayer(modes, self.wavelength, self.length, n_only=True)

    def clear(self):
        """Empties the modes in the ModeSolver to clear memory

        Returns
        -------
        numpy array
            the edited image
        """

        if type(self.mode_solvers) != list:
            self.mode_solvers.clear()
        else:
            for index in range(len(self.mode_solvers)):
                self.mode_solvers[index][0].clear()



class Duplicator(Model):
    def __init__(self, wavelength, modes, label="", **kwargs):
        self.modes = modes
        self.num_modes = len(modes)
        self.wavelength = wavelength

        self.left_pins = ["left" + str(i) for i in range(self.num_modes)] + [
            "left_dup{}{}".format(str(i),label) for i in range(self.num_modes)
        ] 
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)] + [
            "right_dup{}{}".format(str(i),label) for i in range(self.num_modes)
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
            if not "dup" in name:
                pins.append(Pin(self, name))
        self.pins = pins
        super().__init__(**kwargs, pins=pins)
        self.s_params = self.calculate_s_params()

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":

        return self.s_params

    def calculate_s_params(self):
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
        s_matrix_new = np.zeros((4*m,4*m),dtype=complex)
        s_matrix_new[:m,3*m:] = s_matrix[:m,m:]
        s_matrix_new[m:2*m,3*m:] = s_matrix[:m,m:]
        s_matrix_new[3*m:,m:2*m] = s_matrix[:m,m:]
        s_matrix_new[:m,2*m:3*m] = s_matrix[m:,:m]
        s_matrix_new[2*m:3*m,:m] = s_matrix[m:,:m]
        s_matrix_new[3*m:,:m] = s_matrix[m:,:m]
        s_matrix = s_matrix_new

        # Assign number of ports
        self.right_ports = m  # 2 * m - self.which_s * m
        self.left_ports = m  # 2 * m - (1 - self.which_s) * m
        self.num_ports = 2 * m  # 3 * m
        s_matrix = s_matrix.reshape(1,4*m,4*m)

        return s_matrix


class Current(Model):
    """The object that the EME uses to track the s_parameters and cascade them as they come along to save memory"""

    def __init__(self, wavelength, s, **kwargs):
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

    def update_s(self, s, layer):
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

    def s_parameters(self, freq=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """
        return self.s_params


class ActivatedLayer(Model):
    """ActivatedLayer is produced by the Layer class after the ModeSolvers calculate eigenmodes. This is used to create interfaces. This inherits from Simphony's Model class."""

    def __init__(self, modes, wavelength, length, n_only=False, **kwargs):
        """ActivatedLayer class constructor

        Parameters
        ----------
        modes : list [Mode]
            list of solved eigenmodes in Mode class form
        wavelength : number
            the wavelength of the eigenmodes
        length : number
            the length of the layer object that produced the eigenmodes. This number is used for phase propagation.
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

        super().__init__(**kwargs, pins=pins)

    def normalize_fields(self):
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1."""

        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def calculate_s_params(self):
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

    def s_parameters(self, freq=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """

        return self.s_params


class PeriodicLayer(Model):
    """PeriodicLayer behaves similar to ActivatedLayer. However, this class can represent an entire geometry that is repeated in the periodic structure. It also gets constantly updated as it cascades through periods."""

    def __init__(self, left_modes, right_modes, model, n_only=False, **kwargs):
        """PeriodicLayer class constructor

        Parameters
        ----------
        left_modes : list [Mode]
            list of the eigenmodes on the left side of the layer
        right_modes : list [Mode]
            list of the eigenmodes on the right side of the layer
        model :
            the scattering matrix model that represents the layer, which includes both propagation and mode overlaps
        """

        """
            TODO:
            Add capabilities to have the mode source somewhere arbitrarily in the periodic structure
            Test the whole thing
            Reformat this class to allow custom source ports in addition to the rest without adding infinitely many new pins upon cascading the same period over and over
        """

        self.left_modes = left_modes
        self.right_modes = right_modes
        self.left_ports = len(self.left_modes)
        self.right_ports = len(self.right_modes)
        self.left_pins = ["left" + str(i) for i in range(len(self.left_modes))]
        self.right_pins = ["right" + str(i) for i in range(len(self.right_modes))]
        self.s_params = model.s_parameters([0])
        if not n_only:
            self.normalize_fields()
            # self.purge_spurious()

        # create the pins for the model
        # pins = []
        # for name in self.left_pins:
        #     pins.append(Pin(self, name))
        # for name in self.right_pins:
        #     pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=model.pins)

    def purge_spurious(self):
        """Purges all spurious modes in the dataset to prevent EME failure for high mode simulations"""

        for mode in range(len(self.left_modes))[::-1]:
            if self.left_modes[mode].check_spurious():
                self.left_modes.pop(mode)
                self.left_pins.pop(-1)

        for mode in range(len(self.right_modes))[::-1]:
            if self.right_modes[mode].check_spurious():
                self.right_modes.pop(mode)
                self.right_pins.pop(-1)

    def normalize_fields(self):
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1."""

        for mode in range(len(self.left_modes)):
            self.left_modes[mode].normalize()
        for mode in range(len(self.right_modes)):
            self.right_modes[mode].normalize()

    def s_parameters(self, freqs=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """

        return self.s_params


class InterfaceSingleMode(Model):
    """The InterfaceSingleMode class represents the interface between two different layers. This class is an approximation to speed up the process and can ONLY be used during single mode EME."""

    def __init__(self, layer1, layer2, **kwargs):
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

    def s_parameters(self, freqs=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """

        return self.s_params

    def solve(self):
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

    def get_values(self, left, right):
        """Returns the reflection and transmission coefficient based on the two modes

        Parameters
        ----------
        left : Mode
            leftside eigenmode
        right : Mode
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

    def clear(self):
        """Clears the scattering matrix in the object"""

        self.s_params = None


class InterfaceMultiMode(Model):
    """
    The InterfaceMultiMode class represents the interface between two different layers.
    """

    def __init__(self, layer1, layer2, **kwargs):
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
        self.num_ports = layer1.right_ports + layer2.left_ports
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
        self.solve()

    def s_parameters(self, freqs=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix
        """

        return self.s_params

    def solve(self):
        """Solves for the scattering matrix based on transmission and reflection"""

        s = np.zeros((self.num_ports, self.num_ports), dtype=complex)

        for p in range(self.left_ports):

            ts = self.get_t(p, self.layer1, self.layer2, self.left_ports)
            rs = self.get_r(p, ts, self.layer1, self.layer2, self.left_ports)

            for t in range(len(ts)):
                s[self.left_ports + t][p] = ts[t]
            for r in range(len(rs)):
                s[r][p] = rs[r]

        for p in range(self.right_ports):

            ts = self.get_t(p, self.layer2, self.layer1, self.right_ports)
            rs = self.get_r(p, ts, self.layer2, self.layer1, self.right_ports)

            for t in range(len(ts)):
                s[t][self.left_ports + p] = ts[t]
            for r in range(len(rs)):
                s[self.left_ports + r][self.left_ports + p] = rs[r]

        self.s_params = s.reshape((1, self.num_ports, self.num_ports))

    def get_t(self, p, left, right, curr_ports):
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
        t : number
            transmission coefficient
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

    def get_r(self, p, x, left, right, curr_ports):
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

    def clear(self):
        """Clears the scattering matrix in the object"""

        self.s_params = None

class SourceDuplicator(Model):
    def __init__(self, wavelength, modes, length, pk=[],nk=[], label="", special_left=[], special_right=[], **kwargs):
        self.num_modes = len(modes)
        self.wavelength = wavelength
        self.modes = modes
        self.length = length
        self.pk = pk
        self.nk = nk
        self.normalize_fields()

        self.left_pins = ["left" + str(i) for i in range(self.num_modes)] + [
            "left_dup{}{}".format(str(i),label) for i in range(len(pk)*(not len(special_left)))
        ] + special_left
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)] + [
            "right_dup{}{}".format(str(i),label) for i in range(len(nk)*(not len(special_right)))
        ] + special_right
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
            if not "dup" in name:
                pins.append(Pin(self, name))
        self.pins = pins
        super().__init__(**kwargs, pins=pins)
        self.s_params = self.calculate_s_params()

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":

        return self.s_params

    def normalize_fields(self):
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1."""

        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def calculate_s_params(self):
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
        s_matrix_new = np.zeros((4*m,4*m),dtype=complex)
        s_matrix_new[:m,3*m:] = s_matrix[:m,m:]
        s_matrix_new[m:2*m,3*m:] = s_matrix[:m,m:]
        s_matrix_new[3*m:,m:2*m] = s_matrix[:m,m:]
        s_matrix_new[:m,2*m:3*m] = s_matrix[m:,:m]
        s_matrix_new[2*m:3*m,:m] = s_matrix[m:,:m]
        s_matrix_new[3*m:,:m] = s_matrix[m:,:m]
        # s_matrix_new[m:3*m,m:3*m] = s_matrix[:,:]
        s_matrix = s_matrix_new

        # Delete rows and cols
        pk_remove = m-len(self.pk)
        nk_remove = m-len(self.nk)
        for _ in range(nk_remove):
            s_matrix = np.delete(s_matrix,-m-1,axis=0)
            s_matrix = np.delete(s_matrix,-m-1,axis=1)
        for _ in range(pk_remove):
            s_matrix = np.delete(s_matrix,-m-len(self.nk)-1,axis=0)
            s_matrix = np.delete(s_matrix,-m-len(self.nk)-1,axis=1)

        # Assign number of ports
        self.right_ports = m  # 2 * m - self.which_s * m
        self.left_ports = m  # 2 * m - (1 - self.which_s) * m
        self.num_ports = 2 * m  # 3 * m
        s_matrix = s_matrix.reshape(1,2*m + len(self.pk) + len(self.nk),2*m + len(self.pk) + len(self.nk))

        return s_matrix


class CopyModel(Model):

    def __init__(self, model, **kwargs):

        self.num_modes = model.num_modes if hasattr(model,"num_modes") else None
        self.modes = model.modes if hasattr(model,"modes") else []
        self.wavelength = model.wavelength if hasattr(model,"wavelength") else None
        self.length = model.length if hasattr(model,"length") else None
        self.left_ports = model.left_ports if hasattr(model,"left_ports") else []
        self.right_ports = model.right_ports if hasattr(model,"right_ports") else []

        self.S0 = make_copy_model(model.S0) if hasattr(model,"S0") else None
        self.nk = model.nk if hasattr(model,"nk") else []
        self.pk = model.pk if hasattr(model,"pk") else []
        self.pins = self.copy_pins(model.pins) if hasattr(model,"pins") else []
        self.left_pins = model.left_pins if hasattr(model,"left_pins") else [] #[self.find_pin(pin) for pin in model.left_pins] if hasattr(model,"left_pins") else []
        self.right_pins = model.right_pins if hasattr(model,"right_pins") else [] #[self.find_pin(pin) for pin in model.right_pins] if hasattr(model,"right_pins") else []
        self.s_params = model.s_parameters([0])
        
        super().__init__(**kwargs, pins=self.pins)
        return

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":

        return self.s_params

    def copy_pins(self, pins):

        return [Pin(self, p.name) for p in pins]

    def find_pin(self, pin):
        for p in self.pins:
            if p.name == pin.name:
                return p 
        return Pin(self, pin.name)

def make_copy_model(model):
    new_model = CopyModel(model) if not model is None else None
    return new_model

def compute(model, pin_values: "dict", freq : "float") -> "np.ndarray":
    """Takes a dictionary mapping each pin name to a coefficent and multiplies by the S matrix"""
    cfs = np.zeros(len(model.pins), dtype=complex)
    pin_names = [i.name for i in model.pins]
    key_index = dict(zip(pin_names, [pin_names.index(i) for i in pin_names]))
    for key, value in pin_values.items():
        if key in key_index:
            cfs[key_index[key]] = value
    matrix = model.s_parameters(np.array([freq]))
    output = np.matmul(matrix, cfs)[0]
    return dict(zip(pin_names, output))

class ModelTools(object):

    @staticmethod
    def purge_spurious(modes):
        """Purges all spurious modes in the dataset to prevent EME failure for high mode simulations"""

        mm = deepcopy(modes)
        for i, mode in enumerate(mm[::-1]):
            if mode.check_spurious():
                mm.remove(mode)
        
        return mm

    @staticmethod
    def get_sources(sources=[], start=0.0, end=0.0):
        return [i for i in sources if not i.z is None and (((start <= i.z < end) and i.k) or ((start < i.z <= end) and not i.k ))]    

    @staticmethod
    def get_source_system(modes, wavelength, length, custom_sources, start=0.0):

        # Get lengths between components
        lengths = np.diff([start] + [i.z for i in custom_sources] + [start+length])
        dups = []

        # Enumerate through all lengths
        length_tracker = start
        for i, length in enumerate(lengths):

            # Create coefficents indexes to keep in the models
            pk, nk = [[],[]]
            if (i-1) > -1 and custom_sources[i-1].k:
                pk = custom_sources[i-1].mode_coeffs
            if (i) < len(custom_sources) and not custom_sources[i].k:
                nk = custom_sources[i].mode_coeffs

            # Create label
            left = custom_sources[i-1].get_label() if (i-1) > -1 else "n"+str(length_tracker)
            right = custom_sources[i].get_label() if (i) < len(custom_sources) else "n"+str(length_tracker+length)
            label = "_{}_to_{}".format(left,right)

            # Create duplicators
            dups.append(SourceDuplicator(wavelength,modes,length,pk=pk,nk=nk,label=label))
            length_tracker += length

        return dups

    @staticmethod
    def _prop_all(*args):
        layers = [make_copy_model(a) for a in args if not a is None]
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

        return temp_s

    @staticmethod
    def periodic_duplicate_format(model, start, end):
        model = make_copy_model(model)
        wanted = [i for i, pin in enumerate(model.pins) if "dup" in pin.name]
        indices = [i for i, pin in enumerate(model.pins) if "dup" in pin.name]
        model.s_params = model.s_parameters([0])
        
        # Find wanted indices
        for i in wanted[::-1]:
            if not "_to_" in model.pins[i].name:
                pass
            elif "left" in model.pins[i].name:
                n = model.pins[i].name
                n_ = n.split("_")
                l, _ = (float(n_[2][1:]),float(n_[4][1:]))
                if start <= l < end:
                    continue
            else:
                n = model.pins[i].name
                n_ = n.split("_")
                _, r = (float(n_[2][1:]),float(n_[4][1:]))
                if start < r <= end:
                    continue
            wanted.remove(i)

        # Remove unwanted rows and columns
        for i in indices:    
            if not i in wanted:
                model.s_params = np.delete(model.s_params, i, 1)
                model.s_params = np.delete(model.s_params, i, 2)

        # Remove unwanted pins
        model.pins = [i for j, i in enumerate(model.pins) if j in wanted or "dup" not in i.name]
        pin_names = [i.name for i in model.pins]
        model.left_pins = [i for i in model.left_pins if i in pin_names]
        model.right_pins = [i for i in model.right_pins if i in pin_names]

        return model