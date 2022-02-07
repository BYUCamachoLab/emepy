import numpy as np
from simphony import Model
from simphony.pins import Pin
from copy import deepcopy
from emepy.mode import Mode, Mode1D
from emepy.fd import ModeSolver1D

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
        self.activated_layer = None

    def activate_layer(self):
        """Solves for the modes in the layer and creates an ActivatedLayer object"""

        modes = []

        if type(self.mode_solvers) != list:
            self.mode_solvers.solve()
            for mode in range(self.num_modes):
                modes.append(self.mode_solvers.get_mode(mode))

        else:
            for index in range(len(self.mode_solvers)):
                self.mode_solvers[index][0].solve()
                for mode in range(self.mode_solvers[index][1]):
                    modes.append(self.mode_solvers[index][0].get_mode(mode))

        self.activated_layer = ActivatedLayer(modes, self.wavelength, self.length)

    def get_activated_layer(self):
        """Gets the activated layer if it exists or calls activate_layer first

        Returns
        -------
        ActivatedLayer
            the object that stores the eigenmodes
        """

        if self.activated_layer is None:
            self.activate_layer()

        return self.activated_layer

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
    def __init__(self, wavelength, modes, length, which_s=0, **kwargs):
        self.num_modes = len(modes)
        self.wavelength = wavelength
        self.modes = modes
        self.length = length
        self.which_s = which_s
        self.left_pins = ["left" + str(i) for i in range(self.num_modes)] + [
            "left_dup" + str(i) for i in range(self.num_modes * which_s)
        ]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)] + [
            "right_dup" + str(i) for i in range(self.num_modes * (1 - which_s))
        ]

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
        
        # Insert new rows and cols
        if not self.which_s:
            s_matrix = np.insert(s_matrix,[m],s_matrix[m:,:],axis=0)
            s_matrix = np.insert(s_matrix,[m],s_matrix[:,m:],axis=1)
        else:
            s_matrix = np.insert(s_matrix,[m],s_matrix[:m,:],axis=0)
            s_matrix = np.insert(s_matrix,[m],s_matrix[:,:m],axis=1)
        
        # Assign number of ports
        self.right_ports = m  # 2 * m - self.which_s * m
        self.left_ports = m  # 2 * m - (1 - self.which_s) * m
        self.num_ports = 2 * m  # 3 * m
        s_matrix = s_matrix.reshape(1,3*m,3*m)

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
        self.S1 = None
        if not n_only:
            self.normalize_fields()
            self.purge_spurious()
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

    def purge_spurious(self):
        """Purges all spurious modes in the dataset to prevent EME failure for high mode simulations"""

        for mode in range(len(self.modes))[::-1]:
            if self.modes[mode].check_spurious():
                self.modes.pop(mode)
                self.left_pins.pop(-1)
                self.right_pins.pop(-1)
                self.num_modes -= 1

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

    def __init__(self, left_modes, right_modes, s_params, n_only=False, **kwargs):
        """PeriodicLayer class constructor

        Parameters
        ----------
        left_modes : list [Mode]
            list of the eigenmodes on the left side of the layer
        right_modes : list [Mode]
            list of the eigenmodes on the right side of the layer
        s_params :
            the scattering matrix that represents the layer, which includes both propagation and mode overlaps
        """

        self.left_modes = left_modes
        self.right_modes = right_modes
        self.left_ports = len(self.left_modes)
        self.right_ports = len(self.right_modes)
        self.left_pins = ["left" + str(i) for i in range(len(self.left_modes))]
        self.right_pins = ["right" + str(i) for i in range(len(self.right_modes))]
        self.s_params = s_params
        if not n_only:
            self.normalize_fields()
            # self.purge_spurious()

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

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

        return self.s

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

        self.s = s.reshape((1, 2 * self.num_modes, 2 * self.num_modes))

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

        self.s = None


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

        return self.s

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

        self.s = s.reshape((1, self.num_ports, self.num_ports))

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