import numpy as np
from simphony import Model
from simphony.pins import Pin
from simphony.tools import wl2freq
from simphony.models import Subcircuit
from matplotlib import pyplot as plt
from emepy.monitors import Monitor


class Layer(object):
    """Layer objects form the building blocks inside of an EME or PeriodicEME. These represent geometric layers of rectangular waveguides that approximate continuous structures.
    """

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
        """Solves for the modes in the layer and creates an ActivatedLayer object
        """

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


class EME(object):
    """The EME class is the heart of the package. It provides the algorithm that cascades sections modes together to provide the s-parameters for a geometric structure. The object is dependent on the Layer objects that are fed inside.
    """

    def __init__(self, layers=[], num_periods=1):
        """EME class constructor

        Parameters
        ----------
        layers : list [Layer]
            An list of Layer objects, arranged in the order they belong geometrically. (default: [])

        """

        self.reset()
        self.layers = layers
        self.num_periods = num_periods
        self.s_params = None
        self.monitors = []

    def add_layer(self, layer):
        """The add_layer method will add a Layer object to the EME object. The object will be geometrically added to the very right side of the structure. Using this method after propagate is useless as the solver has already been called.

        Parameters
        ----------
        layer : Layer
            Layer object to be appended to the list of Layers inside the EME object.

        """

        self.layers.append(layer)

    def reset(self):
        """Clears out the layers and s params so the user can reuse the object in memory on a new geometry
        """

        self.layers = []
        self.interfaces = []
        self.wavelength = None
        self.s_params = None
        self.interface = None
        self.monitors = []

    def propagate_period(self):
        """The propagate_period method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters for ONE period of the structure. If num_periods is set to 1 (default), this method is the same as propagate, except for it returns values. 

        Returns
        -------
        s_params
            The s_params acquired during propagation
            
        mode_set1
            The set of Mode objects that were solved for on the input layer

        mode_set2
            The set of Mode objects that were solved for on the output layer
        """

        # Propagate the first two layers
        self.layers[0].activate_layer()
        mode_set1 = self.layers[0].get_activated_layer()
        self.layers[1].activate_layer()
        current = Current(self.wavelength, self.layers[0].get_activated_layer())
        
        # Update the monitors
        for m in range(len(self.monitors)):
            while self.monitors[m].remaining_lengths[0] < self.layers[0].length:
                _, y, _ = self.monitors[m].dimensions
                eigenvalue = (2 * np.pi) * mode_set1.modes[0].neff / (self.wavelength)
                phasor = np.exp(self.monitors[m].remaining_lengths[0] * 1j * eigenvalue)
                for c in range(len(self.monitors[m].components)):
                    self.monitors[m][c,0:y,self.monitors[m].cur_prop_index] = getattr(mode_set1.modes[0], self.monitors[m].components[c])[0] * phasor
            self.monitors[m].cur_length += self.layers[0].length

        interface = self.interface(self.layers[0].get_activated_layer(), self.layers[1].get_activated_layer())
        interface.solve()
        self.layers[0].clear()
        current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)
        interface.clear()

        # Propagate the middle layers
        for index in range(1, len(self.layers) - 1):

            layer1_ = self.layers[index]
            layer2_ = self.layers[index + 1]
            layer2_.activate_layer()

            layer1 = layer1_.get_activated_layer()
            layer2 = layer2_.get_activated_layer()

            # Update the monitors
            for m in range(len(self.monitors)):
                while self.monitors[m].remaining_lengths[0] < self.layers[index].length + self.monitors[m].cur_length:
                    _, y, _ = self.monitors[m].dimensions
                    eigenvalue = (2 * np.pi) * layer1.modes[0].neff / (self.wavelength)
                    phasor = np.exp((self.monitors[m].remaining_lengths[0]-self.monitors[m].cur_length) * 1j * eigenvalue)
                    phasor = phasor * current.s_parameters()[0,0,current.left_ports]
                    for c in range(len(self.monitors[m].components)):
                        self.monitors[m][c,0:y,self.monitors[m].cur_prop_index] = getattr(layer1.modes[0], self.monitors[m].components[c])[0] * phasor
                self.monitors[m].cur_length += self.layers[index].length

            interface = self.interface(layer1, layer2)
            interface.solve()

            current.update_s(self.cascade(Current(self.wavelength, current), layer1), layer1)
            layer1_.clear()
            current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)
            interface.clear()

        # Propagate final two layers
        current.update_s(
            self.cascade(Current(self.wavelength, current), self.layers[-1].get_activated_layer()),
            self.layers[-1].get_activated_layer(),
        )

        # Gather and return the s params and edge layers
        mode_set2 = self.layers[-1].get_activated_layer()

        # Update the monitors
        for m in range(len(self.monitors)):
            while self.monitors[m].remaining_lengths[0] < self.layers[-1].length + self.monitors[m].cur_length:
                _, y, _ = self.monitors[m].dimensions
                eigenvalue = (2 * np.pi) * mode_set2.modes[0].neff / (self.wavelength)
                phasor = np.exp((self.monitors[m].remaining_lengths[0]-self.monitors[m].cur_length) * 1j * eigenvalue)
                phasor = phasor * current.s_parameters()[0,0,current.left_ports]
                for c in range(len(self.monitors[m].components)):
                    self.monitors[m][c,0:y,self.monitors[m].cur_prop_index] = getattr(mode_set2.modes[0], self.monitors[m].components[c])[0] * phasor
            self.monitors[m].cur_length += self.layers[-1].length

        self.layers[-1].clear()
        self.s_params = current.s_params

        return (current.s_params, mode_set1, mode_set2)

    def propagate(self):
        """The propagate method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters.
        """

        # Check for layers
        if not len(self.layers):
            raise Exception("Must place layers before propagating")
        else:
            self.wavelength = self.layers[0].wavelength

        # Decide which routine to use
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Run the eme for one period and collect s params and edge layers
        single_period, left, right = self.propagate_period()
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Make memory
        self.layers[0].clear()
        self.layers[1].clear()

        # Cascade params for each period
        for _ in range(self.num_periods - 1):

            current_layer.s_params = self.cascade((current_layer), (interface))
            current_layer.s_params = self.cascade((current_layer), (period_layer))
            # current_layer.s_params = self.cascade(current_layer, period_layer)

        self.s_params = current_layer.s_params

    def cascade(self, first, second):
        """Calculates the s_parameters between two layer objects

        Parameters
        ----------
        first : Layer
            left Layer object
        second : Layer
            right Layer object

        Returns
        -------
        numpy array
            the s_parameters between the two Layers
        """

        Subcircuit.clear_scache()

        # make sure the components are completely disconnected
        first.disconnect()
        second.disconnect()

        # connect the components
        for port in range(first.right_ports):
            first[f"right{port}"].connect(second[f"left{port}"])

        # get the scattering parameters
        return first.circuit.s_parameters(np.array([self.wavelength]))

    def s_parameters(self, freqs=None):
        """Returns the s_parameters if they exist. If they don't exist yet, propagate() will be called first. 

        Returns
        -------
        numpy array
            The s_params acquired during propagation
        """

        if self.s_params is None:
            self.propagate()

        return self.s_params

    def get_total_length(self):
        return np.sum([layer.length for layer in self.layers])

    def add_monitor(self, axes="xz", dimensions=None, components=["E"]):
        """Creates a monitor associated with the eme object BEFORE the simulation is ran

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'xz', 'xyz', 'x', 'y', 'z'. Currently only 'xz' is implemented. Note, propagation is always in z. 
        dimensions : tuple
            the spacial dimensions of the resulting field. Note, if the dimensions are greater than the mesh in axes not along propagation, the fields will be interpolated. (default: mesh density of cross sections and 10 propagation points).
        components : list
            list of the field components to store from ('E','H','Ex','Ey','Ez','Hx','Hy','Hz)
        
        Returns
        -------
        Monitor
            the newly created Monitor object
        """

        if axes == "xz" or axes == "zx":
            if dimensions is None:
                y = self.layers[0].mode_solvers.mesh
                z = 100
                dimensions = [y,z]
                lengths = np.linspace(0,self.get_total_length(),z)
            monitor = Monitor(axes, tuple([len(components)]+dimensions), lengths, components)
        else:
            raise Exception("Monitor setup {} has not yet been implemented. Please choose from the following implemented monitor types: ['xz']".format(axes))
        
        self.monitors.append(monitor)
        return monitor


    def draw(self):
        """The draw method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. Currently, the drawing is very rough. This will change in the future. 
        """

        # Simple drawing to debug geometry
        plt.figure()
        lengths = [0.0] + [i.length for i in self.layers]
        lengths = [sum(lengths[: i + 1]) for i in range(len(lengths))]
        widths = [self.layers[0].mode_solvers.width] + [i.mode_solvers.width for i in self.layers]
        lengths_ = lengths
        widths_ = widths
        for i in range(1, self.num_periods):
            widths_ = widths_ + widths
            lengths_ = lengths_ + (np.array(lengths) + i * lengths[-1]).tolist()

        sub = [0] + np.diff(lengths_).tolist()
        lengths = []
        widths = []
        for i in range(len(lengths_)):
            lengths.append(lengths_[i] - sub[i])
            lengths.append(lengths_[i])
            widths.append(widths_[i])
            widths.append(widths_[i])
        plt.plot(lengths, np.array(widths), "b")
        plt.plot(lengths, -1 * np.array(widths), "b")
        plt.show()


class Current(Model):
    """The object that the EME uses to track the s_parameters and cascade them as they come along to save memory
    """

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
    """ActivatedLayer is produced by the Layer class after the ModeSolvers calculate eigenmodes. This is used to create interfaces. This inherits from Simphony's Model class. 
    """

    def __init__(self, modes, wavelength, length, **kwargs):
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
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(self.num_modes)]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)]
        self.s_params = self.calculate_s_params()

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def normalize_fields(self):
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1. 
        """

        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def calculate_s_params(self):
        """Calculates the s params for the phase propagation and returns it. 

        Returns
        -------
        numpy array
            the scattering matrix for phase propagation. 
        """

        eigenvalues1 = (2 * np.pi) * np.array([mode.neff for mode in self.modes]) / (self.wavelength)

        propagation_matrix1 = np.diag(
            np.exp(self.length * 1j * np.array(eigenvalues1.tolist() + eigenvalues1.tolist()))
        )
        num_cols = len(propagation_matrix1)
        rows = np.array(np.split(propagation_matrix1, num_cols))
        first_half = [rows[j + num_cols // 2][0] for j in range(num_cols // 2)]
        second_half = [rows[j][0] for j in range(num_cols // 2)]
        propagation_matrix = np.array(first_half + second_half).reshape((1, 2 * self.num_modes, 2 * self.num_modes))

        self.right_ports = self.num_modes
        self.left_ports = self.num_modes
        self.num_ports = 2 * self.num_modes

        return propagation_matrix

    def s_parameters(self, freq=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix 
        """

        return self.s_params


class PeriodicLayer(Model):
    """PeriodicLayer behaves similar to ActivatedLayer. However, this class can represent an entire geometry that is repeated in the periodic structure. It also gets constantly updated as it cascades through periods. 
    """

    def __init__(self, left_modes, right_modes, s_params, **kwargs):
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
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(len(self.left_modes))]
        self.right_pins = ["right" + str(i) for i in range(len(self.right_modes))]
        self.s_params = s_params

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def normalize_fields(self):
        """Normalizes all of the eigenmodes such that the overlap with its self, power, is 1. 
        """

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
    """The InterfaceSingleMode class represents the interface between two different layers. This class is an approximation to speed up the process and can ONLY be used during single mode EME. 
    """

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

    def s_parameters(self, freqs=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix 
        """

        return self.s

    def solve(self):
        """Solves for the scattering matrix based on transmission and reflection
        """

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
        """Clears the scattering matrix in the object
        """

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

    def s_parameters(self, freqs=None):
        """Returns the scattering matrix.

        Returns
        -------
        numpy array
            the scattering matrix 
        """

        return self.s

    def solve(self):
        """Solves for the scattering matrix based on transmission and reflection
        """

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
        """Clears the scattering matrix in the object
        """

        self.s_params = None

