import numpy as np
from tqdm import tqdm
from simphony import Model
from simphony.pins import Pin
from simphony.models import Subcircuit
from matplotlib import pyplot as plt
from emepy.monitors import Monitor
from copy import copy
from emepy.mode import Mode, Mode1D
from emepy.fd import MSEMpy, ModeSolver1D


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


class EME(object):
    """The EME class is the heart of the package. It provides the algorithm that cascades sections modes together to provide the s-parameters for a geometric structure. The object is dependent on the Layer objects that are fed inside."""

    states = {
        0: "start",
        1: "mode_solving",
        2: "finished_modes",
        3: "forward_propagating",
        5: "cascading_forward_periods",
        6: "finished_forward",
        7: "reverse_propagating",
        8: "cascading_reverse_periods",
        9: "finished_reverse",
        10: "field_propagating",
        11: "finished",
    }

    def __init__(self, layers=[], num_periods=1):
        """EME class constructor

        Parameters
        ----------
        layers : list [Layer]
            An list of Layer objects, arranged in the order they belong geometrically. (default: [])

        """

        self.reset()
        self.layers = layers[:]
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

    def reset(self, full_reset=True):
        """Clears out the layers and s params so the user can reuse the object in memory on a new geometry"""

        # Erase all information except number of periods
        if full_reset:
            self.layers = []
            self.wavelength = None
            self.update_state(0)

        # Only unsolve everything and erase monitors
        else:
            for i in range(len(self.layers)):
                self.layers[i].clear()
            self.update_state(2)

        self.s_params = None
        self.interface = None
        self.monitors = []

    def solve_modes(self):
        # Check if already solved
        if self.state > 1:
            return

        # Hold still while solving finishes
        while self.state == 1:
            continue

        # Solve modes
        self.update_state(1)
        for layer in tqdm(self.layers):
            layer.activate_layer()
        self.update_state(2)

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
        left, right = (self.layers[0].get_activated_layer(), self.layers[1].get_activated_layer())
        current = Current(self.wavelength, left)
        interface = self.interface(left, right)
        current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)

        # Assign to layer the right sub matrix
        if self.state == 3:
            right.S0 = copy(current)
        elif self.state == 7:
            right.S1 = copy(current)

        # Propagate the middle layers
        for index in tqdm(range(1, len(self.layers) - 1)):

            # Get layers
            layer1 = self.layers[index].get_activated_layer()
            layer2 = self.layers[index + 1].get_activated_layer()

            # Propagate layers together
            interface = self.interface(layer1, layer2)
            current.update_s(self.cascade(Current(self.wavelength, current), layer1), layer1)
            current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)

            # Assign to layer the right sub matrix
            if self.state == 3:
                layer2.S0 = copy(current)
            elif self.state == 7:
                layer2.S1 = copy(current)

        # Propagate final two layers
        current.update_s(
            self.cascade(Current(self.wavelength, current), self.layers[-1].get_activated_layer()),
            self.layers[-1].get_activated_layer(),
        )

        # Assign to layer the right sub matrix
        if self.state == 3:
            self.layers[-1].get_activated_layer().S0 = copy(current)
        elif self.state == 7:
            self.layers[-1].get_activated_layer().S1 = copy(current)

        # Finalize s params
        self.s_params = current.s_params
        for m in range(len(self.monitors)):
            self.monitors[m].normalize()
        return current.s_params

    def propagate_n_only(self):

        # Propagate
        mode_set1 = self.layers[0].get_n_only()
        self.update_monitors(self.layers[0].length, self.layers[0].length, mode_set1, not_first=False, input_array=None)

        # Propagate the middle layers
        for index in tqdm(range(1, len(self.layers) - 1)):

            layer1_ = self.layers[index]
            layer1 = layer1_.get_n_only()
            self.update_monitors(self.layers[index].length, self.layers[index].length, layer1, None, input_array=None)

        mode_set2 = self.layers[-1].get_n_only()
        self.update_monitors(self.layers[-1].length, self.layers[-1].length, mode_set2, None, input_array=None)

        self.interface = InterfaceMultiMode

        # Cascade params for each period
        return self.cascade_periods_n_only()

    def cascade_periods_n_only(self):
        
        # Cascade all periods
        for l in range(self.num_periods - 1):
            self.update_monitors(
                None,
                self.get_total_length(),
                None,
                curr_s=None,
                not_first=True,
                input_array=None,
                periodic=l,
            )

    def cascade_periods(self, current_layer, interface, period_layer):

        # Cascade params for each period
        for l in range(self.num_periods - 1):

            current_layer.s_params = self.cascade((current_layer), (interface))
            current_layer.s_params = self.cascade((current_layer), (period_layer))

        return current_layer.s_params

    def forward_pass(self, input_array):

        # Start forward
        self.update_state(3)

        # Decide routine
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.layers[0].get_activated_layer()
        right = self.layers[-1].get_activated_layer()
        single_period = self.propagate_period()
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self.update_state(5)
        s_params = self.cascade_periods(current_layer, interface, period_layer)

        # Finds the output given the input vector
        while input_array.shape[0] > current_layer.s_params[0].shape[0]:
            input_array = input_array[:-1]
        if input_array.shape[0] < current_layer.s_params[0].shape[0]:
            temp = np.zeros(current_layer.s_params[0].shape[0])
            temp[: input_array.shape[0]] = input_array
            input_array = temp
        self.output = np.matmul(current_layer.s_params[0], input_array)
        self.update_state(6)

        return s_params

    def update_state(self, state):

        self.state = state
        print("current state: {}".format(self.states[self.state]))

    def reverse_pass(self):
        # Assign state
        self.update_state(7)

        # Reverse geometry
        self.layers = self.layers[::-1]

        # Decide routine
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.layers[0].get_activated_layer()
        right = self.layers[-1].get_activated_layer()
        single_period = self.propagate_period()
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self.update_state(8)
        s_params = self.cascade_periods(current_layer, interface, period_layer)

        # Fix geometry
        self.layers = self.layers[::-1]

        return s_params

    def build_input_array(self, input_left, input_middle, input_right):

        # Case 1: input_left > num_modes
        if len(input_left) > self.layers[0].num_modes:
            raise Exception("Too many mode coefficients in the left input")

        # Case 2: input_middle > num_sources
        if len(input_middle):
            raise Exception("Too many mode coefficients in the middle sources")

        # Case 3: input_right > num_modes
        if len(input_left) > self.layers[-1].num_modes:
            raise Exception("Too many mode coefficients in the right input")    

        # Fill input_left
        while len(input_left) < self.layers[0].num_modes:
            input_left += [0]

        # Fill input_middle
        input_middle += []

        # Fill input_right
        while len(input_right) < self.layers[-1].num_modes:
            input_right += [0]

        # Build forward input_array
        forward = np.array(input_left + input_middle + input_right)

        # Built reverse input_array
        reverse = np.array(input_right + input_middle + input_left)

        return forward, reverse

    def field_propagate(self, input_array):
        # Start state
        self.update_state(10)

        # Update all monitors
        for m in tqdm(self.monitors):
            continue

        # Finish state
        self.update_state(11)

    def propagate(self, input_left=[1], input_right=[0]):
        """The propagate method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters.

        Parameters
        ----------
        input_array : numpy array
            the array representing the input to the device in s parameter format: [left port mode 1, left port mode 2, ... left port mode n, right port mode 1, right port mode 2, ... right port mode n] (default : [1,0,0,...]) the default represents sending in the fundamental mode of the left port
        """

        # Format input array
        forward, reverse = self.build_input_array(input_left, [], input_right)

        # Check for layers
        if not len(self.layers):
            raise Exception("Must place layers before propagating")
        else:
            self.wavelength = self.layers[0].wavelength

        # Solve for the modes
        self.solve_modes()

        # Forward pass
        self.forward_pass(forward)

        # Reverse pass
        self.reverse_pass()

        # Update monitors
        self.field_propagate(forward)

    def get_field(self, mode_set, m, c, curr_s, input_array):
        # Index doesn't need s param involvement
        if self.monitors[m].components[c] == "n":
            field = getattr(mode_set.modes[0], self.monitors[m].components[c])
        else:
            # Eigenvalues of all the modes
            eigenvalues = np.array(
                [(2 * np.pi) * mode_set.modes[i].neff / (self.wavelength) for i in range(len(mode_set.modes))]
            ).astype(np.complex128)

            # Phase propagation from the last interfact until the current position
            phase_prop = np.exp(
                (self.monitors[m].remaining_lengths[c][0] - self.monitors[m].cur_length[c]) * 1j * eigenvalues
            )

            # S parameters from mode overlaps and phase prop from previous layers
            if not (curr_s is None):
                while input_array.shape[0] > curr_s.s_parameters()[0].shape[0]:  # Temp solution, deal with later
                    input_array = input_array[:-1]
                if input_array.shape[0] < curr_s.s_parameters()[0].shape[0]:
                    temp = np.zeros(curr_s.s_parameters()[0].shape[0])
                    temp[: input_array.shape[0]] = input_array
                    input_array = temp
                interface_prop = np.matmul(curr_s.s_parameters()[0], input_array)[-len(phase_prop) :]
            else:
                interface_prop = input_array[: len(phase_prop)]

            # Coefficients of each mode
            mode_coefficients = phase_prop * interface_prop

            # Combine the coefficient defined linear combination of modes to get the field profile
            field = getattr(mode_set.modes[0], self.monitors[m].components[c]) * mode_coefficients[0]
            for i in range(1, len(mode_set.modes)):
                field += getattr(mode_set.modes[i], self.monitors[m].components[c]) * mode_coefficients[i]

        return field

    def update_monitors(
        self, location_bonus, adder, mode_set, curr_s=None, not_first=True, input_array=None, periodic=None
    ):

        # Update the monitors
        for m in range(len(self.monitors)):  # Loop the monitors
            for c in range(len(self.monitors[m].components)):  # Loop the field components

                if not (periodic is None):
                    location = (periodic + 2) * self.get_total_length()
                elif not_first:
                    location = location_bonus + self.monitors[m].cur_length[c]
                else:
                    location = location_bonus

                if self.monitors[m].axes in ["xy", "yx"]:

                    # Check if not correct location
                    if (
                        self.monitors[m].cur_length[c]
                        <= self.monitors[m].lengths[c][0]
                        <= self.monitors[m].cur_length[c] + adder
                    ):

                        # Case when repeating layers
                        if not (periodic is None):
                            mode_set = self.monitors[m].layers[0]

                        # Get the field
                        field = self.get_field(mode_set, m, c, curr_s, input_array)

                        self.monitors[m][c, :, :] = field

                    elif periodic is None:
                        # If periodic and the correct length is
                        for n in range(self.num_periods):
                            if (
                                self.monitors[m].cur_length[c] + n * self.get_total_length()
                                <= self.monitors[m].lengths[0]
                                <= self.monitors[m].cur_length[c] + adder + n * self.get_total_length()
                            ):
                                self.monitors[m].layers[0] = mode_set
                else:
                    while (
                        len(self.monitors[m].remaining_lengths[c])
                        and self.monitors[m].remaining_lengths[c][0] < location
                    ):
                        dims = self.monitors[m].dimensions
                        x = dims[1]
                        y = dims[1]
                        if len(dims) == 4:
                            y = dims[2]

                        # Case when repeating layers
                        if not (periodic is None):
                            mode_set = self.monitors[m].layers[
                                round(
                                    self.monitors[m].remaining_lengths[c][0] - (periodic + 1) * self.get_total_length(),
                                    14,
                                )
                            ]

                        # Get the field
                        field = self.get_field(mode_set, m, c, curr_s, input_array)

                        # Only care about the area of the grid of concern
                        if self.monitors[m].axes == "xz":
                            field = field[:, int(len(field) / 2)] if field.ndim > 1 else field[:]
                        elif self.monitors[m].axes == "yz":
                            field = field[int(len(field) / 2), :] if field.ndim > 1 else field[:]
                        else:
                            field = field

                        # Save solved field profiles for referencing from repeated layers
                        if periodic is None:
                            self.monitors[m].layers[round(self.monitors[m].remaining_lengths[c][0], 14)] = copy(
                                mode_set
                            )

                        # Finally, update the monitor
                        if self.monitors[m].axes in ["xz", "yz"]:
                            self.monitors[m][c, 0:y, self.monitors[m].cur_prop_index[c]] = field
                        else:
                            self.monitors[m][c, 0:x, 0:y, self.monitors[m].cur_prop_index[c]] = field

                # Update the current length
                self.monitors[m].cur_length[c] += adder

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

    def add_monitor(self, axes="xz", mesh_z=200, z_range=None, location=None, components=None, excempt=True):
        """Creates a monitor associated with the eme object BEFORE the simulation is ran

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'xz', 'xyz', 'x', 'y', 'z'. Currently only 'xz' is implemented. Note, propagation is always in z.
        mesh_z : int
            number of mesh points in z (for periodic structures, will be z * num_periods)
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        location : float
            z coordinate where to save data for a 'xy' monitor

        Returns
        -------
        Monitor
            the newly created Monitor object
        """

        # Establish mesh # Current workaround...not final
        components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"] if components is None else components
        if excempt and (self.layers[0].mode_solvers.PML and isinstance(self.layers[0].mode_solvers, MSEMpy)):
            x = (
                len(
                    self.layers[0].mode_solvers.x[
                        self.layers[0].mode_solvers.nlayers[1] : -self.layers[0].mode_solvers.nlayers[0]
                    ]
                )
                + 1
            )
            y = (
                len(
                    self.layers[0].mode_solvers.y[
                        self.layers[0].mode_solvers.nlayers[3] : -self.layers[0].mode_solvers.nlayers[2]
                    ]
                )
                + 1
            )
        else:
            x = len(self.layers[0].mode_solvers.after_x)
            y = len(self.layers[0].mode_solvers.after_y)

        # Ensure the axes is not still under development
        if axes in ["xz", "zx", "yz", "zy", "xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:

            # Create lengths
            l = self.get_total_length()
            single_lengths = np.linspace(0, l, mesh_z, endpoint=False).tolist()
            lengths = []
            for i in range(self.num_periods):
                lengths += [np.array(j) + i * l for j in single_lengths]
            lengths = [lengths for i in range(len(components))]

            # Ensure z range is in proper format
            try:
                if z_range is None:
                    start, end = [lengths[0][0], lengths[0][-1]]
                else:
                    start, end = z_range
            except Exception as e:
                raise Exception(
                    "z_range should be a tuple or list of the form (start, end) "
                    "representing the range of the z values to extract where start"
                    "and end are floats such as (0, 1e-6) for a 1 µm range"
                ) from e

            # Fix z mesh if changed by z_range
            difference_start = lambda list_value: abs(list_value - start)
            difference_end = lambda list_value: abs(list_value - end)
            s = min(lengths[0], key=difference_start)
            e = min(lengths[0], key=difference_end)
            z = np.sum((s <= lengths[0]) * (e >= lengths[0]))

            # Create monitor dimensions
            c = len(components)
            if axes in ["xz", "yz"]:
                dimensions = (c, x, z)
            elif axes in ["yz", "zy"]:
                dimensions = (c, y, z)
            elif axes in ["xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
                dimensions = (c, x, y, z)
            elif axes in ["xy", "yx"]:
                dimensions = (c, x, y)
            else:
                raise Exception("Improper axes format")

            # Create grids
            grid_x = self.layers[0].mode_solvers.after_x
            grid_y = self.layers[0].mode_solvers.after_y
            grid_z = np.linspace(s, e, z)

        elif axes in ["xy", "yx"]:

            # No z_range needed
            z_range = None

            # Create single length
            lengths = [[location] for _ in range(len(components))]

            # Create monitor dimensions
            c = len(components)
            dimensions = (c, x, y)

            # Create grids
            grid_x = self.layers[0].mode_solvers.x
            grid_y = self.layers[0].mode_solvers.y
            grid_z = np.array([location])

        else:
            raise Exception(
                "Monitor setup {} has not yet been implemented. Please choose from the following implemented monitor types: ['xz', 'yz', 'xyz']".format(
                    axes
                )
            )

        # Create monitor
        monitor = Monitor(axes, dimensions, lengths, components, z_range, grid_x, grid_y, grid_z)
        self.monitors.append(monitor)
        return monitor

    def draw(self, z_range=None):
        """The draw method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. This will change in the future."""

        temp_storage = self.monitors
        self.monitors = []
        monitor = self.add_monitor(axes="xz", components=["n"], z_range=z_range, excempt=False)
        self.propagate_n_only()
        im = monitor.visualize(component="n")
        self.monitors = temp_storage
        return im


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
        self.s_params = self.calculate_s_params()

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":

        return self.s_params

    def calculate_s_params(self):
        # Create template for final s matrix
        m = self.num_modes
        s_matrix = np.zeros((1, 4 * m, 4 * m), dtype=complex)

        # Create eigenvalue vector
        eigenvalues1 = (2 * np.pi) * np.array([mode.neff for mode in self.modes * 4]) / (self.wavelength)

        # Create propagation diagonal matrix
        propagation_matrix1 = np.diag(np.exp(self.length * 1j * eigenvalues1))

        # Create sub matrix
        temp_s_matrix = np.zeros((2 * m, 2 * m), dtype=complex)
        temp_s_matrix[0:m, 0:m] = propagation_matrix1[m : 2 * m, 0:m]
        temp_s_matrix[m : 2 * m, 0:m] = propagation_matrix1[0:m, 0:m]
        temp_s_matrix[0:m, m : 2 * m] = propagation_matrix1[m : 2 * m, m : 2 * m]
        temp_s_matrix[m : 2 * m, m : 2 * m] = propagation_matrix1[0:m, m : 2 * m]

        # Create final matrix
        s_matrix[0, 0 : 2 * m, 0 : 2 * m] = temp_s_matrix
        s_matrix[0, 2 * m : 4 * m, 0 : 2 * m] = temp_s_matrix
        s_matrix[0, 0 : 2 * m, 2 * m : 4 * m] = temp_s_matrix
        s_matrix[0, 2 * m : 4 * m, 2 * m : 4 * m] = temp_s_matrix

        # Eliminate unnecessary row and column
        starting_rowcol = 2 * m + self.which_s * m
        ending_rowcol = starting_rowcol + m
        s_matrix[0, starting_rowcol:ending_rowcol, :] = 0
        s_matrix[0, :, starting_rowcol:ending_rowcol] = 0

        # Assign number of ports
        self.right_ports = 2 * m - self.which_s * m
        self.left_ports = 2 * m - (1 - self.which_s) * m
        self.num_ports = 3 * m

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
