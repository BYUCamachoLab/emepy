import numpy as np
from tqdm import tqdm
from simphony import Model
from simphony.pins import Pin
from simphony.models import Subcircuit
from matplotlib import pyplot as plt
from emepy.monitors import Monitor
from copy import deepcopy
from emepy.mode import Mode, Mode1D
from emepy.fd import MSEMpy, ModeSolver1D
import time


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
        self.forward_periodic_s = []
        self.reverse_periodic_s = []

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

        # See if only one layer or no layer
        if not len(self.layers):
            raise Exception("No layers in system")
        elif len(self.layers) == 1:
            return self.layers[0].get_activated_layer().s_params

        # Propagate the first two layers
        left, right = (self.layers[0].get_activated_layer(), self.layers[1].get_activated_layer())
        current = Current(self.wavelength, left)
        interface = self.interface(left, right)
        current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)

        # Assign to layer the right sub matrix
        if self.state == 3:
            right.S0 = deepcopy(current)
        elif self.state == 7:
            right.S1 = deepcopy(current)

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
                layer2.S0 = deepcopy(current)
            elif self.state == 7:
                layer2.S1 = deepcopy(current)

        # Propagate final two layers
        current.update_s(
            self.cascade(Current(self.wavelength, current), self.layers[-1].get_activated_layer()),
            self.layers[-1].get_activated_layer(),
        )

        return current.s_params

    def propagate_n_only(self):

        # Update all monitors
        for m in self.monitors:

            # Forward through the device
            cur_len = 0
            for layer in tqdm(self.layers):

                # Get system params
                n = layer.mode_solvers.n
                cur_len += layer.length

                # Iterate through z
                while len(m.remaining_lengths[0]) and m.remaining_lengths[0][0] <= cur_len:
                    z = m.remaining_lengths[0][0]

                    # Get full s params for all periods
                    for i in range(self.num_periods):

                        self.set_monitor(m, i, z, {"n": n}, n=True, last_period=(i==self.num_periods-1))

        return

    def cascade_periods(self, current_layer, interface, period_layer, periodic_s):

        # Cascade params for each period
        for l in range(self.num_periods - 1):

            current_layer.s_params = self.cascade((current_layer), (interface))
            periodic_s.append(deepcopy(current_layer))
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
        s_params = self.cascade_periods(current_layer, interface, period_layer, self.forward_periodic_s)

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
        s_params = self.cascade_periods(current_layer, interface, period_layer, self.reverse_periodic_s)

        # Fix geometry
        self.layers = self.layers[::-1]

        return s_params

    def build_input_array(self, input_left, input_middle, input_right):

        # Case 1: input_left > num_modes
        if len(input_left) > self.layers[0].get_activated_layer().num_modes:
            raise Exception("Too many mode coefficients in the left input")

        # Case 2: input_middle > num_sources
        if len(input_middle):
            raise Exception("Too many mode coefficients in the middle sources")

        # Case 3: input_right > num_modes
        if len(input_left) > self.layers[-1].get_activated_layer().num_modes:
            raise Exception("Too many mode coefficients in the right input")

        # Fill input_left
        while len(input_left) < self.layers[0].get_activated_layer().num_modes:
            input_left += [0]

        # Fill input_middle
        input_middle += []

        # Fill input_right
        while len(input_right) < self.layers[-1].get_activated_layer().num_modes:
            input_right += [0]

        # Build forward input_array
        forward = np.array(input_left + input_middle + input_right)

        # Built reverse input_array
        reverse = np.array(input_right + input_middle + input_left)

        return forward, reverse

    def prop_all(self, *args):
        temp_s = args[0]
        for s in args[1:]:
            Subcircuit.clear_scache()

            # make sure the components are completely disconnected
            temp_s.disconnect()
            s.disconnect()

            # # connect the components
            right_pins = [i for i in temp_s.pins if "dup" not in i.name and "left" not in i.name]
            for port in range(len(right_pins)):
                temp_s[f"right{port}"].connect(s[f"left{port}"])

            temp_s = temp_s.circuit.to_subcircuit()

        pins = temp_s.pins
        s_params = temp_s.s_parameters([0])
        del temp_s
        return s_params[0]

    def swap(self, s):
        # Reformat to be in forward reference frame
        if not s is None:
            for j, pin in enumerate(s.pins):
                if "left" in pin.name:
                    name = pin.name
                    s.pins[j].rename(name.replace("left", "temp"))
                elif "right" in pin.name:
                    name = pin.name
                    s.pins[j].rename(name.replace("right", "left"))
            for j, pin in enumerate(s.pins):
                if "temp" in pin.name:
                    name = pin.name
                    s.pins[j].rename(name.replace("temp", "right"))

        return s

    def field_propagate(self, forward):
        # Start state
        self.update_state(10)

        # Reused params
        num_left = len(self.layers[0].get_activated_layer().left_pins)
        num_right = len(self.layers[-1].get_activated_layer().right_pins)

        # Update all monitors
        for m in self.monitors:
            # Reset monitor
            m.reset_monitor()

            # Get full s params for all periods
            for per in range(self.num_periods):
                cur_len = 0

                # Periodic layers
                f = self.forward_periodic_s[per - 1] if per - 1 > -1 else None
                r = self.reverse_periodic_s[self.num_periods - per - 2] if self.num_periods - per - 2 > -1 else None
                
                # Reformat r to be in forward reference frame
                r = self.swap(r)

                # Forward through the device
                for layer in self.layers:
                    cur_len = self.layer_field_propagate(layer.get_activated_layer(), m, per, r, f, cur_len, forward, num_left, num_right)
                
                # Prepare for new period
                m.soft_reset()
        
        # Finish state
        self.update_state(11)

    
    def layer_field_propagate(self, l, m, per, r, f, cur_len, forward, num_left, num_right):

        # Get length
        cur_last = deepcopy(cur_len)
        cur_len += l.length

        # Get system params
        S0, S1 = (l.S0, l.S1)

        # Reformat S1
        S1 = self.swap(S1)

        # Distance params
        z = m.remaining_lengths[0][0]
        z_temp = z - cur_last
        left = Duplicator(l.wavelength, l.modes, z_temp, which_s=0)
        right = Duplicator(l.wavelength, l.modes, l.length-z_temp, which_s=1)

        # Compute field propagation
        prop = [deepcopy(f), deepcopy(S0), deepcopy(left), deepcopy(right), deepcopy(S1), deepcopy(r)]
        # print(prop)
        S = self.prop_all(
            *[t for t in prop if not (t is None) and not (isinstance(t, list) and not len(t))]
        )
        input_array = np.array(
            forward[:num_left].tolist()
            + [0 for _ in range(2 * len(l.modes))]
            + forward[num_left:].tolist()
        )
        coeffs_ = np.matmul(S, input_array)
        coeffs = coeffs_[num_left : - num_right]
        coeffs_l = coeffs[:len(l.modes)]
        coeffs_r = coeffs[len(l.modes):]
        coe = coeffs_l + coeffs_r
        modes = [[i.Ex, i.Ey, i.Ez, i.Hx, i.Hy, i.Hz] for i in l.modes]
        fields = np.array(modes) * np.array(coe)[:, np.newaxis, np.newaxis, np.newaxis]
        results = {}
        results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
            "Hz"
        ] = fields.sum(0)
        results["n"] = l.modes[0].n
        self.set_monitor(m, per, z, results)

        # Iterate through z
        while len(m.remaining_lengths[0]) and m.remaining_lengths[0][0] <= cur_len:

            # Get coe
            z_old = deepcopy(z)
            z = m.remaining_lengths[0][0]
            z_diff = z - z_old
            eig = (2 * np.pi) * np.array([mode.neff for mode in l.modes]) / (self.wavelength)
            coeffs_l *= np.exp(z_diff * 1j * eig) 
            coeffs_r *= np.exp(-z_diff * 1j * eig) 
            coe =  coeffs_l + coeffs_r 

            # Create field
            fields = np.array(modes) * np.array(coe)[:, np.newaxis, np.newaxis, np.newaxis]
            results = {}
            results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
                "Hz"
            ] = fields.sum(0)
            results["n"] = l.modes[0].n
            self.set_monitor(m, per, z, results)
            
        return cur_len

    # def field_propagate(self, forward):
    #     # Start state
    #     self.update_state(10)

    #     # Reused params
    #     num_left = len(self.layers[0].get_activated_layer().left_pins)
    #     num_right = len(self.layers[-1].get_activated_layer().right_pins)

    #     # Update all monitors
    #     for m in self.monitors:

    #         # Reset monitor
    #         m.reset_monitor()

    #         # Forward through the device
    #         cur_len = 0
    #         for layer in tqdm(self.layers):

    #             # Get system params
    #             l = layer.get_activated_layer()
    #             S0, S1 = (l.S0, l.S1)

    #             # Reformat S1 to be in S0 reference frame
    #             if not S1 is None:
    #                 for i, pin in enumerate(S1.pins):
    #                     if "left" in pin.name:
    #                         name = pin.name
    #                         S1.pins[i].rename(name.replace("left", "temp"))
    #                     elif "right" in pin.name:
    #                         name = pin.name
    #                         S1.pins[i].rename(name.replace("right", "left"))
    #                 for i, pin in enumerate(S1.pins):
    #                     if "temp" in pin.name:
    #                         name = pin.name
    #                         S1.pins[i].rename(name.replace("temp", "right"))

    #             # Get length
    #             cur_last = deepcopy(cur_len)
    #             cur_len += l.length

    #             # Iterate through z
    #             while len(m.remaining_lengths[0]) and m.remaining_lengths[0][0] <= cur_len:
                    
    #                 # Distance params
    #                 z = m.remaining_lengths[0][0]
    #                 z_temp = z - cur_last
    #                 left = Duplicator(l.wavelength, l.modes, z_temp, which_s=0)
    #                 right = Duplicator(l.wavelength, l.modes, l.length-z_temp, which_s=1)

    #                 # Get full s params for all periods
    #                 for per in range(self.num_periods):
                        
    #                     # Periodic layers
    #                     f = self.forward_periodic_s[per - 1] if per - 1 > -1 else None
    #                     r = self.reverse_periodic_s[self.num_periods - per - 2] if self.num_periods - per - 2 > -1 else None
                        
                        
    #                     # Reformat r to be in forward reference frame
    #                     if not r is None:
    #                         for j, pin in enumerate(r.pins):
    #                             if "left" in pin.name:
    #                                 name = pin.name
    #                                 r.pins[j].rename(name.replace("left", "temp"))
    #                             elif "right" in pin.name:
    #                                 name = pin.name
    #                                 r.pins[j].rename(name.replace("right", "left"))
    #                         for j, pin in enumerate(r.pins):
    #                             if "temp" in pin.name:
    #                                 name = pin.name
    #                                 r.pins[j].rename(name.replace("temp", "right"))
                        
    #                     # Compute field propagation
    #                     prop = [deepcopy(f), deepcopy(S0), deepcopy(left), deepcopy(right), deepcopy(S1), deepcopy(r)]
    #                     S = self.prop_all(
    #                         *[t for t in prop if not (t is None) and not (isinstance(t, list) and not len(t))]
    #                     )
    #                     input_array = np.array(
    #                         forward[:num_left].tolist()
    #                         + [0 for _ in range(2 * len(l.modes))]
    #                         + forward[num_left:].tolist()
    #                     )
    #                     coeffs_ = np.matmul(S, input_array)
    #                     coeffs = coeffs_[num_left : - num_right]
    #                     coeffs_l = coeffs[:len(l.modes)]
    #                     coeffs_r = coeffs[len(l.modes):]

    #                     # Create field
    #                     modes = [[i.Ex, i.Ey, i.Ez, i.Hx, i.Hy, i.Hz] for i in l.modes]
    #                     fields = np.array(modes) * np.array(coeffs_l)[:, np.newaxis, np.newaxis, np.newaxis]
    #                     fields += np.array(modes) * np.array(coeffs_r)[:, np.newaxis, np.newaxis, np.newaxis]
    #                     results = {}
    #                     results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
    #                         "Hz"
    #                     ] = fields.sum(0)
    #                     results["n"] = l.modes[0].n
    #                     last_period = (per == self.num_periods - 1)
    #                     self.set_monitor(m, per, z, results, last_period=last_period)


    #     # Finish state
    #     self.update_state(11)

    def set_monitor(self, m, i, z, results, n=False, last_period=True):
        for key, field in results.items():

            key = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"].index(key) if not n else 0

            # Non implemented fields
            if m.axes in ["x"]:
                raise Exception("x not implemented")
            elif m.axes in ["y"]:
                raise Exception("y not implemented")
            elif m.axes in ["z"]:
                raise Exception("z not implemented")

            # z Location
            z_loc = np.argwhere(m.lengths[key] == self.get_total_length() * i + z).sum()

            # Implemented fields
            if m.axes in ["xy", "yx"]:
                m[key, :, :, last_period] = field[:, :]
            elif m.axes in ["xz", "zx"]:
                m[key, :, z_loc, last_period] = field[:, int(len(field) / 2)] if field.ndim > 1 else field[:]
            elif m.axes in ["yz", "zy"]:
                m[key, :, z_loc, last_period] = field[int(len(field) / 2), :] if field.ndim > 1 else field[:]
            elif m.axes in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]:
                m[key, :, :, z_loc, last_period] = field[:, :]

        return

    def propagate(self, input_left=[1], input_right=[0]):
        """The propagate method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters.

        Parameters
        ----------
        input_array : numpy array
            the array representing the input to the device in s parameter format: [left port mode 1, left port mode 2, ... left port mode n, right port mode 1, right port mode 2, ... right port mode n] (default : [1,0,0,...]) the default represents sending in the fundamental mode of the left port
        """

        # Check for layers
        if not len(self.layers):
            raise Exception("Must place layers before propagating")
        else:
            self.wavelength = self.layers[0].wavelength

        # Solve for the modes
        self.solve_modes()

        # Format input array
        forward, reverse = self.build_input_array(input_left, [], input_right)

        # Forward pass
        if self.state == 2:
            self.s_params = self.forward_pass(forward)

        # Reverse pass
        if self.state == 6:
            self.reverse_pass()

        # Update monitors
        self.field_propagate(forward)

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
            lengths = deepcopy(single_lengths)
            
            for i in range(1,self.num_periods):
                lengths += (np.array(single_lengths) + l * i).tolist()
            lengths = [lengths for _ in range(len(components))]
            single_lengths = [single_lengths for _ in range(len(components))]

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
            s = min(np.array(lengths)[0], key=difference_start)
            e = min(np.array(lengths)[0], key=difference_end)
            z = np.sum((s <= np.array(lengths)[0]) * (e >= np.array(lengths)[0]))

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
            single_lengths = lengths

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
        monitor = Monitor(axes, dimensions, lengths, components, z_range, grid_x, grid_y, grid_z, location, single_lengths)
        self.monitors.append(monitor)
        return monitor

    def draw(self, z_range=None, mesh_z=200):
        """The draw method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. This will change in the future."""

        temp_storage = self.monitors
        self.monitors = []
        monitor = self.add_monitor(axes="xz", components=["n"], z_range=z_range, excempt=False,mesh_z=mesh_z)
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
