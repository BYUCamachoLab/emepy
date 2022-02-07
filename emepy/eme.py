import numpy as np
from tqdm import tqdm
from simphony.models import Subcircuit
from emepy.monitors import Monitor
from copy import deepcopy
from emepy.fd import MSEMpy
from emepy.models import Layer, Duplicator, Current, PeriodicLayer, InterfaceSingleMode, InterfaceMultiMode
from emepy.source import Source


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

    def __init__(self, layers=[], num_periods=1, mesh_z=200):
        """EME class constructor

        Parameters
        ----------
        layers : list [Layer]
            An list of Layer objects, arranged in the order they belong geometrically. (default: [])
        num_periods : int
            Number of periods if defining a periodic structure (default: 1)
        mesh_z : int
            Number of mesh points in z per period

        """

        self.reset()
        self.layers = layers[:]
        self.num_periods = num_periods
        self.mesh_z = mesh_z
        self.s_params = None
        self.monitors = []
        self.custom_monitors = []
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
            self._update_state(0)

        # Only unsolve everything and erase monitors
        else:
            for i in range(len(self.layers)):
                self.layers[i].clear()
            self._update_state(2)

        self.s_params = None
        self.interface = None
        self.monitors = []
        self.custom_monitors = []

    def solve_modes(self):
        # Check if already solved
        if self.state > 1:
            return

        # Hold still while solving finishes
        while self.state == 1:
            continue

        # Solve modes
        self._update_state(1)
        for layer in tqdm(self.layers):
            layer.activate_layer()
        self._update_state(2)

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

    def add_monitor(self, axes="xz", sources=[], mesh_z=None, z_range=None, location=None, components=None, exempt=True):
        """Creates a monitor associated with the eme object BEFORE the simulation is ran

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'xz', 'xyz', 'x', 'y', 'z'. Currently only 'xz' is implemented. Note, propagation is always in z.
        sources : list[Source]
            the user can specify custom mode sources to use for this monitor (default: input left)
        mesh_z : int
            number of mesh points in z (for periodic structures, will be z * num_periods), warning: if different than global value for EME, a separate run will have to take place for this monitor (default: EME global defined)
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        location : float
            z coordinate where to save data for a 'xy' monitor
        components : list[string]
            a list of the field components to include. Unless the user is worried about memory, this is best left alone. (Default: ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"])
        exempt : boolean
            flag used for very specific case when using PML for MSEMpy. The user never has to change this value. 

        Returns
        -------
        Monitor
            the newly created Monitor object
        """

        # Establish mesh # Current workaround...not final
        components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"] if components is None else components
        if exempt and (self.layers[0].mode_solvers.PML and isinstance(self.layers[0].mode_solvers, MSEMpy)):
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

        # Default Source
        if not len(sources):
            sources.append(Source())

        # Default mesh_z
        mesh_z if not mesh_z is None else self.mesh_z

        # Ensure the axes is not still under development
        if axes in ["xz", "zx", "yz", "zy", "xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:

            # Create lengths
            l = self._get_total_length()
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
        monitor = Monitor(axes, dimensions, lengths, components, z_range, grid_x, grid_y, grid_z, location, single_lengths, sources=sources)
        
        # Place monitor where it belongs
        if (len(lengths) == self.mesh_z):
            self.monitors.append(monitor)
        else:
            self.custom_monitors.append(monitor)

        return monitor

    def draw(self, z_range=None, mesh_z=200):
        """The draw method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. This will change in the future."""

        temp_storage = [self.monitors, self.custom_monitors]
        self.monitors, self.custom_monitors = [[],[]]
        monitor = self.add_monitor(axes="xz", components=["n"], z_range=z_range, exempt=False, mesh_z=mesh_z)
        self._propagate_n_only()
        im = monitor.visualize(component="n")
        self.monitors, self.custom_monitors = temp_storage
        return im

    def propagate(self):
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

        # Forward pass
        if self.state == 2:
            self.s_params = self._forward_pass()

        # Reverse pass
        if self.state == 6:
            self._reverse_pass()

        # Update monitors
        self._field_propagate()

    def _get_source_locations(self):
        return Source.extract_source_locations(*[i.sources for i in (self.monitors + self.custom_monitors)])

    def _propagate_period(self):
        """The _propagate_period method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters for ONE period of the structure. If num_periods is set to 1 (default), this method is the same as propagate, except for it returns values.

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
        current.update_s(self._cascade(Current(self.wavelength, current), interface), interface)

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
            current.update_s(self._cascade(Current(self.wavelength, current), layer1), layer1)
            current.update_s(self._cascade(Current(self.wavelength, current), interface), interface)

            # Assign to layer the right sub matrix
            if self.state == 3:
                layer2.S0 = deepcopy(current)
            elif self.state == 7:
                layer2.S1 = deepcopy(current)

        # Propagate final two layers
        current.update_s(
            self._cascade(Current(self.wavelength, current), self.layers[-1].get_activated_layer()),
            self.layers[-1].get_activated_layer(),
        )

        return current.s_params

    def _propagate_n_only(self):

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

                        self._set_monitor(m, i, z, {"n": n}, n=True, last_period=(i==self.num_periods-1))

        return

    def _cascade_periods(self, current_layer, interface, period_layer, periodic_s):

        # Cascade params for each period
        for l in range(self.num_periods - 1):

            current_layer.s_params = self._cascade((current_layer), (interface))
            periodic_s.append(deepcopy(current_layer))
            current_layer.s_params = self._cascade((current_layer), (period_layer))

        return current_layer.s_params

    def _forward_pass(self):

        # Start forward
        self._update_state(3)

        # Decide routine
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.layers[0].get_activated_layer()
        right = self.layers[-1].get_activated_layer()
        single_period = self._propagate_period()
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self._update_state(5)
        s_params = self._cascade_periods(current_layer, interface, period_layer, self.forward_periodic_s)

        self._update_state(6)

        return s_params

    def _update_state(self, state):

        self.state = state
        print("current state: {}".format(self.states[self.state]))

    def _reverse_pass(self):
        # Assign state
        self._update_state(7)

        # Reverse geometry
        self.layers = self.layers[::-1]

        # Decide routine
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.layers[0].get_activated_layer()
        right = self.layers[-1].get_activated_layer()
        single_period = self._propagate_period()
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self._update_state(8)
        s_params = self._cascade_periods(current_layer, interface, period_layer, self.reverse_periodic_s)

        # Fix geometry
        self.layers = self.layers[::-1]

        return s_params

    def _build_input_array(self, input_left, input_middle, input_right):

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

    def _prop_all(self, *args):
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

    def _swap(self, s):
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

    def _field_propagate(self):
        # Start state
        self._update_state(10)

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
                r = self._swap(r)

                # Forward through the device
                for layer in self.layers:
                    cur_len = self._layer_field_propagate(layer.get_activated_layer(), m, per, r, f, cur_len, num_left, num_right)
                
                # Prepare for new period
                m.soft_reset()
        
        # Finish state
        self._update_state(11)

    
    def _layer_field_propagate(self, l, m, per, r, f, cur_len, forward, num_right):

        """
            TODO
            1) finish creation of input_array from sources
            2) enable the sources to do something during main propagation
            
        """

        # Get length
        cur_last = deepcopy(cur_len)
        cur_len += l.length

        # Get system params
        S0, S1 = (l.S0, l.S1)

        # Reformat S1
        S1 = self._swap(S1)

        # Distance params
        z = m.remaining_lengths[0][0]
        z_temp = z - cur_last
        left = Duplicator(l.wavelength, l.modes, z_temp, which_s=0)
        right = Duplicator(l.wavelength, l.modes, l.length-z_temp, which_s=1)

        # Compute field propagation
        prop = [deepcopy(f), deepcopy(S0), deepcopy(left), deepcopy(right), deepcopy(S1), deepcopy(r)]
        # print(prop)
        S = self._prop_all(
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
        self._set_monitor(m, per, z, results)

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
            self._set_monitor(m, per, z, results)
            
        return cur_len

    def _set_monitor(self, m, i, z, results, n=False, last_period=True):
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
            z_loc = np.argwhere(m.lengths[key] == self._get_total_length() * i + z).sum()

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

    def _get_total_length(self):
        return np.sum([layer.length for layer in self.layers])

    def _cascade(self, first, second):
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