import numpy as np
from tqdm import tqdm
from simphony.models import Subcircuit
from emepy.monitors import Monitor
from copy import deepcopy

from emepy.fd import *
from emepy.models import *
from emepy.source import *
_prop_all = Layer._prop_all


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
        length = 0.0
        self.activated_layers = []
        sources = self.get_sources()
        for layer in tqdm(self.layers):
            self.activated_layers += (layer.activate_layer(sources,length))
            length += layer.length
        self._update_state(2)

    def get_sources(self):
        srcs = Layer.get_sources([j for i in (self.monitors + self.custom_monitors) for j in i.sources], 0, self._get_total_length())
        srcs.sort(key=lambda s: s.z)
        return srcs

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
        mesh_z = mesh_z if not mesh_z is None else self.mesh_z

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
        if (len(lengths[0]) == self.mesh_z):
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

    def propagate(self, left_coeffs=None, right_coeffs=[]):
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

        # Fix defaults
        if left_coeffs is None:
            if not len(right_coeffs) and not len(self.get_monitors()):
                left_coeffs = [1]
            else:
                left_coeffs = []

        # Solve for the modes
        self.solve_modes()

        # Forward pass
        if self.state == 2:
            self.network = self._forward_pass()

        # Reverse pass
        if self.state == 6:
            self._reverse_pass()

        # Update monitors
        self._field_propagate(left_coeffs, right_coeffs)

    def _get_source_locations(self):
        return Source.extract_source_locations(*[i.sources for i in (self.monitors + self.custom_monitors)])

    def _propagate_period(self, activated_layers):
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
        if not len(activated_layers):
            raise Exception("No activated layers in system")
        elif len(activated_layers) == 1:
            return activated_layers[0].s_params

        # Propagate the first two layers
        left, right = (activated_layers[0], activated_layers[1])
        current = Current(self.wavelength, left)
        interface = self.interface(left, right)
        current = _prop_all(current, interface)

        # Assign to layer the right sub matrix
        if self.state == 3:
            right.S0 = deepcopy(current)
        elif self.state == 7:
            right.S1 = deepcopy(current)

        # Propagate the middle layers
        for index in tqdm(range(1, len(activated_layers) - 1)):

            # Get layers
            layer1 = activated_layers[index]
            layer2 = activated_layers[index + 1]

            # Propagate layers together
            interface = self.interface(layer1, layer2)
            current = _prop_all(current, layer1)
            current = _prop_all(current, interface) 

            # Assign to layer the right sub matrix
            if self.state == 7:
                layer2.S1 = deepcopy(current)
            elif self.state == 3:
                activated_layers[-1].S0 = deepcopy(current)

        # Propagate final two layers
        current = _prop_all(current, activated_layers[-1])

        return current

    def _propagate_n_only(self):

        # Forward through the device
        m = self.monitors[0] if len(self.monitors) else self.custom_monitors[0]
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

        length_tracker = 0.0
        # Cascade params for each period
        for l in range(self.num_periods - 1):

            length_tracker += self._get_total_length()
            current_layer = _prop_all(current_layer, interface)
            periodic_s.append(deepcopy(current_layer))
            
            # Only care about sources between the ends
            sources = self.get_sources()
            custom_sources = self.get_sources(sources, length_tracker, self._get_total_length())

            # If no custom sources
            if not len(custom_sources):
                current_layer = _prop_all(current_layer, period_layer)
            # Other sources
            else:
                current_layers = Layer.parse_activated_layers(self.activated_layer[:], custom_sources, length_tracker)
                current_layer_period = self._propagate_period(current_layers)
                current_layer = _prop_all(current_layer, current_layer_period)

        return current_layer

    def _forward_pass(self):

        # Start forward
        self._update_state(3)

        # Decide routine
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.activated_layers[0]
        right = self.activated_layers[-1]
        single_period = self._propagate_period(self.activated_layers)
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self._update_state(5)
        network = self._cascade_periods(current_layer, interface, period_layer, self.forward_periodic_s)

        self._update_state(6)

        return network

    def _update_state(self, state):

        self.state = state
        print("current state: {}".format(self.states[self.state]))

    def _reverse_pass(self):
        # Assign state
        self._update_state(7)

        # Reverse geometry
        self.activated_layers = self.activated_layers[::-1]

        # Decide routine
        num_modes = max([l.num_modes for l in self.activated_layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Propagate
        left = self.activated_layers[0]
        right = self.activated_layers[-1]
        single_period = self._propagate_period(self.activated_layers)
        self.interface = InterfaceMultiMode

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Cascade periods
        self._update_state(8)
        network = self._cascade_periods(current_layer, interface, period_layer, self.reverse_periodic_s)

        # Fix geometry
        self.activated_layers = self.activated_layers[::-1]

        return network

    def _build_input_array(self, left_coeffs, right_coeffs, model, num_modes=1):

        # Case 1: left_coeffs > num_modes
        if len(left_coeffs) > self.activated_layers[0].num_modes:
            raise Exception("Too many mode coefficients in the left input")

        # Case 2: right_coeffs > num_modes
        if len(left_coeffs) > self.activated_layers[-1].num_modes:
            raise Exception("Too many mode coefficients in the right input")

        # Start mapping
        mapping = {}

        # Get sources
        sources = self.get_sources()

        # Form mapping
        try:
            for pin in model.pins:
                n = pin.name
                
                # Left global input
                if "left" in n and not "dup" in n:
                    ind = int(n[4:])
                    if ind < len(left_coeffs):
                        mapping[n] = left_coeffs[ind]
                    else: 
                        mapping[n] = 0.0

                # Right global input
                if "right" in n and not "dup" in n:
                    ind = int(n[5:])
                    if ind < len(right_coeffs):
                        mapping[n] = right_coeffs[ind]
                    else: 
                        mapping[n] = 0.0

                # Left monitor
                if "left" in n and "dup" in n and not "to" in n:
                    mapping[n] = 0

                # Right monitor
                if "right" in n and "dup" in n and not "to" in n:
                    mapping[n] = 0

                # Custom left source inputs
                if "left" in n and "dup" in n and "to" in n:
                    n = n.split("_")
                    l, _ = (n[2],n[4])
                    ind = int(n[1].replace("dup",""))
                    for i,s in enumerate(sources):
                        if s.match_label(l) and ind < len(s.mode_coeffs):
                            mapping[n] = s.mode_coeffs[ind]
                            break
                        elif i == len(sources) - 1:
                            mapping[n] = 0.0
                    
                # Custom right source inputs
                if "right" in n and "dup" in n and "to" in n:
                    n_ = n.split("_")
                    _, r = (n_[2],n_[4])
                    ind = int(n_[1].replace("dup",""))
                    for i,s in enumerate(sources):
                        if s.match_label(r) and ind < len(s.mode_coeffs):
                            mapping[n] = s.mode_coeffs[ind]
                            break
                        elif i == len(sources) - 1:
                            mapping[n] = 0.0

        except Exception as e:
            raise Exception("Improper format of sources")

        return mapping

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

    def _field_propagate(self, left_coeffs, right_coeffs):
        # Start state
        self._update_state(10)

        # Reused params
        num_left = len(self.activated_layers[0].left_pins)
        num_right = len(self.activated_layers[-1].right_pins)

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
                for layer in self.activated_layers:
                    cur_len = self._layer_field_propagate(layer, m, per, r, f, cur_len, num_left, num_right, left_coeffs, right_coeffs)
                
                # Prepare for new period
                m.soft_reset()
        
        # Finish state
        self._update_state(11)

    
    def _layer_field_propagate(self, l, m, per, r, f, cur_len, num_left, num_right, left_coeffs, right_coeffs):

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
        
        special_left = [pin.name for pin in l.pins if "left" in pin.name and "dup" in pin.name]
        special_right = [pin.name for pin in l.pins if "right" in pin.name and "dup" in pin.name]

        left = Duplicator(l.wavelength, l.modes, z_temp, pk=l.pk,nk=[0 for _ in range(len(l.modes))],special_left=special_left)
        right = Duplicator(l.wavelength, l.modes, l.length-z_temp, pk=[0 for _ in range(len(l.modes))],nk=l.nk,special_right=special_right)

        # Compute field propagation
        prop = [deepcopy(f), deepcopy(S0), deepcopy(left), deepcopy(right), deepcopy(S1), deepcopy(r)]
        S = _prop_all(
            *[t for t in prop if not (t is None) and not (isinstance(t, list) and not len(t))]
        )

        # Get input array
        input_map = self._build_input_array(left_coeffs, right_coeffs, S, num_modes=len(l.modes))
        coeffs_ = S.compute(input_map, 0)
        coeff = np.zeros(len(l.modes), dtype=complex)
        for i in range(len(l.modes)):
            coeff[i] = coeffs_["left_dup{}".format(i)] + coeffs_["right_dup{}".format(i)]

        # Calculate field
        modes = np.array([[i.Ex, i.Ey, i.Ez, i.Hx, i.Hy, i.Hz] for i in l.modes],dtype=complex)
        coeff = np.array(coeff,dtype=complex)
        fields = modes * coeff[:, np.newaxis, np.newaxis, np.newaxis]
        results = {}
        results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
            "Hz"
        ] = fields.sum(0)
        results["n"] = l.modes[0].n
        self._set_monitor(m, per, z, results)

        # Iterate through z
        while len(m.remaining_lengths[0]) and m.remaining_lengths[0][0] <= cur_len:

            # Get coe
            z = m.remaining_lengths[0][0]
            eig = (2 * np.pi) * np.array([mode.neff for mode in l.modes]) / (self.wavelength)
            phase = np.exp(z * 1j * eig) 
            coeff_ = coeff*phase

            # Create field
            fields_ = modes * coeff_[:, np.newaxis, np.newaxis, np.newaxis]
            results = {}
            results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
                "Hz"
            ] = fields_.sum(0)
            results["n"] = l.modes[0].n
            self._set_monitor(m, per, m.remaining_lengths[0][0], results)
            
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
