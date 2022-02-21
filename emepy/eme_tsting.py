import numpy as np
from tqdm import tqdm 
from simphony.models import Subcircuit
from emepy.monitors import Monitor
from copy import deepcopy
import importlib
if not (importlib.util.find_spec("mpi4py") is None):
    from mpi4py import MPI

from emepy.fd import *
from emepy.models import *
from emepy.source import *
_prop_all = ModelTools._prop_all


class EME(object):
    """The EME class is the heart of the package. It provides the algorithm that cascades sections modes together to provide the s-parameters for a geometric structure. The object is dependent on the Layer objects that are fed inside."""

    states = {
        0: "start",
        1: "mode_solving",
        2: "finished_modes",
        3: "layer_propagating",
        4: "finished_layer",
        5: "field_propagating",
        6: "finished",
    }

    def __init__(self, layers=[], num_periods=1, mesh_z=200, parallel=False, quiet=False):
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

        self.parallel=parallel
        if parallel:
            self._configure_parallel_resources()
        self.quiet = quiet or not self.am_master()
        self.reset(parallel=parallel, configure_parallel=False)
        self.layers = layers[:]
        self.num_periods = num_periods
        self.mesh_z = mesh_z
        self.monitors = []
        self.custom_monitors = []
        self.forward_periodic_s = []
        self.reverse_periodic_s = []
        self.parallel = parallel

    def add_layer(self, layer):
        """The add_layer method will add a Layer object to the EME object. The object will be geometrically added to the very right side of the structure. Using this method after propagate is useless as the solver has already been called.

        Parameters
        ----------
        layer : Layer
            Layer object to be appended to the list of Layers inside the EME object.

        """

        self.layers.append(layer)

    def reset(self, full_reset=True, parallel=False, configure_parallel=True):
        """Clears out the layers and s params so the user can reuse the object in memory on a new geometry"""

        # Erase all information except number of periods
        if full_reset:
            self.layers = []
            self.wavelength = None
            self.parallel=parallel
            if parallel and configure_parallel:
                self._configure_parallel_resources()
            self._update_state(0)

        # Only unsolve everything and erase monitors
        else:
            for i in range(len(self.layers)):
                self.layers[i].clear()
            self._update_state(2)

        self.monitors = []
        self.custom_monitors = []
        self.network = None
        self.periodic_network = None

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
        tasks = []
        sources = self.get_sources()
        for layer in tqdm(self.layers,disable=self.quiet):
            tasks.append((layer.activate_layer,[sources,length,self.num_periods,self._get_total_length()],{}))
            length += layer.length

        # Organized solved layers
        self.activated_layers_total = []
        results = self._run_parallel_functions(*tasks)
        self.activated_layers = []
        if not results is None:
            for solved in results:
                self.activated_layers_total.append(solved)
                self.activated_layers += solved[0]

        self._update_state(2)

    def propagate_layers(self):

        tasks = []

        # Check state
        if self.state == 2:
            self._update_state(3)
        else:
            raise Exception("In the wrong place")
        
        # See if only one layer or no layer
        if not len(self.activated_layers):
            raise Exception("No activated layers in system")
        elif len(self.activated_layers) == 1:
            self.periodic_interface = InterfaceMultiMode(self.activated_layers[-1], self.activated_layers[0])
            return self.activated_layers[0]

        # NEED TO FIX ACTIVATED LAYERS TOTAL

        # Configure interface
        num_modes = max([len(l.modes) for l in self.activated_layers])
        interface_type = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Define tasks
        def task(l,r):
            return l, _prop_all(l, interface_type(l, r))

        # Setup parallel layer prop tasks
        for left, right in tqdm(zip(self.activated_layers[:-1],self.activated_layers[1:]),disable=self.quiet):
            tasks.append((task,[left, right],{}))

        # Propagate
        results = self._run_parallel_functions(*tasks)

        # Calculate period interface in case multiple periods
        self.periodic_interface = InterfaceMultiMode(self.activated_layers[-1], self.activated_layers[0])
        
        # Only keep what we need
        for i, result in enumerate(results[:-1]):
            layer, cascaded = result
            attributes = layer.__dict__
            self.activated_layers[i] = cascaded
            self.activated_layers[i].length = attributes['length']
            self.activated_layers[i].wavelength = attributes['wavelength']
            self.activated_layers[i].modes = attributes['modes']
            self.activated_layers[i].num_modes = attributes['num_modes']

        # Finish state
        self._update_state(4)

        return _prop_all(*self.activated_layers)

    def am_master(self):
        return self.rank == 0 if self.parallel else True

    def get_sources(self):
        srcs = []
        period_length = self._get_total_length()
        for i in range(self.num_periods):
            srcs += ModelTools.get_sources([j for i in (self.monitors + self.custom_monitors) for j in i.sources], i*period_length+0, i*period_length+self._get_total_length())
        srcs.sort(key=lambda s: s.z)
        return srcs

    def s_parameters(self, freqs=None):
        """Returns the s_parameters if they exist. If they don't exist yet, propagate() will be called first.

        Returns
        -------
        numpy array
            The s_params acquired during propagation
        """

        if self.network is None and self.periodic_network is None:
            self.propagate()
        elif not self.periodic_network is None:
            return self.periodic_network.s_parameters(freqs)
        else:
            return self.network.s_parameters(freqs)

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
            if not len(right_coeffs) and not len(self.get_sources()):
                left_coeffs = [1]
            else:
                left_coeffs = []

        # Solve for the modes
        self.solve_modes()

        # Forward pass
        if self.state == 2:
            self.network = self.propagate_layers()

        # Periodic ### ToDo, parallelize this process using a log2 n technique
        # if self.num_periods > 1:
        #     self.periodic_network = _prop_all(*([self.network] + [self.periodic_interface, self.network] * (self.num_periods - 1)))

        # Update monitors
        self._field_propagate(left_coeffs, right_coeffs)

    def _run_parallel_functions(self, *args):
        """Args should provide tuples of (function, argument list, kwargument dictionary) and the function will magically compute them in parallel"""
        
        # Initialize empty completed task list
        finished_tasks = []
        finished_tasks_collective = []

        # Complete all tasks and tag based on initial order for either parallel or not
        if not self.parallel:
            # Linearly execute tasks
            for i, a in enumerate(args):
                func, arguments, kwarguments = a
                finished_tasks_collective.append([(i,func(*arguments, **kwarguments))])
        else:

            # Create data
            solve_data = [(i, a) for i, a in enumerate(args)]
            if self.am_master():
                data = []
                for j in range(self.size):
                    subdata = []
                    for i, a in enumerate(args):
                        if self._should_compute(i,j,self.size):
                            subdata.append(i)
                    data.append(subdata)
            else:
                data = None

            # Scatter data
            data = self.comm.scatter(data, root=0)
            new_data = []

            # Compute data
            for i, k in enumerate(data):
                func, arguments, kwarguments = solve_data[k][1]
                new_data.append((k,func(*arguments, **kwarguments)))
            
            # Wait until everyone is finished
            self.comm.Barrier()

            # Gather data
            finished_tasks_collective = self.comm.allgather(new_data)
            for row in finished_tasks_collective:
                finished_tasks += row

            # Wait until everyone is finished
            self.comm.Barrier()

        # Sort by the tag to return in the original order
        finished_tasks = [i for i in finished_tasks if len(i)]
        finished_tasks = sorted(finished_tasks,key=lambda x: x[0])
        finished_tasks = [i[1] for i in finished_tasks]

        return finished_tasks

    def _should_compute(self,i,rank,size):
        return i % size == rank

    def _configure_parallel_resources(self):

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        if self.am_master():
            print("Running in parallel on {} cores".format(self.size))

    def _get_source_locations(self):
        return Source.extract_source_locations(*[i.sources for i in (self.monitors + self.custom_monitors)])


    def _propagate_n_only(self):

        # Forward through the device
        m = self.monitors[0] if len(self.monitors) else self.custom_monitors[0]
        cur_len = 0
        for per in range(self.num_periods):
            for layer in tqdm(self.layers,disable=self.quiet):

                # Get system params
                z_list = m.get_z_list(cur_len, cur_len+layer.length)
                n = layer.mode_solvers.n

                # Iterate through z
                for i, z in z_list:
                    self._set_monitor(m, i, {"n": n}, n=True)

                cur_len += layer.length
        return


    def _update_state(self, state):

        self.state = state
        if self.am_master() and not self.quiet:
            print("current state: {}".format(self.states[self.state]))

    def _build_input_array(self, left_coeffs, right_coeffs, model, num_modes=1, layers=[]):

        # Case 1: left_coeffs > num_modes
        if len(left_coeffs) > layers[0].num_modes:
            raise Exception("Too many mode coefficients in the left input")

        # Case 2: right_coeffs > num_modes
        if len(left_coeffs) > layers[-1].num_modes:
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
                    n_ = n.split("_")
                    l, _ = (n_[2],n_[4])
                    ind = int(n_[1].replace("dup",""))
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
        self._update_state(5)

        # Update all monitors
        for m in self.custom_monitors + self.monitors:
            # Reset monitor
            m.reset_monitor()

            # Get full s params for all periods
            cur_len = 0
            full_z_list = []
            tasks = []
            for per in tqdm(range(self.num_periods),disable=self.quiet):

                # Forward through the device
                layers = self.activated_layers
                for i, layer_ in enumerate(layers):
                    layers[i] = self.activated_layers_total[i][per] if len(self.activated_layers_total[i][per]) else layer_
                layers = [i for ll in layers for i in ll]
                for i, layer in enumerate(layers):
                    z_list = m.get_z_list(cur_len, cur_len+layer.length)
                    just_z_list = [i[1] for i in z_list]
                    task = (self._layer_field_propagate,[i*1, make_copy_model(layer), layers, per*1, left_coeffs[:], right_coeffs[:], cur_len*1, just_z_list[:]],{})
                    task = (self._layer_field_propagate,[i*1, make_copy_model(layer), layers, per*1, left_coeffs[:], right_coeffs[:], cur_len*1, just_z_list[:]],{})
                    full_z_list.append(z_list)
                    tasks.append(task)
                    cur_len += layer.length

            # Get restults
            results = self._run_parallel_functions(*tasks)

            # Assign results to monitor
            for z_l, result_l in zip(full_z_list, results):
                for z, r in zip(z_l, result_l):
                    self._set_monitor(m, z[0], r)
        
        # Finish state
        self._update_state(6)

    def _layer_field_propagate(self, i, l, layers, per, left_coeffs, right_coeffs, cur_len, z_list):

        # Create output
        result_list = []
        
        # Period length
        perf = ModelTools.periodic_duplicate_format
        per_length = self._get_total_length()

        # get SP0
        SP0 = []
        for p in range(per):
            SP0.append(perf(self.network, p*per_length, (p+1)*per_length))
            SP0.append(self.periodic_interface)

        # get SP1
        SP1 = []
        for p in range(self.num_periods - (per) - 1):
            p = per + p + 1
            SP1.append(self.periodic_interface)
            SP1.append(perf(self.network, p*per_length, (p+1)*per_length))

        # Get S0
        S0, S0_length = ([],per*per_length)
        for lay in layers[:i]:
            S0.append(perf(lay, S0_length, S0_length+lay.length))
            S0_length += lay.length

        # Get S1
        S1, S1_length = ([],S0_length+layers[i].length)
        for lay in layers[i+1:]:
            S1.append(perf(lay, S1_length, S1_length+lay.length))
            S1_length += lay.length

        # Distance params
        dup = Duplicator(l.wavelength, l.modes)

        # See if need to remove sources for periodic l
        checked_l = perf(l, cur_len, cur_len+l.length)

        # create all prop layers
        prop = [*SP0, *S0, checked_l, dup, *S1, *SP1] if sum(["_to_" in pin.name and "left" in pin.name for pin in checked_l.pins]) else [*SP0, *S0, dup, checked_l, *S1, *SP1]

        # Compute field propagation
        S = _prop_all(
            *[t for t in prop if not (t is None) and not (isinstance(t, list) and not len(t))]
        )

        # Get input array
        input_map = self._build_input_array(left_coeffs, right_coeffs, S, num_modes=len(l.modes), layers=layers)
        coeffs_ = compute(S, input_map, 0)
        coeff_left = np.zeros(len(l.modes), dtype=complex)
        coeff_right = np.zeros(len(l.modes), dtype=complex)
        modes = np.array([[i.Ex, i.Ey, i.Ez, i.Hx, i.Hy, i.Hz] for i in l.modes])
        for i in range(len(l.modes)):
            coeff_left[i] = 0
            coeff_right[i] = 0
            if "left_dup{}".format(i) in coeffs_:
                coeff_left[i] += coeffs_["left_dup{}".format(i)]
            if "right_dup{}".format(i) in coeffs_:
                coeff_right[i] += coeffs_["right_dup{}".format(i)]

        # Case where layer has size 0 (when source at edge of a layer)
        if not len(z_list):
            return result_list

        # Reverse phase if looking from right end
        diffs = [z_list[0]-cur_len]+np.diff(z_list).tolist()
        eig = (2 * np.pi) * np.array([mode.neff for mode in l.modes]) / (self.wavelength)
        if sum(["_to_" in pin.name and "left" in pin.name for pin in checked_l.pins]):
            coeff_left[i] *= np.exp(1j * eig * l.length)
            coeff_right[i] *= np.exp(-1j * eig * l.length)
        
        # Iterate through z
        for z_diff in diffs:

            # Get coe
            phase_left = np.exp(-z_diff * 1j * eig) 
            phase_right = np.exp(z_diff * 1j * eig) 
            coeff_left = coeff_left*phase_left
            coeff_right = coeff_right*phase_right
            coeff = coeff_left + coeff_right

            # Create field
            fields_ = modes * coeff[:, np.newaxis, np.newaxis, np.newaxis]
            results = {}
            results["Ex"], results["Ey"], results["Ez"], results["Hx"], results["Hy"], results[
                "Hz"
            ] = fields_.sum(0)
            results["n"] = l.modes[0].n
            result_list.append(results)
                
        return result_list
    
    def _set_monitor(self, m, i, results, n=False):
        for key, field in results.items():

            key = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"].index(key) if not n else 0

            # Non implemented fields
            if m.axes in ["x"]:
                raise Exception("x not implemented")
            elif m.axes in ["y"]:
                raise Exception("y not implemented")
            elif m.axes in ["z"]:
                raise Exception("z not implemented")

            # Implemented fields
            if m.axes in ["xy", "yx"]:
                m[key, :, :] = field[:, :]
            elif m.axes in ["xz", "zx"]:
                m[key, :, i] = field[:, int(len(field) / 2)] if field.ndim > 1 else field[:]
            elif m.axes in ["yz", "zy"]:
                m[key, :, i] = field[int(len(field) / 2), :] if field.ndim > 1 else field[:]
            elif m.axes in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]:
                m[key, :, :, i] = field[:, :]

        return

    def _get_total_length(self):
        return np.sum([layer.length for layer in self.layers])
