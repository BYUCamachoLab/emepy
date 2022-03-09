import numpy as np
import numpy
from tqdm import tqdm
from emepy.monitors import Monitor
import matplotlib
import simphony

import importlib

if importlib.util.find_spec("mpi4py") is not None:
    from mpi4py import MPI

from emepy.fd import MSEMpy
from emepy.models import (
    Duplicator,
    Model,
    ModelTools,
    Layer,
    InterfaceSingleMode,
    InterfaceMultiMode,
)

_prop_all = ModelTools._prop_all
purge_spurious = ModelTools.purge_spurious
get_sources = ModelTools.get_sources
get_source_system = ModelTools.get_source_system
periodic_duplicate_format = ModelTools.periodic_duplicate_format
make_copy_model = ModelTools.make_copy_model
compute = ModelTools.compute


class EME(object):
    """The EME class is the heart of the package. It provides the algorithm that cascades sections modes together to provide the s-parameters for a geometric structure. The object is dependent on the Layer objects that are fed inside."""

    states = {
        0: "start",
        1: "mode_solving",
        2: "finished_modes",
        3: "layer_propagating",
        4: "finished_layer",
        5: "network_building",
        6: "finished_network",
        7: "field_propagating",
        8: "finished",
    }

    def __init__(
        self,
        layers: list = [],
        num_periods: int = 1,
        mesh_z: int = 200,
        parallel: bool = False,
        quiet: bool = False,
        **kwargs
    ) -> None:
        """EME class constructor

        Parameters
        ----------
        layers : list [Layer]
            An list of Layer objects, arranged in the order they belong geometrically. (default: [])
        num_periods : int
            Number of periods if defining a periodic structure (default: 1)
        mesh_z : int
            Number of mesh points in z per period for default monitors (default: 200)
        parallel : bool
            If true, will allocate parallelized processes for solving modes, propagating layers, and filling monitors with field data (default: False)
        quiet : bool
            If true, will not print current state and status of the solver (default: False)
        """

        self.parallel = parallel
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
        self.network = None

    def add_layer(self, layer: Layer) -> None:
        """The add_layer method will add a Layer object to the EME object. The object will be geometrically added to the very right side of the structure. Using this method after propagate is useless as the solver has already been called.

        Parameters
        ----------
        layer : Layer
            Layer object to be appended to the list of Layers inside the EME object.
        """

        self.layers.append(layer)

    def add_layers(self, *layers) -> None:
        """Calls add layers for the layers provided"""

        for layer in layers:
            self.add_layer(layer)

    def reset(
        self,
        full_reset: bool = True,
        parallel: bool = False,
        configure_parallel: bool = True,
    ) -> None:
        """Clears out the layers and s params so the user can reuse the object in memory on a new geometry

        Parameters
        ----------
        full_reset : boolean
            If true, will reset everything inside of the object and allow for reinstancing without memory issues (default: True)
        parallel : boolean
            If configure_parallel is True, after reset this method will set the value of parallel. Similar to the constructor (default: False)
        configure_parallel : boolean
            If configure_parallel is True, after reset this method will set the value of parallel. Similar to the constructor (default: True)
        """

        # Erase all information except number of periods
        if full_reset:
            self.layers = []
            self.wavelength = None
            self.parallel = parallel
            if parallel and configure_parallel:
                self._configure_parallel_resources()
            self._update_state(0)

        # Only unsolve everything and erase monitors
        else:
            for i in range(len(self.layers)):
                self.layers[i].clear()
            self.activated_layers = []
            self._update_state(0)

        self.monitors = []
        self.custom_monitors = []
        self.networks = None
        self.network = None

    def solve_modes(self) -> None:
        """Solves for the modes in the system and is the first step in the solver's process all in parallel"""

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
        for layer in tqdm(self.layers, disable=self.quiet):
            tasks.append(
                (layer.activate_layer, [sources, length, self._get_total_length()], {})
            )
            length += layer.length

        # Organized solved layers
        self.activated_layers = dict(
            zip(range(self.num_periods), [[] for _ in range(self.num_periods)])
        )
        results = self._run_parallel_functions(*tasks)
        if results is not None:
            for solved in results:
                for per, layers in solved.items():
                    if not sum(
                        [
                            i is not None
                            for so in results
                            for p, s in so.items()
                            for i in s
                            if p == per
                        ]
                    ):
                        self.activated_layers[per] = None
                    else:
                        self.activated_layers[per] += (
                            layers if layers[0] is not None else solved[0]
                        )

        self._update_state(2)

    def propagate_layers(self) -> None:
        """Propagates each layer with the next by creating interface models and cascading all in parallel. This is the second step for the solver"""

        # Check state
        if self.state == 2:
            self._update_state(3)
        else:
            raise Exception("In the wrong place")

        # Create return dict
        final_activated_layers = dict(
            zip(range(self.num_periods), [None for _ in range(self.num_periods)])
        )
        periodic_interfaces = dict(
            zip(range(self.num_periods), [None for _ in range(self.num_periods)])
        )

        # Loop through all periods in case of custom sources
        for per, activated_layers in self.activated_layers.items():
            tasks = []

            # Most cases
            if per and activated_layers is None:
                continue

            # See if only one layer or no layer
            if not per and activated_layers is None:
                raise Exception("No activated layers in system")
            elif len(activated_layers) == 1:
                periodic_interfaces[per] = InterfaceMultiMode(
                    activated_layers[-1], activated_layers[0]
                )
                final_activated_layers[per] = activated_layers[0]
                continue

            # Configure interface
            num_modes = max([len(l.modes) for l in activated_layers])
            interface_type = (
                InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode
            )

            # Define tasks
            def task(l, r):
                return l, _prop_all(l, interface_type(l, r))

            # Setup parallel layer prop tasks
            for left, right in tqdm(
                zip(activated_layers[:-1], activated_layers[1:]), disable=self.quiet
            ):
                tasks.append((task, [left, right], {}))

            # Propagate
            results = self._run_parallel_functions(*tasks)

            # Calculate period interface in case multiple periods
            periodic_interfaces[per] = InterfaceMultiMode(
                activated_layers[-1], activated_layers[0]
            )

            # Only keep what we need
            for i, result in enumerate(results):
                layer, cascaded = result
                attributes = layer.__dict__
                activated_layers[i] = cascaded
                activated_layers[i].length = attributes["length"]
                activated_layers[i].wavelength = attributes["wavelength"]
                activated_layers[i].modes = attributes["modes"]
                activated_layers[i].num_modes = attributes["num_modes"]

            # Add to cascaded list
            final_activated_layers[per] = _prop_all(*activated_layers)

        # Update
        self.networks = final_activated_layers
        self.periodic_interfaces = periodic_interfaces

        # Finish state
        self._update_state(4)

    def build_network(self) -> None:
        """Builds the full network from the cascaded layers. This is the third step in the solving process."""

        # Initialize building state
        self._update_state(5)

        # Place all periods' proper networks into the final network
        networks = []
        for per, network in self.networks.items():
            start = per * self._get_total_length()
            end = (per + 1) * self._get_total_length()
            n = network if network is not None else self.networks[0]
            networks.append(periodic_duplicate_format(n, start, end))
            if not per:
                networks.append(
                    self.periodic_interfaces[per]
                    if self.periodic_interfaces[per] is not None
                    else self.periodic_interfaces[0]
                )

        # Propagate final network with all periods
        self.network = _prop_all(*networks)

        # Finish building state
        self._update_state(6)

    def field_propagate(self, left_coeffs: list, right_coeffs: list) -> None:
        """Propagates the modes through the device to calculate the field profile everywhere

        Parameters
        ----------
        left_coeffs : list
            A list of floats that represent the mode coefficients for the left side of the full geometry
        right_coeffs : list
            A list of floats that represent the mode coefficients for the right side of the full geometry
        """

        # Start state
        self._update_state(7)

        # Update all monitors
        for m in self.custom_monitors + self.monitors:
            # Reset monitor
            m.reset_monitor()
            m.left_source = len(left_coeffs) > 0
            m.right_source = len(right_coeffs) > 0

            # Get full s params for all periods
            cur_len = 0
            full_z_list = []
            tasks = []
            for per, activated_layers in tqdm(
                self.activated_layers.items(), disable=self.quiet
            ):

                # Forward through the device
                activated_layers = (
                    activated_layers
                    if activated_layers is not None
                    else self.activated_layers[0]
                )
                for i, layer in enumerate(activated_layers):
                    z_list = m.get_z_list(cur_len, cur_len + layer.length)
                    just_z_list = [i[1] for i in z_list]
                    task = (
                        self._layer_field_propagate,
                        [
                            i * 1,
                            make_copy_model(layer),
                            per * 1,
                            left_coeffs[:],
                            right_coeffs[:],
                            cur_len * 1,
                            just_z_list[:],
                        ],
                        {},
                    )
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
        self._update_state(8)

    def am_master(self) -> bool:
        """Returns true for the master process if the user is running a parallel process using mpi. This is essential for I/O"""
        return self.rank == 0 if self.parallel else True

    def get_sources(self) -> dict:
        """Returns a dictionary of each period and the Source objects that can be found inside each"""
        srcs = dict(zip(range(self.num_periods), [[] for _ in range(self.num_periods)]))
        for per in range(self.num_periods):
            start = self._get_total_length() * per
            end = self._get_total_length() * (per + 1)
            src = get_sources(
                [j for i in (self.monitors + self.custom_monitors) for j in i.sources],
                start,
                end,
            )
            src.sort(key=lambda s: s.z)
            srcs[per] = src

        return srcs

    def s_parameters(self, freqs=None) -> "numpy.ndarray":
        """Returns the s_parameters if they exist. If they don't exist yet, propagate() will be called first.

        Returns
        -------
        numpy array
            The s_params acquired during propagation
        """

        if self.network is None:
            self.propagate()
        return self.network.s_parameters(freqs)

    def add_monitor(
        self,
        axes: str = "xz",
        sources: list = [],
        mesh_z: int = None,
        z_range: tuple = None,
        location: float = None,
        components: list = None,
        exempt: bool = True,
    ) -> Monitor:
        """Creates a monitor associated with the eme object BEFORE the simulation is ran

        Parameters
        ----------
        axes : str
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
        exempt : bool
            flag used for very specific case when using PML for MSEMpy. The user never has to change this value.

        Returns
        -------
        Monitor
            the newly created Monitor object
        """

        # Establish mesh # Current workaround...not final
        components = (
            ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "n"]
            if components is None
            else components
        )
        if exempt and (
            self.layers[0].mode_solver.PML
            and isinstance(self.layers[0].mode_solver, MSEMpy)
        ):
            x = (
                len(
                    self.layers[0].mode_solver.x[
                        self.layers[0]
                        .mode_solver.nlayers[1] : -self.layers[0]
                        .mode_solver.nlayers[0]
                    ]
                )
                + 1
            )
            y = (
                len(
                    self.layers[0].mode_solver.y[
                        self.layers[0]
                        .mode_solver.nlayers[3] : -self.layers[0]
                        .mode_solver.nlayers[2]
                    ]
                )
                + 1
            )
        else:
            x = len(self.layers[0].mode_solver.after_x)
            y = len(self.layers[0].mode_solver.after_y)

        # # Default Source
        # if not len(sources):
        #     sources.append(Source())

        # Default mesh_z
        mesh_z = mesh_z if mesh_z is not None else self.mesh_z

        # Ensure the axes is not still under development
        if axes in ["xz", "zx", "yz", "zy", "xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:

            # Create lengths
            l = self._get_total_length()
            single_lengths = np.linspace(0, l, mesh_z, endpoint=False).tolist()
            lengths = np.linspace(0, l, mesh_z, endpoint=False).tolist()

            for i in range(1, self.num_periods):
                lengths += (np.array(single_lengths) + l * i).tolist()
            lengths = [lengths for _ in range(len(components))]
            single_lengths = [single_lengths for _ in range(len(components))]

            # Ensure z range is in proper format
            try:
                start, end = (
                    [lengths[0][0], lengths[0][-1]] if z_range is None else z_range
                )
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
            grid_x = self.layers[0].mode_solver.after_x
            grid_y = self.layers[0].mode_solver.after_y
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
            grid_x = self.layers[0].mode_solver.x
            grid_y = self.layers[0].mode_solver.y
            grid_z = np.array([location])

        else:
            raise Exception(
                "Monitor setup {} has not yet been implemented. Please choose from the following implemented monitor types: ['xz', 'yz', 'xyz']".format(
                    axes
                )
            )

        # Create monitor
        monitor = Monitor(
            axes,
            dimensions,
            components,
            z_range,
            grid_x,
            grid_y,
            grid_z,
            location,
            sources=sources,
        )

        # Place monitor where it belongs
        if len(lengths[0]) == self.mesh_z:
            self.monitors.append(monitor)
        else:
            self.custom_monitors.append(monitor)

        return monitor

    def draw(
        self, z_range: tuple = None, mesh_z: int = 200
    ) -> "matplotlib.image.AxesImage":
        """The draw method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. This will change in the future.

        Parameters
        ----------
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        mesh_z : int
            the number of mesh points in z to calculate index profiles for

        Returns
        -------
        matplotlib.image.AxesImage
            the image used to plot the index profile
        """

        temp_storage = [self.monitors, self.custom_monitors]
        self.monitors, self.custom_monitors = [[], []]
        monitor = self.add_monitor(
            axes="xz", components=["n"], z_range=z_range, exempt=False, mesh_z=mesh_z
        )
        self._propagate_n_only()
        im = monitor.visualize(component="n")
        self.monitors, self.custom_monitors = temp_storage
        return im

    def propagate(
        self, left_coeffs: list = None, right_coeffs: list = []
    ) -> "simphony.models.Model":
        """The propagate method should be called once all Layer objects have been added. This method will call the EME solver and produce s-parameters. The defulat

        Parameters
        ----------
        left_coeffs : list
            A list of floats that represent the mode coefficients for the left side of the full geometry. The default is determined on whether or not any custom mode sources or right_coeffs are defined. If they are, (default:[]) else (default"[1])
        right_coeffs : list
            A list of floats that represent the mode coefficients for the right side of the full geometry. (default:[])

        Returns
        -------
        simphony.models.Model
            The simphony model that represents the entire device
        """

        # Check for layers
        if not len(self.layers):
            raise Exception("Must place layers before propagating")
        else:
            self.wavelength = self.layers[0].wavelength

        # Fix defaults
        if left_coeffs is None:
            left_coeffs = (
                [1]
                if not len(right_coeffs)
                and not len([i for i in self.get_sources().values() if len(i)])
                else []
            )

        # Solve for the modes
        if self.state == 0:
            self.solve_modes()

        # Forward pass
        if self.state == 2:
            self.propagate_layers()

        # Periodic ### ToDo, parallelize this process using a log2 n technique
        if self.state == 4:
            self.build_network()

        # Update monitors
        if self.state >= 6:
            self.field_propagate(left_coeffs, right_coeffs)

        return self.network

    def _run_parallel_functions(self, *tasks) -> list:
        """Takes a series of "tasks" as arguments and returns a list of "results" after running in parallel.

        Arguments
        ---------
        *tasks : tuple
            Each "task" should take the following form: (function, list of arguments, dictionary of kwarguments) and the function will magically compute them in parallel

        Returns
        -------
        list
            Each "result" in the return list will be whatever is returned by the function in the corresponding task
        """

        # Initialize empty completed task list
        finished_tasks = []
        finished_tasks_collective = []

        # Complete all tasks and tag based on initial order for either parallel or not
        if not self.parallel:
            # Linearly execute tasks
            for i, a in enumerate(tasks):
                func, arguments, kwarguments = a
                finished_tasks.append(func(*arguments, **kwarguments))
        else:

            # Create data
            solve_data = [(i, a) for i, a in enumerate(tasks)]
            if self.am_master():
                data = []
                for j in range(self.size):
                    subdata = []
                    for i, a in enumerate(tasks):
                        if self._should_compute(i, j, self.size):
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
                new_data.append((k, func(*arguments, **kwarguments)))

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
            finished_tasks = sorted(finished_tasks, key=lambda x: x[0])
            finished_tasks = [i[1] for i in finished_tasks]

        return finished_tasks

    def _should_compute(self, i: int, rank: int, size: int) -> bool:
        """Returns whether or not the rank (process) should be used to complete the given task

        Parameters
        ----------
        i : int
            index of the task
        rank : int
            process rank
        size : int
            number of ranks

        Returns
        -------
        bool:
            true if the process should compute the task
        """
        return i % size == rank

    def _configure_parallel_resources(self) -> None:
        """Sets up the resources for parallel computing"""

        # Use mpi4py to gather resources
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        if self.am_master():
            print("Running in parallel on {} cores".format(self.size))

    def _propagate_n_only(self) -> None:
        """Propagates all the layers but only gathers the refractive index profiles"""

        # Forward through the device
        m = self.monitors[0] if len(self.monitors) else self.custom_monitors[0]
        cur_len = 0
        for per in range(self.num_periods):
            for layer in tqdm(self.layers, disable=self.quiet):

                # Get system params
                z_list = m.get_z_list(cur_len, cur_len + layer.length)
                n = layer.mode_solver.n

                # Iterate through z
                for i, z in z_list:
                    self._set_monitor(m, i, {"n": n}, n=True)

                cur_len += layer.length

    def _update_state(self, state: int) -> None:
        """Updates the state machine to the provided state

        Parameters
        ----------
        state : int
            state to move to
        """

        self.state = state
        if self.am_master() and not self.quiet:
            print("current state: {}".format(self.states[self.state]))

    def _build_input_array(
        self, left_coeffs: list, right_coeffs: list, model: Model
    ) -> dict:
        """Builds the properly formatted input array to be used to calculate field profiles from s matrices and mode coefficients

        Parameters
        ----------
        left_coeffs : list
            A list of floats that represent the mode coefficients for the left side of the full geometry
        right_coeffs : list
            A list of floats that represent the mode coefficients for the right side of the full geometry
        model : simphony.models.Model
            The simphony model base class for storing pins and scattering parameters that will be used to multiply the resulting input array

        Returns
        -------
        dict
            a mapping of the pin names found in the model to their corrisponding weights (mode coefficients) for the field profiles
        """

        # Case 1: left_coeffs > num_modes
        if len(left_coeffs) > self.activated_layers[0][0].num_modes:
            raise Exception("Too many mode coefficients in the left input")

        # Case 2: right_coeffs > num_modes
        if len(left_coeffs) > self.activated_layers[0][-1].num_modes:
            raise Exception("Too many mode coefficients in the right input")

        # Start mapping
        mapping = {}

        # Get sources
        sources = [x for i in self.get_sources().values() for x in i]

        # Form mapping
        try:
            for pin in model.pins:
                n = pin.name

                # Left global input
                if "left" in n and "dup" not in n:
                    ind = int(n[4:])
                    if ind < len(left_coeffs):
                        mapping[n] = left_coeffs[ind]
                    else:
                        mapping[n] = 0.0

                # Right global input
                if "right" in n and "dup" not in n:
                    ind = int(n[5:])
                    if ind < len(right_coeffs):
                        mapping[n] = right_coeffs[ind]
                    else:
                        mapping[n] = 0.0

                # Left monitor
                if "left" in n and "dup" in n and "to" not in n:
                    mapping[n] = 0

                # Right monitor
                if "right" in n and "dup" in n and "to" not in n:
                    mapping[n] = 0

                # Custom left source inputs
                if "left" in n and "dup" in n and "to" in n:
                    n_ = n.split("_")
                    l, _ = (n_[2], n_[4])
                    ind = int(n_[1].replace("dup", ""))
                    for i, s in enumerate(sources):
                        if s.match_label(l) and ind < len(s.mode_coeffs):
                            mapping[n] = s.mode_coeffs[ind]
                            break
                        elif i == len(sources) - 1:
                            mapping[n] = 0.0

                # Custom right source inputs
                if "right" in n and "dup" in n and "to" in n:
                    n_ = n.split("_")
                    _, r = (n_[2], n_[4])
                    ind = int(n_[1].replace("dup", ""))
                    for i, s in enumerate(sources):
                        if s.match_label(r) and ind < len(s.mode_coeffs):
                            mapping[n] = s.mode_coeffs[ind]
                            break
                        elif i == len(sources) - 1:
                            mapping[n] = 0.0

        except Exception as e:
            print(e)
            raise Exception("Improper format of sources")

        return mapping

    def _layer_field_propagate(
        self,
        i: int,
        l: Model,
        per: int,
        left_coeffs: list,
        right_coeffs: list,
        cur_len: float,
        z_list: list,
    ) -> list:
        """Propagates the fields through the current layer only. Implements the "field spider/peaker" technique of extracting fields at an arbitrary location without approximating the reflections and actually finding the fully cascaded values.

        Parameters
        ----------
        i : int
            The index of the current layer in the period
        l : Model
            The model that represents the solved layer
        per : int
            The index of the period
        left_coeffs : list
            A list of floats that represent the mode coefficients for the left side of the full geometry
        right_coeffs : list
            A list of floats that represent the mode coefficients for the right side of the full geometry
        cur_len : float
            The current length within the geometry (As it iterates through all the layers)
        z_list : list
            A list of the z points that matter for this layer

        Returns
        -------
        list
            A list of the fields for each point in the input z_list

        """

        # Create output
        result_list = []
        activated_layers = (
            self.activated_layers[per]
            if self.activated_layers[per] is not None
            else self.activated_layers[0]
        )

        # Period length
        perf = periodic_duplicate_format
        per_length = self._get_total_length()

        # get SP0
        SP0 = []
        for p in range(per):
            network = (
                self.networks[p] if self.networks[p] is not None else self.networks[0]
            )
            interface = (
                self.periodic_interfaces[p]
                if self.periodic_interfaces[p] is not None
                else self.periodic_interfaces[0]
            )
            SP0.append(perf(network, p * per_length, (p + 1) * per_length))
            SP0.append(interface)

        # get SP1
        SP1 = []
        for p in range(self.num_periods - (per) - 1):
            p = per + p + 1
            network = (
                self.networks[p] if self.networks[p] is not None else self.networks[0]
            )
            interface = (
                self.periodic_interfaces[p]
                if self.periodic_interfaces[p] is not None
                else self.periodic_interfaces[0]
            )
            SP1.append(interface)
            SP1.append(perf(network, p * per_length, (p + 1) * per_length))

        # Get S0
        S0, S0_length = ([], per * per_length)
        for lay in activated_layers[:i]:
            S0.append(perf(lay, S0_length, S0_length + lay.length))
            S0_length += lay.length

        # Get S1
        S1, S1_length = ([], S0_length + activated_layers[i].length)
        for lay in activated_layers[i + 1 :]:
            S1.append(perf(lay, S1_length, S1_length + lay.length))
            S1_length += lay.length

        # Distance params
        dup = Duplicator(l.wavelength, len(l.modes))

        # See if need to remove sources for periodic l
        checked_l = perf(l, cur_len, cur_len + l.length)

        # create all prop layers
        prop = (
            [*SP0, *S0, checked_l, dup, *S1, *SP1]
            if sum(
                ["_to_" in pin.name and "left" in pin.name for pin in checked_l.pins]
            )
            else [*SP0, *S0, dup, checked_l, *S1, *SP1]
        )

        # Compute field propagation
        S = _prop_all(
            *[
                t
                for t in prop
                if (t is not None) and not (isinstance(t, list) and not len(t))
            ]
        )

        # Get input array
        input_map = self._build_input_array(left_coeffs, right_coeffs, S)
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
        diffs = [z_list[0] - cur_len] + np.diff(z_list).tolist()
        eig = (
            (2 * np.pi) * np.array([mode.neff for mode in l.modes]) / (self.wavelength)
        )
        if sum(["_to_" in pin.name and "left" in pin.name for pin in checked_l.pins]):
            coeff_left *= np.exp(1j * eig * l.length)
            coeff_right *= np.exp(-1j * eig * l.length)

        # Iterate through z
        for z_diff in diffs:

            # Get coe
            phase_left = np.exp(-z_diff * 1j * eig)
            phase_right = np.exp(z_diff * 1j * eig)
            coeff_left = coeff_left * phase_left
            coeff_right = coeff_right * phase_right
            coeff = coeff_left + coeff_right

            # Create field
            fields_ = modes * coeff[:, np.newaxis, np.newaxis, np.newaxis]
            results = {}
            (
                results["Ex"],
                results["Ey"],
                results["Ez"],
                results["Hx"],
                results["Hy"],
                results["Hz"],
            ) = fields_.sum(0)
            results["n"] = l.modes[0].n
            result_list.append(results)

        return result_list

    def _set_monitor(self, m: Monitor, i: int, results: dict, n: bool = False) -> None:
        """Adds provided field data to the provides monitor

        Parameters
        ----------
        m : Monitor
            the monitor which to add field data to
        i : int
            the index within the monitor which to add field data
        results : dict
            a dictionary that maps field types (i.e. "Ex", "Hy", "n", etc.) to the field values as a numpy array
        n : bool
            if true, will only care about adding refractive index data to the monitor. Used for propagate_n_only and draw
        """

        # Iterate through all the results
        for key, field in results.items():

            # Really we care about where in the monitor array the field represents
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
                m[key, :, i] = (
                    field[:, int(len(field) / 2)] if field.ndim > 1 else field[:]
                )
            elif m.axes in ["yz", "zy"]:
                m[key, :, i] = (
                    field[int(len(field) / 2), :] if field.ndim > 1 else field[:]
                )
            elif m.axes in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]:
                m[key, :, :, i] = field[:, :]

    def _get_total_length(self) -> float:
        """Returns the total length of a single period of the device"""
        return np.sum([layer.length for layer in self.layers])
