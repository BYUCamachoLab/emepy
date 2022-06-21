import numpy as np
import numpy
from matplotlib import pyplot as plt
import matplotlib
from scipy.interpolate import griddata
plt.rcParams.update({'figure.max_open_warning': 0})


class Monitor(object):
    """Monitor objects store fields during propagation for user visualization. Three types of monitors exist: 3D, 2D, and 1D."""

    def __init__(
        self,
        axes: str = "xz",
        dimensions: tuple = (1, 1),
        components: list = ["E"],
        z_range: tuple = None,
        grid_x: np.array = None,
        grid_y: np.array = None,
        grid_z: np.array = None,
        location: float = None,
        sources: list = [],
        adjoint_n: bool = True,
        total_length: float = 0.0,
    ) -> None:
        """Monitor class constructor0

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z. (default: "xy")
        dimensions : tuple
            the spacial dimensions of the resulting field (default: (1,1))
        components : list
            list of the field components to store from ('E','H','Ex','Ey','Ez','Hx','Hy','Hz) (default: ["E"])
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract (default: None)
        grid_x : numpy array (default: None)
            1d x grid
        grid_y : numpy array (default: None)
            1d y grid
        grid_z : numpy array (default: None)
            1d z grid
        location : float
            the location in z if the monitor represents "xy" axes (default: None)
        sources : list[Source]
            sources to use for the monitor (default:[])
        adjoint_n : bool
            if true will use the "continuous" n used for adjoint
        """

        # Ensure z range is in proper format
        if not (axes in ["xy", "yx"]):
            try:
                if z_range is None:
                    self.start, self.end = [grid_z[0], grid_z[-1]]
                else:
                    self.start, self.end = z_range
            except Exception as e:
                raise Exception(
                    "z_range should be a tuple or list of the form (start, end)"
                    " representing the range of the z values to extract where start "
                    "and end are floats such as (0, 1) for a 1 µm range"
                ) from e

        # Check axes
        if axes == "xz" or axes == "zx":
            self.axes = "xz"
        elif axes == "yz" or axes == "zy":
            self.axes = "yz"
        elif axes in ["xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
            self.axes = "xyz"
        elif axes in ["xy", "yx"]:
            self.axes = "xy"
        else:
            raise Exception(
                f"Monitor setup {axes} has not yet been implemented. Please choose from"
                " the following implemented monitor types: ['xy','yz','xz','xyz']"
            )

        # Set parameters globally
        self.adjoint_n = adjoint_n
        self.dimensions = dimensions
        self.field = np.zeros(dimensions).astype(complex)
        self.lengths = grid_z.tolist()
        self.sources = sources
        self.left_source = False
        self.right_source = False
        self.components = components
        self.layers = {}
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.location = location
        self.total_length = total_length
        self.xy_monitors = []

    def reset_monitor(self) -> None:
        """Resets the fields in the monitor"""
        self.field *= 0
        # self.sources = []
        # self.left_source = False
        # self.right_source = False

    def get_z_list(self, start: float, end: float) -> list:
        """Finds all the points in z between start and end

        Parameters
        ----------
        start : float
            starting point in z
        end : float
            ending point in z

        Returns
        -------
        list[tuples]
            A list of tuples that take the format (i, l) where i is the index of the z point and l is the z point for all z points in the range
        """
        return [(i, l) for i, l in enumerate(self.lengths) if start <= l <= end]

    def normalize(self) -> None:
        """Normalizes the entire field to 1"""
        self.field[:-1] /= 1  # np.max(np.abs(self.field[:-1, :, 0]))

    def get_array(
        self,
        component: str = "Hy",
        axes: str = None,
        location: float = None,
        z_range: tuple = None,
        grid_x: np.array = None,
        grid_y: np.array = None,
    ) -> "numpy.ndarray":
        """Creates a matplotlib axis displaying the provides field component

        Parameters
        ----------
        component : str
            field component from "['Ex','Ey','Ez','Hx','Hy','Hz','E','H']"
        axes : str
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z.
        location : float
            if taken from 3D fields, users can specify where to take their 2D slice. If axes is 'xz', location refers to the location in y and 'yz' refers to a location in x and 'xy' refers to a location in z
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        grid_x : numpy array
            custom x grid to interpolate onto
        grid_y : numpy array
            custom y grid to interpolate onto

        Returns
        -------
        numpy array
            the requested field
        """

        # Default axes is created upon class creation
        if axes is None:
            axes = self.axes

        # Cannot get plane that was never stored during simulation
        if (self.axes != axes) and (self.axes != "xyz"):
            raise Exception(
                "Different 2D plane was stored during simulation than is being requested. Stored axes: {} different from Requested axes: {}".format(
                    self.axes, axes
                )
            )

        if axes in ["xz", "zx", "yz", "zy", "xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
            # Ensure z range is in proper format
            try:
                if z_range is None:
                    start, end = [self.start, self.end]
                else:
                    start, end = z_range
            except Exception as e:
                raise Exception(
                    "z_range should be a tuple or list of the form (start, end)"
                    "representing the range of the z values to extract where start"
                    "and end are floats such as (0, 1) for a 1 µm range"
                ) from e

            # Get start and end
            def difference_start(list_value):
                return abs(list_value - start)

            def difference_end(list_value):
                return abs(list_value - end)

            s = self.lengths.index(min(self.lengths, key=difference_start))
            e = self.lengths.index(min(self.lengths, key=difference_end)) + 1
            default_grid_z = self.lengths[s:e]

            def m(list_value):
                return abs(list_value - self.grid_z[0])

            m = self.lengths.index(min(self.lengths, key=m))
            s -= m
            e -= m

        elif axes in ["xy", "yx"]:
            default_grid_z = self.grid_z
        else:
            raise Exception("Incorrect axes format")

        # Get default x, y, z grids
        default_grid_x = self.grid_x
        default_grid_y = self.grid_y

        interp_x = False if (grid_x is None) else True
        interp_y = False if (grid_y is None) else True

        # Identify components; components=['Ex','Ey','Ez','Hx','Hy','Hz','n','E','H']
        results = {}
        for i, c in zip(range(len(self.components)), self.components):

            # Perform if needed interpolations and place components
            if self.axes == "xyz":
                # xz plane
                if axes in ["xz", "zx"]:
                    if location:

                        def d(list_value):
                            return abs(list_value - location)

                        index = list(default_grid_x).index(min(default_grid_x, key=d))
                    else:
                        index = int(len(self.field[i][0]) / 2)
                    results[c] = self.field[i][:, index, s:e]

                # yz plane
                elif axes in ["yz", "zy"]:
                    if location:

                        def d(list_value):
                            return abs(list_value - location)

                        index = list(default_grid_y).index(min(default_grid_y, key=d))
                    else:
                        index = int(len(self.field[i]) / 2)
                    results[c] = self.field[i][index, :, s:e]

                # xy plane
                elif axes in ["xy", "yx"]:
                    if location:

                        def d(list_value):
                            return abs(list_value - location)

                        index = list(default_grid_z).index(min(default_grid_z, key=d))
                    else:
                        index = 0
                    results[c] = self.field[i][:, :, index]

                # xyz volume
                elif axes in ["xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
                    results[c] = self.field[i][:, :, s:e]

            elif self.axes in ["xz", "yz", "zx", "zy"]:
                results[c] = self.field[i][:, s:e]

            elif self.axes in ["xy", "yx"]:
                results[c] = self.field[i][:, :]

        # Create E and H fields
        if component == "E":
            results["E"] = np.abs(results["Ex"]) ** 2 + np.abs(results["Ey"]) ** 2 + np.abs(results["Ez"]) ** 2
        if component == "H":
            results["H"] = np.abs(results["Hx"]) ** 2 + np.abs(results["Hy"]) ** 2 + np.abs(results["Hz"]) ** 2

        # List to return
        grid_field = []

        # Custom 2D interpolation function
        def custom_interp2d(field, old_a, old_b, new_a, new_b):
            aa, bb = np.meshgrid(new_a, new_b)
            aa_old, bb_old = np.meshgrid(old_a, old_b)
            points = np.array((aa_old.flatten(), bb_old.flatten())).T
            real = griddata(points, np.real(field).flatten(), (aa, bb)).astype(np.complex128)
            imag = griddata(points, np.real(field).flatten(), (aa, bb)).astype(np.complex128)
            return real + 1j * imag

        # Custom 3D interpolation function
        def custom_interp3d(field, old_a, old_b, old_c, new_a, new_b, new_c):
            aa, bb, cc = np.meshgrid(new_a, new_b, new_c)
            aa_old, bb_old, cc_old = np.meshgrid(old_a, old_b, old_c)
            points = np.array((aa_old.flatten(), bb_old.flatten(), cc_old.flatten())).T
            return griddata(points, np.real(field), (aa, bb, cc)).astype(np.complex128) + 1j * griddata(
                points, np.real(field), (aa, bb)
            ).astype(np.complex128)

        # Add to return list the grid
        if axes in ["xz", "zx"]:
            x = default_grid_x if not interp_x else grid_x
            z = default_grid_z
            grid_field.append(np.array(x))
            grid_field.append(np.array(z))
            if interp_x:
                results[component] = custom_interp2d(results[component], default_grid_z, default_grid_x, z, x)
        elif axes in ["yz", "zy"]:
            y = default_grid_y if not interp_y else grid_y
            z = default_grid_z
            grid_field.append(np.array(y))
            grid_field.append(np.array(z))
            if interp_y:
                results[component] = custom_interp2d(results[component], default_grid_z, default_grid_y, z, y)
        elif axes in ["xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
            x = default_grid_x if not interp_x else grid_x
            y = default_grid_y if not interp_y else grid_y
            z = default_grid_z
            grid_field.append(np.array(x))
            grid_field.append(np.array(y))
            grid_field.append(np.array(z))
            if interp_x or interp_y:
                results[component] = custom_interp3d(
                    results[component], default_grid_x, default_grid_y, default_grid_z, x, y, z
                )
        elif axes in ["xy", "yx"]:
            x = default_grid_x if not interp_x else grid_x
            y = default_grid_y if not interp_y else grid_y
            grid_field.append(np.array(x))
            grid_field.append(np.array(y))
            if interp_x or interp_y:
                results[component] = custom_interp2d(results[component], default_grid_x, default_grid_y, x, y)
        else:
            raise Exception("Please choose valid axes")

        # Add to return list the field
        grid_field.append(np.array(results[component]))

        return grid_field

    def get_source_visual(self, min, max) -> "numpy.ndarray":
        """Returns a mask with lines indicating where a source is"""
        srcs = []
        if self.right_source:
            srcx = [self.total_length, self.total_length]
            srcy = [min, max]
            srcs.append((srcx, srcy))
        if self.left_source:
            srcx = [0, 0]
            srcy = [min, max]
            srcs.append((srcx, srcy))
        for source in self.sources:
            srcx = [source.z, source.z]
            srcy = [min, max]
            srcs.append((srcx, srcy))

        return srcs

    def get_xy_monitor_visual(self, min, max) -> "numpy.ndarray":
        """Returns a mask with lines indicating where a source is"""
        xy_monitors = []
        for monitor in self.xy_monitors:
            srcx = [monitor.location, monitor.location]
            srcy = [min, max]
            xy_monitors.append((srcx, srcy))

        return xy_monitors

    def visualize(
        self,
        ax: matplotlib.image.AxesImage = None,
        component: str = "Hy",
        axes: str = None,
        location: float = 0,
        z_range: tuple = None,
        show_geometry: bool = True,
        show_sources: bool = True,
        show_xy_monitors: bool = False,
    ) -> "matplotlib.image.AxesImage":
        """Creates a matplotlib axis displaying the provides field component

        Parameters
        ----------
        ax : matplotlib axis
            the axis object created when calling plt.figure() or plt.subplots(), if None (default) then the plt interface will be used
        component : string
            field component from "['Ex','Ey','Ez','Hx','Hy','Hz','E','H']"
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z.
        location : float
            if taken from 3D fields, users can specify where to take their 2D slice. If axes is 'xz', location refers to the location in y and 'yz' refers to a location in x and 'xy' refers to a location in z.
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        show_geometry : bool
            if true, will display the geometry faintly under the field profiles (default: True)
        show_sources : bool
            if true, will display a red line indicating where a source is (default: True)

        Returns
        -------
        matplotlib.image.AxesImage
            the image used to plot the index profile
        """

        # Only 2D fields can be generated
        if axes in ["xyz", "yxz", "xzy", "yzx", "zxy", "zyx"]:
            raise Exception(
                "3D fields can be extracted using get_array or visualized on a 2D plane via an axes of xz or yz"
            )

        if axes is None:
            axes = self.axes

        if axes in ["xz", "zx"]:
            axes = "xz"
        elif axes in ["yz", "zy"]:
            axes = "yz"
        elif axes in ["xy", "yx"]:
            axes = "xy"
        else:
            raise Exception("Incorrect axes format")

        yn, zn, n = self.get_array(component="n", axes=axes, location=location, z_range=z_range)
        y, z, field = self.get_array(component=component, axes=axes, location=location, z_range=z_range)

        # Color map lookup table
        cmap_lookup = {
            "Ex": "RdBu",
            "Ey": "RdBu",
            "Ez": "RdBu",
            "Hx": "RdBu",
            "Hy": "RdBu",
            "Hz": "RdBu",
            "E": "Blues",
            "H": "Blues",
            "n": "Greys",
        }

        # Create plots
        if axes in ["xz", "zx", "yz", "zy"]:

            # Underlay the geometry
            show = ax if ax else plt
            if show_geometry:
                show.imshow(
                    np.real(n[::-1]),
                    extent=[np.real(zn[0]), np.real(zn[-1]), np.real(yn[0]), np.real(yn[-1])],
                    cmap=cmap_lookup["n"],
                )
            vmin, vmax = (np.real(np.min(field)), np.real(np.max(field)))
            alpha = 1 if not show_geometry else 0.85

            # Plot sources
            if show_sources:
                srcs = self.get_source_visual(np.real(y[0]), np.real(y[-1]))
                for srcx, srcy in srcs:
                    plt.plot(srcx, srcy, color="red", linewidth=2)

            # Plot xy monitors
            if show_xy_monitors:
                xy_monitors = self.get_xy_monitor_visual(np.real(y[0]), np.real(y[-1]))
                for locx, locy in xy_monitors:
                    plt.plot(locx, locy, color="blue", linewidth=2)

            im = show.imshow(
                np.real(field[::-1]),
                extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                cmap=cmap_lookup[component],
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )

            # Assign labels
            if not ax:
                plt.xlabel(np.real(axes[1]))
                plt.ylabel(np.real(axes[0]))
                plt.title(component)
            else:
                ax.set_xlabel(np.real(axes[1]))
                ax.set_ylabel(np.real(axes[0]))
                ax.set_title(component)

        # xy fields
        else:
            if ax:
                im = ax.imshow(
                    np.rot90(np.real(field)),
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                ax.set_xlabel(np.real(axes[0]))
                ax.set_ylabel(np.real(axes[1]))
                ax.set_title(component)
            else:
                im = plt.imshow(
                    np.rot90(np.real(field)),
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                plt.xlabel(np.real(axes[0]))
                plt.ylabel(np.real(axes[1]))
                plt.title(component)

        return im

    def __getitem__(self, subscript):
        return self.field[subscript]

    def __setitem__(self, subscript, item):
        self.field[subscript] = item

    def __delitem__(self, subscript):
        del self.field[subscript]
