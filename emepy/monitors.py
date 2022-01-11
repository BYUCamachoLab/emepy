import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from scipy.interpolate import griddata


class Monitor(object):
    """Monitor objects store fields during propagation for user visualization. Three types of monitors exist: 3D, 2D, and 1D."""

    def __init__(
        self,
        axes="xz",
        dimensions=(1, 1),
        lengths=[],
        components=["E"],
        z_range=None,
        grid_x=None,
        grid_y=None,
        grid_z=None,
        location=None,
    ):
        """Monitor class constructor

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z.
        dimensions : tuple
            the spacial dimensions of the resulting field
        lengths : list
            list of the remaining points in z at which to calculate the fields
        components : list
            list of the field components to store from ('E','H','Ex','Ey','Ez','Hx','Hy','Hz)
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract
        grid_x : numpy array
            1d x grid
        grid_y : numpy array
            1d y grid
        grid_z : numpy array
            1d z grid
        """

        # Ensure z range is in proper format
        if not (axes in ["xy", "yx"]):
            try:
                if z_range is None:
                    self.start, self.end = [lengths[0][0], lengths[0][-1]]
                else:
                    self.start, self.end = z_range
            except Exception as e:
                raise Exception(
                    "z_range should be a tuple or list of the form (start, end)"
                    " representing the range of the z values to extract where start "
                    "and end are floats such as (0, 1e-6) for a 1 µm range"
                ) from e

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

        self.dimensions = dimensions
        self.field = np.zeros(dimensions).astype(complex)
        self.lengths = copy(lengths)
        self.remaining_lengths = copy(lengths)

        self.cur_prop_index = [0 for i in range(len(components))]
        self.cur_length = [0 for i in range(len(components))]
        self.components = components
        self.layers = {}
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

    def __getitem__(self, subscript):
        return self.field[subscript]

    def __setitem__(self, subscript, item):

        # Only set item if it's in the valid z_range
        if self.axes in ["xy", "yx"]:
            self.field[subscript] = item
        elif (self.lengths[0][subscript[-1]] >= self.start) and (self.lengths[0][subscript[-1]] <= self.end):
            difference_start = lambda list_value: abs(list_value - self.start)
            s = self.lengths[0].index(min(self.lengths[0], key=difference_start))
            subscript = tuple(list(subscript[:-1]) + [subscript[-1] - s])
            self.field[subscript] = item

        if len(self.remaining_lengths[int(subscript[0])]) > 1:
            self.remaining_lengths[int(subscript[0])] = self.remaining_lengths[int(subscript[0])][1:]
            self.cur_prop_index[int(subscript[0])] += 1
        else:
            self.remaining_lengths[int(subscript[0])] = []
            self.cur_prop_index[int(subscript[0])] += 1

    def __delitem__(self, subscript):
        del self.field[subscript]

    def normalize(self):
        self.field[:-1] /= 1  # np.max(np.abs(self.field[:-1, :, 0]))

    def get_array(self, component="Hy", axes=None, location=None, z_range=None, grid_x=None, grid_y=None):
        """Creates a matplotlib axis displaying the provides field component

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z.
        component : string
            field component from "['Ex','Ey','Ez','Hx','Hy','Hz','E','H']"
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
                    "and end are floats such as (0, 1e-6) for a 1 µm range"
                ) from e

            # Get start and end
            def difference_start(list_value):
                return abs(list_value - start)

            def difference_end(list_value):
                return abs(list_value - end)

            s = self.lengths[0].index(min(self.lengths[0], key=difference_start))
            e = self.lengths[0].index(min(self.lengths[0], key=difference_end)) + 1
            default_grid_z = self.lengths[0][s:e]

            def m(list_value):
                return abs(list_value - self.grid_z[0])

            m = self.lengths[0].index(min(self.lengths[0], key=m))
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

    def visualize(self, ax=None, component="Hy", axes=None, location=0, z_range=None):
        """Creates a matplotlib axis displaying the provides field component

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Note, propagation is always in z.
        ax : matplotlib axis
            the axis object created when calling plt.figure() or plt.subplots(), if None (default) then the plt interface will be used
        component : string
            field component from "['Ex','Ey','Ez','Hx','Hy','Hz','E','H']"
        location : float
            if taken from 3D fields, users can specify where to take their 2D slice. If axes is 'xz', location refers to the location in y and 'yz' refers to a location in x and 'xy' refers to a location in z.
        z_range : tuple
            tuple or list of the form (start, end) representing the range of the z values to extract

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
            if ax:
                im = ax.imshow(
                    np.real(field),
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                ax.set_xlabel(np.real(axes[1]))
                ax.set_ylabel(np.real(axes[0]))
                ax.set_title(component)
            else:
                im = plt.imshow(
                    np.real(field),
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                plt.xlabel(np.real(axes[1]))
                plt.ylabel(np.real(axes[0]))
                plt.title(component)
        else:
            if ax:
                im = ax.imshow(
                    np.real(field).T,
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                ax.set_xlabel(np.real(axes[0]))
                ax.set_ylabel(np.real(axes[1]))
                ax.set_title(component)
            else:
                im = plt.imshow(
                    np.real(field).T,
                    extent=[np.real(z[0]), np.real(z[-1]), np.real(y[0]), np.real(y[-1])],
                    cmap=cmap_lookup[component],
                )
                plt.xlabel(np.real(axes[0]))
                plt.ylabel(np.real(axes[1]))
                plt.title(component)

        return im
