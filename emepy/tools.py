import numpy as np
import os
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
import EMpy_gpu
import collections
from emepy.materials import *
from typing import Callable
from shapely.geometry import Polygon, Point


def polygon_to_n_2D(
    polygon: "Polygon", x: list, y: list, subpixel=True, core_index=3.4, cladding_index=1.4
) -> "np.ndarray":
    # Create grid
    xx, yy = np.meshgrid(x, y)
    n = np.zeros(xx.shape)[:-1, :-1].T

    # Apply subpixel
    xlower, xupper = (x[:-1], x[1:])
    zlower, zupper = (y[:-1], y[1:])
    for i, xp in enumerate(zip(xlower, xupper)):
        for j, zp in enumerate(zip(zlower, zupper)):

            # Upper and lower points
            xl, xu = xp
            zl, zu = zp
            total_area = (xu - xl) * (zu - zl)

            # Create polygon of the pixel
            pixel_poly = Polygon([(xl, zl), (xl, zu), (xu, zu), (xu, zl)])

            # Get overlapping area
            overlapping_area = 0
            if pixel_poly.intersects(polygon):
                overlapping_area = pixel_poly.intersection(polygon).area

            # Calculate effective index
            if subpixel:
                n[i, j] = (
                    overlapping_area / total_area * core_index + (1 - overlapping_area / total_area) * cladding_index
                )
            elif overlapping_area:
                n[i, j] = core_index
            else:
                n[i, j] = cladding_index

    return n


def vertices_to_n(vertices: list, x: list, y: list, subpixel=True, core_index=3.4, cladding_index=1.4) -> "np.ndarray":
    """
    Takes vertices of a polygon and maps it to a grid using or not using subpixel smoothing
    """

    polygon = Polygon(vertices)
    return polygon_to_n_2D(polygon, x, y, subpixel, core_index, cladding_index)


def rectangle_to_n(
    center: tuple, width: float, thickness: float, x: list, y: list, subpixel=True, core_index=3.4, cladding_index=1.4
) -> "np.ndarray":

    xc, yc = center
    vertices = [
        (xc - width / 2, yc - thickness / 2),
        (xc - width / 2, yc + thickness / 2),
        (xc + width / 2, yc + thickness / 2),
        (xc + width / 2, yc - thickness / 2),
    ]
    return vertices_to_n(vertices, x, y, subpixel, core_index, cladding_index)


def circle_to_n(
    center: tuple, radius: float, x: list, y: list, subpixel=True, core_index=3.4, cladding_index=1.4
) -> "np.ndarray":

    polygon = Point(*center).buffer(radius)
    return polygon_to_n_2D(polygon, x, y, subpixel, core_index, cladding_index)


class get_epsfunc(object):
    """Callable class for getting epsilon on a grid"""

    def __init__(
        self,
        width: float,
        thickness: float,
        cladding_width: float,
        cladding_thickness: float,
        core_index: float,
        cladding_index: float,
        compute: bool = False,
        profile: "np.ndarray" = None,
        nx: int = None,
        ny: int = None,
    ):
        """Returns the epsfunc for given parameters for a rectangular waveguide

        Parameters
        ----------
        width: float
            the width of the geometry
        thickness: float
            the thickess of the geometry
        cladding_width: float
            the width of the surrounding cladding (note this is total width and should be > core width)
        cladding_thickness: float
            the thickness of the surrounding cladding (note this is total thickness and should be > core thickness)
        core_index: float
            refractive index of the core
        cladding_index: float
            refractive index of the cladding
        compute: bool = False
            if true, will not place rectangle at center. This should only be necessary for compute_other_fields
        profile: "np.ndarray" = None
            the refractive index profile (note if providing a width, this should be left None)
        nx: int = None
            number of points in the x direction
        ny: int = None
            number of points in the y direction
        subpixel: bool = True
            if true, will use subpixel smoothing, assuming asking for a waveguide cross section and not providing an index map (recommended)

        Returns
        -------
        function
            an epsfunc function that takes an x,y grid arrays and returns the refractive index profile
        """
        self.width = width
        self.thickness = thickness
        self.cladding = cladding_width
        self.cladding = cladding_thickness
        self.core_index = core_index
        self.cladding_index = cladding_index
        self.compute = compute
        self.profile = profile
        self.nx = nx
        self.ny = ny

        if (width is not None) and (thickness is not None):
            self.epsfunction = self.epsfunc_2D_1

        elif (width is None) and (thickness is not None) and (profile is not None):
            self.epsfunction = self.epsfunc_2D_2

        elif (width is None) and (thickness is None) and (profile is not None):
            self.epsfunction = self.epsfunc_2D_3

        elif (thickness is None) and (width is not None):
            self.epsfunction = self.epsfunc_1D_1

        elif (thickness is None) and (width is None) and (profile is not None):
            self.epsfunction = self.epsfunc_1D_2

        else:
            raise Exception(
                "Need to provide width & thickness, or 1D profile and thickness, or 2D profile, or width for 1D, or 1D profile for 1D"
            )

    def __call__(self, x_, y_):
        return self.epsfunction(x_, y_)

    # Case 1 : width and thickness are defined
    def epsfunc_2D_1(self, x_, y_):
        """Return a matrix describing a 2d material.

        Parameters
        ----------
        x_: numpy array
            x values
        y_: numpy array
            y values

        Returns
        -------
        numpy array
            2d-matrix
        """
        xx, yy = np.meshgrid(x_, y_)
        if self.compute:
            n = np.where(
                (np.abs(np.real(xx.T) - self.cladding_width * 0.5) <= self.width * 0.5)
                * (np.abs(np.real(yy.T) - self.cladding_thickness * 0.5) <= self.thickness * 0.5),
                self.core_index ** 2 + 0j,
                self.cladding_index ** 2 + 0j,
            )
        else:
            n = np.where(
                (np.abs(np.real(xx.T)) <= self.width * 0.5) * (np.abs(np.real(yy.T)) <= self.thickness * 0.5),
                self.core_index ** 2 + 0j,
                self.cladding_index ** 2 + 0j,
            )
        return n

    # Case 2 : thickness and 1D n is defined
    def epsfunc_2D_2(self, x_, y_):

        n = self.profile
        xx, yy = np.meshgrid(np.real(x_), np.real(y_))
        n = np.interp(np.real(x_), np.real(self.nx), n).astype(complex)
        n = np.repeat(n, len(y_)).reshape((len(n), len(y_)))
        n = np.where((np.abs(np.real(yy.T)) <= self.thickness * 0.5), n, self.cladding_index + 0j) ** 2
        return n

    # Case 3 : 2D n is defined
    def epsfunc_2D_3(self, x_, y_):

        xxn, yyn = np.meshgrid(np.real(self.nx), np.real(self.ny))
        points = np.array((xxn.flatten(), yyn.flatten())).T
        n = self.profile.flatten()
        xx, yy = np.meshgrid(np.real(x_), np.real(y_))
        n_real = griddata(points, np.real(n), (xx, yy))
        n_imag = griddata(points, np.imag(n), (xx, yy))
        n = (n_real + 1j * n_imag) ** 2
        return n

    # Case 4: width only
    def epsfunc_1D_1(self, x_, y_):

        n = np.where(
            (np.abs(np.real(x_)) <= self.width * 0.5), self.core_index ** 2 + 0j, self.cladding_index ** 2 + 0j
        ).reshape(len(x_), 1)

        return n

    # Case 5: 1D n only
    def epsfunc_1D_2(self, x_, y_):

        n = np.interp(np.real(x_), np.real(self.nx), self.profile).astype(complex).reshape(len(x_), 1)
        return n


def get_isotropic_epsfunc(
    epsfunc_xx, epsfunc_yy, epsfunc_zz, epsfunc_xy=None, epsfunc_yx=None
) -> Callable[["np.ndarray", "np.ndarray"], "np.ndarray"]:
    """Returns an epsfunction for an isotropic medium"""

    def epsfunc_iso(x_, y_):
        xx = epsfunc_xx(x_, y_)
        yy = epsfunc_yy(x_, y_)
        zz = epsfunc_zz(x_, y_)
        xy = epsfunc_xy(x_, y_) if epsfunc_xy is not None else 0 * xx
        yx = epsfunc_yx(x_, y_) if epsfunc_yx is not None else 0 * xx

        return np.array([xx, xy, yx, yy, zz])

    return epsfunc_iso


def create_polygon(x: "np.ndarray", y: "np.ndarray", n: "np.ndarray", detranslate: bool = True) -> list:
    """Given a grid and a refractive index profile, will return the vertices of the polygon for importing into libraries such as Lumerical

    Parameters
    ----------
    x : "np.ndarray"
        the x grid
    y : "np.ndarray"
        the y grid
    n :"np.ndarray"
        the refractive index profile
    detranslate : bool
        if True, will detranslate the vertices

    Returns
    -------
    list[tuples]
        the resulting vertices

    """

    x0, y0 = [x.copy(), y.copy()]
    diff = np.abs(np.diff(n, axis=1))
    where = np.argwhere(diff > np.mean(n))
    tx = np.mean(x[where[:, 0]])
    ty = np.mean(y[where[:, 1]])
    x0 -= tx
    y0 -= ty
    diff = diff[1:, :] + diff[:-1, :]
    diff2 = np.abs(np.diff(n, axis=0))
    diff2 = diff2[:, 1:] + diff2[:, :-1]
    diff = diff + diff2
    diff = np.where(diff, 1, 0)

    newd = {}
    for i in np.argwhere(diff):
        angle = np.angle(x0[i[0]] + 1j * y0[i[1]])
        if detranslate:
            newd[angle] = [y0[i[1]] - ty, x0[i[0]] - tx]
        else:
            newd[angle] = [y0[i[1]], x0[i[0]]]

    od = collections.OrderedDict(sorted(newd.items()))

    if detranslate:
        return np.array(list(od.values())).astype(float)
    else:
        return [np.array(list(od.values())).astype(float), tx, ty]


def interp(
    x: "np.ndarray", y: "np.ndarray", x0: "np.ndarray", y0: "np.ndarray", f: "np.ndarray", centered: bool
) -> "np.ndarray":
    """Interpolate a 2D complex array.

    Parameters
    ----------
    x:"np.ndarray"
        the new x grid
    y:"np.ndarray"
        the new y grid
    x0:"np.ndarray"
        the original x grid
    y0:"np.ndarray"
        the original y grid
    f:"np.ndarray"
        the field to interpolate
    centered:bool
        whether or not it needs to stil be shifted

    Returns
    -------
    np.ndarray
        the interpolated field
    """

    if centered:
        # electric fields and intensity are centered
        x0 = EMpy_gpu.utils.centered1d(x0)
        y0 = EMpy_gpu.utils.centered1d(y0)

    f1r = np.zeros((len(x0), len(y)))
    f1i = np.zeros((len(x0), len(y)))
    for ix0 in range(len(x0)):
        f1r[ix0, :] = np.interp(y, y0, np.real(f[ix0, :]))
        f1i[ix0, :] = np.interp(y, y0, np.imag(f[ix0, :]))
    fr = np.zeros((len(x), len(y)))
    fi = np.zeros((len(x), len(y)))
    for iy in range(len(y)):
        fr[:, iy] = np.interp(x, x0, f1r[:, iy])
        fi[:, iy] = np.interp(x, x0, f1i[:, iy])
    return fr + 1j * fi


def interp1d(x: "np.ndarray", x0: "np.ndarray", f: "np.ndarray", centered: bool) -> "np.ndarray":
    """Interpolate a 1D complex array.

    Parameters
    ----------
    x:"np.ndarray"
        the new grid
    x0:"np.ndarray"
        the original grid
    f:"np.ndarray"
        the field to interpolate
    centered:bool
        whether or not it needs to stil be shifted

    Returns
    -------
    np.ndarray
        the interpolated field
    """

    if centered:
        # electric fields and intensity are centered
        x0 = EMpy_gpu.utils.centered1d(x0)

    f1r = np.interp(x, x0, np.real(f[:]))
    f1i = np.interp(x, x0, np.imag(f[:]))
    return f1r + 1j * f1i


def into_chunks(location: str, name: str, chunk_size: int = 20000000) -> None:
    """Takes a large serialized file and breaks it up into smaller chunk files

    Parameters
    ---------
    location : string
        the absolute or relative path of the large file
    name : string
        the name of the serialized smaller components (will have _chunk_# appended to it)
    chunk_size : int
        how big each save chunk should be
    """
    CHUNK_SIZE = chunk_size
    f = open(location, "rb")
    chunk = f.read(CHUNK_SIZE)
    count = 0
    while chunk:  # loop until the chunk is empty (the file is exhausted)
        with open(name + "_chunk_" + str(count), "wb+") as w:
            w.write(chunk)
        count += 1
        chunk = f.read(CHUNK_SIZE)  # read the next chunk
    f.close()


def from_chunks(location: str, name: str) -> None:
    """Takes a directory of serialized chunks that were made using into_chunks and combines them back into a large serialized file

    Parameters
    ----------
    location : string
        the path of the directory where the chunks are located
    name : string
        the name of the serialized file to create (make sure to include file extension if it matters)
    """
    if location[-1] != "/":
        location += "/"
    f = open(name, "wb+")
    direc = os.listdir(location)
    keys = [int(d[9:]) for d in direc]
    dic = dict(zip(keys, direc))
    for i in sorted(dic):
        f.write(open(location + dic[i], "rb").read())
    f.close()


def _get_eps(
    xc: "np.ndarray", yc: "np.ndarray", epsfunc: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"]
) -> tuple:
    """Used by compute_other_fields and adapted from the EMpy library"""
    tmp = epsfunc(xc, yc)

    def _reshape(tmp):
        """pads the array by duplicating edge values"""
        tmp = np.c_[tmp[:, 0:1], tmp, tmp[:, -1:]]
        tmp = np.r_[tmp[0:1, :], tmp, tmp[-1:, :]]
        return tmp

    if tmp.ndim == 2:
        tmp = _reshape(tmp)
        epsxx = epsyy = epszz = tmp
        epsxy = epsyx = np.zeros_like(epsxx)

    elif tmp.ndim == 3:
        assert tmp.shape[2] == 5, "eps must be NxMx5"
        epsxx = _reshape(tmp[:, :, 0])
        epsxy = _reshape(tmp[:, :, 1])
        epsyx = _reshape(tmp[:, :, 2])
        epsyy = _reshape(tmp[:, :, 3])
        epszz = _reshape(tmp[:, :, 4])

    else:
        raise ValueError("Invalid eps")

    return epsxx, epsxy, epsyx, epsyy, epszz


def compute_other_fields_2D(
    neff: float,
    Hx: "np.ndarray",
    Hy: "np.ndarray",
    wl: float,
    x: "np.ndarray",
    y: "np.ndarray",
    boundary: str,
    epsfunc: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"],
) -> list:
    """Given the Hx and Hy fields, will calculate the rest

    Parameters
    ----------
    neff: float
        the effective index of the EigenMode
    Hx: "np.ndarray"
        the Hx field profile
    Hy: "np.ndarray"
        the Hy field profile
    wl: float
        the wavelength of the simulation
    x: "np.ndarray"
        the x numpy grid array
    y: "np.ndarray"
        the y numpy grid array
    boundary: str
        a string representing the boundary conditions as defined for the electromagneticpython library
    epsfunc: function
        a function of (x,y) that can be called to return the refractive index field profile for the domain

    Returns
    -------
    list
        a list of the field profile np.ndarrays [Hx,Hy,Hz,Ex,Ey,Ez]
    """

    dx = np.diff(x)
    dy = np.diff(y)

    dx = np.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
    dy = np.r_[dy[0], dy, dy[-1]].reshape(1, -1)

    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    epsxx, epsxy, epsyx, epsyy, epszz = _get_eps(xc, yc, epsfunc)

    nx = len(x)
    ny = len(y)

    k = 2 * np.pi / wl

    ones_nx = np.ones((nx, 1))
    ones_ny = np.ones((1, ny))

    n = np.dot(ones_nx, dy[:, 1:]).flatten()
    s = np.dot(ones_nx, dy[:, :-1]).flatten()
    e = np.dot(dx[1:, :], ones_ny).flatten()
    w = np.dot(dx[:-1, :], ones_ny).flatten()

    exx1 = epsxx[:-1, 1:].flatten()
    exx2 = epsxx[:-1, :-1].flatten()
    exx3 = epsxx[1:, :-1].flatten()
    exx4 = epsxx[1:, 1:].flatten()

    eyy1 = epsyy[:-1, 1:].flatten()
    eyy2 = epsyy[:-1, :-1].flatten()
    eyy3 = epsyy[1:, :-1].flatten()
    eyy4 = epsyy[1:, 1:].flatten()

    exy1 = epsxy[:-1, 1:].flatten()
    exy2 = epsxy[:-1, :-1].flatten()
    exy3 = epsxy[1:, :-1].flatten()
    exy4 = epsxy[1:, 1:].flatten()

    eyx1 = epsyx[:-1, 1:].flatten()
    eyx2 = epsyx[:-1, :-1].flatten()
    eyx3 = epsyx[1:, :-1].flatten()
    eyx4 = epsyx[1:, 1:].flatten()

    ezz1 = epszz[:-1, 1:].flatten()
    ezz2 = epszz[:-1, :-1].flatten()
    ezz3 = epszz[1:, :-1].flatten()
    ezz4 = epszz[1:, 1:].flatten()

    b = neff * k

    bzxne = (
        0.5
        * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
        * eyx4
        / ezz4
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy3
        * eyy1
        * w
        * eyy2
        + 0.5
        * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
        * (1 - exx4 / ezz4)
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * exx1
        * s
    ) / b

    bzxse = (
        -0.5
        * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
        * eyx3
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy1
        * w
        * eyy2
        + 0.5
        * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
        * (1 - exx3 / ezz3)
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * n
        * exx1
        * exx4
    ) / b

    bzxnw = (
        -0.5
        * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
        * eyx1
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy2
        * e
        - 0.5
        * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
        * (1 - exx1 / ezz1)
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * exx4
        * s
    ) / b

    bzxsw = (
        0.5
        * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
        * eyx2
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * e
        - 0.5
        * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
        * (1 - exx2 / ezz2)
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx3
        * n
        * exx1
        * exx4
    ) / b

    bzxn = (
        (
            0.5
            * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * n
            * ezz1
            * ezz2
            / eyy1
            * (2 * eyy1 / ezz1 / n ** 2 + eyx1 / ezz1 / n / w)
            + 0.5
            * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * n
            * ezz4
            * ezz3
            / eyy4
            * (2 * eyy4 / ezz4 / n ** 2 - eyx4 / ezz4 / n / e)
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * (
                0.5
                * ezz4
                * ((1 - exx1 / ezz1) / n / w - exy1 / ezz1 * (2.0 / n ** 2 - 2 / n ** 2 * s / (n + s)))
                / exx1
                * ezz1
                * w
                + (ezz4 - ezz1) * s / n / (n + s)
                + 0.5
                * ezz1
                * (-(1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (2.0 / n ** 2 - 2 / n ** 2 * s / (n + s)))
                / exx4
                * ezz4
                * e
            )
            - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * (
                -ezz3 * exy2 / n / (n + s) / exx2 * w
                + (ezz3 - ezz2) * s / n / (n + s)
                - ezz2 * exy3 / n / (n + s) / exx3 * e
            )
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzxs = (
        (
            0.5
            * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * s
            * ezz2
            * ezz1
            / eyy2
            * (2 * eyy2 / ezz2 / s ** 2 - eyx2 / ezz2 / s / w)
            + 0.5
            * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * s
            * ezz3
            * ezz4
            / eyy3
            * (2 * eyy3 / ezz3 / s ** 2 + eyx3 / ezz3 / s / e)
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * (
                -ezz4 * exy1 / s / (n + s) / exx1 * w
                - (ezz4 - ezz1) * n / s / (n + s)
                - ezz1 * exy4 / s / (n + s) / exx4 * e
            )
            - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * (
                0.5
                * ezz3
                * (-(1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (2.0 / s ** 2 - 2 / s ** 2 * n / (n + s)))
                / exx2
                * ezz2
                * w
                - (ezz3 - ezz2) * n / s / (n + s)
                + 0.5
                * ezz2
                * ((1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (2.0 / s ** 2 - 2 / s ** 2 * n / (n + s)))
                / exx3
                * ezz3
                * e
            )
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzxe = (
        (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
        * (
            0.5 * n * ezz4 * ezz3 / eyy4 * (2.0 / e ** 2 - eyx4 / ezz4 / n / e)
            + 0.5 * s * ezz3 * ezz4 / eyy3 * (2.0 / e ** 2 + eyx3 / ezz3 / s / e)
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            -0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz1 * (1 - exx4 / ezz4) / n / exx4 * ezz4
            - 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz2 * (1 - exx3 / ezz3) / s / exx3 * ezz3
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzxw = (
        (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
        * (
            0.5 * n * ezz1 * ezz2 / eyy1 * (2.0 / w ** 2 + eyx1 / ezz1 / n / w)
            + 0.5 * s * ezz2 * ezz1 / eyy2 * (2.0 / w ** 2 - eyx2 / ezz2 / s / w)
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            0.5 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) * ezz4 * (1 - exx1 / ezz1) / n / exx1 * ezz1
            + 0.5 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) * ezz3 * (1 - exx2 / ezz2) / s / exx2 * ezz2
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzxp = (
        (
            (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * (
                0.5
                * n
                * ezz1
                * ezz2
                / eyy1
                * (-2.0 / w ** 2 - 2 * eyy1 / ezz1 / n ** 2 + k ** 2 * eyy1 - eyx1 / ezz1 / n / w)
                + 0.5
                * s
                * ezz2
                * ezz1
                / eyy2
                * (-2.0 / w ** 2 - 2 * eyy2 / ezz2 / s ** 2 + k ** 2 * eyy2 + eyx2 / ezz2 / s / w)
            )
            + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * (
                0.5
                * n
                * ezz4
                * ezz3
                / eyy4
                * (-2.0 / e ** 2 - 2 * eyy4 / ezz4 / n ** 2 + k ** 2 * eyy4 + eyx4 / ezz4 / n / e)
                + 0.5
                * s
                * ezz3
                * ezz4
                / eyy3
                * (-2.0 / e ** 2 - 2 * eyy3 / ezz3 / s ** 2 + k ** 2 * eyy3 - eyx3 / ezz3 / s / e)
            )
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * (
                0.5
                * ezz4
                * (
                    -(k ** 2) * exy1
                    - (1 - exx1 / ezz1) / n / w
                    - exy1 / ezz1 * (-2.0 / n ** 2 - 2 / n ** 2 * (n - s) / s)
                )
                / exx1
                * ezz1
                * w
                + (ezz4 - ezz1) * (n - s) / n / s
                + 0.5
                * ezz1
                * (
                    -(k ** 2) * exy4
                    + (1 - exx4 / ezz4) / n / e
                    - exy4 / ezz4 * (-2.0 / n ** 2 - 2 / n ** 2 * (n - s) / s)
                )
                / exx4
                * ezz4
                * e
            )
            - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * (
                0.5
                * ezz3
                * (
                    -(k ** 2) * exy2
                    + (1 - exx2 / ezz2) / s / w
                    - exy2 / ezz2 * (-2.0 / s ** 2 + 2 / s ** 2 * (n - s) / n)
                )
                / exx2
                * ezz2
                * w
                + (ezz3 - ezz2) * (n - s) / n / s
                + 0.5
                * ezz2
                * (
                    -(k ** 2) * exy3
                    - (1 - exx3 / ezz3) / s / e
                    - exy3 / ezz3 * (-2.0 / s ** 2 + 2 / s ** 2 * (n - s) / n)
                )
                / exx3
                * ezz3
                * e
            )
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzyne = (
        0.5
        * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
        * (1 - eyy4 / ezz4)
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy3
        * eyy1
        * w
        * eyy2
        + 0.5
        * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
        * exy4
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * exx1
        * s
    ) / b

    bzyse = (
        -0.5
        * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
        * (1 - eyy3 / ezz3)
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy1
        * w
        * eyy2
        + 0.5
        * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
        * exy3
        / ezz3
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * n
        * exx1
        * exx4
    ) / b

    bzynw = (
        -0.5
        * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
        * (1 - eyy1 / ezz1)
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy2
        * e
        - 0.5
        * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
        * exy1
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * exx4
        * s
    ) / b

    bzysw = (
        0.5
        * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
        * (1 - eyy2 / ezz2)
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * e
        - 0.5
        * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
        * exy2
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx3
        * n
        * exx1
        * exx4
    ) / b

    bzyn = (
        (
            0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz1 * ezz2 / eyy1 * (1 - eyy1 / ezz1) / w
            - 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz4 * ezz3 / eyy4 * (1 - eyy4 / ezz4) / e
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
        * (
            0.5 * ezz4 * (2.0 / n ** 2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w
            + 0.5 * ezz1 * (2.0 / n ** 2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzys = (
        (
            -0.5 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * ezz2 * ezz1 / eyy2 * (1 - eyy2 / ezz2) / w
            + 0.5 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * ezz3 * ezz4 / eyy3 * (1 - eyy3 / ezz3) / e
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
        * (
            0.5 * ezz3 * (2.0 / s ** 2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w
            + 0.5 * ezz2 * (2.0 / s ** 2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzye = (
        (
            (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * (
                -n * ezz2 / eyy1 * eyx1 / e / (e + w)
                + (ezz1 - ezz2) * w / e / (e + w)
                - s * ezz1 / eyy2 * eyx2 / e / (e + w)
            )
            + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * (
                0.5
                * n
                * ezz4
                * ezz3
                / eyy4
                * (-(1 - eyy4 / ezz4) / n / e - eyx4 / ezz4 * (2.0 / e ** 2 - 2 / e ** 2 * w / (e + w)))
                + 0.5
                * s
                * ezz3
                * ezz4
                / eyy3
                * ((1 - eyy3 / ezz3) / s / e - eyx3 / ezz3 * (2.0 / e ** 2 - 2 / e ** 2 * w / (e + w)))
                + (ezz4 - ezz3) * w / e / (e + w)
            )
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            0.5
            * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * ezz1
            * (2 * exx4 / ezz4 / e ** 2 - exy4 / ezz4 / n / e)
            / exx4
            * ezz4
            * e
            - 0.5
            * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * ezz2
            * (2 * exx3 / ezz3 / e ** 2 + exy3 / ezz3 / s / e)
            / exx3
            * ezz3
            * e
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzyw = (
        (
            (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * (
                0.5
                * n
                * ezz1
                * ezz2
                / eyy1
                * ((1 - eyy1 / ezz1) / n / w - eyx1 / ezz1 * (2.0 / w ** 2 - 2 / w ** 2 * e / (e + w)))
                - (ezz1 - ezz2) * e / w / (e + w)
                + 0.5
                * s
                * ezz2
                * ezz1
                / eyy2
                * (-(1 - eyy2 / ezz2) / s / w - eyx2 / ezz2 * (2.0 / w ** 2 - 2 / w ** 2 * e / (e + w)))
            )
            + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * (
                -n * ezz3 / eyy4 * eyx4 / w / (e + w)
                - s * ezz4 / eyy3 * eyx3 / w / (e + w)
                - (ezz4 - ezz3) * e / w / (e + w)
            )
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            0.5
            * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * ezz4
            * (2 * exx1 / ezz1 / w ** 2 + exy1 / ezz1 / n / w)
            / exx1
            * ezz1
            * w
            - 0.5
            * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * ezz3
            * (2 * exx2 / ezz2 / w ** 2 - exy2 / ezz2 / s / w)
            / exx2
            * ezz2
            * w
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    bzyp = (
        (
            (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
            * (
                0.5
                * n
                * ezz1
                * ezz2
                / eyy1
                * (
                    -(k ** 2) * eyx1
                    - (1 - eyy1 / ezz1) / n / w
                    - eyx1 / ezz1 * (-2.0 / w ** 2 + 2 / w ** 2 * (e - w) / e)
                )
                + (ezz1 - ezz2) * (e - w) / e / w
                + 0.5
                * s
                * ezz2
                * ezz1
                / eyy2
                * (
                    -(k ** 2) * eyx2
                    + (1 - eyy2 / ezz2) / s / w
                    - eyx2 / ezz2 * (-2.0 / w ** 2 + 2 / w ** 2 * (e - w) / e)
                )
            )
            + (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
            * (
                0.5
                * n
                * ezz4
                * ezz3
                / eyy4
                * (
                    -(k ** 2) * eyx4
                    + (1 - eyy4 / ezz4) / n / e
                    - eyx4 / ezz4 * (-2.0 / e ** 2 - 2 / e ** 2 * (e - w) / w)
                )
                + 0.5
                * s
                * ezz3
                * ezz4
                / eyy3
                * (
                    -(k ** 2) * eyx3
                    - (1 - eyy3 / ezz3) / s / e
                    - eyx3 / ezz3 * (-2.0 / e ** 2 - 2 / e ** 2 * (e - w) / w)
                )
                + (ezz4 - ezz3) * (e - w) / e / w
            )
        )
        / ezz4
        / ezz3
        / (n * eyy3 + s * eyy4)
        / ezz2
        / ezz1
        / (n * eyy2 + s * eyy1)
        / (e + w)
        * eyy4
        * eyy3
        * eyy1
        * w
        * eyy2
        * e
        + (
            (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
            * (
                0.5
                * ezz4
                * (-2.0 / n ** 2 - 2 * exx1 / ezz1 / w ** 2 + k ** 2 * exx1 - exy1 / ezz1 / n / w)
                / exx1
                * ezz1
                * w
                + 0.5
                * ezz1
                * (-2.0 / n ** 2 - 2 * exx4 / ezz4 / e ** 2 + k ** 2 * exx4 + exy4 / ezz4 / n / e)
                / exx4
                * ezz4
                * e
            )
            - (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
            * (
                0.5
                * ezz3
                * (-2.0 / s ** 2 - 2 * exx2 / ezz2 / w ** 2 + k ** 2 * exx2 + exy2 / ezz2 / s / w)
                / exx2
                * ezz2
                * w
                + 0.5
                * ezz2
                * (-2.0 / s ** 2 - 2 * exx3 / ezz3 / e ** 2 + k ** 2 * exx3 - exy3 / ezz3 / s / e)
                / exx3
                * ezz3
                * e
            )
        )
        / ezz3
        / ezz2
        / (w * exx3 + e * exx2)
        / ezz4
        / ezz1
        / (w * exx4 + e * exx1)
        / (n + s)
        * exx2
        * exx3
        * n
        * exx1
        * exx4
        * s
    ) / b

    ii = np.arange(nx * ny).reshape(nx, ny)

    # NORTH boundary

    ib = ii[:, -1]

    if boundary[0] == "S":
        sign = 1
    elif boundary[0] == "A":
        sign = -1
    elif boundary[0] == "0":
        sign = 0
    else:
        raise ValueError("unknown boundary conditions")

    bzxs[ib] += sign * bzxn[ib]
    bzxse[ib] += sign * bzxne[ib]
    bzxsw[ib] += sign * bzxnw[ib]
    bzys[ib] -= sign * bzyn[ib]
    bzyse[ib] -= sign * bzyne[ib]
    bzysw[ib] -= sign * bzynw[ib]

    # SOUTH boundary

    ib = ii[:, 0]

    if boundary[1] == "S":
        sign = 1
    elif boundary[1] == "A":
        sign = -1
    elif boundary[1] == "0":
        sign = 0
    else:
        raise ValueError("unknown boundary conditions")

    bzxn[ib] += sign * bzxs[ib]
    bzxne[ib] += sign * bzxse[ib]
    bzxnw[ib] += sign * bzxsw[ib]
    bzyn[ib] -= sign * bzys[ib]
    bzyne[ib] -= sign * bzyse[ib]
    bzynw[ib] -= sign * bzysw[ib]

    # EAST boundary

    ib = ii[-1, :]

    if boundary[2] == "S":
        sign = 1
    elif boundary[2] == "A":
        sign = -1
    elif boundary[2] == "0":
        sign = 0
    else:
        raise ValueError("unknown boundary conditions")

    bzxw[ib] += sign * bzxe[ib]
    bzxnw[ib] += sign * bzxne[ib]
    bzxsw[ib] += sign * bzxse[ib]
    bzyw[ib] -= sign * bzye[ib]
    bzynw[ib] -= sign * bzyne[ib]
    bzysw[ib] -= sign * bzyse[ib]

    # WEST boundary

    ib = ii[0, :]

    if boundary[3] == "S":
        sign = 1
    elif boundary[3] == "A":
        sign = -1
    elif boundary[3] == "0":
        sign = 0
    else:
        raise ValueError("unknown boundary conditions")

    bzxe[ib] += sign * bzxw[ib]
    bzxne[ib] += sign * bzxnw[ib]
    bzxse[ib] += sign * bzxsw[ib]
    bzye[ib] -= sign * bzyw[ib]
    bzyne[ib] -= sign * bzynw[ib]
    bzyse[ib] -= sign * bzysw[ib]

    # Assemble sparse matrix

    iall = ii.flatten()
    i_s = ii[:, :-1].flatten()
    i_n = ii[:, 1:].flatten()
    i_e = ii[1:, :].flatten()
    i_w = ii[:-1, :].flatten()
    i_ne = ii[1:, 1:].flatten()
    i_se = ii[1:, :-1].flatten()
    i_sw = ii[:-1, :-1].flatten()
    i_nw = ii[:-1, 1:].flatten()

    Izx = np.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
    Jzx = np.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
    Vzx = np.r_[
        bzxp[iall], bzxe[i_w], bzxw[i_e], bzxn[i_s], bzxs[i_n], bzxsw[i_ne], bzxnw[i_se], bzxne[i_sw], bzxse[i_nw]
    ]

    Izy = np.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
    Jzy = np.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
    Vzy = np.r_[
        bzyp[iall], bzye[i_w], bzyw[i_e], bzyn[i_s], bzys[i_n], bzysw[i_ne], bzynw[i_se], bzyne[i_sw], bzyse[i_nw]
    ]

    I = np.r_[Izx, Izy]
    J = np.r_[Jzx, Jzy]
    V = np.r_[Vzx, Vzy]
    B = coo_matrix((V, (I, J))).tocsr()

    HxHy = np.r_[Hx, Hy]
    Hz = B * HxHy.ravel() / 1j
    Hz = Hz.reshape(Hx.shape)

    # in xc e yc
    exx = epsxx[1:-1, 1:-1]
    exy = epsxy[1:-1, 1:-1]
    eyx = epsyx[1:-1, 1:-1]
    eyy = epsyy[1:-1, 1:-1]
    ezz = epszz[1:-1, 1:-1]
    edet = exx * eyy - exy * eyx

    h = e.reshape(nx, ny)[:-1, :-1]
    v = n.reshape(nx, ny)[:-1, :-1]

    # in xc e yc
    Dx = neff * EMpy_gpu.utils.centered2d(Hy) + (Hz[:-1, 1:] + Hz[1:, 1:] - Hz[:-1, :-1] - Hz[1:, :-1]) / (2j * k * v)
    Dy = -neff * EMpy_gpu.utils.centered2d(Hx) - (Hz[1:, :-1] + Hz[1:, 1:] - Hz[:-1, 1:] - Hz[:-1, :-1]) / (2j * k * h)
    Dz = (
        (Hy[1:, :-1] + Hy[1:, 1:] - Hy[:-1, 1:] - Hy[:-1, :-1]) / (2 * h)
        - (Hx[:-1, 1:] + Hx[1:, 1:] - Hx[:-1, :-1] - Hx[1:, :-1]) / (2 * v)
    ) / (1j * k)

    Ex = (eyy * Dx - exy * Dy) / edet
    Ey = (exx * Dy - eyx * Dx) / edet
    Ez = Dz / ezz

    return [Hx, Hy, Hz, Ex, Ey, Ez]
