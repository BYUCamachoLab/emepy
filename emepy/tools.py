import numpy as np
import os
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

def interp2d(x, y, xx, yy, f, sci=False):

    def interp2d_partial(x, xp, fp):
        j = np.searchsorted(xp, x)
        j = np.clip(j, 0, len(xp) - 2)
        alpha = (x - xp[j]) / (xp[j + 1] - xp[j])
        alpha = np.repeat(alpha[:, np.newaxis], fp.shape[1], axis=1)
        return np.clip((fp[j, :] * alpha + fp[j+1, :] * (1 - alpha)), np.min(fp), np.max(fp))

    def on_real(x, y, xx, yy, f):

        if sci:
            from scipy.interpolate import griddata
            xxn, yyn = np.meshgrid(np.real(xx), np.real(yy))
            points = np.array((xxn.flatten(), yyn.flatten())).T
            n = f.flatten()
            xx, yy = np.meshgrid(np.real(x), np.real(y))
            return griddata(points, n, (xx, yy))
        else:
            f1 = interp2d_partial(x, xx, f)
            f2 = interp2d_partial(y, yy, f1.T)
            return f2.T

    return on_real(x, y, xx, yy, np.real(f)) + 1j * on_real(x, y, xx, yy, np.imag(f)) if np.iscomplexobj(f) else on_real(x, y, xx, yy, f)

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

        # from scipy.interpolate import griddata
        # xxn, yyn = np.meshgrid(np.real(self.nx), np.real(self.ny))
        # points = np.array((xxn.flatten(), yyn.flatten())).T
        # n = self.profile.flatten()
        # xx, yy = np.meshgrid(np.real(x_), np.real(y_))
        # n_real = griddata(points, np.real(n), (xx, yy))
        # n_imag = griddata(points, np.imag(n), (xx, yy))
        # n = (n_real + 1j * n_imag) ** 2

        return interp2d(x_, y_, self.nx, self.ny, self.profile) ** 2

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

