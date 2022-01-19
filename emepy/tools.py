import numpy as np
import scipy
import os
from scipy.interpolate import griddata
import EMpy_gpu
import collections
from matplotlib import pyplot as plt


def get_epsfunc(
    width,
    thickness,
    cladding_width,
    cladding_thickness,
    core_index,
    cladding_index,
    compute=False,
    profile=None,
    nx=None,
    ny=None,
):
    """Returns the epsfunc for given parameters"""

    # Case 1 : width and thickness are defined
    def epsfunc_1(x_, y_):
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
        if compute:
            n = np.where(
                (np.abs(np.real(xx.T) - cladding_width * 0.5) <= width * 0.5)
                * (np.abs(np.real(yy.T) - cladding_thickness * 0.5) <= thickness * 0.5),
                core_index ** 2 + 0j,
                cladding_index ** 2 + 0j,
            )
        else:
            n = np.where(
                (np.abs(np.real(xx.T)) <= width * 0.5) * (np.abs(np.real(yy.T)) <= thickness * 0.5),
                core_index ** 2 + 0j,
                cladding_index ** 2 + 0j,
            )

        return n

    # Case 2 : thickness and 1D n is defined
    def epsfunc_2(x_, y_):

        n = profile
        xx, yy = np.meshgrid(np.real(x_), np.real(y_))
        n = np.interp(np.real(x_), np.real(nx), n).astype(complex)
        n = np.repeat(n, len(y_)).reshape((len(n), len(y_)))
        n = np.where((np.abs(np.real(yy.T)) <= thickness * 0.5), n, cladding_index + 0j) ** 2
        return n

    # Case 3 : 2D n is defined
    def epsfunc_3(x_, y_):

        xxn, yyn = np.meshgrid(np.real(nx), np.real(ny))
        points = np.array((xxn.flatten(), yyn.flatten())).T
        n = profile.flatten()
        xx, yy = np.meshgrid(np.real(x_), np.real(y_))
        n_real = griddata(points, np.real(n), (xx, yy))
        n_imag = griddata(points, np.imag(n), (xx, yy))
        n = (n_real + 1j * n_imag) ** 2

        return n

    if not (width is None) and not (thickness is None):
        return epsfunc_1

    elif (width is None) and not (thickness is None) and not (profile is None):
        return epsfunc_2

    elif (width is None) and (thickness is None) and not (profile is None):
        return epsfunc_3

    raise Exception("Need to provide width & thickness, or 1D profile and thickness, or 2D profile")


def create_polygon(x, y, n, detranslate=True):

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


Si_lambda = [
    1.2,
    1.22,
    1.24,
    1.26,
    1.28,
    1.3,
    1.32,
    1.34,
    1.36,
    1.38,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.8,
    1.9,
    2,
    2.25,
    2.5,
    2.75,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
]

Si_n = [
    3.5167,
    3.5133,
    3.5102,
    3.5072,
    3.5043,
    3.5016,
    3.499,
    3.4965,
    3.4941,
    3.4918,
    3.4896,
    3.4845,
    3.4799,
    3.4757,
    3.4719,
    3.4684,
    3.4653,
    3.4597,
    3.455,
    3.451,
    3.4431,
    3.4375,
    3.4334,
    3.4302,
    3.4229,
    3.4195,
    3.4177,
    3.4165,
    3.4158,
    3.4153,
    3.415,
    3.4147,
    3.4145,
    3.4144,
    3.4142,
]

SiO2_lambda = [
    0.21,
    0.2174,
    0.2251,
    0.233,
    0.2412,
    0.2497,
    0.2585,
    0.2676,
    0.277,
    0.2868,
    0.2969,
    0.3074,
    0.3182,
    0.3294,
    0.341,
    0.353,
    0.3655,
    0.3783,
    0.3917,
    0.4055,
    0.4197,
    0.4345,
    0.4498,
    0.4657,
    0.4821,
    0.4991,
    0.5167,
    0.5349,
    0.5537,
    0.5732,
    0.5934,
    0.6143,
    0.636,
    0.6584,
    0.6816,
    0.7056,
    0.7305,
    0.7562,
    0.7829,
    0.8104,
    0.839,
    0.8686,
    0.8992,
    0.9308,
    0.9636,
    0.9976,
    1.033,
    1.069,
    1.107,
    1.146,
    1.186,
    1.228,
    1.271,
    1.316,
    1.362,
    1.41,
    1.46,
    1.512,
    1.565,
    1.62,
    1.677,
    1.736,
    1.797,
    1.861,
    1.926,
    1.994,
    2.064,
    2.137,
    2.212,
    2.29,
    2.371,
    2.454,
    2.541,
    2.63,
    2.723,
    2.819,
    2.918,
    3.021,
    3.128,
    3.238,
    3.352,
    3.47,
    3.592,
    3.719,
    3.85,
    3.986,
    4.126,
    4.271,
    4.422,
    4.578,
    4.739,
    4.906,
    5.079,
    5.258,
    5.443,
    5.635,
    5.833,
    6.039,
    6.252,
    6.472,
    6.7,
]

SiO2_n = [
    1.5383576204905,
    1.530846431063,
    1.5240789072975,
    1.5180417677275,
    1.5125721155558,
    1.5076095872199,
    1.5031009629039,
    1.498999218542,
    1.495262719426,
    1.4918215034661,
    1.488683281387,
    1.4857914366574,
    1.4831504333467,
    1.4807144415912,
    1.4784676522483,
    1.4763951298847,
    1.4744682820342,
    1.4727046797948,
    1.4710525123802,
    1.4695286500209,
    1.4681218218832,
    1.4668048202486,
    1.465580829975,
    1.4644360310913,
    1.4633719346282,
    1.4623764385944,
    1.4614449911601,
    1.4605730794883,
    1.4597562854765,
    1.4589865613939,
    1.4582607881284,
    1.4575758068817,
    1.4569256013294,
    1.4563104080175,
    1.4557246986958,
    1.4551660299221,
    1.4546298754538,
    1.4541161650842,
    1.4536188536247,
    1.4531396102638,
    1.4526712743322,
    1.4522138205731,
    1.4517653834747,
    1.4513240787777,
    1.4508853971281,
    1.450447735732,
    1.4500069615101,
    1.4495710901504,
    1.4491214616538,
    1.448668310738,
    1.4482096590065,
    1.4477322458328,
    1.4472455774929,
    1.4467363524093,
    1.4462138519129,
    1.445664578157,
    1.4450861470109,
    1.4444759883488,
    1.4438434020915,
    1.4431739285381,
    1.4424645759259,
    1.4417121733672,
    1.4409133669095,
    1.4400509385522,
    1.43914806608,
    1.4381729429417,
    1.4371349603433,
    1.4360139618555,
    1.4348196176837,
    1.433529881253,
    1.4321372109342,
    1.4306516985468,
    1.4290287283802,
    1.427296093753,
    1.4254044329951,
    1.4233613684394,
    1.4211544161909,
    1.4187459519934,
    1.416117293379,
    1.4132741569442,
    1.4101696832452,
    1.4067782146466,
    1.4030708962299,
    1.3989819997456,
    1.3945035722002,
    1.3895553417944,
    1.3841208059058,
    1.3780997735118,
    1.3713701305288,
    1.36388143366,
    1.3555262189157,
    1.3461171232165,
    1.3354823573874,
    1.3234105439689,
    1.3096384003386,
    1.2937460280032,
    1.2753723963511,
    1.2537289561387,
    1.2280888354422,
    1.1973256716307,
    1.1596494139777,
]


def Si(wavelength):
    """Return the refractive index for Silicon given the wavelength in microns.

    Parameters
    ----------
    wavelength : number
        wavelength (microns)

    Returns
    -------
    number
        refractive index
    """

    f = scipy.interpolate.interp1d(Si_lambda, Si_n)
    return f([wavelength, wavelength])[0]


def SiO2(wavelength):
    """Return the refractive index for Silicon Dioxide given the wavelength in microns.

    Parameters
    ----------
    wavelength : number
        the optical wavelength (microns)

    Returns
    -------
    number
        refractive index
    """

    f = scipy.interpolate.interp1d(SiO2_lambda, SiO2_n)
    return f([wavelength, wavelength])[0]


def interp(x, y, x0, y0, f, centered):
    """Interpolate a 2D complex array."""

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


def into_chunks(location, name, chunk_size=20000000):
    """Takes a large serialized file and breaks it up into smaller chunk files

    Paramters
    ---------
    location : string
        the absolute or relative path of the large file
    name : string
        the name of the serialized smaller components (will have _chunk_# appended to it)
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


def from_chunks(location, name):
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
