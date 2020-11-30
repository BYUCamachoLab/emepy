import numpy as np
import pandas as pd
import scipy
from EMpy.modesolvers.FD import stretchmesh


def get_epsfunc(width, thickness, cladding_width, cladding_thickness, core_index, cladding_index):
    """Returns the epsfunc for given parameters
    """

    def epsfunc(x_, y_):
        """Return a matrix describing a 2d material.

        :param x_: x values
        :param y_: y values
        :return: 2d-matrix
        """
        layer = int(len(x_) / 20)
        nlayers = [layer, layer, layer, layer]
        factor = 1 + 2j
        x, y, _, _, _, _ = stretchmesh(x_, y_, nlayers, factor)
        xx, yy = np.meshgrid(x, y)
        n = np.where(
            (np.abs(xx.T - cladding_width * 0.5) <= width * 0.5)
            * (np.abs(yy.T - cladding_thickness * 0.5) <= thickness * 0.5),
            core_index ** 2 + 0j,
            cladding_index ** 2 + 0j,
        )

        return n

    return epsfunc


def Si(wavelength):
    """Return the refractive index for Silicon given the wavelength in microns.

    :param wavelength (float): wavelength (microns)
    :return (float): refractive index
    """
    a = pd.read_csv("./refractive_index/Si.csv")
    lambdas = [i for i in a["Wavelength, µm"]]
    n = [i for i in a["n"]]
    f = scipy.interpolate.interp1d(lambdas, n)
    return f([wavelength, wavelength])[0]


def SiO2(wavelength):
    """Return the refractive index for Silicon Dioxide given the wavelength in microns.

    :param wavelength (float): wavelength (microns)
    :return (float): refractive index
    """
    a = pd.read_csv("./refractive_index/SiO2.csv")
    lambdas = [i for i in a["Wavelength, µm"]]
    n = [i for i in a["n"]]
    f = scipy.interpolate.interp1d(lambdas, n)
    return f([wavelength, wavelength])[0]
