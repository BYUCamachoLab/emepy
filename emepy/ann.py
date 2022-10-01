import os
import numpy as np
import pickle

import torch
import torch.nn as nn

from sklearn.preprocessing import PolynomialFeatures
import sklearn

from emepy.mode import Mode, EigenMode
from emepy.fd import ModeSolver, EMpy_gpu
from emepy.tools import from_chunks, get_epsfunc, _get_eps, interp
from emepy.materials import Si, SiO2
from scipy.sparse import coo_matrix

from typing import Callable

FIELD_WIDTH = 128
FIELD_SIZE = FIELD_WIDTH ** 2


def getUpConvLayer(i_size: int, o_size: int, kernal: int, channels: int, first: bool = False, last: bool = False):
    """Returns the right size for up convolutional sampling"""

    def out_size(in_size, kernal, stride, padding, output_padding):
        return (in_size - 1) * stride - 2 * padding + kernal + output_padding

    stride = 0
    padding = 0
    output_padding = 0

    if not out_size(i_size, kernal, stride, padding, output_padding) == o_size:
        while out_size(i_size, kernal, stride, padding, output_padding) < o_size:
            stride += 1

    if not out_size(i_size, kernal, stride, padding, output_padding) == o_size:
        while out_size(i_size, kernal, stride, padding, output_padding) > o_size:
            padding += 1

    if not out_size(i_size, kernal, stride, padding, output_padding) == o_size:
        while out_size(i_size, kernal, stride, padding, output_padding) < o_size:
            output_padding += 1

    if first:
        return nn.ConvTranspose2d(
            1, channels, stride=stride, padding=padding, kernel_size=kernal, output_padding=output_padding
        )

    if last:
        return nn.ConvTranspose2d(
            channels, 1, stride=stride, padding=padding, kernel_size=kernal, output_padding=output_padding
        )

    return nn.ConvTranspose2d(
        channels, channels, stride=stride, padding=padding, kernel_size=kernal, output_padding=output_padding
    )


def getDownConvLayer(i_size: int, o_size: int, kernal: int, channels: int, first: bool = False, last: bool = False):
    """Returns the right size for down convolutional sampling"""

    def out_size(in_size, kernal, stride, padding, dilation):
        return (in_size + 2 * padding - dilation * (kernal - 1) - 1) / stride + 1

    stride = 1
    padding = 0
    dilation = 1

    if not out_size(i_size, kernal, stride, padding, dilation) == o_size:
        while out_size(i_size, kernal, stride, padding, dilation) > o_size:
            dilation += 1

    if not out_size(i_size, kernal, stride, padding, dilation) == o_size:
        while out_size(i_size, kernal, stride, padding, dilation) < o_size:
            padding += 1

    if not out_size(i_size, kernal, stride, padding, dilation) == o_size:
        raise Exception("Choose a different kernal size")

    if first:
        return nn.Conv2d(1, channels, stride=stride, padding=padding, kernel_size=kernal, dilation=dilation)

    if last:
        return nn.Conv2d(channels, 1, stride=stride, padding=padding, kernel_size=kernal, dilation=dilation)

    return nn.Conv2d(channels, channels, stride=stride, padding=padding, kernel_size=kernal, dilation=dilation)


class Network(nn.Module):
    """The pytorch inherited class that defines and represents the physical neural network"""

    def __init__(self, code_size: int, channels: int, component: str) -> None:
        """Network constructor

        Parameters
        ----------
        code_size : int
            the number of inputs
        channels : int
            the number of channels
        component : string
            "Hx" or "Hy"
        """
        super().__init__()
        self.channels = channels

        self.linear_up_1 = nn.Linear(code_size, int(FIELD_WIDTH / 20) ** 2)
        self.linear_up_2 = nn.Linear(int(FIELD_WIDTH / 20) ** 2, int(FIELD_WIDTH / 7) ** 2)
        self.linear_up_3 = nn.Linear(int(FIELD_WIDTH / 7) ** 2, int(FIELD_WIDTH / 4) ** 2)
        self.linear_up_4 = nn.Linear(int(FIELD_WIDTH / 4) ** 2, int(FIELD_WIDTH / 3) ** 2)
        self.conv_up_1 = getUpConvLayer(int(FIELD_WIDTH / 3), int(FIELD_WIDTH / 2), 3, channels, first=True)
        self.conv_up_2 = getUpConvLayer(int(FIELD_WIDTH / 2), int(5 * FIELD_WIDTH / 8), 5, channels)
        self.conv_up_3 = getUpConvLayer(int(5 * FIELD_WIDTH / 8), int(6 * FIELD_WIDTH / 8), 7, channels)
        self.conv_up_4 = getUpConvLayer(int(6 * FIELD_WIDTH / 8), int(7 * FIELD_WIDTH / 8), 7, channels)
        self.conv_up_5 = getUpConvLayer(int(7 * FIELD_WIDTH / 8), int(FIELD_WIDTH), 9, channels, last=True)
        self.linear_up_5 = nn.Linear(FIELD_WIDTH ** 2, FIELD_WIDTH ** 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_normal_(self.linear_up_1.weight)
        torch.nn.init.xavier_normal_(self.linear_up_2.weight)
        torch.nn.init.xavier_normal_(self.linear_up_3.weight)

        torch.nn.init.xavier_normal_(self.conv_up_1.weight)
        torch.nn.init.xavier_normal_(self.conv_up_2.weight)
        torch.nn.init.xavier_normal_(self.conv_up_3.weight)

        self.component = component

    def forward(self, field) -> tuple:
        """Performs the network propagation

        Parameters
        ----------
        field : array
            The inputs to the network

        Returns
        -------
        tuple (torch array, torch array)
            Returns a tuple of the outputs, inputs
        """

        out = self.tanh(self.linear_up_1(field)).view(-1, 1, int(FIELD_WIDTH / 20) ** 2)
        out = self.tanh(self.linear_up_2(out)).view(-1, 1, int(FIELD_WIDTH / 7) ** 2)
        out = self.tanh(self.linear_up_3(out)).view(-1, 1, int(FIELD_WIDTH / 4) ** 2)
        out = self.tanh(self.linear_up_4(out)).view(-1, 1, int(FIELD_WIDTH / 3), int(FIELD_WIDTH / 3))
        out = self.tanh(self.conv_up_1(out)).view(-1, self.channels, int(FIELD_WIDTH / 2), int(FIELD_WIDTH / 2))
        out = self.tanh(self.conv_up_2(out)).view(-1, self.channels, int(5 * FIELD_WIDTH / 8), int(5 * FIELD_WIDTH / 8))
        out = self.tanh(self.conv_up_3(out)).view(-1, self.channels, int(6 * FIELD_WIDTH / 8), int(6 * FIELD_WIDTH / 8))
        out = self.tanh(self.conv_up_4(out)).view(-1, self.channels, int(7 * FIELD_WIDTH / 8), int(7 * FIELD_WIDTH / 8))
        out = self.tanh(self.conv_up_5(out)).view(-1, 1, FIELD_WIDTH ** 2)

        out = self.linear_up_5(out).view(-1, FIELD_WIDTH, FIELD_WIDTH)

        out = out / 1000.0 if self.component == "Hx" else out / 100.0

        return out, field


class ANN(object):
    """Object that loads the neural network; Users are heavily encouraged to design their own networks and rewrite their own ANN to match their needs"""

    def __init__(self,) -> None:
        """Constructor for Mode Object"""
        self.x = np.linspace(0, 2.5, FIELD_WIDTH)
        self.y = np.linspace(0, 2.5, FIELD_WIDTH)

        self.Hx_model = self.Hx_network()
        self.Hy_model = self.Hy_network()
        self.neff_model = self.neff_regression()

    def neff_regression(self) -> "sklearn.linear_model._base.LinearRegression":
        """Return the opened regression model for the effective index"""

        with open(os.path.dirname(os.path.abspath(__file__)) + "/models/neff_pickle/model.pk", "rb") as f:
            model = pickle.load(f)

        return model

    def Hx_network(self) -> Network:
        """Return the opened network model for the Hx component"""

        from_chunks(os.path.dirname(os.path.abspath(__file__)) + "/models/Hx_chunks/", "hx_temp.pt")
        with open("hx_temp.pt", "rb") as f:
            model = Network(3, 1, "Hx")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

            model.eval()

        os.system("rm hx_temp.pt")
        return model

    def Hy_network(self) -> Network:
        """Return the opened network model for the Hy component"""

        from_chunks(os.path.dirname(os.path.abspath(__file__)) + "/models/Hy_chunks/", "hy_temp.pt")
        with open("hy_temp.pt", "rb") as f:
            model = Network(3, 1, "Hy")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

            model.eval()

        os.system("rm hy_temp.pt")
        return model


class MSNeuralNetwork(ModeSolver):
    """ModeSolver object for the sample neural networks, parameterizes the cross section components. Currently designed only for single mode calculations in Silicon on SiO2"""

    def __init__(self, ann: ANN, wl: float, width: float, thickness: float) -> None:
        """MSNeuralNetwork constructor

        Parameters
        ----------
        ann : ANN
            The ANN object that contains the network and regression models
        wl : number
            The wavelength (most accurate around 1.55 µm)
        width : number
            The width of the cross section (most accurate around 550 nm)
        thickness : number
            The thickness of the cross section (most accurate around 250 nm)
        """

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.ann = ann
        self.Hx_model = ann.Hx_model
        self.Hy_model = ann.Hy_model
        self.neff_model = ann.neff_model
        self.num_modes = 1
        self.x = ann.x
        self.y = ann.y
        self.after_x = self.x
        self.after_y = self.y
        self.mesh = len(self.x) - 1
        self.PML = False
        self.n = get_epsfunc(self.width, self.thickness, 2.5, 2.5, Si(self.wl), SiO2(self.wl), compute=True)(
            self.x, self.y
        )

    def solve(self) -> None:
        """Solves for the eigenmode using the neural networks"""

        self.mode = None

        Hx, Hy, neff = self.data(self.width, self.thickness, self.wl)
        self.mode = (Hx, Hy, neff)

    def data(self, width: float, thickness: float, wl: float) -> tuple:
        """Propagates the inputs into the neural networks and regression models and returns the outputs

        Parameters
        ----------
        width : number
            The width of the cross section (most accurate around 550 nm)
        thickness : number
            The thickness of the cross section (most accurate around 250 nm)
        wl : number
            The wavelength (most accurate around 1.55 µm)

        Returns
        -------
        tuple (numpy array, numpy array, number)
            Returns Hx, Hy, and neff
        """

        neff = self.neff_regression(width, thickness, wl, self.neff_model)
        Hx = self.Hx_network(width, thickness, wl, self.Hx_model)
        Hy = self.Hy_network(width, thickness, wl, self.Hy_model)

        return Hx, Hy, neff

    def neff_regression(
        self, width: float, thickness: float, wl: float, model: "sklearn.linear_model._base.LinearRegression"
    ) -> float:
        """Calculates the effective index using a regression model

        Parameters
        ----------
        width : number
            The width of the cross section (most accurate around 550 nm)
        thickness : number
            The thickness of the cross section (most accurate around 250 nm)
        wl : number
            The wavelength (most accurate around 1.55 µm)
        model : sklearn regression model
            The model that performs the regression

        Returns
        -------
        number
            Returns the effective index
        """

        poly = PolynomialFeatures(degree=8)
        X = poly.fit_transform([[width, thickness, wl]])
        neff = model.predict(X)

        return neff[0]

    def Hx_network(self, width: float, thickness: float, wl: float, model: Network) -> "np.ndarray":
        """Calculates the Hx component using a network model

        Parameters
        ----------
        width : number
            The width of the cross section (most accurate around 550 nm)
        thickness : number
            The thickness of the cross section (most accurate around 250 nm)
        wl : number
            The wavelength (most accurate around 1.55 µm)
        model : pytorch model
            The model that performs the ann calculation

        Returns
        -------
        numpy array
            Returns the Hx field
        """

        with torch.no_grad():
            parameters = torch.Tensor([[[width, thickness, wl]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128, 128)

        return output

    def Hy_network(self, width: float, thickness: float, wl: float, model: Network) -> "np.ndarray":
        """Calculates the Hy component using a network model

        Parameters
        ----------
        width : number
            The width of the cross section (most accurate around 550 nm)
        thickness : number
            The thickness of the cross section (most accurate around 250 nm)
        wl : number
            The wavelength (most accurate around 1.55 µm)
        model : pytorch model
            The model that performs the ann calculation

        Returns
        -------
        numpy array
            Returns the Hy field
        """

        with torch.no_grad():
            parameters = torch.Tensor([[[width, thickness, wl]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128, 128)

        return output

    def clear(self) -> None:
        """Clears the mode in the object"""
        self.mode = None

    def get_mode(self, mode_num: int = 0) -> EigenMode:
        """Returns the solved eigenmode

        Parameters
        ----------
        mode_num : int
            mode index to return mode of

        Returns
        -------
        EigenMode
            the EigenMode corresponding to the provdided mode index
        """
        Hx, Hy, neff = self.mode

        epsfunc_before = get_epsfunc(self.width, self.thickness, 2.5, 2.5, Si(self.wl), SiO2(self.wl), compute=True)
        epsfunc_after = get_epsfunc(self.width, self.thickness, 2.5, 2.5, Si(self.wl), SiO2(self.wl))
        m = Mode(
            self.x,
            self.y,
            self.wl,
            neff,
            Hx + 0j,
            Hy + 0j,
            None,
            None,
            None,
            None,
            np.sqrt(epsfunc_after(self.x, self.y)),
        )
        m.compute_other_fields(epsfunc_before, epsfunc_after)
        m.normalize()

        return m


def compute_other_fields(
    mode,
    epsfunc_1: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"],
    epsfunc_2: Callable[["np.ndarray", "np.ndarray"], "np.ndarray"],
    boundary: str = "0000",
) -> None:
    """Given the Hx and Hy fields, maxwell's curl relations can be used to calculate the remaining field; adapted from the EMpy

    Parameters
    ----------
    epsfunc_1 : function
        epsfunc for computing other fields
    epsfunc_2 : function
        epsfunc for computing final refractive index profile
    boundary : str
        the boundary conditions as defined by electromagneticpython
    """

    (
        mode.Hx,
        mode.Hy,
        mode.Hz,
        mode.Ex,
        mode.Ey,
        mode.Ez,
    ) = compute_other_fields_2D(
        mode.neff, mode.Hx, mode.Hy, mode.wl, mode.x, mode.y, boundary, epsfunc_1
    )
    x_ = (mode.x[1:] + mode.x[:-1]) / 2.0
    y_ = (mode.y[1:] + mode.y[:-1]) / 2.0
    mode.Ex = interp(mode.x, mode.y, x_, y_, mode.Ex, False)
    mode.Ey = interp(mode.x, mode.y, x_, y_, mode.Ey, False)
    mode.Ez = interp(mode.x, mode.y, x_, y_, mode.Ez, False)
    mode.x = mode.x - mode.x[int(len(mode.x) / 2)]
    mode.y = mode.y - mode.y[int(len(mode.y) / 2)]
    mode.H = np.sqrt(
        np.abs(mode.Hx) ** 2 + np.abs(mode.Hy) ** 2 + np.abs(mode.Hz) ** 2
    )
    mode.E = np.sqrt(
        np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    )
    mode.n = np.sqrt(epsfunc_2(mode.x, mode.y))


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
