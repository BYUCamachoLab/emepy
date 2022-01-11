import numpy as np
import os

from emepy.mode import Mode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from sklearn.preprocessing import PolynomialFeatures
from emepy.fd import ModeSolver
from emepy.tools import from_chunks


FIELD_WIDTH = 128
FIELD_SIZE = FIELD_WIDTH ** 2


def getUpConvLayer(i_size, o_size, kernal, channels, first=False, last=False):
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
            1,
            channels,
            stride=stride,
            padding=padding,
            kernel_size=kernal,
            output_padding=output_padding,
        )

    if last:
        return nn.ConvTranspose2d(
            channels,
            1,
            stride=stride,
            padding=padding,
            kernel_size=kernal,
            output_padding=output_padding,
        )

    return nn.ConvTranspose2d(
        channels,
        channels,
        stride=stride,
        padding=padding,
        kernel_size=kernal,
        output_padding=output_padding,
    )


def getDownConvLayer(i_size, o_size, kernal, channels, first=False, last=False):
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
        return nn.Conv2d(
            1,
            channels,
            stride=stride,
            padding=padding,
            kernel_size=kernal,
            dilation=dilation,
        )

    if last:
        return nn.Conv2d(
            channels,
            1,
            stride=stride,
            padding=padding,
            kernel_size=kernal,
            dilation=dilation,
        )

    return nn.Conv2d(
        channels,
        channels,
        stride=stride,
        padding=padding,
        kernel_size=kernal,
        dilation=dilation,
    )


class Network(nn.Module):
    """The pytorch inherited class that defines and represents the physical neural network"""

    def __init__(self, code_size, channels, component):
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
        self.linear_up_2 = nn.Linear(
            int(FIELD_WIDTH / 20) ** 2, int(FIELD_WIDTH / 7) ** 2
        )
        self.linear_up_3 = nn.Linear(
            int(FIELD_WIDTH / 7) ** 2, int(FIELD_WIDTH / 4) ** 2
        )
        self.linear_up_4 = nn.Linear(
            int(FIELD_WIDTH / 4) ** 2, int(FIELD_WIDTH / 3) ** 2
        )
        self.conv_up_1 = getUpConvLayer(
            int(FIELD_WIDTH / 3), int(FIELD_WIDTH / 2), 3, channels, first=True
        )
        self.conv_up_2 = getUpConvLayer(
            int(FIELD_WIDTH / 2), int(5 * FIELD_WIDTH / 8), 5, channels
        )
        self.conv_up_3 = getUpConvLayer(
            int(5 * FIELD_WIDTH / 8), int(6 * FIELD_WIDTH / 8), 7, channels
        )
        self.conv_up_4 = getUpConvLayer(
            int(6 * FIELD_WIDTH / 8), int(7 * FIELD_WIDTH / 8), 7, channels
        )
        self.conv_up_5 = getUpConvLayer(
            int(7 * FIELD_WIDTH / 8), int(FIELD_WIDTH), 9, channels, last=True
        )
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

    def forward(self, field):
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
        out = self.tanh(self.linear_up_4(out)).view(
            -1, 1, int(FIELD_WIDTH / 3), int(FIELD_WIDTH / 3)
        )
        out = self.tanh(self.conv_up_1(out)).view(
            -1, self.channels, int(FIELD_WIDTH / 2), int(FIELD_WIDTH / 2)
        )
        out = self.tanh(self.conv_up_2(out)).view(
            -1, self.channels, int(5 * FIELD_WIDTH / 8), int(5 * FIELD_WIDTH / 8)
        )
        out = self.tanh(self.conv_up_3(out)).view(
            -1, self.channels, int(6 * FIELD_WIDTH / 8), int(6 * FIELD_WIDTH / 8)
        )
        out = self.tanh(self.conv_up_4(out)).view(
            -1, self.channels, int(7 * FIELD_WIDTH / 8), int(7 * FIELD_WIDTH / 8)
        )
        out = self.tanh(self.conv_up_5(out)).view(-1, 1, FIELD_WIDTH ** 2)

        out = self.linear_up_5(out).view(-1, FIELD_WIDTH, FIELD_WIDTH)

        out = out / 1000.0 if self.component == "Hx" else out / 100.0

        return out, field


class MSNeuralNetwork(ModeSolver):
    """ModeSolver object for the sample neural networks, parameterizes the cross section components. Currently designed only for single mode calculations in Silicon on SiO2"""

    def __init__(
        self,
        ann,
        wl,
        width,
        thickness,
    ):
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

    def solve(self):
        """Solves for the eigenmode using the neural networks"""

        self.mode = None

        Hx, Hy, neff = self.data(self.width, self.thickness, self.wl)
        self.mode = (Hx, Hy, neff)

    def data(self, width, thickness, wl):
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

    def neff_regression(self, width, thickness, wl, model):
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
        X = poly.fit_transform([[width * 1e6, thickness * 1e6, wl * 1e6]])
        neff = model.predict(X)

        return neff[0]

    def Hx_network(self, width, thickness, wl, model):
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
            parameters = torch.Tensor([[[width * 1e6, thickness * 1e6, wl * 1e6]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128, 128)

        return output

    def Hy_network(self, width, thickness, wl, model):
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
            parameters = torch.Tensor([[[width * 1e6, thickness * 1e6, wl * 1e6]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128, 128)

        return output

    def clear(self):
        """Clears the mode in the object"""
        self.mode = None

    def get_mode(self, mode_num=0):
        """Returns the solved eigenmode"""
        Hx, Hy, neff = self.mode
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
            width=self.width,
            thickness=self.thickness,
        )
        m.compute_other_fields()
        m.normalize()

        return m


class ANN(object):
    """Object that loads the neural network; Users are heavily encouraged to design their own networks and rewrite their own ANN to match their needs"""

    def __init__(
        self,
    ):
        """Constructor for Mode Object"""
        self.x = np.linspace(0, 2.5e-6, FIELD_WIDTH)
        self.y = np.linspace(0, 2.5e-6, FIELD_WIDTH)

        self.Hx_model = self.Hx_network()
        self.Hy_model = self.Hy_network()
        self.neff_model = self.neff_regression()

    def neff_regression(self):
        """Return the opened regression model for the effective index"""

        with open(
            os.path.dirname(os.path.abspath(__file__)) + "/models/neff_pickle/model.pk",
            "rb",
        ) as f:
            model = pickle.load(f)

        return model

    def Hx_network(self):
        """Return the opened network model for the Hx component"""

        from_chunks(
            os.path.dirname(os.path.abspath(__file__)) + "/models/Hx_chunks/",
            "hx_temp.pt",
        )
        with open("hx_temp.pt", "rb") as f:
            model = Network(3, 1, "Hx")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

            model.eval()

        os.system("rm hx_temp.pt")
        return model

    def Hy_network(self):
        """Return the opened network model for the Hy component"""

        from_chunks(
            os.path.dirname(os.path.abspath(__file__)) + "/models/Hy_chunks/",
            "hy_temp.pt",
        )
        with open("hy_temp.pt", "rb") as f:
            model = Network(3, 1, "Hy")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

            model.eval()

        os.system("rm hy_temp.pt")
        return model
