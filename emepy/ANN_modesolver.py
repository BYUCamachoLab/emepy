import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import time

from mode import Mode

import pyMode as pm

import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from EMpy.modesolvers.FD import stretchmesh
from EMpy.utils import centered2d
import EMpy


FIELD_WIDTH = 500
FIELD_SIZE = FIELD_WIDTH ** 2


def deNormalizeHx(field):

    # field -= .5
    field /= 250
    field_return = np.zeros((500, 500))
    field_return[200:300, 200:300] = field

    return field_return


def deNormalizeHy(field):

    # field -= 1
    # field *= 18
    # field = 10 ** field
    field /= 100
    field_return = np.zeros((500, 500))
    field_return[200:300, 200:300] = field

    return field_return


def getUpConvLayer(i_size, o_size, kernal, channels, first=False, last=False):
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


def getDownConvLayer(i_size, o_size, kernal, channels, first=False, last=False):
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
    def __init__(self, code_size, channels, field_name):
        super().__init__()
        self.channels = channels
        self.field_name = field_name

        self.linear_up_1 = nn.Linear(code_size, 5 ** 2)
        self.linear_up_2 = nn.Linear(5 ** 2, 15 ** 2)
        self.linear_up_3 = nn.Linear(15 ** 2, 25 ** 2)
        self.linear_up_4 = nn.Linear(25 ** 2, 30 ** 2)
        self.conv_up_1 = getUpConvLayer(30, 48, 3, channels, first=True)
        self.conv_up_2 = getUpConvLayer(48, 64, 3, channels)
        self.conv_up_3 = getUpConvLayer(64, 78, 3, channels)
        self.conv_up_4 = getUpConvLayer(78, 90, 5, channels)
        self.conv_up_5 = getUpConvLayer(90, 100, 5, channels, last=True)
        self.linear_up_5 = nn.Linear(100 ** 2, 100 ** 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_normal_(self.linear_up_1.weight)
        torch.nn.init.xavier_normal_(self.linear_up_2.weight)
        torch.nn.init.xavier_normal_(self.linear_up_3.weight)

        torch.nn.init.xavier_normal_(self.conv_up_1.weight)
        torch.nn.init.xavier_normal_(self.conv_up_2.weight)
        torch.nn.init.xavier_normal_(self.conv_up_3.weight)

    def forward(self, field):

        if self.field_name == "Hx":

            out = self.tanh(self.linear_up_1(field)).view(-1, 1, 5 ** 2)
            out = self.tanh(self.linear_up_2(out)).view(-1, 1, 15 ** 2)
            out = self.tanh(self.linear_up_3(out)).view(-1, 1, 25 ** 2)
            out = self.tanh(self.linear_up_4(out)).view(-1, 1, 30, 30)
            out = self.tanh(self.conv_up_1(out)).view(-1, self.channels, 48, 48)
            out = self.tanh(self.conv_up_2(out)).view(-1, self.channels, 64, 64)
            out = self.tanh(self.conv_up_3(out)).view(-1, self.channels, 78, 78)
            out = self.tanh(self.conv_up_4(out)).view(-1, self.channels, 90, 90)
            out = self.tanh(self.conv_up_5(out)).view(-1, 1, 100 ** 2)

            out = self.linear_up_5(out).view(-1, 100, 100)

            return out, field

        else:

            out = self.sigmoid(self.linear_up_1(field)).view(-1, 1, 5 ** 2)
            out = self.sigmoid(self.linear_up_2(out)).view(-1, 1, 15 ** 2)
            out = self.sigmoid(self.linear_up_3(out)).view(-1, 1, 25 ** 2)
            out = self.sigmoid(self.linear_up_4(out)).view(-1, 1, 30, 30)
            out = self.sigmoid(self.conv_up_1(out)).view(-1, self.channels, 48, 48)
            out = self.sigmoid(self.conv_up_2(out)).view(-1, self.channels, 64, 64)
            out = self.sigmoid(self.conv_up_3(out)).view(-1, self.channels, 78, 78)
            out = self.sigmoid(self.conv_up_4(out)).view(-1, self.channels, 90, 90)
            out = self.sigmoid(self.conv_up_5(out)).view(-1, 1, 100 ** 2)

            out = self.linear_up_5(out).view(-1, 100, 100)

            return out, field


class ModeSolver_Network(object):
    def __init__(
        self,
        wavelength,
        width,
        thickness,
        sklearn_save,
        torch_save_x,
        torch_save_y,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        x=None,
        y=None,
    ):

        self.wavelength = wavelength
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.x = x
        self.y = y
        self.sklearn_save = sklearn_save
        self.torch_save_x = torch_save_x
        self.torch_save_y = torch_save_y

        if x == None:
            self.x = np.linspace(0, cladding_width, FIELD_WIDTH)
        if y == None:
            self.y = np.linspace(0, cladding_width, FIELD_WIDTH)

    def solve(self):

        self.modes = []

        for i in range(self.num_modes):
            Hx, Hy, neff = self.data(
                i, self.width, self.thickness, self.wavelength, self.sklearn_save, self.torch_save_x, self.torch_save_y
            )
            self.modes.append((Hx, Hy, neff))

    def data(self, mode_num, width, thickness, wavelength, sklearn_save, torch_save_x, torch_save_y):

        neff = self.neff_regression(mode_num, width, thickness, wavelength, sklearn_save)
        Hx = self.Hx_network(mode_num, width, thickness, wavelength, torch_save_x)
        Hy = self.Hy_network(mode_num, width, thickness, wavelength, torch_save_y)

        return Hx, Hy, neff

    def neff_regression(self, mode_num, width, thickness, wavelength, sklearn_save):

        with open(sklearn_save, "rb") as f:
            model = pickle.load(f)

        poly = PolynomialFeatures(degree=8)
        X = poly.fit_transform([[width * 1e6, thickness * 1e6, wavelength * 1e6]])
        neff = model.predict(X)

        return neff[0]

    def Hx_network(self, mode_num, width, thickness, wavelength, torch_save):

        with open(torch_save, "rb") as f:
            model = Network(3, 5, "Hx")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

            model.eval()

        with torch.no_grad():
            parameters = torch.Tensor([[[self.width * 1e6, self.thickness * 1e6, self.wavelength * 1e6]]])
            output, _ = model(parameters)
            output = deNormalizeHx(output)

        return output

    def Hy_network(self, mode_num, width, thickness, wavelength, torch_save):

        with open(torch_save, "rb") as f:
            model = Network(3, 15, "Hy")

            # original saved file with DataParallel
            state_dict = torch.load(f)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

            model.eval()

        with torch.no_grad():
            parameters = torch.Tensor([[[self.width * 1e6, self.thickness * 1e6, self.wavelength * 1e6]]])
            output, _ = model(parameters)
            output = deNormalizeHy(output)

        return output

    def clear(self):
        self.modes = []

    def getMode(self, mode_num=0):

        Hx, Hy, neff = self.modes[mode_num]
        m = Mode(self.x, self.y, self.wavelength, neff, Hx, Hy, None, None, None, None)
        m.compute_other_fields(self.width, self.thickness)

        return m

