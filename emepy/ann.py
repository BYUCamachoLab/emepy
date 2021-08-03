import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import time

from emepy.mode import Mode
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
from fd import Modesolver


FIELD_WIDTH = 128
FIELD_SIZE = FIELD_WIDTH ** 2


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
    def __init__(self, code_size, channels):
        super().__init__()
        self.channels = channels

        self.linear_up_1 = nn.Linear(code_size, int(FIELD_WIDTH/20)**2)
        self.linear_up_2 = nn.Linear(int(FIELD_WIDTH/20)**2, int(FIELD_WIDTH/7)**2)
        self.linear_up_3 = nn.Linear(int(FIELD_WIDTH/7)**2, int(FIELD_WIDTH/4)**2)
        self.linear_up_4 = nn.Linear(int(FIELD_WIDTH/4)**2, int(FIELD_WIDTH/3)**2)
        self.conv_up_1 = getUpConvLayer(int(FIELD_WIDTH/3), int(FIELD_WIDTH/2), 3, channels, first=True)
        self.conv_up_2 = getUpConvLayer(int(FIELD_WIDTH/2), int(5*FIELD_WIDTH/8), 5, channels)
        self.conv_up_3 = getUpConvLayer(int(5*FIELD_WIDTH/8), int(6*FIELD_WIDTH/8), 7, channels)
        self.conv_up_4 = getUpConvLayer(int(6*FIELD_WIDTH/8), int(7*FIELD_WIDTH/8), 7, channels)
        self.conv_up_5 = getUpConvLayer(int(7*FIELD_WIDTH/8), int(FIELD_WIDTH), 9, channels,last=True)
        self.linear_up_5 = nn.Linear(FIELD_WIDTH**2, FIELD_WIDTH**2)

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


        out = self.tanh(self.linear_up_1(field)).view(-1,1,int(FIELD_WIDTH/20)**2)
        out = self.tanh(self.linear_up_2(out)).view(-1,1,int(FIELD_WIDTH/7)**2)
        out = self.tanh(self.linear_up_3(out)).view(-1,1,int(FIELD_WIDTH/4)**2)
        out = self.tanh(self.linear_up_4(out)).view(-1,1,int(FIELD_WIDTH/3),int(FIELD_WIDTH/3))
        out = self.tanh(self.conv_up_1(out)).view(-1,self.channels,int(FIELD_WIDTH/2),int(FIELD_WIDTH/2))
        out = self.tanh(self.conv_up_2(out)).view(-1,self.channels,int(5*FIELD_WIDTH/8),int(5*FIELD_WIDTH/8))
        out = self.tanh(self.conv_up_3(out)).view(-1,self.channels,int(6*FIELD_WIDTH/8),int(6*FIELD_WIDTH/8))
        out = self.tanh(self.conv_up_4(out)).view(-1,self.channels,int(7*FIELD_WIDTH/8),int(7*FIELD_WIDTH/8))
        out = self.tanh(self.conv_up_5(out)).view(-1,1,FIELD_WIDTH**2)
        
        out = self.linear_up_5(out).view(-1,FIELD_WIDTH,FIELD_WIDTH)

        return out, field


class MSNeuralNetwork(Modesolver):
    def __init__(
        self,
        networkBaseObject,
        wl,
        width,
        thickness
    ):

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.networkBaseObject = networkBaseObject
        self.Hx_model = networkBaseObject.Hx_model
        self.Hy_model = networkBaseObject.Hy_model
        self.neff_model = networkBaseObject.neff_model
        self.num_modes = 1
        self.x = networkBaseObject.x
        self.y = networkBaseObject.y


    def solve(self):

        self.modes = []

        for i in range(self.num_modes):
            Hx, Hy, neff = self.data(
                i, self.width, self.thickness, self.wl
            )
            self.modes.append((Hx, Hy, neff))

    def data(self, mode_num, width, thickness, wl):

        neff = self.neff_regression(mode_num, width, thickness, wl, self.neff_model)
        Hx = self.Hx_network(mode_num, width, thickness, wl, self.Hx_model)
        Hy = self.Hy_network(mode_num, width, thickness, wl, self.Hy_model)

        return Hx, Hy, neff

    def neff_regression(self, mode_num, width, thickness, wl, model):

        poly = PolynomialFeatures(degree=8)
        X = poly.fit_transform([[width * 1e6, thickness * 1e6, wl * 1e6]])
        neff = model.predict(X)

        return neff[0]

    def Hx_network(self, mode_num, width, thickness, wl, model):

        with torch.no_grad():
            parameters = torch.Tensor([[[self.width * 1e6, self.thickness * 1e6, self.wl * 1e6]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128,128)

        return output

    def Hy_network(self, mode_num, width, thickness, wl, model):

        with torch.no_grad():
            parameters = torch.Tensor([[[self.width * 1e6, self.thickness * 1e6, self.wl * 1e6]]])
            output, _ = model(parameters)
            output = output.numpy()
            output = output.reshape(128,128)

        return output

    def clear(self):
        self.modes = []

    def get_mode(self, mode_num=0):

        Hx, Hy, neff = self.modes[mode_num]
        m = Mode(self.x, self.y, self.wl, neff, Hx+0j, Hy+0j, None, None, None, None, pickle.load(open("/fslhome/ihammond/GitHub/ANNEME/ANN/Network/output/03_good/n_profile", "rb")))
        m.compute_other_fields(self.width, self.thickness)

        return m


class ANN(object):
    def __init__(
        self,
        sklearn_save,
        torch_save_x,
        torch_save_y,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        x=None,
        y=None,
    ):
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

        self.Hx_model = self.Hx_network()
        self.Hy_model = self.Hy_network()
        self.neff_model = self.neff_regression()

    def neff_regression(self):

        with open(self.sklearn_save, "rb") as f:
            model = pickle.load(f)

        return model

    def Hx_network(self):

        with open(self.torch_save_x, "rb") as f:
            model = Network(3, 1)

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

        return model

    def Hy_network(self):

        with open(self.torch_save_y, "rb") as f:
            model = Network(3, 1)

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

        return model
