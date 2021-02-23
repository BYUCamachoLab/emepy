from emepy.mode import Mode

from simphony.elements import Model
from simphony.tools import interpolate
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation, wl2freq
import simphony

import numpy as np
from matplotlib import pyplot as plt


class Current(Model):
    freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def __init__(self, wavelength, s):
        self.left_ports = s.left_ports
        self.left_pins = s.left_pins
        self.update_s(s.s_params, s)
        self.wavelength = wavelength

    def update_s(self, s, layer):

        self.s_params = s
        self.right_ports = layer.right_ports
        self.num_ports = self.right_ports + self.left_ports
        self.right_pins = layer.right_pins
        self.pins = tuple(self.left_pins + self.right_pins)

    def s_parameters(self, freq):
        return self.s_params


class ActivatedLayer(Model):
    freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def __init__(self, modes, wavelength, length):
        self.num_modes = len(modes)
        self.modes = modes
        self.wavelength = wavelength
        self.length = length
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(self.num_modes)]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)]
        self.pins = tuple(self.left_pins + self.right_pins)
        self.s_params = self.get_s_params()
        self.freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def normalize_fields(self):
        for mode in range(len(self.modes)):
            self.modes[mode].normalize()

    def get_s_params(self):
        eigenvalues1 = (2 * np.pi) * np.array([mode.neff for mode in self.modes]) / (self.wavelength)

        propagation_matrix1 = np.diag(
            np.exp(self.length * 1j * np.array(eigenvalues1.tolist() + eigenvalues1.tolist()))
        )
        num_cols = len(propagation_matrix1)
        rows = np.array(np.split(propagation_matrix1, num_cols))
        first_half = [rows[j + num_cols // 2][0] for j in range(num_cols // 2)]
        second_half = [rows[j][0] for j in range(num_cols // 2)]
        propagation_matrix = np.array(first_half + second_half).reshape((1, 2 * self.num_modes, 2 * self.num_modes))

        self.right_ports = self.num_modes
        self.left_ports = self.num_modes
        self.num_ports = 2 * self.num_modes

        return propagation_matrix

    def s_parameters(self, freq):
        return self.s_params


class Layer(object):
    def __init__(self, mode_solvers, num_modes, wavelength, length):

        self.num_modes = num_modes
        self.mode_solvers = mode_solvers
        self.wavelength = wavelength
        self.length = length

    def activate_layer(self):

        modes = []

        if type(self.mode_solvers) != list:
            self.mode_solvers.solve()
            for mode in range(self.num_modes):
                modes.append(self.mode_solvers.get_mode(mode))

        else:
            for index in range(len(self.mode_solvers)):
                self.mode_solvers[index][0].solve()
                for mode in range(self.mode_solvers[index][1]):
                    modes.append(self.mode_solvers[index][0].get_mode(mode))

        self.activated_layer = ActivatedLayer(modes, self.wavelength, self.length)

    def get_activated_layer(self):
        return self.activated_layer

    def clear(self):

        if type(self.mode_solvers) != list:
            self.mode_solvers.clear()
        else:
            for index in range(len(self.mode_solvers)):
                self.mode_solvers[index][0].clear()


class PeriodicLayer(Model):
    freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def __init__(self, left_modes, right_modes, s_params):
        self.left_modes = left_modes
        self.right_modes = right_modes
        self.left_ports = len(self.left_modes)
        self.right_ports = len(self.right_modes)
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(len(self.left_modes))]
        self.right_pins = ["right" + str(i) for i in range(len(self.right_modes))]
        self.pins = tuple(self.left_pins + self.right_pins)
        self.s_params = s_params
        self.freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def normalize_fields(self):
        for mode in range(len(self.left_modes)):
            self.left_modes[mode].normalize()
        for mode in range(len(self.right_modes)):
            self.right_modes[mode].normalize()

    def s_parameters(self, freq=1.55e-6):
        return self.s_params


class PeriodicEME(object):
    def __init__(self, layers=[], num_periods=1):

        self.layers = layers
        self.num_periods = num_periods

    def add_layer(self, layer):
        self.layers.append(layer)

    def propagate(self):

        if not len(self.layers):
            raise Exception("Must place layers before propagating")

        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceMultiMode if num_modes == 1 else InterfaceMultiMode
        eme = EME(keep_modeset=True, layers=self.layers)
        eme.propagate()
        self.single_period = eme.get_s_params()

        left = eme.mode_set1
        right = eme.mode_set2

        eme.clear()

        period_layer = PeriodicLayer(left.modes, right.modes, self.single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, self.single_period)
        interface = self.interface(right, left)
        interface.solve()

        self.layers[0].clear()
        self.layers[1].clear()

        for _ in range(self.num_periods - 1):

            current_layer.s_params = self.cascade(current_layer, interface)
            current_layer.s_params = self.cascade(current_layer, period_layer)

        self.s_params = current_layer.s_params

    def cascade(self, first, second):

        circuit = Subcircuit("Device")

        circuit.add([(first, "first"), (second, "second")])
        for port in range(first.right_ports):
            circuit.connect("first", "right" + str(port), "second", "left" + str(port))

        simulation = SweepSimulation(circuit, 1.55e-6, 1.55e-6, num=1)
        result = simulation.simulate()

        return result.s

    def s_parameters(self):

        return self.s_params

    def draw(self):

        plt.figure()
        lengths = [0.0] + [i.length for i in self.layers]
        lengths = [sum(lengths[: i + 1]) for i in range(len(lengths))]
        widths = [self.layers[0].mode_solvers.width] + [i.mode_solvers.width for i in self.layers]

        # lengths = lengths[:-1]
        # widths = widths[:-1]

        lengths_ = lengths
        widths_ = widths
        for i in range(1, self.num_periods):
            widths_ = widths_ + widths
            lengths_ = lengths_ + (np.array(lengths) + i * lengths[-1]).tolist()

        sub = [0] + np.diff(lengths_).tolist()
        lengths = []
        widths = []
        for i in range(len(lengths_)):
            lengths.append(lengths_[i] - sub[i])
            lengths.append(lengths_[i])
            widths.append(widths_[i])
            widths.append(widths_[i])

        plt.plot(lengths, np.array(widths), "b")
        plt.plot(lengths, -1 * np.array(widths), "b")

        plt.show()


class EME(object):
    def __init__(self, layers=[], keep_modeset=False):
        self.layers = layers
        self.interfaces = []
        self.wavelength = None if not len(self.layers) else layers[0].wavelength
        self.keep_modeset = keep_modeset

    def add_layer(self, layer):
        self.layers.append(layer)
        if not self.wavelength:
            self.wavelength = layer.wavelength

    def propagate(self):

        if not len(self.layers):
            raise Exception("Must place layers before propagating")

        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        self.layers[0].activate_layer()
        self.mode_set1 = self.layers[0].get_activated_layer() if self.keep_modeset else None
        self.layers[1].activate_layer()
        current = Current(self.wavelength, self.layers[0].get_activated_layer())
        interface = self.interface(self.layers[0].get_activated_layer(), self.layers[1].get_activated_layer())
        interface.solve()
        self.layers[0].clear()
        current.update_s(self.cascade(current, interface), interface)
        interface.clear()

        for index in range(1, len(self.layers) - 1):

            layer1_ = self.layers[index]
            layer2_ = self.layers[index + 1]
            layer2_.activate_layer()

            layer1 = layer1_.get_activated_layer()
            layer2 = layer2_.get_activated_layer()

            interface = self.interface(layer1, layer2)
            interface.solve()

            current.update_s(self.cascade(current, layer1), layer1)
            layer1_.clear()
            current.update_s(self.cascade(current, interface), interface)
            interface.clear()

        current.update_s(
            self.cascade(current, self.layers[-1].get_activated_layer()), self.layers[-1].get_activated_layer()
        )

        self.mode_set2 = self.layers[-1].get_activated_layer() if self.keep_modeset else None

        self.layers[-1].clear()
        self.s_matrix = current.s_params

    def draw(self):

        plt.figure()
        lengths = [0.0] + [i.length for i in self.layers]
        lengths = [sum(lengths[: i + 1]) for i in range(len(lengths))]
        widths = [self.layers[0].mode_solvers.width] + [i.mode_solvers.width for i in self.layers]

        # lengths = lengths[:-1]
        # widths = widths[:-1]

        lengths_ = lengths
        widths_ = widths
        # for i in range(1, self.num_periods):
        #     widths_ = widths_ + widths
        #     lengths_ = lengths_ + (np.array(lengths) + i * lengths[-1]).tolist()

        sub = [0] + np.diff(lengths_).tolist()
        lengths = []
        widths = []
        for i in range(len(lengths_)):
            lengths.append(lengths_[i] - sub[i])
            lengths.append(lengths_[i])
            widths.append(widths_[i])
            widths.append(widths_[i])

        plt.plot(lengths, np.array(widths), "b")
        plt.plot(lengths, -1 * np.array(widths), "b")

        plt.show()

    def cascade(self, first, second):

        circuit = Subcircuit("Device")

        circuit.add([(first, "first"), (second, "second")])
        for port in range(first.right_ports):
            circuit.connect("first", "right" + str(port), "second", "left" + str(port))

        simulation = SweepSimulation(circuit, self.wavelength, self.wavelength, num=1)
        result = simulation.simulate()

        return result.s

    def clear(self):
        self.layers = None
        self.interfaces = []
        self.wavelength = None

    def get_s_params(self):
        return self.s_matrix


class InterfaceSingleMode(Model):
    def __init__(self, layer1, layer2, num_modes=1):
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_modes = num_modes
        self.left_ports = num_modes
        self.right_ports = num_modes
        self.num_ports = self.left_ports + self.right_ports
        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]
        self.pins = tuple(self.left_pins + self.right_pins)
        self.freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def s_parameters(self, freq):
        return self.s

    def solve(self):

        s = np.zeros((2 * self.num_modes, 2 * self.num_modes), dtype=complex)

        for inp in range(len(self.layer1.modes)):
            for outp in range(len(self.layer2.modes)):

                left_mode = self.layer1.modes[inp]
                right_mode = self.layer2.modes[outp]

                r, t = self.get_values(left_mode, right_mode)

                s[outp, inp] = r
                s[outp + self.num_modes, inp] = t

        for inp in range(len(self.layer2.modes)):
            for outp in range(len(self.layer1.modes)):

                left_mode = self.layer1.modes[outp]
                right_mode = self.layer2.modes[inp]

                r, t = self.get_values(right_mode, left_mode)

                s[outp, inp + self.num_modes] = t
                s[outp + self.num_modes, inp + self.num_modes] = r

        self.s = s.reshape((1, 2 * self.num_modes, 2 * self.num_modes))

    def get_values(self, left, right):

        a = 0.5 * left.inner_product(right) + 0.5 * right.inner_product(left)
        b = 0.5 * left.inner_product(right) - 0.5 * right.inner_product(left)

        t = (a ** 2 - b ** 2) / a
        r = 1 - t / (a + b)

        return -r, t

    def clear(self):
        self.s = None


class InterfaceMultiMode(Model):
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_ports = layer1.right_ports + layer2.left_ports
        self.left_ports = layer1.right_ports
        self.right_ports = layer2.left_ports

        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]
        self.pins = tuple(self.left_pins + self.right_pins)
        self.freq_range = (wl2freq(2000e-9), wl2freq(1000e-9))

    def s_parameters(self, freq):
        return self.s

    def solve(self):

        s = np.zeros((self.num_ports, self.num_ports), dtype=complex)

        for p in range(self.left_ports):

            ts = self.get_t(p, self.layer1, self.layer2, self.left_ports)
            rs = self.get_r(p, ts, self.layer1, self.layer2, self.left_ports)

            for t in range(len(ts)):
                s[self.left_ports + t][p] = ts[t]
            for r in range(len(rs)):
                s[r][p] = rs[r]

        for p in range(self.right_ports):

            ts = self.get_t(p, self.layer2, self.layer1, self.right_ports)
            rs = self.get_r(p, ts, self.layer2, self.layer1, self.right_ports)

            for t in range(len(ts)):
                s[t][self.left_ports + p] = ts[t]
            for r in range(len(rs)):
                s[self.left_ports + r][self.left_ports + p] = rs[r]

        self.s = s.reshape((1, self.num_ports, self.num_ports))

    def get_t(self, p, left, right, curr_ports):

        # Ax = b
        A = np.array(
            [
                [
                    right.modes[k].inner_product(left.modes[i]) + left.modes[i].inner_product(right.modes[k])
                    for k in range(self.num_ports - curr_ports)
                ]
                for i in range(curr_ports)
            ]
        )
        b = np.array([0 if i != p else 2 * left.modes[p].inner_product(left.modes[p]) for i in range(curr_ports)])
        x = np.matmul(np.linalg.pinv(A), b)

        return x

    def get_r(self, p, x, left, right, curr_ports):

        rs = np.array(
            [
                np.sum(
                    [
                        (right.modes[k].inner_product(left.modes[i]) - left.modes[i].inner_product(right.modes[k]))
                        * x[k]
                        for k in range(self.num_ports - curr_ports)
                    ]
                )
                / (2 * left.modes[i].inner_product(left.modes[i]))
                for i in range(curr_ports)
            ]
        )

        return rs

    def clear(self):

        self.s_params = None

