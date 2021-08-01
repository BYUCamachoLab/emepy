import numpy as np
from simphony import Model
from simphony.pins import Pin
from simphony.tools import wl2freq
from simphony.models import Subcircuit


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


class EME(object):
    def __init__(self, layers=[], num_periods=1):

        self.reset()
        self.layers = layers
        self.num_periods = num_periods

    def add_layer(self, layer):
        self.layers.append(layer)

    def reset(self):
        self.layers = []
        self.interfaces = []
        self.wavelength = None
        self.s_params = None
        self.interface = None

    def propagate_period(self):

        # Propagate the first two layers
        self.layers[0].activate_layer()
        mode_set1 = self.layers[0].get_activated_layer()
        self.layers[1].activate_layer()
        current = Current(self.wavelength, self.layers[0].get_activated_layer())
        interface = self.interface(self.layers[0].get_activated_layer(), self.layers[1].get_activated_layer())
        interface.solve()
        self.layers[0].clear()
        current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)
        interface.clear()

        # Propagate the middle layers
        for index in range(1, len(self.layers) - 1):

            layer1_ = self.layers[index]
            layer2_ = self.layers[index + 1]
            layer2_.activate_layer()

            layer1 = layer1_.get_activated_layer()
            layer2 = layer2_.get_activated_layer()

            interface = self.interface(layer1, layer2)
            interface.solve()

            current.update_s(self.cascade(Current(self.wavelength, current), layer1), layer1)
            layer1_.clear()
            current.update_s(self.cascade(Current(self.wavelength, current), interface), interface)
            interface.clear()

        # Propagate final two layers
        current.update_s(
            self.cascade(Current(self.wavelength, current), self.layers[-1].get_activated_layer()),
            self.layers[-1].get_activated_layer(),
        )

        # Gather and return the s params and edge layers
        mode_set2 = self.layers[-1].get_activated_layer()
        self.layers[-1].clear()
        self.s_params = current.s_params

        return (current.s_params, mode_set1, mode_set2)

    def propagate(self):

        # Check for layers
        if not len(self.layers):
            raise Exception("Must place layers before propagating")
        else:
            self.wavelength = self.layers[0].wavelength

        # Decide which routine to use
        num_modes = max([l.num_modes for l in self.layers])
        self.interface = InterfaceSingleMode if num_modes == 1 else InterfaceMultiMode

        # Run the eme for one period and collect s params and edge layers
        single_period, left, right = self.propagate_period()

        # Create an interface between the two returns layers
        period_layer = PeriodicLayer(left.modes, right.modes, single_period)
        current_layer = PeriodicLayer(left.modes, right.modes, single_period)
        interface = self.interface(right, left)
        interface.solve()

        # Make memory
        self.layers[0].clear()
        self.layers[1].clear()

        # print(current_layer.s_params)

        # Cascade params for each period
        from copy import copy
        for _ in range(self.num_periods - 1):

            current_layer.s_params = self.cascade((current_layer), (interface))
            current_layer.s_params = self.cascade((current_layer), (period_layer))
            # current_layer.s_params = self.cascade(current_layer, period_layer)
            
        self.s_params = current_layer.s_params

    def cascade(self, first, second):
        Subcircuit.clear_scache()

        # make sure the components are completely disconnected
        first.disconnect()
        second.disconnect()

        # connect the components
        for port in range(first.right_ports):
            first[f"right{port}"].connect(second[f"left{port}"])

        # get the scattering parameters
        return first.circuit.s_parameters(np.array([self.wavelength]))

    def get_s_params(self):

        return self.s_params

    def draw(self):

        # Simple drawing to debug geometry
        plt.figure()
        lengths = [0.0] + [i.length for i in self.layers]
        lengths = [sum(lengths[: i + 1]) for i in range(len(lengths))]
        widths = [self.layers[0].mode_solvers.width] + [i.mode_solvers.width for i in self.layers]
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


class Current(Model):
    def __init__(self, wavelength, s, **kwargs):
        self.left_ports = s.left_ports
        self.left_pins = s.left_pins
        self.s_params = s.s_params
        self.right_ports = s.right_ports
        self.num_ports = self.right_ports + self.left_ports
        self.right_pins = s.right_pins
        self.wavelength = wavelength

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def update_s(self, s, layer):

        self.s_params = s
        self.right_ports = layer.right_ports
        self.num_ports = self.right_ports + self.left_ports
        self.right_pins = layer.right_pins

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(pins=pins)

    def s_parameters(self, freq):
        return self.s_params


class ActivatedLayer(Model):
    def __init__(self, modes, wavelength, length, **kwargs):
        self.num_modes = len(modes)
        self.modes = modes
        self.wavelength = wavelength
        self.length = length
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(self.num_modes)]
        self.right_pins = ["right" + str(i) for i in range(self.num_modes)]
        self.s_params = self.get_s_params()

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

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

    def s_parameters(self, freqs):
        return self.s_params


class PeriodicLayer(Model):
    def __init__(self, left_modes, right_modes, s_params, **kwargs):
        self.left_modes = left_modes
        self.right_modes = right_modes
        self.left_ports = len(self.left_modes)
        self.right_ports = len(self.right_modes)
        self.normalize_fields()
        self.left_pins = ["left" + str(i) for i in range(len(self.left_modes))]
        self.right_pins = ["right" + str(i) for i in range(len(self.right_modes))]
        self.s_params = s_params

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

    def normalize_fields(self):
        for mode in range(len(self.left_modes)):
            self.left_modes[mode].normalize()
        for mode in range(len(self.right_modes)):
            self.right_modes[mode].normalize()

    def s_parameters(self, freqs):
        return self.s_params


class InterfaceSingleMode(Model):
    def __init__(self, layer1, layer2, num_modes=1, **kwargs):
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_modes = num_modes
        self.left_ports = num_modes
        self.right_ports = num_modes
        self.num_ports = self.left_ports + self.right_ports
        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))
        super().__init__(**kwargs, pins=pins)

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
    def __init__(self, layer1, layer2, **kwargs):
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_ports = layer1.right_ports + layer2.left_ports
        self.left_ports = layer1.right_ports
        self.right_ports = layer2.left_ports
        self.left_pins = ["left" + str(i) for i in range(self.left_ports)]
        self.right_pins = ["right" + str(i) for i in range(self.right_ports)]

        # create the pins for the model
        pins = []
        for name in self.left_pins:
            pins.append(Pin(self, name))
        for name in self.right_pins:
            pins.append(Pin(self, name))

        super().__init__(**kwargs, pins=pins)

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

