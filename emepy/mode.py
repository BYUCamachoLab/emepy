import numpy as np
from matplotlib import pyplot as plt
import pickle
import random
import EMpy
from emepy import tools
from EMpy.modesolvers.FD import stretchmesh


class Mode(object):
    """Object that holds the field profiles and effective index for an eigenmode
    """

    def __init__(self, x, y, wl, neff, Hx, Hy, Hz, Ex, Ey, Ez):
        """Constructor for Mode Object
            :param x (ndarray float): array of grid points in x direction (propogation in z)
            :param y (ndarray float): array of grid points in y direction (propogation in z)
            :param wl (float): wavelength (meters)
            :param neff (float): effective index
            :param Hx (ndarray float): Hx field profile
            :param Hy (ndarray float): Hy field profile
            :param Hz (ndarray float): Hz field profile
            :param Ex (ndarray float): Ex field profile
            :param Ey (ndarray float): Ey field profile
            :param Ez (ndarray float): Ez field profile
        """

        self.x = x
        self.y = y
        self.wl = wl
        self.neff = neff
        self.Hx = np.array(Hx)
        self.Hy = np.array(Hy)
        self.Hz = np.array(Hz)
        self.Ex = np.array(Ex)
        self.Ey = np.array(Ey)
        self.Ez = np.array(Ez)

    def plot(self, value_type="Real", colorbar=True):

        self /= max([np.abs(np.real(np.amax(i))) for i in [self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz]])

        if value_type == "Imaginary":
            Hx = np.imag(self.Hx).T
            Hy = np.imag(self.Hy).T
            Hz = np.imag(self.Hz).T
            Ex = np.imag(self.Ex).T
            Ey = np.imag(self.Ey).T
            Ez = np.imag(self.Ez).T
        elif value_type == "Abs":
            Hx = np.abs(self.Hx).T
            Hy = np.abs(self.Hy).T
            Hz = np.abs(self.Hz).T
            Ex = np.abs(self.Ex).T
            Ey = np.abs(self.Ey).T
            Ez = np.abs(self.Ez).T
        elif value_type == "Abs^2":
            Hx = np.abs(self.Hx).T ** 2
            Hy = np.abs(self.Hy).T ** 2
            Hz = np.abs(self.Hz).T ** 2
            Ex = np.abs(self.Ex).T ** 2
            Ey = np.abs(self.Ey).T ** 2
            Ez = np.abs(self.Ez).T ** 2
        elif value_type == "Real":
            Hx = np.real(self.Hx).T
            Hy = np.real(self.Hy).T
            Hz = np.real(self.Hz).T
            Ex = np.real(self.Ex).T
            Ey = np.real(self.Ey).T
            Ez = np.real(self.Ez).T
        else:
            raise Exception("Invalid value_type entered. Please choose from ('Imaginary', 'Abs', 'Abs^2', 'Real')")

        plt.subplot(2, 3, 4, adjustable="box", aspect=Ex.shape[0] / Ex.shape[1])
        v = max(abs(Ex.min()), abs(Ex.max()))
        plt.imshow(
            Ex,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Ex)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.subplot(2, 3, 5, adjustable="box", aspect=Ey.shape[0] / Ey.shape[1])
        v = max(abs(Ey.min()), abs(Ey.max()))
        plt.imshow(
            Ey,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Ey)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.subplot(2, 3, 6, adjustable="box", aspect=Ez.shape[0] / Ez.shape[1])
        v = max(abs(Ez.min()), abs(Ez.max()))
        plt.imshow(
            Ez,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Ez)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.subplot(2, 3, 1, adjustable="box", aspect=Hx.shape[0] / Hx.shape[1])
        v = max(abs(Hx.min()), abs(Hx.max()))
        plt.imshow(
            Hx,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Hx)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.subplot(2, 3, 2, adjustable="box", aspect=Hy.shape[0] / Hy.shape[1])
        v = max(abs(Hy.min()), abs(Hy.max()))
        plt.imshow(
            Hy,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Hy)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.subplot(2, 3, 3, adjustable="box", aspect=Hz.shape[0] / Hz.shape[1])
        v = max(abs(Hz.min()), abs(Hz.max()))
        plt.imshow(
            Hz,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6],
        )
        plt.title(value_type + "(Hz)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.tight_layout()

    def _inner_product(self, mode1, mode2):

        res = mode1.Ex.shape[0] * mode1.Ex.shape[1]
        Ex = mode1.Ex.reshape(1, res)
        Ey = mode1.Ey.reshape(1, res)
        Ez = mode1.Ez.reshape(1, res)
        L = np.stack([Ex, Ey, Ez], axis=-1)[0]

        res = mode2.Hx.shape[0] * mode2.Hx.shape[1]
        Hx = np.conj(mode2.Hx).reshape(1, res)
        Hy = np.conj(mode2.Hy).reshape(1, res)
        Hz = np.conj(mode2.Hz).reshape(1, res)
        R = np.stack([Hx, Hy, Hz], axis=-1)[0]

        cross = np.cross(L, R)[:, 2]
        size = int(np.sqrt(cross.shape[0]))
        cross = cross.reshape((size, size))

        return 0.5 * np.trapz(np.trapz(cross, mode1.x), mode1.y)

    def inner_product(self, mode2):
        return self._inner_product(self, mode2)
        # return self._inner_product(self, mode2) * self._inner_product(mode2, self) / self._inner_product(self, self)

    def __str__(self):

        return "Mode Object with effective index of " + str(self.neff)

    def __mul__(self, other):

        self.Hx *= other
        self.Hy *= other
        self.Hz *= other
        self.Ex *= other
        self.Ey *= other
        self.Ez *= other

        return self

    def __add__(self, other):

        self.Hx += other
        self.Hy += other
        self.Hz += other
        self.Ex += other
        self.Ey += other
        self.Ez += other

        return self

    def __truediv__(self, other):

        self.Hx /= other
        self.Hy /= other
        self.Hz /= other
        self.Ex /= other
        self.Ey /= other
        self.Ez /= other

        return self

    def __sub__(self, other):

        self.Hx -= other
        self.Hy -= other
        self.Hz -= other
        self.Ex -= other
        self.Ey -= other
        self.Ez -= other

        return self

    def normalize(self):

        factor = self.inner_product(self)
        self /= np.sqrt(factor)

    def zero_phase(self):

        index = int(self.Hy.shape[0] / 2)
        phase = np.angle(np.array(self.Hy))[index][index]
        self *= np.exp(-1j * phase)
        if (np.sum(np.real(self.Hy))) < 0:
            self *= -1

    def get_fields(self):

        return [self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez]

    def get_H(self):

        return [self.Hx, self.Hy, self.Hz]

    def get_E(self):

        return [self.Ex, self.Ey, self.Ez]

    def get_neff(self):

        return self.neff

    def get_wavelength(self):

        return self.wl

    def save(self, path=None, other=None):

        data = (self.x, self.y, self.wl, self.neff, self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez, other)

        if path:
            pickle.dump(data, open(path, "wb+"))
        else:
            pickle.dump(data, open("ModeObject_" + str(random.random()) + ".pk", "wb+"))

    def load(self, path):

        self.x, self.y, self.wl, self.neff, self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez, self.other = pickle.load(
            open(path, "rb")
        )

    def compute_other_fields(self, width, thickness):
        """Adapted from the EMpy library"""

        from scipy.sparse import coo_matrix

        core_index = tools.Si(self.wl * 1e6)
        cladding_index = tools.SiO2(self.wl * 1e6)
        self.epsfunc = tools.get_epsfunc(
            width, thickness, 2.5e-6, 2.5e-6, tools.Si(self.wl * 1e6), tools.SiO2(self.wl * 1e6)
        )

        wl = self.wl
        x = np.array(self.x)
        y = np.array(self.y)
        boundary = "0000"  # "A0"

        neffs = [self.neff]
        Hxs = [np.array(self.Hx)]
        Hys = [np.array(self.Hy)]

        Hzs = []
        Exs = []
        Eys = []
        Ezs = []
        for neff, Hx, Hy in zip(neffs, Hxs, Hys):

            dx = np.diff(x)
            dy = np.diff(y)

            dx = np.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
            dy = np.r_[dy[0], dy, dy[-1]].reshape(1, -1)

            xc = (x[:-1] + x[1:]) / 2
            yc = (y[:-1] + y[1:]) / 2
            epsxx, epsxy, epsyx, epsyy, epszz = self._get_eps(xc, yc)

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
                    -0.5
                    * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e)
                    * ezz1
                    * (1 - exx4 / ezz4)
                    / n
                    / exx4
                    * ezz4
                    - 0.5
                    * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
                    * ezz2
                    * (1 - exx3 / ezz3)
                    / s
                    / exx3
                    * ezz3
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
                    + 0.5
                    * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e)
                    * ezz3
                    * (1 - exx2 / ezz2)
                    / s
                    / exx2
                    * ezz2
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
                            -k ** 2 * exy1
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
                            -k ** 2 * exy4
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
                            -k ** 2 * exy2
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
                            -k ** 2 * exy3
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
                    0.5
                    * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
                    * ezz1
                    * ezz2
                    / eyy1
                    * (1 - eyy1 / ezz1)
                    / w
                    - 0.5
                    * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
                    * ezz4
                    * ezz3
                    / eyy4
                    * (1 - eyy4 / ezz4)
                    / e
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
                    -0.5
                    * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3)
                    * ezz2
                    * ezz1
                    / eyy2
                    * (1 - eyy2 / ezz2)
                    / w
                    + 0.5
                    * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2)
                    * ezz3
                    * ezz4
                    / eyy3
                    * (1 - eyy3 / ezz3)
                    / e
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
                            -k ** 2 * eyx1
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
                            -k ** 2 * eyx2
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
                            -k ** 2 * eyx4
                            + (1 - eyy4 / ezz4) / n / e
                            - eyx4 / ezz4 * (-2.0 / e ** 2 - 2 / e ** 2 * (e - w) / w)
                        )
                        + 0.5
                        * s
                        * ezz3
                        * ezz4
                        / eyy3
                        * (
                            -k ** 2 * eyx3
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
                bzxp[iall],
                bzxe[i_w],
                bzxw[i_e],
                bzxn[i_s],
                bzxs[i_n],
                bzxsw[i_ne],
                bzxnw[i_se],
                bzxne[i_sw],
                bzxse[i_nw],
            ]

            Izy = np.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
            Jzy = np.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
            Vzy = np.r_[
                bzyp[iall],
                bzye[i_w],
                bzyw[i_e],
                bzyn[i_s],
                bzys[i_n],
                bzysw[i_ne],
                bzynw[i_se],
                bzyne[i_sw],
                bzyse[i_nw],
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
            Dx = neff * EMpy.utils.centered2d(Hy) + (Hz[:-1, 1:] + Hz[1:, 1:] - Hz[:-1, :-1] - Hz[1:, :-1]) / (
                2j * k * v
            )
            Dy = -neff * EMpy.utils.centered2d(Hx) - (Hz[1:, :-1] + Hz[1:, 1:] - Hz[:-1, 1:] - Hz[:-1, :-1]) / (
                2j * k * h
            )
            Dz = (
                (Hy[1:, :-1] + Hy[1:, 1:] - Hy[:-1, 1:] - Hy[:-1, :-1]) / (2 * h)
                - (Hx[:-1, 1:] + Hx[1:, 1:] - Hx[:-1, :-1] - Hx[1:, :-1]) / (2 * v)
            ) / (1j * k)

            Ex = (eyy * Dx - exy * Dy) / edet
            Ey = (exx * Dy - eyx * Dx) / edet
            Ez = Dz / ezz

            Hzs.append(Hz)
            Exs.append(Ex)
            Eys.append(Ey)
            Ezs.append(Ez)

        self.Hz = Hzs[0]
        self.Ex = Exs[0]
        self.Ey = Eys[0]
        self.Ez = Ezs[0]

    def _get_eps(self, xc, yc):
        tmp = self.epsfunc(xc, yc)

        def _reshape(tmp):
            """
            pads the array by duplicating edge values
            """
            tmp = np.c_[tmp[:, 0:1], tmp, tmp[:, -1:]]
            tmp = np.r_[tmp[0:1, :], tmp, tmp[-1:, :]]
            return tmp

        if tmp.ndim == 2:  # isotropic refractive index
            tmp = _reshape(tmp)
            epsxx = epsyy = epszz = tmp
            epsxy = epsyx = np.zeros_like(epsxx)

        elif tmp.ndim == 3:  # anisotropic refractive index
            assert tmp.shape[2] == 5, "eps must be NxMx5"
            epsxx = _reshape(tmp[:, :, 0])
            epsxy = _reshape(tmp[:, :, 1])
            epsyx = _reshape(tmp[:, :, 2])
            epsyy = _reshape(tmp[:, :, 3])
            epszz = _reshape(tmp[:, :, 4])

        else:
            raise ValueError("Invalid eps")

        return epsxx, epsxy, epsyx, epsyy, epszz

