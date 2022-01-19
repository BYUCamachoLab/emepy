import numpy as np
from matplotlib import pyplot as plt
import pickle
import random
import EMpy_gpu
from emepy import tools
from scipy.signal import convolve2d


class Mode(object):
    """Object that holds the field profiles and effective index for an eigenmode"""

    def __init__(self, x, y, wl, neff, Hx, Hy, Hz, Ex, Ey, Ez, n=None, width=None, thickness=None):
        """Constructor for Mode Object

        Parameters
        ----------
        x : (ndarray float)
            array of grid points in x direction (propogation in z)
        y : (ndarray float)
            array of grid points in y direction (propogation in z)
        wl : (float)
            wavelength (meters)
        neff : (float)
            effective index
        Hx : (ndarray float)
            Hx field profile
        Hy : (ndarray float)
            Hy field profile
        Hz : (ndarray float)
            Hz field profile
        Ex : (ndarray float)
            Ex field profile
        Ey : (ndarray float)
            Ey field profile
        Ez : (ndarray float)
            Ez field profile
        width : number
            The core width
        thickness : number
            The core thickness
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
        self.n = n
        self.width = width
        self.thickness = thickness
        if True in [
            self.Ex is None,
            self.Ey is None,
            self.Ez is None,
            self.Hx is None,
            self.Hy is None,
            self.Hz is None,
        ]:
            self.H = np.sqrt(np.abs(self.Hx) ** 2 + np.abs(self.Hy) ** 2 + np.abs(self.Hz) ** 2)
            self.E = np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2)
        if self.n is None:
            eps_func = tools.get_epsfunc(
                self.width, self.thickness, 2.5e-6, 2.5e-6, tools.Si(self.wl * 1e6), tools.SiO2(self.wl * 1e6)
            )
            self.n = eps_func(self.x, self.y)

    def plot(self, operation="Real", colorbar=True, normalize=True):
        """Plots the fields in the mode using pyplot. Should call plt.figure() before and plt.show() or plt.savefig() after

        Parameters
        ----------
        operation : string or function
            the operation to perform on the fields from ("Real", "Imaginary", "Abs", "Abs^2") (default:"Real") or a function such as np.abs
        colorbar : bool
            if true, will show a colorbar for each field
        normalize : bool
            if true, will normalize biggest field to 1
        """

        temp = (
            self / max([np.abs(np.real(np.amax(i))) for i in [self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz]])
            if normalize
            else self / 1
        )

        if operation == "Imaginary":
            Hx = np.imag(temp.Hx).T
            Hy = np.imag(temp.Hy).T
            Hz = np.imag(temp.Hz).T
            Ex = np.imag(temp.Ex).T
            Ey = np.imag(temp.Ey).T
            Ez = np.imag(temp.Ez).T
        elif operation == "Abs":
            Hx = np.abs(temp.Hx).T
            Hy = np.abs(temp.Hy).T
            Hz = np.abs(temp.Hz).T
            Ex = np.abs(temp.Ex).T
            Ey = np.abs(temp.Ey).T
            Ez = np.abs(temp.Ez).T
        elif operation == "Abs^2":
            Hx = np.abs(temp.Hx).T ** 2
            Hy = np.abs(temp.Hy).T ** 2
            Hz = np.abs(temp.Hz).T ** 2
            Ex = np.abs(temp.Ex).T ** 2
            Ey = np.abs(temp.Ey).T ** 2
            Ez = np.abs(temp.Ez).T ** 2
        elif operation == "Real":
            Hx = np.real(temp.Hx).T
            Hy = np.real(temp.Hy).T
            Hz = np.real(temp.Hz).T
            Ex = np.real(temp.Ex).T
            Ey = np.real(temp.Ey).T
            Ez = np.real(temp.Ez).T
        else:
            try:
                Hx = operation(temp.Hx).T
                Hy = operation(temp.Hy).T
                Hz = operation(temp.Hz).T
                Ex = operation(temp.Ex).T
                Ey = operation(temp.Ey).T
                Ez = operation(temp.Ez).T
            except:
                raise Exception(
                    "Invalid operation provided. Please choose from ('Imaginary', 'Abs', 'Abs^2', 'Real') or provide a function"
                )

        if hasattr(operation, "__name__"):
            operation = operation.__name__
        else:
            operation = str(operation)
        plt.subplot(2, 3, 4, adjustable="box", aspect=Ex.shape[0] / Ex.shape[1])
        v = max(abs(Ex.min()), abs(Ex.max()))
        plt.imshow(
            Ex,
            cmap="RdBu",
            vmin=-v,
            vmax=v,
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Ex)")
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
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Ey)")
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
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Ez)")
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
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Hx)")
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
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Hy)")
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
            extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
        )
        plt.title(operation + "(Hz)")
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("x µm")
        plt.ylabel("y µm")

        plt.tight_layout()

    def _inner_product(self, mode1, mode2, mask=None):
        """Helper function that takes the inner product between Modes mode1 and mode2

        Parameters
        ----------
        mode1 : Mode
            first eigenmode in the operation
        mode2 : Mode
            second eigenmode in the operation

        Returns
        -------
        number
            the inner product between the two input modes
        """

        mask = 1 if mask is None else mask

        Ex = mode1.Ex * mask
        Hy = np.conj(mode2.Hy) * mask
        Ey = mode1.Ey * mask
        Hx = np.conj(mode2.Hx) * mask

        cross = Ex * Hy - Ey * Hx

        return np.trapz(
            np.trapz(cross, np.real(mode1.x)), np.real(mode1.y)
        )  # /np.trapz(np.trapz(cross, mode1.x), mode1.y) ### HEY

    def inner_product(self, mode2):
        """Takes the inner product between self and the provided Mode

        Parameters
        ----------
        mode2 : Mode
            second eigenmode in the operation

        Returns
        -------
        number
            the inner product between the modes
        """

        return self._inner_product(self, mode2)  # / self._inner_product(self, self)

    def check_spurious(self, threshold_power=0.05, threshold_neff=0.9):
        """Takes in a mode and determine whether the mode is likely spurious based on the ratio of confined to not confined power

        Parameters
        ----------
        threshold_power : float
            threshold of power percentage of Pz in the core to total
        threshold_neff : float
            threshold of real to abs neff percentage

        Returns 
        -------
        boolean
            True if likely spurious 
        """

        power_bool = self.get_confined_power() < threshold_power
        neff_bool = (np.real(self.neff) / np.abs(self.neff)) < threshold_neff
        return power_bool or neff_bool

    def get_confined_power(self, num_pixels=None):
        """Takes in a mode and returns the percentage of power confined in the core

        Parameters
        ----------
        num_pixels : int
            number of pixels outside of the core to expand the mask to capture power just outside the core

        Returns
        -------
        float
            Percentage of confined power
        """

        # Increase core by 5% to capture slight leaks
        if num_pixels is None:
            num_pixels = int(len(self.x) * 0.05)

        mask = np.where(self.n > np.mean(self.n), 1, 0)
        kernel = np.ones((num_pixels + 1, num_pixels + 1))
        mask = convolve2d(mask, kernel, "same")
        mask = np.where(mask > 0, 1, 0)
        ratio = self._inner_product(self, self, mask=mask) / self._inner_product(self, self, mask=None)
        return ratio

    def __str__(self):
        return "Mode Object with effective index of " + str(self.neff)

    def __mul__(self, other):
        if isinstance(other, Mode):
            self.Hx *= self.Hx
            self.Hy *= self.Hy
            self.Hz *= self.Hz
            self.Ex *= self.Ex
            self.Ey *= self.Ey
            self.Ez *= self.Ez
        else:
            self.Hx *= other
            self.Hy *= other
            self.Hz *= other
            self.Ex *= other
            self.Ey *= other
            self.Ez *= other

            return self

    def __add__(self, other):
        if isinstance(other, Mode):
            self.Hx += self.Hx
            self.Hy += self.Hy
            self.Hz += self.Hz
            self.Ex += self.Ex
            self.Ey += self.Ey
            self.Ez += self.Ez
        else:
            self.Hx += other
            self.Hy += other
            self.Hz += other
            self.Ex += other
            self.Ey += other
            self.Ez += other

        return self

    def __truediv__(self, other):

        if isinstance(other, Mode):
            self.Hx /= self.Hx
            self.Hy /= self.Hy
            self.Hz /= self.Hz
            self.Ex /= self.Ex
            self.Ey /= self.Ey
            self.Ez /= self.Ez
        else:
            self.Hx /= other
            self.Hy /= other
            self.Hz /= other
            self.Ex /= other
            self.Ey /= other
            self.Ez /= other

        return self

    def __sub__(self, other):

        if isinstance(other, Mode):
            self.Hx -= self.Hx
            self.Hy -= self.Hy
            self.Hz -= self.Hz
            self.Ex -= self.Ex
            self.Ey -= self.Ey
            self.Ez -= self.Ez
        else:
            self.Hx -= other
            self.Hy -= other
            self.Hz -= other
            self.Ex -= other
            self.Ey -= other
            self.Ez -= other

        return self

    def normalize(self):
        """Normalizes the Mode to power 1."""
        self.zero_phase()
        factor = self.inner_product(self)
        self /= np.sqrt(factor)
        return

    def zero_phase(self):
        """Changes the phase such that the z components are all imaginary and the xy components are all real."""

        index = int(self.Hy.shape[0] / 2)
        phase = np.angle(np.array(self.Hy))[index][index]
        self *= np.exp(-1j * phase)
        if (np.sum(np.real(self.Hy))) < 0:
            self *= -1

    def get_fields(self):
        """Returns an array [self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez]."""

        return [self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez]

    def get_H(self):
        """Returns an array [self.Hx, self.Hy, self.Hz]."""

        return [self.Hx, self.Hy, self.Hz]

    def get_E(self):
        """Returns an array [self.Ex, self.Ey, self.Ez]."""

        return [self.Ex, self.Ey, self.Ez]

    def get_neff(self):
        """Returns the effective index as a complex number."""

        return self.neff

    def get_wavelength(self):
        """Returns the wavelength."""

        return self.wl

    def save(self, path=None):
        """Serializes the mode into a pickle file

        Parameters
        ----------
        path : string
            The path (including name) to save the file.
        """

        if path:
            pickle.dump(self, open(path, "wb+"))
        else:
            pickle.dump(self, open("./ModeObject_" + str(random.random()) + ".pk", "wb+"))

    def plot_material(self):
        """Plots the index of refraction profile"""
        n = self.n

        plt.imshow(np.sqrt(np.real(n)).T, extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6])
        plt.colorbar()
        plt.title("Index of Refraction")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")

    def compute_other_fields(self):
        """Given the Hx and Hy fields, maxwell's curl relations can be used to calculate the remaining field; adapted from the EMpy"""

        from scipy.sparse import coo_matrix

        if self.n is None:
            self.epsfunc = tools.get_epsfunc(
                self.width,
                self.thickness,
                2.5e-6,
                2.5e-6,
                tools.Si(self.wl * 1e6),
                tools.SiO2(self.wl * 1e6),
                compute=True,
            )
        else:
            self.epsfunc = lambda x, y: self.n[: len(x), : len(y)]

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
            Dx = neff * EMpy_gpu.utils.centered2d(Hy) + (Hz[:-1, 1:] + Hz[1:, 1:] - Hz[:-1, :-1] - Hz[1:, :-1]) / (
                2j * k * v
            )
            Dy = -neff * EMpy_gpu.utils.centered2d(Hx) - (Hz[1:, :-1] + Hz[1:, 1:] - Hz[:-1, 1:] - Hz[:-1, :-1]) / (
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

        # self.Hx = (Hx[1:, 1:] + Hx[1:, :-1] + Hx[:-1, 1:] + Hx[:-1, :-1]) / 4.0 + 0j
        # self.Hy = (Hy[1:, 1:] + Hy[1:, :-1] + Hy[:-1, 1:] + Hy[:-1, :-1]) / 4.0 + 0j
        # self.Hz = (Hzs[0][1:, 1:] + Hzs[0][1:, :-1] + Hzs[0][:-1, 1:] + Hzs[0][:-1, :-1]) / 4.0 + 0j
        self.Hx = Hxs[0]
        self.Hy = Hys[0]
        self.Hz = Hzs[0]
        self.Ex = Exs[0]
        self.Ey = Eys[0]
        self.Ez = Ezs[0]
        x_ = (self.x[1:] + self.x[:-1]) / 2.0
        y_ = (self.y[1:] + self.y[:-1]) / 2.0
        self.Ex = tools.interp(self.x, self.y, x_, y_, Exs[0], False)
        self.Ey = tools.interp(self.x, self.y, x_, y_, Eys[0], False)
        self.Ez = tools.interp(self.x, self.y, x_, y_, Ezs[0], False)
        self.x = self.x - self.x[int(len(self.x) / 2)]
        self.y = self.y - self.y[int(len(self.y) / 2)]
        self.H = np.sqrt(np.abs(self.Hx) ** 2 + np.abs(self.Hy) ** 2 + np.abs(self.Hz) ** 2)
        self.E = np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2)
        eps_func = tools.get_epsfunc(
            self.width, self.thickness, 2.5e-6, 2.5e-6, tools.Si(self.wl * 1e6), tools.SiO2(self.wl * 1e6)
        )
        self.n = eps_func(self.x, self.y)

        # self.Hz = np.zeros((self.Hx.shape[0],self.Hx.shape[1]),dtype="complex")
        # self.Ex = np.zeros((self.Hx.shape[0],self.Hx.shape[1]),dtype="complex")
        # self.Ey = np.zeros((self.Hx.shape[0],self.Hx.shape[1]),dtype="complex")
        # self.Ez = np.zeros((self.Hx.shape[0],self.Hx.shape[1]),dtype="complex")

    def _get_eps(self, xc, yc):
        """Used by compute_other_fields and adapted from the EMpy library"""
        tmp = self.epsfunc(xc, yc)

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
