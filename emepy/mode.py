import numpy as np
from matplotlib import pyplot as plt
import pickle
import random
import EMpy_gpu
from emepy import tools
from scipy.signal import convolve2d
from copy import copy


class EigenMode(object):
    """Virtual class representing an eigenmode"""

    def __init__(self):
        return NotImplementedError()

    def inner_product(self):
        return NotImplementedError()

    def plot(self):
        return NotImplementedError()

    def check_spurious(self):
        return NotImplementedError()

    def get_confined_power(self):
        return NotImplementedError()

    def normalize(self):
        return NotImplementedError()

    def zero_phase(self):
        return NotImplementedError()

    def plot_material(self):
        return NotImplementedError()

    def compute_other_fields(self):
        return NotImplementedError()

    def save(self, path=("./ModeObject_" + str(random.random()) + ".pk")):
        """Serializes the mode into a pickle file"""
        pickle.dump(self, open(path, "wb+"))

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

    def get_Hx(self):
        return self.Hx

    def get_Hy(self):
        return self.Hy

    def get_Hz(self):
        return self.Hz

    def get_Ex(self):
        return self.Ex

    def get_Ey(self):
        return self.Ey

    def get_Ez(self):
        return self.Ez

    def get_wavelength(self):
        """Returns the wavelength."""

        return self.wl

    def __str__(self):
        return "Mode Object with effective index of " + str(self.neff)

    def change_fields(self, start, other, func):
        for comp in ["Hx", "Hy", "Hz", "Ex", "Ey", "Ez"]:
            cur = getattr(start, comp)
            new = getattr(other, comp) if isinstance(other, EigenMode) else other
            setattr(start, comp, func(cur, new))
        return start

    def __mul__(self, other):
        return self.change_fields(copy(self), other, lambda a, b: a * b)

    def __add__(self, other):
        return self.change_fields(copy(self), other, lambda a, b: a + b)

    def __truediv__(self, other):
        return self.change_fields(copy(self), other, lambda a, b: a / b)

    def __sub__(self, other):
        return self.change_fields(copy(self), other, lambda a, b: a - b)

    def __imul__(self, other):
        return self.change_fields(self, other, lambda a, b: a * b)

    def __iadd__(self, other):
        return self.change_fields(self, other, lambda a, b: a + b)

    def __itruediv__(self, other):
        return self.change_fields(self, other, lambda a, b: a / b)

    def __isub__(self, other):
        return self.change_fields(self, other, lambda a, b: a - b)


# class Mode1D(EigenMode):
#     """Object that holds the field profiles and effective index for a 1D eigenmode"""


class Mode(EigenMode):
    """Object that holds the field profiles and effective index for a 2D eigenmode"""

    def __init__(self, x, y, wl, neff, Hx, Hy, Hz, Ex, Ey, Ez, n):
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
        self.Hx = Hx if not (Hx is None) else np.zeros((10, 10))
        self.Hy = Hy if not (Hy is None) else self.Hx * 0
        self.Hz = Hz if not (Hz is None) else self.Hx * 0
        self.Ex = Ex if not (Ex is None) else self.Hx * 0
        self.Ey = Ey if not (Ey is None) else self.Hx * 0
        self.Ez = Ez if not (Ez is None) else self.Hx * 0
        self.n = n
        self.H = np.sqrt(np.abs(self.Hx) ** 2 + np.abs(self.Hy) ** 2 + np.abs(self.Hz) ** 2)
        self.E = np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2)

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

        # Parse operation
        op_name = operation.__name__ if hasattr(operation, "__name__") else str(operation)
        if operation == "Imaginary":
            operation = lambda a: np.imag(a)
        elif operation == "Abs":
            operation = lambda a: np.abs(a)
        elif operation == "Abs^2":
            operation = lambda a: np.abs(a) ** 2
        elif operation == "Real":
            operation = lambda a: np.real(a)
        try:
            t = self.change_fields(copy(temp), copy(temp), lambda a, b: operation(b))
            Hx, Hy, Hz, Ex, Ey, Ez = [t.Hx, t.Hy, t.Hz, t.Ex, t.Ey, t.Ez]
        except:
            raise Exception(
                "Invalid operation provided. Please choose from ('Imaginary', 'Abs', 'Abs^2', 'Real') or provide a function"
            )

        # Plot fields
        fields = ["Hx", "Hy", "Hz", "Ex", "Ey", "Ez"]
        for i, field in enumerate([Hx, Hy, Hz, Ex, Ey, Ez]):
            plt.subplot(2, 3, i + 1, adjustable="box", aspect=field.shape[0] / field.shape[1])
            v = max(abs(field.min()), abs(field.max()))
            plt.imshow(
                field.T,
                cmap="RdBu",
                vmin=-v,
                vmax=v,
                extent=[temp.x[0] * 1e6, temp.x[-1] * 1e6, temp.y[0] * 1e6, temp.y[-1] * 1e6],
            )
            plt.title("{}({})".format(op_name, fields[i]))
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
        return (power_bool or neff_bool)

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

    def plot_material(self):
        """Plots the index of refraction profile"""
        n = self.n

        plt.imshow(np.sqrt(np.real(n)).T, extent=[self.x[0] * 1e6, self.x[-1] * 1e6, self.y[0] * 1e6, self.y[-1] * 1e6])
        plt.colorbar()
        plt.title("Index of Refraction")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")

    def compute_other_fields(self, epsfunc_1, epsfunc_2, boundary="0000"):
        """Given the Hx and Hy fields, maxwell's curl relations can be used to calculate the remaining field; adapted from the EMpy"""

        self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez = tools.compute_other_fields_2D(
            self.neff, self.Hx, self.Hy, self.wl, self.x, self.y, boundary, epsfunc_1
        )
        x_ = (self.x[1:] + self.x[:-1]) / 2.0
        y_ = (self.y[1:] + self.y[:-1]) / 2.0
        self.Ex = tools.interp(self.x, self.y, x_, y_, self.Ex, False)
        self.Ey = tools.interp(self.x, self.y, x_, y_, self.Ey, False)
        self.Ez = tools.interp(self.x, self.y, x_, y_, self.Ez, False)
        self.x = self.x - self.x[int(len(self.x) / 2)]
        self.y = self.y - self.y[int(len(self.y) / 2)]
        self.H = np.sqrt(np.abs(self.Hx) ** 2 + np.abs(self.Hy) ** 2 + np.abs(self.Hz) ** 2)
        self.E = np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2)
        self.n = epsfunc_2(self.x, self.y)

