from emepy import Monitor, EME, Layer, MSEMpy
import numpy as np
from matplotlib import pyplot as plt
import unittest
import os

# Params
plot = False
num_periods = 5  # Number of Periods for Bragg Grating
length = 0.159  # Length of each segment of BG, Period = Length * 2
wavelength = 1.55  # Wavelength
num_modes = 1  # Number of Modes
mesh = 128  # Number of mesh points
width1 = 0.46  # Width of first core block
width2 = 0.54  # Width of second core block
thickness = 0.22  # Thicnkess of the core
modesolver = MSEMpy  # Which modesolver to use


# Bragg grating simulation
def get_simulation():

    # ModeSolvers
    mode_solver1 = modesolver(
        # ann,
        wavelength,
        width1,
        thickness,
    )  # First half of bragg grating

    mode_solver2 = modesolver(
        # ann,
        wavelength,
        width2,
        thickness,
    )  # Second half of bragg grating

    # Create simulation
    eme = EME(num_periods=num_periods, quiet=True)
    eme.add_layer(Layer(mode_solver1, num_modes, wavelength, length))  # First half of bragg grating
    eme.add_layer(Layer(mode_solver2, num_modes, wavelength, length))  # Second half of bragg grating

    return eme


class TestMonitors(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        super(TestMonitors, self).__init__(*args, **kwargs)

    def analyze_monitor(self, monitor: Monitor, axes=None, key=""):

        # Plot index
        plt.figure()
        im1 = monitor.visualize(component="n", axes=axes)
        plt.colorbar()
        plt.show() if plot else plt.savefig(os.path.join(self.data_dir, "test_monitor_n_{}.png".format(key)))
        self.assertTrue(im1 is not None)

        # Plot field
        plt.figure()
        im2 = monitor.visualize(component="Hy", axes=axes)
        plt.colorbar()
        plt.show() if plot else plt.savefig(os.path.join(self.data_dir, "test_monitor_Hy_{}.png".format(key)))
        self.assertTrue(im2 is not None)

    def test_xy(self):

        # Get eme
        eme = get_simulation()

        # Create monitors
        axes = "xy"
        monitor_xy = eme.add_monitor(axes=axes, location=0.2)

        # Run simulation
        eme.propagate()

        # Analyze monitors
        self.analyze_monitor(monitor_xy, axes=axes, key="xy")

    def test_xz(self):

        # Get eme
        eme = get_simulation()

        # Create monitors
        axes = "xz"
        monitor_xz = eme.add_monitor(axes=axes)

        # Run simulation
        eme.propagate()

        # Analyze monitors
        self.analyze_monitor(monitor_xz, axes=axes, key="xz")

    def test_yz(self):

        # Get eme
        eme = get_simulation()

        # Create monitors
        axes = "yz"
        monitor_yz = eme.add_monitor(axes=axes)

        # Run simulation
        eme.propagate()

        # Analyze monitors
        self.analyze_monitor(monitor_yz, axes=axes, key="yz")

    def test_xyz(self):
        # Get eme
        eme = get_simulation()

        # Create monitors
        axes = "xyz"
        monitor_custom_z = eme.add_monitor(axes=axes, z_range=(1 * 0.159, 5 * 0.159))

        # Run simulation
        eme.propagate()

        # Analyze monitors
        self.analyze_monitor(monitor_custom_z, axes="xz", key="custom_z")

        # Analyze 3D field
        x, y, z, f = monitor_custom_z.get_array(component="Hy", axes="xyz", z_range=(2 * 0.159, 4 * 0.159))
        self.assertTrue(f is not None)

    def test_custom_z_range(self):

        # Get eme
        eme = get_simulation()

        # Create monitors
        axes = "xz"
        monitor_custom_z = eme.add_monitor(axes=axes, z_range=(1 * 0.159, 5 * 0.159))

        # Run simulation
        eme.propagate()

        # Analyze monitors
        self.analyze_monitor(monitor_custom_z, axes=axes, key="custom_z")

        # Analyze custom z range
        plt.figure()
        im = monitor_custom_z.visualize(component="Hy", axes="xz", z_range=(2 * 0.159, 4 * 0.159))
        plt.colorbar()
        plt.show() if plot else plt.savefig(os.path.join(self.data_dir, "test_monitor_custom_z.png"))
        self.assertTrue(im is not None)


if __name__ == "__main__":
    unittest.main()
