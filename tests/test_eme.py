from emepy import Layer, EME, Source, MSEMpy
import numpy as np
from matplotlib import pyplot as plt


def test_eme():
    num_periods = 15  # Number of Periods for Bragg Grating
    length = 0.159e-6  # Length of each segment of BG, Period = Length * 2
    wavelength = 1.55e-6  # Wavelength
    num_modes = 1  # Number of Modes
    mesh = 100  # Number of mesh points
    width1 = 0.46e-6  # Width of first core block
    width2 = 0.54e-6  # Width of second core block
    thickness = 0.22e-6  # Thicnkess of the core
    modesolver = MSEMpy  # Which modesolver to use

    eme = EME(num_periods=num_periods, quiet=True)

    mode_solver1 = modesolver(wavelength, width1, thickness, mesh=mesh)  # First half of bragg grating

    mode_solver2 = modesolver(wavelength, width2, thickness, mesh=mesh)  # Second half of bragg grating

    eme.add_layer(Layer(mode_solver1, num_modes, wavelength, length))  # First half of bragg grating
    eme.add_layer(Layer(mode_solver2, num_modes, wavelength, length))  # Second half of bragg grating

    positive_source = Source(1.2e-6, [1], -1)
    negative_source = Source(3.2e-6, [1], 1)
    monitor = eme.add_monitor(axes="xz", mesh_z=200, sources=[positive_source, negative_source])

    eme.propagate(left_coeffs=[], right_coeffs=[])  # propagate at given wavelength

    assert monitor.get_array()


test_eme()
