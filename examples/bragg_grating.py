from emepy.fd import MSLumerical # Requires Lumerical API
from emepy.fd import MSEMpy  # Open source
from emepy.ann import ANN, MSNeuralNetwork
from emepy.eme import Layer, EME

import numpy as np
from matplotlib import pyplot as plt

num_periods = 50  # Number of Periods for Bragg Grating
length = 0.155  # Length of each segment of BG, Period = Length * 2
num_wavelengths = 30  # Number of wavelengths to sweep
wl_lower = 1.5  # Lower wavelength bound
wl_upper = 1.6  # Upper wavelength bound
num_modes = 1  # Number of Modes
mesh = 128
modesolver = MSLumerical
t = []  # Array that holds the transmission coefficients for different wavelengths


eme = EME(num_periods=num_periods)
ann = ANN()

for wavelength in np.linspace(wl_lower, wl_upper, num_wavelengths):

    eme.reset()

    mode_solver1 = MSNeuralNetwork(
        ann,
        wavelength * 1e-6,
        0.46e-6,
        0.22e-6,
    )  # First half of bragg grating

    mode_solver2 = MSNeuralNetwork(
        ann,
        wavelength * 1e-6,
        0.54e-6,
        0.22e-6,
        # neff = mode_solver2.get_mode().neff,
        # Hx = mode_solver2.get_mode().Hx,
        # Hy = mode_solver2.get_mode().Hy
    )  # Second half of bragg grating

    eme.add_layer(Layer(mode_solver1, num_modes, wavelength * 1e-6, length * 1e-6))  # First half of bragg grating
    eme.add_layer(Layer(mode_solver2, num_modes, wavelength * 1e-6, length * 1e-6))  # Second half of bragg grating

    eme.propagate()  # propagate at given wavelength

    t.append(np.abs((eme.s_parameters()))[0,0,1]**2)  # Grab the transmission coefficient
    print(t[-1])

# Plot the results
plt.plot(np.linspace(wl_lower, wl_upper, num_wavelengths), t)
plt.title("BG Bode Plot Periods=" + str(num_periods))
plt.xlabel("Wavelength (microns)")
plt.ylabel("t")
plt.savefig('plot1')
