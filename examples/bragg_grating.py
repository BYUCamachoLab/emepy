import emepy
from emepy.FD_modesolvers import ModeSolver_Lumerical  # Requires Lumerical API
from emepy.FD_modesolvers import ModeSolver_EMpy  # Open source
from emepy.eme import Layer, EMERunner, PeriodicEME
from emepy.mode import Mode

import numpy as np
from matplotlib import pyplot as plt

num_periods = 300  # Number of Periods for Bragg Grating
length = 0.16  # Length of each segment of BG, Period = Length * 2
num_wavelengths = 50  # Number of wavelengths to sweep
wl_lower = 1.5  # Lower wavelength bound
wl_upper = 1.6  # Upper wavelength bound
num_modes = 2  # Number of Modes

t = []  # Array that holds the transmission coefficients for different wavelengths

for wavelength in np.linspace(wl_lower, wl_upper, num_wavelengths):

    mode_solver1 = ModeSolver_EMpy(
        wavelength * 1e-6,
        0.46e-6,
        0.22e-6,
        mesh=500,
        num_modes=num_modes,
        # lumapi_location="/Applications/Lumerical v202.app/Contents/API/Python",
    )  # First half of bragg grating

    mode_solver2 = ModeSolver_EMpy(
        wavelength * 1e-6,
        0.54e-6,
        0.22e-6,
        mesh=500,
        num_modes=num_modes,
        # lumapi_location="/Applications/Lumerical v202.app/Contents/API/Python",
    )  # Second half of bragg grating

    for layer in range(num_periods):

        layer1 = Layer(mode_solver1, num_modes, wavelength * 1e-6, length * 1e-6)  # First half of bragg grating
        layer2 = Layer(mode_solver2, num_modes, wavelength * 1e-6, length * 1e-6)  # Second half of bragg grating

        eme = PeriodicEME(
            [layer1, layer2], num_periods
        )  # Periodic EME will save computational time for repeated geometry

    # eme.draw() # Draw the structure

    eme.propagate()  # propagate at given wavelength

    t.append(np.abs((eme.s_parameters())[0, 0, 2]) ** 2)  # Grab the transmission coefficient

# Plot the results
plt.plot(np.linspace(wl_lower, wl_upper, num_wavelengths), 20 * np.log(t))
plt.title("BG Bode Plot Periods=" + str(num_periods))
plt.xlabel("Wavelength (microns)")
plt.ylabel("dB")
plt.savefig("./p2_" + str(num_periods) + ".jpg")
