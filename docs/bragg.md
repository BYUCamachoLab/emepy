# Example: Bragg Grating

This example shows the capabilities of the EME object by designing a Bragg Grating. We begin by importing the similar classes and libraries as the [last example](index.md). The script for this example can be found [here](https://github.com/BYUCamachoLab/emepy/blob/master/examples/taper.py)

    import emepy
    from emepy.fd import MSEMpy
    from emepy.eme import Layer, EME
    from emepy.mode import Mode

    import numpy as np
    from matplotlib import pyplot as plt

Next we'll define parameters for our device. In this example, we will sweep over a set of wavelengths and visualize the transfer function.

    num_periods = 10  # Number of Periods for Bragg Grating
    length = 0.16  # Length of each segment of BG, Period = Length * 2
    num_wavelengths = 50  # Number of wavelengths to sweep
    wl_lower = 1.5  # Lower wavelength bound
    wl_upper = 1.6  # Upper wavelength bound
    num_modes = 1  # Number of Modes
    mesh = 256
    modesolver = MSEMpy

This example utilizes the ability to calculate transmission values from the resulting s-matrix. Because EMEpy operates only in the frequency domain, we will run a simulation for each wavelength we care about. Let's begin by creating an array to hold our transmission values.

    t = []

We will now sweep over our set of wavelengths and create modesolvers and layers for both steps of the bragg grating period.

    for wavelength in np.linspace(wl_lower, wl_upper, num_wavelengths):

        mode_solver1 = modesolver(
            wavelength * 1e-6,
            0.46e-6,
            0.22e-6,
            mesh=mesh,
            num_modes=num_modes,
        )  # First half of bragg grating

        mode_solver2 = modesolver(
            wavelength * 1e-6,
            0.54e-6,
            0.22e-6,
            mesh=mesh,
            num_modes=num_modes,
        )  # Second half of bragg grating

        layer1 = Layer(mode_solver1, num_modes, wavelength * 1e-6, length * 1e-6)  # First half of bragg grating
        layer2 = Layer(mode_solver2, num_modes, wavelength * 1e-6, length * 1e-6)  # Second half of bragg grating

Still in our loop, we will create a EME object and assign a number of periods. The solver will utilize this by only solving for the modes of one period, and will cascade the resulting s-parameters together a number of times matching the period count.

        eme = EME([layer1, layer2], num_periods)

Let's draw our structure just once and make sure we designed it correctly.

        if wavelength == wl_lower:
            eme.draw()

Finally, let's propagate our results and grab the absolute value of the transmission value and append to our list.

        eme.propagate()  # propagate at given wavelength

        t.append(np.abs((eme.s_parameters())[0, 0, num_modes]) ** 2)  # Grab the transmission coefficient

Once the solver finishes for each wavelength of concern, we can plot our transfer function.

    plt.plot(np.linspace(wl_lower, wl_upper, num_wavelengths), 20 * np.log(t))
    plt.title("BG Bode Plot Periods=" + str(num_periods))
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("dB")
    plt.show()
