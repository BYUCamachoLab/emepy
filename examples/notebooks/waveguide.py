from emepy.eme import EME, Layer
from emepy.monitors import Monitor
from emepy.ann import ANN, MSNeuralNetwork
import numpy as np
from matplotlib import pyplot as plt

wavelength = 1.55
width = 0.4
thickness = 0.22
num_modes = 1
length = wavelength 

eme = EME()
ann = ANN()

mode_solvers = [
    MSNeuralNetwork(
        ann,
        wavelength * 1e-6,
        width * 1e-6,
        thickness * 1e-6,
    )  for i in range(5)
]

mode_solvers[0].solve()
plt.figure()
mode_solvers[0].get_mode().plot()
plt.show()
quit()

for i in range(5):
    eme.add_layer(Layer(mode_solvers[i], num_modes, wavelength * 1e-6, length * 1e-6))

monitor = eme.add_monitor(axes="xz")

eme.propagate() 

monitor.visualize()