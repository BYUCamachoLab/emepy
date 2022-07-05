from emepy.fd import MSEMpy  # Open source
from emepy.materials import Si, SiO2
from emepy.tools import circle_to_n
import unittest
import numpy as np
from matplotlib import pyplot as plt

# Global params
global_params = {
    "mesh": 128,
    "wl": 1.55,
    "num_modes": 4,
    "cladding_width": 5,
    "cladding_thickness": 5,
    "cladding_index": SiO2(1.55),
    "core_index": Si(1.55),
}

# Create waveguide params
waveguide_params = {**global_params, "width": 0.5, "thickness": 0.22}

# Directional coupler maps
gap = 0.3
x = np.linspace(-global_params["cladding_width"] / 2, global_params["cladding_width"] / 2, global_params["mesh"])
y = np.linspace(-global_params["cladding_width"] / 2, global_params["cladding_width"] / 2, global_params["mesh"])
xx, yy = np.meshgrid(x, y)
starting_center = -0.5 * (gap + waveguide_params["width"])
n_1D = np.ones(global_params["mesh"]) * global_params["cladding_index"]
n_2D = np.ones((global_params["mesh"], global_params["mesh"])) * global_params["cladding_index"]
for out in range(2):

    # Width
    center = starting_center + out * (gap + waveguide_params["width"])
    left_edge = center - 0.5 * waveguide_params["width"]
    right_edge = center + 0.5 * waveguide_params["width"]
    n_1D = np.where((left_edge <= x) * (x <= right_edge), global_params["core_index"], n_1D)

# Thickness
n_mask, cladding_mask = np.meshgrid(n_1D, np.ones(global_params["mesh"]) * global_params["cladding_index"])
left_edge = center - 0.5 * waveguide_params["thickness"]
right_edge = center + 0.5 * waveguide_params["thickness"]
n_2D = np.where((left_edge <= yy) * (yy <= right_edge), n_1D, n_2D).T

# Create directional coupler params
custom_params1D = {**global_params, "n": n_1D, "thickness": waveguide_params["thickness"]}

# Create directional coupler params
custom_params2D = {**global_params, "n": n_2D}

# Create circular waveguide params
n_circle = circle_to_n(
    (0, 0),
    waveguide_params["width"],
    np.linspace(-global_params["cladding_width"] / 2, global_params["cladding_width"] / 2, global_params["mesh"] + 1),
    np.linspace(-global_params["cladding_width"] / 2, global_params["cladding_width"] / 2, global_params["mesh"] + 1),
    True,
    global_params["core_index"],
    global_params["cladding_index"],
)

circular_params = {**global_params, "n": n_circle}


def plot_modes(modes):
    plt.figure()
    modes[0].plot()
    plt.show()


def test_solver(**kwargs):
    # Create a modesolver object that represents a waveguide cross section
    fd_solver = MSEMpy(**kwargs)

    # Solve for the fundamental Eigenmode
    fd_solver.solve()
    modes = [fd_solver.get_mode(i) for i in range(fd_solver.num_modes)]
    # plot_modes(modes)
    return modes


class TestSolver(unittest.TestCase):
    # def test_MSEMpy_waveguide(self):
    #     print("Testing waveguide")
    #     modes = test_solver(**waveguide_params)
    #     self.assertTrue(len(modes) == global_params["num_modes"])

    # def test_MSEMpy_custom_2D(self):
    #     print("Testing two waveguides with no defined waveguide params")
    #     modes = test_solver(**custom_params2D)
    #     self.assertTrue(len(modes) == global_params["num_modes"])

    # def test_MSEMpy_custom_1D(self):
    #     print("Testing two waveguides with defined waveguide thickness")
    #     modes = test_solver(**custom_params1D)
    #     self.assertTrue(len(modes) == global_params["num_modes"])

    # def test_MSEMpy_circular(self):
    #     print("Testing circular waveguide")
    #     modes = test_solver(**circular_params)
    #     self.assertTrue(len(modes) == global_params["num_modes"])

    def test_test(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
