from emepy import Layer, EME, MSEMpy, ModelTools, am_master
import numpy as np
from matplotlib import pyplot as plt
import unittest
from copy import deepcopy

# Global params
quiet = True # Suppress output
modesolver = MSEMpy  # Which modesolver to use

###### Bragg grating ######

# Resolution
b_num_modes = 3  # Number of Modes
b_mesh = 150  # Number of mesh points

# Geometry
b_width1 = 0.46  # Width of first core block
b_width2 = 0.54  # Width of second core block
b_thickness = 0.22  # Thicnkess of the core
b_length = 0.159  # Length of each segment of BG, Period = Length * 2
b_wavelength = 1.55  # Wavelength

# ModeSolvers
b_mode_solver1 = modesolver(b_wavelength, b_width1, b_thickness, mesh=b_mesh, num_modes=b_num_modes)  # First half of bragg grating
b_mode_solver2 = modesolver(b_wavelength, b_width2, b_thickness, mesh=b_mesh, num_modes=b_num_modes)  # Second half of bragg grating

###### Taper ######

# Resolution
t_num_modes = 5  # Number of Modes
t_mesh = 100  # Number of mesh points
t_num_layers = 10 # Number of layers

# Geometry
t_input_width = 0.30  # Width of input
t_output_width = 1  # Width of output
t_thickness = 0.22  # Thickness
t_input_length = 0.5  # Length of input
t_output_length = 0.5  # Length of output
t_taper_length = 3 # Length of taper
t_wavelength = 1.55  # Wavelength

# Input and Output ModeSolvers
t_mode_solver_input = modesolver(t_wavelength, t_input_width, t_thickness, mesh=t_mesh, num_modes=t_num_modes)  # Input waveguide
t_mode_solver_output = modesolver(t_wavelength, t_output_width, t_thickness, mesh=t_mesh, num_modes=t_num_modes)  # Output waveguide

# Taper ModeSolvers
widths = np.linspace(t_input_width, t_output_width, t_num_layers) # Widths of each layer
t_mode_solvers_taper = [modesolver(t_wavelength, width, t_thickness, mesh=t_mesh, num_modes=t_num_modes) for width in widths] # Taper waveguides

# Bragg grating run
def bragg_grating(parallel:bool, num_periods:int):

    # Create simulation
    b_eme = EME(num_periods=num_periods, quiet=quiet, parallel=parallel)
    b_eme.add_layer(Layer(deepcopy(b_mode_solver1), b_num_modes, b_wavelength, b_length))  # First half of bragg grating
    b_eme.add_layer(Layer(deepcopy(b_mode_solver2), b_num_modes, b_wavelength, b_length))  # Second half of bragg grating

    # Propagate
    b_eme.propagate()
    power = np.abs(ModelTools.compute(b_eme.network, {"left0":1})["right0"]) ** 2

    return power

# Taper run
def taper(parallel:bool):
    
    # Create simulation
    t_eme = EME(num_periods=1, quiet=quiet, parallel=parallel)
    t_eme.add_layer(Layer(deepcopy(t_mode_solver_input), t_num_modes, t_wavelength, t_input_length))  # Input waveguide
    for mode_solver in t_mode_solvers_taper:
        t_eme.add_layer(Layer(deepcopy(mode_solver), t_num_modes, t_wavelength, t_taper_length))  # Taper waveguide
    t_eme.add_layer(Layer(deepcopy(t_mode_solver_output), t_num_modes, t_wavelength, t_output_length))  # Output waveguide

    # Propagate
    t_eme.propagate()
    power = np.abs(ModelTools.compute(t_eme.network, {"left0":1})["right0"]) ** 2

    return power


class TestEME(unittest.TestCase):

    def test_eme(self):
        # Setup
        parallel = False
        if am_master(parallel):
            print("Running EME test single period not parallel")

        # Get power from simulation
        power = bragg_grating(parallel=parallel, num_periods=1)
        
        # Assert a proper result
        self.assertTrue(0 <= power < 1.1)

    def test_eme_parallel(self):
        # Setup
        parallel = True
        if am_master(parallel):
            print("Running EME test single period with parallel")

        # Get power from simulation
        power = bragg_grating(parallel=parallel, num_periods=1)
        
        # Assert a proper result
        self.assertTrue(0 <= power <= 1.1)

    def test_eme_periodic(self):
        # Setup
        parallel = False
        if am_master(parallel):
            print("Running EME test many periods not parallel")

        # Get power from simulation
        power = bragg_grating(parallel=parallel, num_periods=50)
        
        # Assert a proper result
        self.assertTrue(0 <= power <= 1.1)

    def test_eme_periodic_parallel(self):
        # Setup
        parallel = True
        if am_master(parallel):
            print("Running EME test many periods with parallel")

        # Get power from simulation
        power = bragg_grating(parallel=parallel, num_periods=50)
        
        # Assert a proper result
        self.assertTrue(0 <= power <= 1.1)

    def test_eme_many_layers(self):
        # Setup
        parallel = False
        if am_master(parallel):
            print("Running EME test many layers not parallel")

        # Get power from simulation
        power = taper(parallel=parallel)

        # Assert a proper result
        self.assertTrue(0 <= power <= 1.1)

    def test_eme_many_layers_parallel(self):
        # Setup
        parallel = True
        if am_master(parallel):
            print("Running EME test many layers with parallel")

        # Get power from simulation
        power = taper(parallel=parallel)

        # Assert a proper result
        self.assertTrue(0 <= power <= 1.1)


if __name__ == '__main__':
  unittest.main()