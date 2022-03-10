import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
from nlopt import opt

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55e-6,
    cladding_width=2.5e-6,
    cladding_thickness=2.5e-6,
    core_index=3.4,
    cladding_index=1.4,
    mesh=50,
)

# Create an input waveguide
input_waveguide = Waveguide(rect_params, 0.5e-6, 0.22e-6, 0.5e-6, center=(0, 0))

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=0.5e-6,
    length=2e-6,
    num_modes=1,
    num_params=300,
    symmetry=False,
    subpixel=True,
    mesh_z=10,
)

# Make the initial design a linear taper

# Create an output waveguide
output_waveguide = Waveguide(rect_params, 0.5e-6, 0.22e-6, 0.5e-6, center=(0, 0))

# Create the EME and Optimization
eme = EME()
opt = Optimization(eme, [input_waveguide, dynamic_rect, output_waveguide])

# Create an nlopt optimizater