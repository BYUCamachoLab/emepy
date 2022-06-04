from matplotlib import pyplot as plt
import numpy as np
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
from emepy.materials import Si, SiO2


# Define materials
wavelength = 1.55
matSi = Si(wavelength)
matSiO2 = SiO2(wavelength)

# Define geometry parameters
rect_params = EMpyGeometryParameters(
    wavelength=wavelength, cladding_width=5, cladding_thickness=2.5, core_index=matSi, cladding_index=matSiO2, mesh=150
)

# Define input waveguide
input_waveguide = Waveguide(rect_params, width=1.0, thickness=0.22, length=0.5, center=(0, 0), num_modes=1)


# Define output waveguide
output_waveguide = Waveguide(rect_params, width=1.75, thickness=0.22, length=0.5, center=(0, 0), num_modes=1)

# Define waveguide taper
taper = DynamicRect2D(
    params=rect_params,
    input_width=input_waveguide.width,
    output_width=output_waveguide.width,
    length=2,
    num_modes=1,
    num_params=100,
    mesh_z=5,
)

# Define the EME model
eme = EME(quiet=True, parallel=False, mesh_z=100)
opt = Optimization(eme, [input_waveguide, taper, output_waveguide])

# Get the EMEPy eps model
plt.figure()
opt.draw()
plt.show()


# Get the analytic model
