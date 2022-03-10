import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
import nlopt

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55e-6, cladding_width=4e-6, cladding_thickness=2.5e-6, core_index=3.4, cladding_index=1.4, mesh=60
)

# Create an input waveguide
input_waveguide = Waveguide(rect_params, 1.0e-6, 0.22e-6, 0.5e-6, center=(0, 0), num_modes=3)

# Create an output waveguide
output_waveguide = Waveguide(rect_params, width=1.5e-6, thickness=0.22e-6, length=0.5e-6, center=(0, 0), num_modes=5)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=input_waveguide.width,
    length=2e-6,
    num_modes=3,
    num_params=10,
    symmetry=True,
    subpixel=True,
    mesh_z=5,
    input_width=input_waveguide.width,
    output_width=output_waveguide.width,
)

# Create the EME and Optimization
eme = EME()
optimizer = Optimization(eme, [input_waveguide, dynamic_rect, output_waveguide])

# Make the initial design a linear taper
design_x, design_z = optimizer.get_design_readable()
linear_taper = np.linspace(input_waveguide.width, output_waveguide.width, len(design_x)) / 2.0
design_x[:] = linear_taper[:]
optimizer.set_design_readable(design_x, None, design_z)

# plt.figure()
# optimizer.draw()
# # n = dynamic_rect.get_n(np.linspace(-1.5e-6, 1.5e-6, 100), np.linspace(0, 2e-6, 100))
# # plt.imshow(n, cmap="Greys", extent=[0, 2e-6, -1.5e-6, 1.5e-6], interpolation="none")
# plt.show()
# quit()

# Evaluation history
evaluation_history = []


# Create gradient assigning function
def f(design, grad):
    f0, dJ_du = optimizer.optimize(design)
    if grad.size > 0:
        grad[:] = np.squeeze(dJ_du)
    evaluation_history.append(np.real(f0))
    return np.real(np.real(f0))


# Create an nlopt optimizater
algorithm = nlopt.LD_MMA
design = optimizer.get_design()
n = len(design)
maxeval = 10

solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(0)
solver.set_upper_bounds(1e10)
solver.set_max_objective(f)
solver.set_maxeval(maxeval)
x = solver.optimize(design)

plt.figure()
plt.plot(evaluation_history)
plt.show()
