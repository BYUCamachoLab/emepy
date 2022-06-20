import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
import nlopt
from emepy.materials import Si, SiO2
import emepy

Si = Si(1.55)
SiO2 = SiO2(1.55)

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55, cladding_width=4, cladding_thickness=2.5, core_index=Si, cladding_index=SiO2, mesh=80
)

# Create an input waveguide
input_waveguide = Waveguide(rect_params, 1.0, 0.22, 0.5, center=(0, 0), num_modes=4)

# Create an output waveguide
output_waveguide = Waveguide(rect_params, width=1.7, thickness=0.22, length=0.5, center=(0, 0), num_modes=4)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=input_waveguide.width,
    length=2,
    num_modes=4,
    num_params=10,
    symmetry=True,
    subpixel=True,
    mesh_z=5,
    input_width=input_waveguide.width,
    output_width=output_waveguide.width,
)

# Create the EME and Optimization
eme = EME(quiet=True)
optimizer = Optimization(eme, [input_waveguide, dynamic_rect, output_waveguide], mesh_z=100)

# Make the initial design a linear taper
design_x, design_z = optimizer.get_design_readable()
linear_taper = np.linspace(input_waveguide.width, output_waveguide.width, len(design_x)) / 2.0
design_x[:] = linear_taper[:]
optimizer.set_design_readable(design_x, None, design_z)


# source = emepy.Source(z=0.25, mode_coeffs=[1], k=1)  # Hard coded
# monitor = eme.add_monitor(mesh_z=5, sources=[source])
# eme.propagate()
# plt.figure()
# monitor.visualize()
# plt.show()
# quit()

# plt.figure()
# optimizer.draw()
# # n = dynamic_rect.get_n(np.linspace(-1.5, 1.5, 100), np.linspace(0, 2, 100))
# # plt.imshow(n, cmap="Greys", extent=[0, 2, -1.5, 1.5], interpolation="none")
# plt.show()
# quit()

# Evaluation history
evaluation_history = []

# plt.figure()
# eme.draw()
# plt.show()
# quit()


# Create gradient assigning function
def f(design, grad):
    f0, dJ_du, monitor = optimizer.optimize(design)
    if eme.am_master():
        print("Run {} finished with value {}".format(len(evaluation_history), f0))
    if grad.size > 0:
        grad[:] = np.squeeze(dJ_du)
    evaluation_history.append(np.real(f0))

    # plt.figure()
    # monitor.visualize(component="n")
    # plt.show()

    grid_z = np.linspace(dynamic_rect.grid_z[0], dynamic_rect.grid_z[-1], 300)
    n = dynamic_rect.get_n(dynamic_rect.grid_x, grid_z)
    plt.figure()
    plt.imshow(
        n[::-1],
        cmap="Greys",
        extent=[grid_z[0], grid_z[-1], dynamic_rect.grid_x[0], dynamic_rect.grid_x[-1]],
        interpolation="none",
    )
    if eme.am_master():
        plt.savefig("test_images/design{}.jpg".format(len(evaluation_history)))

    return np.real(np.real(f0))


# Create an nlopt optimizater
algorithm = nlopt.LD_MMA
design = optimizer.get_design()
n = len(design)
maxeval = 1

solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(0)
solver.set_upper_bounds(1e10)
solver.set_max_objective(f)
solver.set_maxeval(maxeval)
x = solver.optimize(design)

plt.figure()
plt.plot(evaluation_history)
plt.xlabel("iteration")
plt.ylabel("power")
if eme.am_master():
    plt.savefig("test_images/evaluation_history.jpg")
