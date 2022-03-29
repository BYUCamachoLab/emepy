import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
import nlopt
from emepy.materials import Si, SiO2
import emepy

# Materials
Si = Si(1.55)
SiO2 = SiO2(1.55)

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55, cladding_width=4.5, cladding_thickness=3, core_index=Si, cladding_index=SiO2, mesh=120
)

# Create an input waveguide
input_waveguide = Waveguide(rect_params, 1.0, 0.22, 0.5, center=(0, 0), num_modes=10)

# Create an output waveguide
output_waveguide = Waveguide(rect_params, width=2.5, thickness=0.22, length=0.5, center=(0, 0), num_modes=10)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=input_waveguide.width,
    length=2,
    num_modes=10,
    num_params=30,
    symmetry=True,
    subpixel=True,
    mesh_z=12,
    input_width=input_waveguide.width,
    output_width=output_waveguide.width,
)

# Create the EME and Optimization
eme = EME(quiet=True, parallel=False, mesh_z=150)
optimizer = Optimization(eme, [input_waveguide, dynamic_rect, output_waveguide], mesh_z=150)

# Make the initial design a linear taper
design_x, design_z = optimizer.get_design_readable()
linear_taper = np.linspace(input_waveguide.width, output_waveguide.width, len(design_x)) / 2.0
design_x[:] = linear_taper[:]
optimizer.set_design_readable(design_x, None, design_z)

# Evaluation history
evaluation_history = []

# Create gradient assigning function
def f(design, grad):

    # Calculate gradients
    f0, dJ_du, monitor = optimizer.optimize(design)
    design_x, design_z = optimizer.get_design_readable()

    # Assign gradients
    if eme.am_master():
        print("Run {} finished with value {}".format(len(evaluation_history), f0))
    if grad.size > 0:
        grad[:] = np.squeeze(dJ_du)
    evaluation_history.append(np.real(f0))

    # Plot the eme field
    plt.figure()
    monitor.visualize(axes="xz",component="n")
    if eme.am_master():
        plt.savefig("test_images/i{}eme.jpg".format(len(evaluation_history)))

    # Plot the field
    plt.figure()
    monitor.visualize(axes="xz",component="Hy")
    if eme.am_master():
        plt.savefig("test_images/i{}field.jpg".format(len(evaluation_history)))

    # Plot the design
    grid_z = np.linspace(dynamic_rect.grid_z[0], dynamic_rect.grid_z[-1], 300)
    n = dynamic_rect.get_n(dynamic_rect.grid_x, None, grid_z)
    plt.figure()
    plt.imshow(
        n[::-1],
        cmap="Greys",
        extent=[grid_z[0], grid_z[-1], dynamic_rect.grid_x[0], dynamic_rect.grid_x[-1]],
        interpolation="none",
    )
    if eme.am_master():
        plt.savefig("test_images/i{}design.jpg".format(len(evaluation_history)))

    # Get the gradients
    vertices_gradients = np.array(
        [[z, x] for z, x in zip(dJ_du[1::2], dJ_du[::2])]
    )
    vertices_origins = np.array(
        [[z + input_waveguide.length, x] for z, x in zip(design_z, design_x)]
    ).T

    # Plot the gradients
    plt.figure()
    monitor.visualize(axes="xz", component="n")
    plt.quiver(
        *vertices_origins, vertices_gradients[:, 0], vertices_gradients[:, 1], color="r"
    )
    # plt.imshow(np.real(image), alpha=0.7, cmap="RdBu", extent=[z[0],z[-1],x[0],x[-1]])
    if eme.am_master():
        plt.savefig("test_images/i{}gradients.jpg".format(len(evaluation_history)))

    return np.real(np.real(f0))


# Create an nlopt optimizater
algorithm = nlopt.LD_LBFGS
design = optimizer.get_design()
n = len(design)
maxeval = 20

# Create limits
lower_limit, upper_limit = np.array(design),np.array(design)
lower_limit[::2] *= 0.7
lower_limit[1::2] *= 0.9
upper_limit[::2] *= 1.3
upper_limit[1::2] *= 1.1

# Define optimizer
solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(lower_limit)
solver.set_upper_bounds(upper_limit)
solver.set_max_objective(f)
solver.set_maxeval(maxeval)
x = solver.optimize(design)

# Plot the iteration history
plt.figure()
plt.plot(evaluation_history)
plt.xlabel("iteration")
plt.ylabel("power")
if eme.am_master():
    plt.savefig("test_images/evaluation_history.jpg")
