import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55,
    cladding_width=2.5,
    cladding_thickness=2.5,
    core_index=3.4,
    cladding_index=1.4,
    mesh=50,
)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=0.5,
    length=2,
    num_modes=1,
    num_params=300,
    symmetry=False,
    subpixel=True,
    mesh_z=10,
)

# Create a normal waveguide
waveguide = Waveguide(rect_params, 0.7, 0.22, 0.5, center=(0, 0))

# Create the EME and Optimization
eme = EME()
opt = Optimization(eme, [waveguide, dynamic_rect, waveguide])

# Visualize
plt.figure()
eme.draw()
plt.show()

# Get design params
design_x, design_z = opt.get_design_readable()
top_x, bottom_x = (
    np.array(design_x[: len(design_x) // 2]),
    np.array(design_x[len(design_x) // 2 :]),
)
top_z, bottom_z = (
    np.array(design_z[: len(design_z) // 2]),
    np.array(design_z[len(design_z) // 2 :]),
)

# Add a sin arc on the top
top_x += np.sin(top_z / np.max(top_z) * np.pi) * 0.3

# Update design
design_x = top_x.tolist() + bottom_x.tolist()
opt.set_design_readable(design_x, None, design_z)
opt.update_eme()

# Visualize
plt.figure()
eme.draw()
plt.show()

# Continuous grid
grid_z = np.linspace(dynamic_rect.grid_z[0], dynamic_rect.grid_z[-1], 300)
n = dynamic_rect.get_n(dynamic_rect.grid_x, grid_z)
plt.figure()
plt.imshow(
    n[::-1],
    cmap="Greys",
    extent=[grid_z[0], grid_z[-1], dynamic_rect.grid_x[0], dynamic_rect.grid_x[-1]],
    interpolation="none",
)
plt.show()
