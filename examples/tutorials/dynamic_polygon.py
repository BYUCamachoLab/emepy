import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters
from emepy import EME

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55e-6,
    cladding_width=2.5e-6,
    cladding_thickness=2.5e-6,
    core_index=3.4,
    cladding_index=1.4,
    mesh=30,
)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=0.5e-6,
    length=1e-6,
    num_modes=1,
    num_params=10,
    symmetry=False,
    subpixel=True,
    mesh_z=10,
)

# Create eme layers
layers = [*dynamic_rect]

# Create the EME
eme = EME()
eme.add_layers(*layers)

# Visualize
plt.figure()
eme.draw()
plt.show()
