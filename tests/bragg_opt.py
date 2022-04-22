import emepy as em
from emepy import optimization
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization

import numpy as np
from matplotlib import pyplot as plt

# Parameters
mesh = 150
num_periods = 50
num_params = 20
num_modes = 3
parallel = True
mesh_z = 200
length = .159e-6
width1 = 0.46e-6
width2 = 0.54e-6
thickness = 0.22e-6

# Materials
matSi = em.Si(1.55)
matSiO2 = em.SiO2(1.55)

# Create goemetry params
rect_params = EMpyGeometryParameters(
    wavelength=1.55,
    cladding_width=5,
    cladding_thickness=2.5,
    core_index=matSi,
    cladding_index=matSiO2,
    mesh=mesh,
)

# Create the geometry
small = DynamicRect2D(
    params=rect_params,
    width=width1,
    length=length,
    num_modes=num_modes,
    num_params=num_params,
    symmetry=True,
    subpixel=True,
    mesh_z=1,
)
big = DynamicRect2D(
    params=rect_params,
    width=width2,
    length=length,
    num_modes=num_modes,
    num_params=num_params,
    symmetry=True,
    subpixel=True,
    mesh_z=1,
)
geometry = [small, big]

# Create the EME
eme = EME(quiet=True, parallel=parallel, mesh_z=mesh_z, num_periods=num_periods)
optimizer = Optimization(eme, geometry, mesh_z=mesh_z)
