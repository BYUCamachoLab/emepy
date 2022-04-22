import numpy as np
from matplotlib import pyplot as plt
import meep as mp
import pickle
import emepy as em

# Parameters
mesh = 150
num_layers = 15
mesh_z = 100

# Materials
matSi = mp.Medium(index=3.4757)
matSiO2 = mp.Medium(index=1.4440224359379057)

# Create geometry
input_waveguide = mp.Block(size=mp.Vector3(1, 0.22, 0.5), material=matSi, center=mp.Vector3(0, 0, -(0.25-1.5)))
layer_widths = np.linspace(1, 1.75, num_layers)
layers = []
length = 2/num_layers
cur_len = 0.5-1.5 - length/2
for i in range(num_layers):
    cur_len += length
    layers.append(mp.Block(size=mp.Vector3(layer_widths[i], 0.22, length), material=matSi, center=mp.Vector3(0, 0, -cur_len)))
output_waveguide = mp.Block(size=mp.Vector3(1.75, 0.22, 0.5), material=matSi, center=mp.Vector3(0, 0, -(cur_len+0.25)))
geometry = [input_waveguide, *layers, output_waveguide]

# Calculate resolution
resolution = 30#mesh / 5 / 2.5

# Create source
source = mp.EigenModeSource(mp.ContinuousSource(frequency=1/1.55), center=mp.Vector3(0,0,-(2.75-1.5)), size=mp.Vector3(5, 2.5, 0))

# Create simulation object
sim = mp.Simulation(
    cell_size=mp.Vector3(7, 4.5, 5),
    boundary_layers=[mp.PML(1.0)],
    sources=[source],
    resolution=resolution,
    geometry=geometry,
    default_material=matSiO2
)

# Plot the geometry
# sim.init_sim()
# plt.figure()
# sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(7, 0, 5)))
# if mp.am_master():
#     plt.show()

# Run simulation
sim.run(until=0.1)

# Get the field
index = sim.get_array(component=mp.Dielectric, center=mp.Vector3(0, 0, 0), size=mp.Vector3(5, 0, 3))
field = sim.get_array(component=mp.Hy, center=mp.Vector3(0, 0, 0), size=mp.Vector3(5, 0, 3))

# Plot the field
plt.figure()
plt.imshow(index[:,::-1], extent=[-1.5, 1.5, -2.5, 2.5], cmap="Greys")
plt.imshow(field[:,::-1], extent=[-1.5, 1.5, -2.5, 2.5], cmap="RdBu", alpha=0.8)
if mp.am_master():
    plt.show()

# # Save to an eme monitor
# axes, dimensions, components, z_range, grid_x, grid_y, grid_z, location, sources=sources, total_length=total_length
# monitor = em.Monitor(axes="xyz", components=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], dimensions=(c, x, y, z), grid_x, grid_y, grid_z)

# pickle.dump(field, open("adjoint.pk", "wb+"))

