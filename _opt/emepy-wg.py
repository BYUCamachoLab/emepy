import emepy as em
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk

# Parameters
wavelength = 1.55
data = {}

# Create a waveguide
geometry = em.Waveguide(
    params = em.EMpyGeometryParameters(
        wavelength=wavelength,
        cladding_width=5.0,
        cladding_thickness=5.0,
        mesh=151,
    ),
    width = 0.44,
    thickness = 0.22,
    length = 7.0,
    num_modes=1
)

# Create an EMEpy simulation
eme = em.EME(layers = [*geometry], mesh_z=210, parallel=True, quiet=True)

# Add a monitor
source = em.Source(z=1.25, mode_coeffs=[1])
monitor = eme.add_monitor(axes="xz",sources=[source], z_range=[0,7])

# Visualize the geometry
plt.figure()
eme.draw()
if eme.am_master():
    plt.savefig("emepy-figures/n_profile.png")

# Run the simulation
eme.propagate(left_coeffs=[])

# Plot the fields
for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
    plt.figure()
    monitor.visualize(component=comp, colorbar=True)
    x, z, field = monitor.get_array(component=comp)
    data[comp] = field
    data["x"] = x
    data["z"] = z
    if eme.am_master():
        plt.title("freq_{}.png".format(comp))
        plt.savefig("emepy-figures/freq_{}.png".format(comp))

# Save the data
if eme.am_master():
    pk.dump(data, open("data/emepy_data.pk", "wb+"))