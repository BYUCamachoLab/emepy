import meep as mp
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk
from emepy.materials import Si, SiO2

# Parameters
wavelength = 1.55
fields = dict(zip(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]))

# Create a waveguide
geometry = [
    mp.Block(
        size=mp.Vector3(0.44, 0.22, mp.inf),
        material=mp.Medium(index=Si(wavelength))
    )
]

# Add a mode source
sources = [
    mp.EigenModeSource(
        mp.ContinuousSource(frequency=1/wavelength), 
        center=mp.Vector3(0,0,-2.25), 
        size=mp.Vector3(4,4,0),
        direction=mp.Z
    )
]

# Set up a simulation
sim = mp.Simulation(
    cell_size=mp.Vector3(5, 5, 7),
    resolution=30,
    geometry=geometry,
    sources=sources,
    eps_averaging=True,
    boundary_layers=[mp.PML(thickness=1.0)],
    default_material=mp.Medium(index=SiO2(wavelength))
)

# Add a dft monitor
dft = sim.add_dft_fields(list(fields.values()), 1/wavelength, 0, 1, center=mp.Vector3(0,0,0), size=mp.Vector3(5, 0, 7))

# Plot the geometry
plt.figure()
sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(5, 0, 7), center=mp.Vector3(0,0,0)))
if mp.am_master():
    plt.savefig("meep-figures/n_profile.png")

# Run the simulation
sim.run(until=200)

# Plot the fields
for string, field in fields.items():
    plt.figure()
    sim.plot2D(fields=field, output_plane=mp.Volume(size=mp.Vector3(5, 0, 7), center=mp.Vector3(0,0,0)))
    if mp.am_master():
        plt.title("time_{}.png".format(string))
        plt.savefig("meep-figures/time_{}.png".format(string))

# Plot the dft fields
data = {}
mpx, mpy, mpz, _ = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(5, 0, 7)))
data["mpx"] = mpx
data["mpy"] = mpy
data["mpz"] = mpz
for string, field in fields.items():
    plt.figure()
    profile = sim.get_dft_array(dft, component=field, num_freq=0)
    data[string] = profile
    plt.imshow(profile.real, cmap="RdBu", extent=[0, mpz[-1]-mpz[0], mpx[0], mpx[-1]])
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("freq_{}.png".format(string))
    if mp.am_master():
        plt.savefig("meep-figures/freq_{}.png".format(string))

# Save the data
if mp.am_master():
    pk.dump(data, open("data/meep_data.pk", "wb+"))