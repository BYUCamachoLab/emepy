import numpy as np
from matplotlib import pyplot as plt
import meep as mp

mp.quiet(True)

geometry = []
thickness = 0.2
wavelength = 0.6 * 3
freq = 1 / wavelength

sources = [
    mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-1.2, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+1.2, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-0.6, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+0.6, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-1.95, 0.15)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+1.95, 0.15)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-0.6, 0.15)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+0.6, 0.15)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-1.95, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+1.95, 1.93)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(-1.3, 1)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(0, 1)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(0, 1.9)),
    # mp.Source(mp.ContinuousSource(frequency=freq), component=mp.Hz, center=mp.Vector3(+1.3, 1)),
]


pml_layers = [mp.PML(1.0)]
resolution = 35

# Params
height = 2
length = 1
thickness = 0.3
translation = 0.5

######## E
# Bottom
geometry.append(
    mp.Block(
        center=mp.Vector3(translation - 2 * length + 0.25 * length, 0.1),
        size=mp.Vector3(length * 1.5, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Side
geometry.append(
    mp.Block(
        center=mp.Vector3(translation - 2 * length - length / 2.2, height / 2),
        size=mp.Vector3(thickness, height),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Middle
geometry.append(
    mp.Block(
        center=mp.Vector3(translation - 2 * length, height / 2),
        size=mp.Vector3(length * 0.75, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Top
geometry.append(
    mp.Block(
        center=mp.Vector3(translation - 2 * length, height - 0.1),
        size=mp.Vector3(length, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)

######## M
# Left
geometry.append(
    mp.Block(
        center=mp.Vector3(-translation - length * 0.1, height / 2),
        size=mp.Vector3(thickness, height),
        material=mp.Medium(epsilon=4 ** 2),
    )
)

# Middle
geometry.append(
    mp.Block(
        center=mp.Vector3(0, 2.1 * height / 3),
        size=mp.Vector3(thickness, 2 * height / 3),
        material=mp.Medium(epsilon=4 ** 2),
    )
)

# Top
geometry.append(
    mp.Block(center=mp.Vector3(0, height - 0.1), size=mp.Vector3(length, thickness), material=mp.Medium(epsilon=4 ** 2))
)

# Right
geometry.append(
    mp.Block(
        center=mp.Vector3(translation + length * 0.1, height / 2),
        size=mp.Vector3(thickness, height),
        material=mp.Medium(epsilon=4 ** 2),
    )
)

######## E
# Bottom
geometry.append(
    mp.Block(
        center=mp.Vector3(-translation + 2 * length - 0.25 * length, 0.1),
        size=mp.Vector3(length * 1.5, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Side
geometry.append(
    mp.Block(
        center=mp.Vector3(-translation + 2 * length + length / 2.2, height / 2),
        size=mp.Vector3(thickness, height),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Middle
geometry.append(
    mp.Block(
        center=mp.Vector3(-translation + 2 * length, height / 2),
        size=mp.Vector3(length * 0.75, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)
# Top
geometry.append(
    mp.Block(
        center=mp.Vector3(-translation + 2 * length, height - 0.1),
        size=mp.Vector3(length, thickness),
        material=mp.Medium(epsilon=4 ** 2),
    )
)

cellsxy = mp.Vector3(10 * length, 10 * length, 0)
sim = mp.Simulation(
    cell_size=cellsxy,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    default_material=mp.Medium(epsilon=1 ** 2),
)

sim.run(until=800)

# Visualize geometry
# plt.figure()
# sim.plot2D(
#     fields=mp.Ex,
#     output_plane=mp.Volume(size=mp.Vector3(10, 10, 0)),
#     plot_sources_flag=False,
#     plot_boundaries_flag=False,
# )
# plt.show()

# plt.figure()
# sim.plot2D(
#     fields=mp.Ey,
#     output_plane=mp.Volume(size=mp.Vector3(10, 10, 0)),
#     plot_sources_flag=False,
#     plot_boundaries_flag=False,
# )
# plt.show()

plt.figure()
sim.plot2D(
    fields=mp.Hz,
    output_plane=mp.Volume(size=mp.Vector3(10, 10, 0)),
    plot_sources_flag=False,
    plot_boundaries_flag=False,
)
plt.show()
