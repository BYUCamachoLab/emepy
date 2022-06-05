from emepy.eme import EME
from emepy.geometries import Waveguide, EMpyGeometryParameters
from matplotlib import pyplot as plt

# Create wg params
wg_params = EMpyGeometryParameters(
    wavelength=1.55, cladding_width=5, cladding_thickness=2.5, core_index=3.4, cladding_index=1.5, mesh=90
)

# Create left wg
left_wg = Waveguide(wg_params, width=0.7, thickness=0.22, length=1.55)

# Create right wg
right_wg = Waveguide(wg_params, width=1.2, thickness=0.22, length=1.55)

# Create eme
eme = EME(quiet=True, parallel=False, mesh_z=100)
eme.add_layers(*left_wg, *right_wg)

# Create monitor
monitor = eme.add_monitor()

# Propagate eme
eme.propagate()

# Plot
plt.figure()
monitor.visualize()
plt.show()
