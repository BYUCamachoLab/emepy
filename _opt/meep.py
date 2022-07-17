import meep as mp
import emepy as em
import numpy as np
from matplotlib import pyplot as plt

# Constants
WAVELENGTH = 1.55
CLADDING_WIDTH = 2.5
CORE_INDEX = 3.4
CLADDING_INDEX = 1.4
NUM_PARAMS = 1
MESH_XY = 128
NUM_LAYERS = 5
NUM_MODES = 4
DP = 1e-6 # Finite difference size

# Geometry params
INPUT_WIDTH = 0.5
OUTPUT_WIDTH = 0.5
THICKNESS = 0.22
INPUT_LENGTH = 0.25
SHAPE_LENGTH = 2.0
OUTPUT_LENGTH = 0.25
DELTA_WIDTH = 0.5 # Percentage

# Flags
USE_EME_GRID = False
VISUALIZE_N = False
SYMMETRY = True

# Derived params
CLADDING_THICKNESS = CLADDING_WIDTH
SOURCE_LOCATION = INPUT_LENGTH / 2
FOM_LOCATION = INPUT_LENGTH + SHAPE_LENGTH + OUTPUT_LENGTH / 2
TOTAL_LENGTH = INPUT_LENGTH + SHAPE_LENGTH + OUTPUT_LENGTH
MESH_DENSITY = MESH_XY / CLADDING_WIDTH 
MESH_Z = int(TOTAL_LENGTH * MESH_DENSITY)

def main():

    # Define the emepy geometry
    params = em.EMpyGeometryParameters(WAVELENGTH, CLADDING_WIDTH, CLADDING_THICKNESS, CORE_INDEX, CLADDING_INDEX, mesh=MESH_XY+1)
    input_wg = em.Waveguide(params, width=INPUT_WIDTH, thickness=THICKNESS, length=INPUT_LENGTH, num_modes=NUM_MODES)
    shape = em.DynamicRect2D(params, input_width=INPUT_WIDTH, output_width=OUTPUT_WIDTH, length=SHAPE_LENGTH, thickness=THICKNESS, num_modes=NUM_MODES, mesh_z=NUM_LAYERS, num_params=NUM_PARAMS, symmetry=SYMMETRY)
    output_wg = em.Waveguide(params, width=OUTPUT_WIDTH, thickness=THICKNESS, length=OUTPUT_LENGTH, num_modes=NUM_MODES)
    geometry = [input_wg, shape, output_wg]
    layers = [layer for geom in geometry for layer in geom]

    # Define the emepy optimization 
    eme = em.EME(layers=layers, mesh_z=MESH_Z, quiet=True, parallel=False)
    opt = em.Optimization(eme, geometries=geometry, mesh_z=MESH_Z, fom_location=FOM_LOCATION, source_location=SOURCE_LOCATION)

    # Initialize the optimization geometry
    design_x, design_z = opt.get_design_readable()
    design_x[:] = (np.array(design_x[:]) * (1+DELTA_WIDTH))[:]
    opt.set_design_readable(design_x, None, design_z)
    opt.update_eme()

    # Get the epsilon profile
    monitor:em.Monitor = eme.draw(mesh_z=MESH_Z, plot=False)
    x, y, z, n = monitor.get_array("n", axes="xyz")
    if not USE_EME_GRID:
        xd, yd, zd, _ = monitor.get_array("n", axes="xyz", z_range=(INPUT_LENGTH, INPUT_LENGTH + SHAPE_LENGTH))
        xd = np.linspace(xd[0], xd[-1], xd.shape[0]+1)
        yd = np.linspace(yd[0], yd[-1], yd.shape[0]+1) 
        design_profile = opt.get_n(xd, yd, zd)
        start, end = np.where(z==zd[0])[0].item(0), np.where(z==zd[-1])[0].item(0)
        n[:, :, start:end] = design_profile[:,:,:]

    # Visualize the epsilon profile
    if VISUALIZE_N:
        plt.figure()
        plt.imshow(n[:, 63, :].real, extent=[z[0], z[-1], x[0], x[-1]], cmap="Greys")
        plt.show()

    # Get Au 
    xd, yd, zd, _ = monitor.get_array("n", axes="xyz", z_range=(INPUT_LENGTH, INPUT_LENGTH + SHAPE_LENGTH))
    xd = np.linspace(xd[0], xd[-1], xd.shape[0]+1)
    yd = np.linspace(yd[0], yd[-1], yd.shape[0]+1) 
    A_u = opt.gradient(xd, yd, zd, dp=DP)  

    # Create meep objects
    mp_geometry = [mp.Block(
        size=mp.Vector3(x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]),
        material=n
    )]
    boundary_layers = [mp.PML(1.0)]
    sources = [mp.Source(mp.EigenModeSource(mp.ContinuousSource(frequency=1/WAVELENGTH),))]

    # Set up MEEP simulation
    sim = mp.Simulation(
        cell_size=mp.Vector3(x[-1]-x[0]+1, y[-1]-y[0]+1, z[-1]-z[0]+1),
        resolution=MESH_DENSITY,
        geometry=mp_geometry,
        sources=sources,
        eps_averaging=False,
        boundary_layers=boundary_layers,
        default_material=mp.Medium(epsilon=CLADDING_INDEX)
    )



if __name__ == "__main__":
    main()