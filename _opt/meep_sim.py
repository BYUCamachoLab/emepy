import meep as mp
import emepy as em
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk

# Constants
WAVELENGTH = 1.55
CLADDING_WIDTH = 2.5
CORE_INDEX = 3.4
CLADDING_INDEX = 1.4
NUM_PARAMS = 1
MESH_XY = 250
NUM_LAYERS = 10
NUM_MODES = 10
DP = 1e-6 # Finite difference size
TOTAL_TIME = 500
DATA_FILE = "data_new.pk"

# Geometry params
INPUT_WIDTH = 0.4
OUTPUT_WIDTH = 0.4
THICKNESS = 0.15
INPUT_LENGTH = 0.25
SHAPE_LENGTH = 2.0
OUTPUT_LENGTH = 0.25
DELTA_WIDTH = 0.1 # Percentage

# Flags
USE_EME_GRID = False
VISUALIZE_N = False
SYMMETRY = True
DUMP_DATA = True

# Derived params
CLADDING_THICKNESS = CLADDING_WIDTH
SOURCE_LOCATION = INPUT_LENGTH / 2
FOM_LOCATION = INPUT_LENGTH + SHAPE_LENGTH + OUTPUT_LENGTH / 2
TOTAL_LENGTH = INPUT_LENGTH + SHAPE_LENGTH + OUTPUT_LENGTH
MESH_DENSITY = MESH_XY / CLADDING_WIDTH 
MESH_Z = int(TOTAL_LENGTH * MESH_DENSITY)

# MPI
AM_MASTER = [False]

data = {
    "n": None,
    "eme_n": None,
    "A_u": None,
    "em_forward_fields": None,
    "em_backward_fields": None,
    "eme_grid_x": None,
    "eme_grid_y": None,
    "eme_grid_z": None,
    "emx": None,
    "emy": None,
    "emz": None,
    "forward_meep_fields": None,
    "backward_meep_fields": None,
    "mpx": None,
    "mpy": None,
    "mpz": None,
}

def eme():

    # Define the emepy geometry
    params = em.EMpyGeometryParameters(WAVELENGTH, CLADDING_WIDTH, CLADDING_THICKNESS, CORE_INDEX, CLADDING_INDEX, mesh=MESH_XY+1)
    input_wg = em.Waveguide(params, width=INPUT_WIDTH, thickness=THICKNESS, length=INPUT_LENGTH, num_modes=NUM_MODES)
    shape = em.DynamicRect2D(params, input_width=INPUT_WIDTH, output_width=OUTPUT_WIDTH, length=SHAPE_LENGTH, thickness=THICKNESS, num_modes=NUM_MODES, mesh_z=NUM_LAYERS, num_params=NUM_PARAMS, symmetry=SYMMETRY)
    output_wg = em.Waveguide(params, width=OUTPUT_WIDTH, thickness=THICKNESS, length=OUTPUT_LENGTH, num_modes=NUM_MODES)
    geometry = [input_wg, shape, output_wg]
    layers = [layer for geom in geometry for layer in geom]

    # Define the emepy optimization 
    eme = em.EME(layers=layers, mesh_z=MESH_Z, quiet=True, parallel=True)
    AM_MASTER[0] = eme.am_master()
    opt = em.Optimization(eme, geometries=geometry, mesh_z=MESH_Z, fom_location=FOM_LOCATION, source_location=SOURCE_LOCATION)

    # Initialize the optimization geometry
    design_x, design_z = opt.get_design_readable()
    design_x[:] = (np.array(design_x[:]) * (1+DELTA_WIDTH))[:]
    opt.set_design_readable(design_x, None, design_z)
    opt.update_eme()

    # Get the epsilon profile
    monitor:em.Monitor = eme.draw(mesh_z=MESH_Z, plot=False)
    x, y, z, n = monitor.get_array("n", axes="xyz")
    eme_n = n[:,:,:]
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
        plt.imshow(n[:, MESH_XY//2, :].real, extent=[z[0], z[-1], x[0], x[-1]], cmap="Greys")
        plt.show()

    # Get EMEPy fields 
    emx, emy, emz = (x, y, z)

    # Get Au 
    xd, yd, zd, _ = monitor.get_array("n", axes="xyz", z_range=(INPUT_LENGTH, INPUT_LENGTH + SHAPE_LENGTH))
    xd = np.linspace(xd[0], xd[-1], xd.shape[0]+1)
    yd = np.linspace(yd[0], yd[-1], yd.shape[0]+1) 
    A_u = opt.gradient(xd, yd, zd, dp=DP) 

    # Opt eme
    _, _, _, _, forward_monitor, fom_monitor = opt.forward_run()
    f_x, fom = opt.objective_gradient(fom_monitor)
    _, _, _, _, adjoint_monitor = opt.adjoint_run(None)
    em_forward_fields = forward_monitor.field
    em_backward_fields = adjoint_monitor.field
    eme_grid_x = forward_monitor.grid_x
    eme_grid_y = forward_monitor.grid_y
    eme_grid_z = forward_monitor.grid_z

    # Set data
    if AM_MASTER[0]:
        data["n"] = n
        data["eme_n"] = eme_n
        data["A_u"] = A_u
        data["em_forward_fields"] = em_forward_fields
        data["em_backward_fields"] = em_backward_fields
        data["eme_grid_x"] = eme_grid_x
        data["eme_grid_y"] = eme_grid_y
        data["eme_grid_z"] = eme_grid_z
        data["emx"] = emx
        data["emy"] = emy
        data["emz"] = emz

    return n


def meep(n, forward):

    # Create meep objects
    silicon = mp.Medium(index=CORE_INDEX)
    oxide = mp.Medium(index=CLADDING_INDEX)
    design_variables = mp.MaterialGrid(mp.Vector3(MESH_XY,MESH_XY,MESH_Z),oxide,silicon,grid_type='U_MEAN')
    design_variables.update_weights(((n.real-CLADDING_INDEX)/(CORE_INDEX-CLADDING_INDEX)).flatten())
    mp_geometry = [mp.Block(
        size=mp.Vector3(CLADDING_WIDTH, CLADDING_THICKNESS, TOTAL_LENGTH),
        material=design_variables
    )]
    boundary_layers = [mp.PML(thickness=0.5, direction=mp.X), mp.PML(thickness=0.5, direction=mp.Y), mp.PML(thickness=0.5, direction=mp.Z)]
    if forward:
        sources = [mp.EigenModeSource(mp.ContinuousSource(frequency=1/WAVELENGTH), center=mp.Vector3(0,0,-TOTAL_LENGTH/2+SOURCE_LOCATION), size=mp.Vector3(CLADDING_WIDTH,CLADDING_THICKNESS,0), direction=mp.Z)]
    else:
        sources = [mp.EigenModeSource(mp.ContinuousSource(frequency=1/WAVELENGTH), center=mp.Vector3(0,0,-TOTAL_LENGTH/2+FOM_LOCATION), size=mp.Vector3(CLADDING_WIDTH,CLADDING_THICKNESS,0), direction=-mp.Z)]

    # Set up MEEP simulation
    sx, sy, sz = CLADDING_WIDTH+1, CLADDING_THICKNESS+1, TOTAL_LENGTH+1
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        resolution=MESH_DENSITY,
        geometry=mp_geometry,
        sources=sources,
        eps_averaging=False,
        boundary_layers=boundary_layers,
        default_material=oxide
    )

    # Add a dft monitor
    dtf_vol = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(CLADDING_WIDTH, CLADDING_THICKNESS, TOTAL_LENGTH))
    dft = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz], WAVELENGTH, 0, 1, center=dtf_vol.center, size=dtf_vol.size)

    # Plot meep profile
    if VISUALIZE_N:
        plt.figure()
        sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(sx, 0, sz), center=mp.Vector3(0,0,0)))
        plt.show()

    # Run the simulation
    sim.run(until=TOTAL_TIME)

    # Get the dft fields
    fields = []
    for field in [mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz]:
        fields.append(sim.get_dft_array(dft,component=field,num_freq=0))
        mpx, mpy, mpz, _ = sim.get_array_metadata(vol=mp.Volume(center=dtf_vol.center, size=dtf_vol.size))
    meep_fields = np.array(fields)

    # Set data
    if AM_MASTER[0]:
        if forward:
            data["forward_meep_fields"] = meep_fields
        else:
            data["backward_meep_fields"] = meep_fields
        data["mpx"] = mpx
        data["mpy"] = mpy
        data["mpz"] = mpz


def dump(data):
    pk.dump(data, open(DATA_FILE, "wb+"))


def main():

    # First run eme
    n = eme()

    # Then dump data
    if DUMP_DATA and AM_MASTER[0]:
        dump(data)

    # Then run meep foward
    meep(n, True)

    # Then dump data
    if DUMP_DATA and AM_MASTER[0]:
        dump(data)

    # Then run meep backward
    meep(n, False)

    # Then dump data
    if DUMP_DATA and AM_MASTER[0]:
        dump(data)

    if AM_MASTER[0]:
        for key, value in data.items():
            print(key, value.shape)

if __name__ == "__main__":
    main()