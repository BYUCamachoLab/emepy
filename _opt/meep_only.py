from meep_sim import *
import sys

VISUALIZE_N = False
TOTAL_TIME = 200
forward = False
print("forward: ", forward)
data = pk.load(open(DATA_FILE, "rb"))
n = data["n"] 

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
dft = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz], WAVELENGTH, 0, 1, center=mp.Vector3(0,0,0), size=mp.Vector3(sx, 0, sz))

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
    mpx, mpy, mpz, _ = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx, 0, sz)))
meep_fields = np.array(fields)
pk.dump(fields, open("freq_{}.pk".format(forward), "wb+"))

if mp.am_really_master():
    plt.figure()
    sim.plot2D(fields=mp.Hy, output_plane=mp.Volume(size=mp.Vector3(sx, 0, sz), center=mp.Vector3(0,0,0)))
    plt.savefig('time_{}'.format(forward))