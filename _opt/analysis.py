import pickle as pk
import numpy as np
from matplotlib import pyplot as plt
from both import DP
from emepy.interface import OverlapTools

# Load data
data = pk.load(open("data_new.pk", "rb"))

# for key, value in data.items():
#     print(key, value.shape if value is not None else None)

# quit()

# EME data
n = data["n"] #  (200, 200, 200)
eme_n = data["eme_n"] #  (200, 200, 200)
A_u = data["A_u"] #  (3, 3, 200, 200, 200, 2)
em_forward_fields = data["em_forward_fields"] #  (7, 200, 200, 200)
em_backward_fields = data["em_backward_fields"] #  (7, 200, 200, 200)
eme_grid_x = data["eme_grid_x"] #  (250,)
eme_grid_y = data["eme_grid_y"] #  (250,)
eme_grid_z = data["eme_grid_z"] #  (250,)
emx = data["emx"] #  (250,)
emy = data["emy"] #  (250,)
emz = data["emz"] #  (250,)

# # MEEP data
forward_meep_fields = data["forward_meep_fields"] #  (6, 202, 202, 202)
# backward_meep_fields = data["backward_meep_fields"] #  (6, 202, 202, 202)
backward_meep_fields = forward_meep_fields[:, :, :, ::-1]
mpx = data["mpx"] #  (202,)
mpy = data["mpy"] #  (202,)
mpz = data["mpz"] #  (202,)

# # Plot EME n
# plt.figure()
# plt.imshow(n[:,100,:].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="Greys")
# plt.xlabel("z")
# plt.ylabel("x")
# plt.savefig("analysis/eme_n.png")

# # Plot EME eme_n
# plt.figure()
# plt.imshow(eme_n[:,100,:].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="Greys")
# plt.xlabel("z")
# plt.ylabel("x")
# plt.savefig("analysis/eme_eme_n.png")

# # Plot EME forward fields
# for i, field in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]):
#     plt.figure()
#     plt.imshow(em_forward_fields[i,:,100,:].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="RdBu")
#     plt.xlabel("z")
#     plt.ylabel("x")
#     plt.colorbar()
#     plt.savefig("analysis/eme_forward_" + field + ".png")
    
#     plt.figure()
#     plt.imshow(em_backward_fields[i,:,100,:].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="RdBu")
#     plt.xlabel("z")
#     plt.ylabel("x")
#     plt.colorbar()
#     plt.savefig("analysis/eme_adjoint_" + field + ".png")

# # Plot MEEP n
# plt.figure()
# plt.imshow(n[:,100,:].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="Greys")
# plt.xlabel("z")
# plt.ylabel("x")
# plt.savefig("analysis/meep_n.png")

# # Plot MEEP fields
# for i, field in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]):
#     plt.figure()
#     plt.imshow(forward_meep_fields[i,:,100,:].real, extent=[mpz[0], mpz[-1], mpx[0], mpx[-1]], cmap="RdBu")
#     plt.xlabel("z")
#     plt.ylabel("x")
#     plt.colorbar()
#     plt.savefig("analysis/meep_forward_" + field + ".png")
    
#     plt.figure()
#     plt.imshow(backward_meep_fields[i,:,100,:].real, extent=[mpz[0], mpz[-1], mpx[0], mpx[-1]], cmap="RdBu")
#     plt.xlabel("z")
#     plt.ylabel("x")
#     plt.colorbar()
#     plt.savefig("analysis/meep_backward_" + field + ".png")

# Gradient function
def compute_final_gradient(lamdagger: "np.ndarray", A_u: "np.ndarray", X: "np.ndarray"):
    """Computes the final gradient using the adjoint formulation and loops to conserve memory"""

    # Initialize final result
    f_u = np.zeros(A_u.shape[-1], dtype=float)

    # Reshape
    lamdagger = np.transpose(np.conj(lamdagger), axes=(1,2,3,0))
    A_u = A_u
    X = X

    # Loop through all params
    for p in range(len(f_u)):
        A_u_temp = A_u[..., p]

        # Compute all 9 components of the matrix
        A_u_x = np.zeros([3] + list(A_u.shape[2:-1]), dtype=complex)
        for i, mi in enumerate(A_u_temp):
            for j, mij in enumerate(mi):
                A_u_x[i] += mij * X[j]

        # Compute lambda * A_u_x matrix multiplication for the pth column
        for i in range(3):
            f_u[p] += - 2 * np.real(np.sum(A_u_x[j] * lamdagger[..., i]))
    return f_u

def xy_csection(z_array, field, where):
    """Computes the xy cross section of a field"""
    idx = (np.abs(z_array - where)).argmin()
    return field[..., idx]

from both import *
# params = em.EMpyGeometryParameters(WAVELENGTH, CLADDING_WIDTH, CLADDING_THICKNESS, CORE_INDEX, CLADDING_INDEX, mesh=MESH_XY+1)
# # input_wg = em.Waveguide(params, width=INPUT_WIDTH, thickness=THICKNESS, length=INPUT_LENGTH, num_modes=NUM_MODES)[0].mode_solver
# output_wg = [*em.Waveguide(params, width=OUTPUT_WIDTH, thickness=THICKNESS, length=OUTPUT_LENGTH, num_modes=NUM_MODES)][0].mode_solver
# # input_wg.solve()
# output_wg.solve()
# # input_wg = input_wg.get_mode(0)
# output_wg = output_wg.get_mode(0)
# pk.dump(output_wg, open("output_wg.pk", "wb+"))

output_wg = pk.load(open("output_wg.pk", "rb"))
output_wg.normalize()


def overlap(x, y, field, mode=output_wg):
    field = field / np.sqrt(np.abs(OverlapTools.fom_overlap(field[0], field[1], field[3], field[4], field[0], field[1], field[3], field[4], x, y)))
    return OverlapTools.fom_overlap(field[0], field[1], field[3], field[4], mode.Ex, mode.Ey, mode.Hx, mode.Hy, x, y)


# # Calculate the gradient according to the EME data
# mask = ((emz>0.7) * (emz<2.7))
# em_backward_fields = em_backward_fields[:3,:,:,mask]
# em_forward_fields = em_forward_fields[:3,:,:,mask]
# eme_gradient = compute_final_gradient(em_backward_fields, A_u, em_forward_fields)

# Calculate the gradient according to the MEEP data
mpz -= np.min(mpz)
mpx_mask = ((mpx>=emx[0]) * (mpx<=emx[-1]))
mpx_mask[1] = True
mpy_mask = ((mpy>=emy[0]) * (mpy<=emy[-1]))
mpy_mask[1] = True
mpz_mask = ((mpz>=0.7) * (mpz<2.7))
mpz = mpz[mpz_mask]
mpx = mpx[mpx_mask]
mpy = mpy[mpy_mask]

forward_meep_fields = forward_meep_fields[:6,:,:,mpz_mask]
forward_meep_fields = forward_meep_fields[:6,:,mpy_mask,:]
forward_meep_fields = forward_meep_fields[:6,mpx_mask,:,:]
backward_meep_fields = backward_meep_fields[:6,:,:,mpz_mask]
backward_meep_fields = backward_meep_fields[:6,:,mpy_mask,:]
backward_meep_fields = backward_meep_fields[:6,mpx_mask,:,:]
# meep_gradient = compute_final_gradient(backward_meep_fields[:3], A_u, forward_meep_fields[:3])

from emepy.mode import Mode

_forward_meep_fields = xy_csection(mpz, forward_meep_fields, FOM_LOCATION)
_em_forward_fields = xy_csection(emz, em_forward_fields, FOM_LOCATION)
_n = xy_csection(emz, eme_n, FOM_LOCATION)
meep_mode = Mode(mpx, mpy, WAVELENGTH, 3, _forward_meep_fields[0], _forward_meep_fields[1], _forward_meep_fields[2], _forward_meep_fields[3], _forward_meep_fields[4], _forward_meep_fields[5], _n)
eme_mode = Mode(emx, emy, WAVELENGTH, 3, _em_forward_fields[0], _em_forward_fields[1], _em_forward_fields[2], _em_forward_fields[3], _em_forward_fields[4], _em_forward_fields[5], _n)
meep_mode.normalize()
eme_mode.normalize()

# print(overlap(mpx, mpy, _forward_meep_fields))
# print(overlap(emx, emy, _em_forward_fields))
plt.figure()
meep_mode.plot()
plt.figure()
eme_mode.plot()
plt.show()
quit()
plt.figure()
plt.imshow(np.rot90(_forward_meep_fields[4, :, :].real), extent=[mpx[0], mpx[-1], mpy[0], mpy[-1]], cmap="RdBu")
plt.show()

# plt.figure()
# output_wg.plot()
# plt.show()

# # Grab the finite difference data
fd_data2 = pk.load(open("fd_data2.pk","rb"))
# for key, value in fd_data.items():
#     print(key)

# print(np.sum(fd_data["eme_lower_n"]-fd_data["eme_upper_n"]))

# plt.figure()
# plt.imshow((fd_data["eme_lower_n"]-fd_data["eme_upper_n"])[:, 100, :].real, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="Greys")
# plt.show()

# print((fd_data['eme_upper_fom']-fd_data['eme_lower_fom'])/(2*DP))
# print((fd_data['meep_upper_fom']-fd_data['meep_lower_fom'])/(2*DP))

# plt.figure()
# print(np.sum(A_u[0,0,:, :, :, 0][84:86], axis=0).sum(axis=1))
# plt.imshow(A_u[0,0,:, 100, :, 0].imag, extent=[emz[0], emz[-1], emx[0], emx[-1]], cmap="Greys")
# plt.show()



# params = em.EMpyGeometryParameters(WAVELENGTH, CLADDING_WIDTH, CLADDING_THICKNESS, CORE_INDEX, CLADDING_INDEX, mesh=MESH_XY+3)
# # input_wg = em.Waveguide(params, width=INPUT_WIDTH, thickness=THICKNESS, length=INPUT_LENGTH, num_modes=NUM_MODES)[0].mode_solver
# output_wg = [*em.Waveguide(params, width=OUTPUT_WIDTH, thickness=THICKNESS, length=OUTPUT_LENGTH, num_modes=1)][0].mode_solver
# # input_wg.solve()
# output_wg.solve()
# # input_wg = input_wg.get_mode(0)
# output_wg = output_wg.get_mode(0)
# pk.dump(output_wg, open("output_wg_second.pk", "wb"))
output_wg = pk.load(open("output_wg_second.pk", "rb"))
output_wg.normalize()

fd_data = pk.load(open("fd_data.pk","rb"))
meep_upper_fields = np.array(fd_data["meep_upper_fields"])
meep_lower_fields = np.array(fd_data["meep_lower_fields"])
meep_mpz = fd_data["meep_mpz"]
meep_mpx = fd_data["meep_mpx"]
meep_mpy = fd_data["meep_mpy"]
upper_meep = xy_csection(meep_mpz, meep_upper_fields, FOM_LOCATION)
lower_meep = xy_csection(meep_mpz, meep_lower_fields, FOM_LOCATION)
upper_fom = overlap(meep_mpx, meep_mpy, upper_meep, output_wg)
lower_fom = overlap(meep_mpx, meep_mpy, lower_meep, output_wg)


# plt.figure()
# vals = [np.abs(overlap(meep_mpx, meep_mpy, upper_meep*np.exp(1j*phi), output_wg)) for phi in np.linspace(0,2*np.pi,100)]
# plt.plot(np.linspace(0,2*np.pi,100), vals)
# plt.show()
# quit()

print((np.abs(upper_fom)**2-np.abs(lower_fom)**2)/(2*DP))
lower_fom = fd_data2["eme_lower_f_x"]
upper_fom = fd_data2["eme_upper_f_x"]
print((np.abs(upper_fom)**2-np.abs(lower_fom)**2)/(2*DP))