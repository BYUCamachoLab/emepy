import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
import nlopt
from emepy.materials import Si, SiO2
import emepy

def get_geometry():
    matSi = Si(1.55)
    matSiO2 = SiO2(1.55)

    # Create goemetry params
    rect_params = EMpyGeometryParameters(
        wavelength=1.55, cladding_width=4, cladding_thickness=2.5, core_index=matSi, cladding_index=matSiO2, mesh=120
    )

    # Create an input waveguide
    input_waveguide = Waveguide(rect_params, width=1.0, thickness=0.22, length=0.5, center=(0, 0), num_modes=10)

    # Create an output waveguide
    output_waveguide = Waveguide(rect_params, width=1.5, thickness=0.22, length=0.5, center=(0, 0), num_modes=10)

    # Create the design region geometry
    dynamic_rect = DynamicRect2D(
        params=rect_params,
        width=input_waveguide.width,
        length=2,
        num_modes=10,
        num_params=30,
        symmetry=True,
        subpixel=True,
        mesh_z=10,
        input_width=input_waveguide.width,
        output_width=output_waveguide.width,
    )

    return [input_waveguide, dynamic_rect, output_waveguide]

def adjoint(geometry):
    input_waveguide, dynamic_rect, output_waveguide = geometry

    # Create the EME and Optimization
    eme = EME(quiet=True, parallel=True,mesh_z=100)
    optimizer = Optimization(eme, geometry, mesh_z=100)


    # Make the initial design a linear taper
    design_x, design_z = optimizer.get_design_readable()
    linear_taper = np.linspace(input_waveguide.width, output_waveguide.width, len(design_x)+2)[1:-1] / 2.0
    design_x[:] = linear_taper[:]
    optimizer.set_design_readable(design_x, None, design_z)

    # Get gradient
    f0, dJ_du, monitor = optimizer.optimize(optimizer.get_design())

    return dJ_du

def finite_difference(geometry, dp):
    input_waveguide, dynamic_rect, output_waveguide = geometry
    
    # Create the EME and Optimization
    eme = EME(quiet=True, parallel=True,mesh_z=100)
    optimizer = Optimization(eme, geometry, mesh_z=100)

    # Make the initial design a linear taper
    design_x, design_z = optimizer.get_design_readable()
    linear_taper = np.linspace(input_waveguide.width, output_waveguide.width, len(design_x)+2)[1:-1] / 2.0
    design_x[:] = linear_taper[:]
    optimizer.set_design_readable(design_x, None, design_z)

    # Run
    f0, _, monitor1 = optimizer.optimize(optimizer.get_design())

    # Change geometry
    design_x, design_z = optimizer.get_design_readable()
    design_x[:] = np.array(design_x) + dp
    optimizer.set_design_readable(design_x, None, design_z)

    # Run
    f1, _, monitor2 = optimizer.optimize(optimizer.get_design())
    x2, z2, field2 = monitor2.get_array(axes="xz")

    # Compute finite difference
    return (f1 - f0)

def main(dp=1e-10):

    # ms = get_geometry()[-1].layers[0].mode_solver
    # ms.solve()
    # for i in range(ms.num_modes):
    #     print(i, ms.get_mode(i).check_spurious())
    #     plt.figure()
    #     ms.get_mode(i).plot()
    #     plt.show()
    # quit()

    # Create vector dp
    input_waveguide, dynamic_rect, output_waveguide = get_geometry()
    dpv = np.random.rand(dynamic_rect.num_params*2) * dp
    dpv[1::2] = 0

    # Get finite difference differential
    fd_df = finite_difference(get_geometry(), dpv[::2])

    # Get adjoint differential
    adjoint_dfdp = adjoint(get_geometry())
    adjoint_df = np.sum(adjoint_dfdp @ dpv)

    # Compare 
    print("Adjoint run differential: {}".format(adjoint_df))
    print("Finite difference differential: {}".format(fd_df))

if __name__ == "__main__":
    main()