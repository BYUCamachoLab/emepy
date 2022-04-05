import numpy as np
from matplotlib import pyplot as plt
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization
import nlopt
from emepy.materials import Si, SiO2
import emepy

parallel = True

def get_geometry(mesh=100, num_layers=10):
    matSi = Si(1.55)
    matSiO2 = SiO2(1.55)

    # Create goemetry params
    rect_params = EMpyGeometryParameters(
        wavelength=1.55,
        cladding_width=5,
        cladding_thickness=2.5,
        core_index=matSi,
        cladding_index=matSiO2,
        mesh=mesh,
    )

    # Create an input waveguide
    input_waveguide = Waveguide(
        rect_params, width=1.0, thickness=0.22, length=0.5, center=(0, 0), num_modes=10
    )

    # Create an output waveguide
    output_waveguide = Waveguide(
        rect_params, width=1.75, thickness=0.22, length=0.5, center=(0, 0), num_modes=10
    )

    # Create the design region geometry
    dynamic_rect = DynamicRect2D(
        params=rect_params,
        width=input_waveguide.width,
        length=2,
        num_modes=10,
        num_params=30,
        symmetry=True,
        subpixel=True,
        mesh_z=num_layers,
        input_width=input_waveguide.width,
        output_width=output_waveguide.width,
    )

    return [input_waveguide, dynamic_rect, output_waveguide]


def adjoint(geometry, dp):
    input_waveguide, dynamic_rect, output_waveguide = geometry

    # Create the EME and Optimization
    eme = EME(quiet=False, parallel=parallel, mesh_z=100)
    optimizer = Optimization(eme, geometry, mesh_z=100)

    # Make the initial design a linear taper
    design_x, design_z = optimizer.get_design_readable()
    linear_taper = (
        np.linspace(input_waveguide.width, output_waveguide.width, len(design_x) + 2)[
            1:-1
        ]
        / 2.0
    )
    design_x[:] = linear_taper[:]
    optimizer.set_design_readable(design_x, None, design_z)

    # Get gradient
    f0, dJ_du, monitor = optimizer.optimize(optimizer.get_design(), dp)
    design_x, design_z = optimizer.get_design_readable()

    return dJ_du, eme, monitor, design_x, design_z


def finite_difference(geometry, dp):
    input_waveguide, dynamic_rect, output_waveguide = geometry

    # Create the EME and Optimization
    eme = EME(quiet=True, parallel=parallel, mesh_z=100)
    optimizer = Optimization(eme, geometry, mesh_z=100)

    # Make the initial design a linear taper
    design_x, design_z = optimizer.get_design_readable()
    linear_taper = (
        np.linspace(input_waveguide.width, output_waveguide.width, len(design_x) + 2)[
            1:-1
        ]
        / 2.0
    )
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
    return f1 - f0


def main(dp=1e-8):

    # Create arrays
    mesh_array = np.linspace(50, 250, 10)[1:]
    num_layers_array = np.linspace(5, 25, 10)[1:]

    # Create empty results
    adjoint_results = []
    finite_difference_results = []

    # Get differental
    _, dynamic_rect, _ = get_geometry(50, 5)
    dpv = np.random.rand(dynamic_rect.num_params * 2) * dp
    dpv[1::2] = 0

    # Loop over mesh and num_layers
    for mesh, num_layers in zip(mesh_array, num_layers_array):

        # intify
        mesh, num_layers = int(mesh), int(num_layers)

        # Create vector dp
        _, dynamic_rect, _ = get_geometry(mesh, num_layers)

        # Get adjoint differential
        adjoint_dfdp, _, _, _, _ = adjoint(get_geometry(mesh, num_layers), dp)
        adjoint_df = np.sum(adjoint_dfdp @ dpv)

        # Get finite difference differential
        fd_df = finite_difference(get_geometry(mesh, num_layers), dpv[::2])

        # Compare
        if emepy.am_master(parallel):
            print("Adjoint run differential: {}".format(adjoint_df))
            print("Finite difference differential: {}\n".format(fd_df))

        # Append results
        adjoint_results.append(adjoint_df)
        finite_difference_results.append(fd_df)

    # Plot
    plt.figure()
    plt.plot(mesh_array, adjoint_results, label="Adjoint")
    plt.plot(mesh_array, finite_difference_results, label="Finite difference")
    plt.xlabel("Mesh")
    plt.ylabel("Differential")
    plt.legend()
    plt.savefig("test_adjoint_converge.png")

    plt.figure()
    plt.semilogy(mesh_array, np.abs(adjoint_results), label="Adjoint")
    plt.semilogy(mesh_array, np.abs(finite_difference_results), label="Finite difference")
    plt.xlabel("Mesh")
    plt.ylabel("Differential")
    plt.legend()
    plt.savefig("test_adjoint_converge_log.png")



if __name__ == "__main__":
    main()
