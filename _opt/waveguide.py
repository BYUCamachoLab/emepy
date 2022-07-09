import numpy as np
import argparse
from emepy.geometries import EMpyGeometryParameters, DynamicRect2D
from emepy.eme import EME
from emepy.optimization import Optimization

def get_geometry(**kwargs):

    # Get geometry parameters
    geometry_params = EMpyGeometryParameters(
        wavelength = kwargs["wavelength"],
        cladding_width = kwargs["cladding_width"],
        cladding_thickness = kwargs["cladding_thickness"],
        core_index = kwargs["core_index"],
        cladding_index = kwargs["cladding_index"],
        mesh = kwargs["mesh"],
    )

    # Create waveguide
    waveguide = DynamicRect2D(
        geometry_params,
        width = kwargs["width"],
        thickness = kwargs["thickness"],
        length = kwargs["length"],
        num_modes = kwargs["num_modes"],
        num_params = kwargs["num_params"],
        symmetry = True,
        subpixel = True,
        mesh_z = kwargs["num_slabs"],
        input_width = kwargs["input_width"],
        output_width = kwargs["output_width"],
    )

    return waveguide

def get_simulation(geometry, **kwargs):

    # Create an EME
    eme = EME(
        layers = [*geometry],
        num_periods = 1,
        quiet=kwargs["quiet"],
    )
    
    return eme

def get_optimization(geometries, eme, **kwargs):

    # Create an optimization
    optimization = Optimization(
        eme = eme,
        geometries = geometries,
        mesh_z = kwargs["mesh_z"],
    )

    # Initialize design
    x, z = optimization.get_design_readable()
    x = np.array(x)
    x[1:-1] = x[1:-1] * (1 + np.random.rand(len(x[1:-1])) * kwargs["random_strength"])
    x = x.tolist()
    optimization.set_design_readable(x, None, z)

    return optimization

def run_optimization(optimization:Optimization, **kwargs):

    # Run optimization
    fom, f_u, monitor_forward = optimization.optimize(optimization.get_design(), dp=kwargs["dp"])
    return fom, f_u

def run_finite_difference(optimization:Optimization, **kwargs):

    # Run optimization
    fd_gradient, fd_gradient_idx, avg, fom_monitor = optimization.calculate_fd_gradient(
        num_gradients=kwargs["num_params"], 
        dp=kwargs["dp"]
    )
    return fd_gradient, fd_gradient_idx, avg


def main(**kwargs):

    # Set random seed
    np.random.seed(kwargs["seed"])

    # Get geometry
    geometry = get_geometry(**kwargs)

    # Get simulation
    eme = get_simulation(geometry, **kwargs)

    # Get optimization
    optimization = get_optimization(geometry, eme, **kwargs)

    # Run adjoint run
    fom, f_u = run_optimization(optimization, **kwargs)
    
    # Run finite difference run
    fd_gradient, fd_gradient_idx, avg = run_finite_difference(optimization, **kwargs)

    # Print results
    print("Adjoint FOM: {}; FD FOM: {}".format(fom, avg))
    print("Adjoint gradient: {}; FD gradient: {}".format(f_u, fd_gradient))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Script params
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--random_strength", type=float, default=0.1)

    # General params
    parser.add_argument("--wavelength", help="wavelength (µm)", type=float, default=1.55)
    parser.add_argument("--cladding_width", help="cladding width (µm)", type=float, default=2.5)
    parser.add_argument("--cladding_thickness", help="cladding thickness (µm)", type=float, default=2.5)
    parser.add_argument("--core_index", help="core index", type=float, default=None)
    parser.add_argument("--cladding_index", help="cladding index", type=float, default=None)
    parser.add_argument("--mesh", help="mesh", type=int, default=128)

    # Waveguide params
    parser.add_argument("--width", help="width (µm)", type=float, default=0.5)
    parser.add_argument("--thickness", help="thickness (µm)", type=float, default=0.22)
    parser.add_argument("--length", help="length (µm)", type=float, default=1.0)
    parser.add_argument("--num_modes", help="number of modes", type=int, default=4)
    parser.add_argument("--num_params", help="number of parameters", type=int, default=1)
    parser.add_argument("--num_slabs", help="mesh_z", type=int, default=3)
    parser.add_argument("--input_width", help="input width (µm)", type=float, default=0.46)
    parser.add_argument("--output_width", help="output width (µm)", type=float, default=0.54)

    # Simulation and Optimization params
    parser.add_argument("--quiet", help="quiet", type=bool, default=False)
    parser.add_argument("--mesh_z", help="number of points in z", type=int, default=100)
    parser.add_argument("--dp", help="finite difference size", type=float, default=1e-6)

    args = parser.parse_args()
    main(**args)