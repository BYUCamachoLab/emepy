import numpy as np
import argparse
from emepy.geometries import EMpyGeometryParameters, DynamicRect2D, Waveguide
from emepy.eme import EME
from emepy.optimization import Optimization
from emepy.materials import Si, SiO2
from matplotlib import pyplot as plt

def get_geometry(**kwargs):

    # Get geometry parameters
    geometry_params = EMpyGeometryParameters(
        wavelength = kwargs["wavelength"],
        cladding_width = kwargs["cladding_width"],
        cladding_thickness = kwargs["cladding_thickness"],
        core_index = kwargs["core_index"] if kwargs["core_index"] is not None else Si(kwargs["wavelength"]),
        cladding_index = kwargs["cladding_index"] if kwargs["cladding_index"] is not None else SiO2(kwargs["wavelength"]),
        mesh = kwargs["mesh"],
    )

    # Input waveguide
    input_waveguide = Waveguide(
        params = geometry_params,
        width = kwargs["width"],
        thickness = kwargs["thickness"],
        length = 0.2,
        num_modes = kwargs["num_modes"]
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

    # Output waveguide
    output_waveguide = Waveguide(
        params = geometry_params,
        width = kwargs["width"],
        thickness = kwargs["thickness"],
        length = 0.2,
        num_modes = kwargs["num_modes"]
    )

    return [input_waveguide, waveguide, output_waveguide]

def get_simulation(geometry:list, **kwargs):

    # Create an EME
    layers = [i for geom in geometry for i in geom]
    eme = EME(
        layers = layers,
        num_periods = 1,
        quiet=kwargs["quiet"],
    )
    
    return eme

def get_optimization(geometries, eme, r, **kwargs):

    # Create an optimization
    optimization = Optimization(
        eme = eme,
        geometries = geometries,
        mesh_z = kwargs["mesh_z"],
        fom_location = kwargs["fom_location"],
        source_location = kwargs["source_location"],
    )

    # Initialize design
    x, z = optimization.get_design_readable()
    x = np.array(x)
    x[:] = x[:] * (1 + r) * kwargs["random_strength"]
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


def main(r, **kwargs):

    # Get geometry
    geometry = get_geometry(**kwargs)

    # Get simulation
    eme = get_simulation(geometry, **kwargs)

    # Get optimization
    optimization = get_optimization(geometry, eme, r, **kwargs)

    # Run adjoint run
    fom, f_u = run_optimization(optimization, **kwargs)
    return fom, f_u
    
    # # Run finite difference run
    # fd_gradient, fd_gradient_idx, avg = run_finite_difference(optimization, **kwargs)

    # # Print results
    # print("Adjoint FOM: {}; FD FOM: {}".format(fom, avg))
    # print("Adjoint gradient: {}; FD gradient: {}".format(f_u, fd_gradient))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Script params
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--random_strength", type=float, default=0.75)

    # General params
    parser.add_argument("--wavelength", help="wavelength (µm)", type=float, default=1.55)
    parser.add_argument("--cladding_width", help="cladding width (µm)", type=float, default=2.5)
    parser.add_argument("--cladding_thickness", help="cladding thickness (µm)", type=float, default=2.5)
    parser.add_argument("--core_index", help="core index", type=float, default=None)
    parser.add_argument("--cladding_index", help="cladding index", type=float, default=None)
    parser.add_argument("--mesh", help="mesh", type=int, default=200)

    # Waveguide params
    parser.add_argument("--width", help="width (µm)", type=float, default=0.5)
    parser.add_argument("--thickness", help="thickness (µm)", type=float, default=0.22)
    parser.add_argument("--length", help="length (µm)", type=float, default=1.0)
    parser.add_argument("--num_modes", help="number of modes", type=int, default=1)
    parser.add_argument("--num_params", help="number of parameters", type=int, default=1)
    parser.add_argument("--num_slabs", help="mesh_z", type=int, default=3)
    parser.add_argument("--input_width", help="input width (µm)", type=float, default=0.46)
    parser.add_argument("--output_width", help="output width (µm)", type=float, default=0.54)

    # Simulation and Optimization params
    parser.add_argument("--quiet", help="quiet", type=bool, default=True)
    parser.add_argument("--mesh_z", help="number of points in z", type=int, default=100)
    parser.add_argument("--dp", help="finite difference size", type=float, default=1e-6)
    parser.add_argument("--fom_location", help="fom location", type=float, default=1.3)
    parser.add_argument("--source_location", help="source location", type=float, default=0.1)

    args = parser.parse_args()
    d = vars(args)

    # Set random seed
    np.random.seed(d["random_seed"])
    r = 1.5#np.random.rand(d["num_params"])

    # for mesh in np.arange(60,500,20).astype(int):
    #     d["mesh"] = 60
    #     fom, f_u = main(r, **d)
    #     print("mesh: {}, fom: {}, f_u: {}".format(mesh, fom, f_u))


    # data = [
    #     [60, (0.9999940400831391+0j), [128850.37888565 , 17631.21707566]],
    #     [80, (0.9998958278398605+0j), [46846.61920109, 13087.76137277]],
    #     [100, (0.9999999032578732+0j),[-29404.68739404 , 36416.15770947]],
    #     [120, (0.0006355178729881811+0j), [22730.35867229 , 1030.23911606]],
    #     [140, (0.9999980037904592+0j), [1708082.06675059  , 19460.37182551]],
    #     [160, (0.9999989051977589+0j), [1924285.35242667  , 45249.84763741]],
    #     [180, (0.9999986659535003+0j), [1736422.28665092 , 101454.59937267]],
    #     [200, (0.9999998824684009+0j), [1194372.60113441 , 174634.95456064]],
    #     [220, (0.9999999999598641+0j), [419634.53901722, 257400.92918473]],
    #     [240, (0.29316091340179945+0j), [2305627.5982443   , 98102.48920399]],
    #     [260, (0.30696602643086107+0j), [2715243.29637686  , 80209.28460327]],
    #     [280, (0.04238400031904539+0j), [1113458.12572923  , 33531.28313765]],
    # ]

    # mesh = np.array([i[0] for i in data])
    # fom = np.array([i[1] for i in data])
    # gradient = np.array([i[2][0]+1j*i[2][1] for i in data])

    # plt.figure()
    # plt.plot(mesh, np.abs(gradient))
    # plt.xlabel("mesh")
    # plt.ylabel("gradient angle")
    # plt.show()


    # # num modes = 3
    nm3 = [
        [60, (0.9999555959879836+0j), [ 4345.64437559, -6719.33196686]],
        [80, (0.9999184352200832+0j), [  9220.07243116, -12652.08183507]],
        [100, (0.9999999262764483+0j), [ 12427.42513831, -21534.02893509]],
        [120, (0.9999963870532885+0j), [ 13457.49775298, -31528.60747992]],
        [140, (0.9999971894021131+0j), [ 19337.19170555, -42395.4642594 ]],
        [160, (0.9999985480817566+0j), [ 25321.28498604, -56145.91538109]],
        [180, (0.9999979784446504+0j), [ 31126.48958078, -70770.14290234]],
        [200, (0.9999998952492339+0j), [ 40647.60393154, -86783.16437394]],
        [220, (0.9999999929747491+0j), [  50371.54884826, -104570.94576274]],
        [240, (0.9999999998311053+0j), [  67924.8024761,  -136160.32824796]],
        [260, (0.9999992107424002+0j), [  76342.30131751, -148421.51966059]],
        [280, (0.9999995006803118+0j), [  83342.15306903, -173658.26277443]],
        [300, (0.9999989340734562+0j), [  93025.76468629, -199407.20057637]],
        [320, (0.9999999146803092+0j), [ 107438.25411438, -226485.9975513 ]],
        [340, (0.9999999542482377+0j), [ 118808.59698262, -255702.93652429]],
    ]
    mesh = np.array([i[0] for i in nm3])
    fom = np.array([i[1] for i in nm3])
    gradient = np.array([i[2][0]+1j*i[2][1] for i in nm3])

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(mesh, np.abs(gradient))
    plt.xlabel("mesh")
    plt.ylabel("gradient abs")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(mesh, np.angle(gradient))
    plt.xlabel("mesh")
    plt.ylabel("gradient angle")
    plt.grid()

    plt.show()


    # # num modes = 8
    # nm8 = [
    #     [60, (0.9999777736528301+0j), [ 3890.20894562, -5457.76510733]],
    #     [80, (0.9999223355255626+0j), [  8311.7909657,  -12105.70917772]],
    #     [100, (0.999999926096966+0j), [ 10982.19285304, -21177.49202131]],
    #     [120, (0.999997370550053+0j), [ 11406.61704207, -31616.24318961]],
    #     [140, (0.9999975679365479+0j), [ 16917.18391219, -43304.89726168]],
    #     [160, (1.000274119624781+0j), [ 20324.84446516, -55510.1808651 ]],
    #     [180, (1.0002689723462475+0j), [ 24866.29416851, -69500.66835061]],
    #     [200, (1.000282158186632+0j), [ 32804.11764687, -85008.89290317]],
    #     [220, (1.0003009887911398+0j), [  40529.35886499, -101800.42911281]],
    # ]