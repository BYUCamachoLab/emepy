import emepy as em
from emepy import optimization
from emepy.geometries import DynamicRect2D, EMpyGeometryParameters, Waveguide
from emepy import EME
from emepy.optimization import Optimization

import numpy as np
import unittest
from utils import ApproxComparisonTestCase
from matplotlib import pyplot as plt

# Resolution
mesh = 100
num_layers = 6
num_params = 30
num_modes = 10
parallel = False
mesh_z = 100

# Materials
matSi = em.Si(1.55)
matSiO2 = em.SiO2(1.55)

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
    rect_params, width=1.0, thickness=0.22, length=0.5, center=(0, 0), num_modes=num_modes
)

# Create an output waveguide
output_waveguide = Waveguide(
    rect_params, width=1.75, thickness=0.22, length=0.5, center=(0, 0), num_modes=num_modes
)

# Create the design region geometry
dynamic_rect = DynamicRect2D(
    params=rect_params,
    width=input_waveguide.width,
    length=2,
    num_modes=num_modes,
    num_params=num_params,
    symmetry=True,
    subpixel=True,
    mesh_z=num_layers,
    input_width=input_waveguide.width,
    output_width=output_waveguide.width,
)

# Define geometry
geometry =  [input_waveguide, dynamic_rect, output_waveguide]

# Create the EME and Optimization
eme = EME(quiet=False, parallel=parallel, mesh_z=mesh_z)
optimizer = Optimization(eme, geometry, mesh_z=mesh_z)

## ensure reproducible results
seed = 9861548
rng = np.random.RandomState(seed)

## start with a linear taper
design_x, design_z = optimizer.get_design_readable()
design_x, design_z = np.array(design_x), np.array(design_z)
linear_taper = (
    np.linspace(input_waveguide.width, output_waveguide.width, len(design_x) + 2)[
        1:-1
    ]
    / 2.0
)
design_x[:] = linear_taper[:]

## vertex constraints
ul_x = 1.2 * design_x
ll_x = 0.8 * design_x
ul_z = 1.05 * design_z
ll_z = 0.95 * design_z

## random design region
design_x_i = (ul_x-ll_x)*rng.rand(design_x.shape[0]) + ll_x
design_z_i = (ul_z-ll_z)*rng.rand(design_z.shape[0]) + ll_z

## random epsilon perturbation for design region
deps = 1e-5
dp_x = deps*rng.rand(design_x.shape[0])
dp_z = deps*rng.rand(design_z.shape[0])
dp = np.zeros(design_x.shape[0]+design_z.shape[0])
dp[::2] = dp_x
dp[1::2] = dp_z


def forward_simulation(design_x, design_y):

    # Get geometry
    design = np.zeros(len(design_x) + len(design_y))
    design[::2] = design_x
    design[1::2] = design_y
    dynamic_rect.set_design(design.tolist())
    geometry =  [input_waveguide, dynamic_rect, output_waveguide]
    layers = [l for g in geometry for l in g]

    # Run the simulation
    eme = EME(layers=layers, quiet=False, parallel=parallel, mesh_z=mesh_z)
    eme.propagate()

    # Get overlap
    f0 = np.abs(em.ModelTools.compute(eme.network, {"left0":1})["right0"]) ** 2

    return f0
    

def adjoint_solver(design_x, design_y):

    # Set the design region
    optimizer.set_design_readable(design_x, None, design_y)

    # Run the simulation
    f0, dJ_du, _ = optimizer.optimize(optimizer.get_design(), dp=deps)

    return f0, dJ_du


class TestAdjointSolver(ApproxComparisonTestCase):

    def test_adjoint_solver_overlap(self):

        if True:
            return

        print("*** TESTING OVERLAP ADJOINT ***")

        ## compute gradient using adjoint solver
        adjsol_obj, adjsol_grad = adjoint_solver(design_x_i, design_z_i)

        ## compute unperturbed |Ez|^2
        Ez2_unperturbed = forward_simulation(design_x_i, design_z_i)

        ## compare objective results
        print("|Ez|^2 -- adjoint solver: {}, traditional simulation: {}".format(adjsol_obj,Ez2_unperturbed))
        self.assertClose(adjsol_obj,Ez2_unperturbed,epsilon=1e-6)

        ## compute perturbed Ez2
        Ez2_perturbed = forward_simulation(design_x + dp_x, design_z + dp_z)

        ## compare gradients
        if adjsol_grad.ndim < 2:
            adjsol_grad = np.expand_dims(adjsol_grad,axis=1)
        adj_scale = (dp[None,:]@adjsol_grad).flatten()
        fd_grad = Ez2_perturbed-Ez2_unperturbed
        print("Directional derivative -- adjoint solver: {}, FD: {}".format(adj_scale,fd_grad))
        tol = 0.006
        self.assertClose(adj_scale,fd_grad,epsilon=tol)

    def test_adjoint_solver_gradient(self):

        if False:
            return

        print("*** TESTING GRADIENT ADJOINT ***")

        ## compute gradient using adjoint solver
        f0, dJ_du = adjoint_solver(design_x_i, design_z_i)

        ## finite difference gradient
        dp = 1e-3
        num_gradients = 10
        optimizer.set_design_readable(design_x_i, None, design_z_i)
        g_discrete, idx = optimizer.calculate_fd_gradient(num_gradients=num_gradients,dp=dp,rand=rng)

        # compare gradients
        (m, b) = np.polyfit(np.squeeze(g_discrete), dJ_du[idx], 1)

        # plot results
        min_g = np.min(g_discrete)
        max_g = np.max(g_discrete)

        fig = plt.figure(figsize=(12,6))

        plt.subplot(1,2,1)
        plt.plot([min_g, max_g],[min_g, max_g],label='y=x comparison')
        plt.plot([min_g, max_g],[m*min_g+b, m*max_g+b],'--',label='Best fit')
        plt.plot(g_discrete,dJ_du[idx],'o',label='Adjoint comparison')
        plt.xlabel('Finite Difference Gradient')
        plt.ylabel('Adjoint Gradient')
        plt.legend()
        plt.grid(True)
        plt.axis("square")

        plt.subplot(1,2,2)
        rel_err = np.abs(np.squeeze(g_discrete) - np.squeeze(dJ_du[idx])) / np.abs(np.squeeze(g_discrete)) * 100
        plt.semilogy(g_discrete,rel_err,'o')
        plt.grid(True)
        plt.xlabel('Finite Difference Gradient')
        plt.ylabel('Relative Error (%)')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Resolution: {} Seed: {} Np: {}'.format(mesh,seed,num_params))
        plt.show()

        


if __name__ == '__main__':
    unittest.main()
