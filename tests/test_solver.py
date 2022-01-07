from emepy.fd import MSEMpy  # Open source


def test_solver():
    # Create a modesolver object that represents a waveguide cross section
    fd_solver = MSEMpy(
        1.55e-6,  # Set the wavelength of choice
        0.46e-6,  # Define the width of the waveguide
        0.22e-6,  # Define the thickness of the waveguide
        mesh=128,  # Set the mesh density
        num_modes=1,  # Set the number of modes to solve for
        cladding_width=2.5e-6,  # Define the cladding width (default is same as ANN)
        cladding_thickness=2.5e-6,  # Define the cladding thickness (default is same as ANN)
        core_index=None,  # Define the core refractive index (leave None (default) for Silicon)
        cladding_index=None,  # Define the cladding refractive index (leave None (default) for Silicon Dioxide)
    )

    # Solve for the fundamental Eigenmode
    fd_solver.solve()
    fd_mode = fd_solver.get_mode()
    assert fd_mode
