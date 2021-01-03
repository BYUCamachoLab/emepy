# EMEpy Complete User Library

## Tools

    def get_epsfunc(
        width, 
        thickness, 
        cladding_width, 
        cladding_thickness, 
        core_index, 
        cladding_index
    )

- **`width` []**
- **`thickness` []**
- **`cladding_width` []**
- **`cladding_thickness` []**
- **`core_index` []**
- **`cladding_index` []**

---

    def Si(wavelength)

- **`wavelength` []**

---

    def SiO2(wavelength)

- **`wavelength` []**

## Modesolvers

        class ModeSolver_EMpy(object):

ModeSolver_EMpy is based on the electromagnetic python module. It's open-source and fairly easy to use. This is the recommended class for users who want to use a finite difference solver. 

    def __init__(
        self,
        wl,
        width,
        thickness,
        num_modes=1,
        cladding_width=2.5e-6,
        cladding_thickness=2.5e-6,
        core_index=None,
        cladding_index=None,
        x=None,
        y=None,
        mesh=300,
        accuracy=1e-8,
        boundary="0000",
        epsfunc=None,
    )

- **`wl` []**
- **`width` []**
- **`thickness` []**
- **`num_modes` []**
- **`cladding_width` []**
- **`cladding_thickness` []**
- **`core_index` []**
- **`cladding_index` []**
- **`x` []**
- **`y` []**
- **`mesh` []**
- **`accuracy` []**
- **`boundary` []**
- **`epsfunc` []**

---

        class ModeSolver_Lumerical(object):

ModeSolver_Lumerical requires the Lumerical API. Licensing for such is not free. Therefore users are encouraged to use the other classes which work just as well. 

    def __init__(
        self,
        wl,
        width,
        thickness,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        core_index=None,
        cladding_index=None,
        mesh=300,
        lumapi_location=None,
    )

- **`wl` []**
- **`width` []**
- **`thickness` []**
- **`num_modes` []**
- **`cladding_width` []**
- **`cladding_thickness` []**
- **`core_index` []**
- **`cladding_index` []**
- **`mesh` []**
- **`lumapi_location` []**

---

        class ModeSolver_Pickle(object):

ModeSolver_Pickle simply uses the pickle library to open files with presaved field profiles and effective indices. This requires no mode solving during the EME process, however requires saved fields beforehand. 

    def __init__(
        self, 
        filename, 
        index=None, 
        width=None, 
        thickness=None
    )

- **`filename` []**
- **`index` []**
- **`width` []**
- **`thickness` []**

---

        class ModeSolver_ANN(object):

ModeSolver_ANN is an example class for what users may design if they chose to use neural networks to generate modes. This is computationally the fasted of the solvers themselves, but requires a pretrained network. 

    def __init__(
        self,
        wavelength,
        width,
        thickness,
        sklearn_save,
        torch_save_x,
        torch_save_y,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        x=None,
        y=None,
    )

- **`wavelength` []**
- **`width` []**
- **`thickness` []**
- **`sklearn_save` []**
- **`torch_save_x` []**
- **`torch_save_y` []**
- **`num_modes` []**
- **`cladding_width` []**
- **`cladding_thickness` []**
- **`x` []**
- **`y` []**

## EME Simulation

        class EME(object):

    def __init__(
        self, 
        keep_modeset=False
    )

- **`keep_modeset` []**


---

        class PeriodicEME(object):

    def __init__(
        self, 
        layers, 
        num_periods
    )

- **`layers` []**
- **`num_periods` []**

---

        class Layer(object):

    def __init__(
        self, 
        mode_solvers, 
        num_modes, 
        wavelength, 
        length
    )   

- **`mode_solvers` []**
- **`num_modes` []**
- **`wavelength` []**
- **`length` []**

---

