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

- **`width` [number]** The width of the cross sectional core (m).
- **`thickness` [number]** The thickness of the cross sectional core (m).
- **`cladding_width` [number]** The width of the cross sectional cladding (m).
- **`cladding_thickness` [number]** The thickness of the cross sectional cladding (m).
- **`core_index` [number]** Index of refraction of the cross sectional core.
- **`cladding_index` [number]** Index of refraction of the cross sectional cladding.

The `get_epsfunc` function takes in a geometry and index of refraction for core and cladding of a simple rectangular waveguide and outputs another function. This new function can be used to extract the cross sectional square of the index of refraction for a given x,y space. This is only necessary for the EMpy modesolver, which is handled in the backend. Therefore users do not need to use this function, but can if they wish. 

**Example**

    from emepy.tools import get_epsfunc
    from matplotlib import pyplot as plt
    import numpy as np

    index_func = get_epsfunc(
        width = .5e-6, 
        thickness = .22e-6, 
        cladding_width = 5e-6, 
        cladding_thickness = 5e-6, 
        core_index = np.sqrt(3.5), 
        cladding_index = np.sqrt(1.4)
    )

    x = np.linspace(0, 5e-6, 128)
    y = np.linspace(0, 5e-6, 128)

    index = index_func(x,y)

    plt.imshow(np.real(index),extent=[0, 5, 0, 5])
    plt.colorbar()
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.title('index')
    plt.show()

![](images/eps_func.png)

---

    def Si(wavelength)

- **`wavelength` [number]** The wavelength of light propagating through silicon (µm).

The `Si` function provides an index of refraction in silicon given a specific wavelength. The function uses a regression on a dataset and is thus only valid for a range of wavelengths: (1.2µm - 14µm).

**Example**

    from emepy.tools import Si
    import numpy as np

    lambdas = np.linspace(1.5,1.6,10)
    index_array = [Si(i) for i in lambdas]

    print(index_array)

**Output**

    [3.4799, 3.478966666666667, 3.4780333333333333, 3.4771, 3.4761666666666664, 3.4752777777777775, 3.4744333333333333, 3.473588888888889, 3.4727444444444444, 3.4719]

---

    def SiO2(wavelength)

- **`wavelength` [number]** The wavelength of light propagating through silicon dioxide (glass) (µm).

The `SiO2` function provides an index of refraction in silicon given a specific wavelength. The function uses a regression on a dataset and is thus only valid for a range of wavelengths: (0.21µm - 6.7µm).

**Example**

    from emepy.tools import SiO2
    import numpy as np

    lambdas = np.linspace(1.5,1.6,10)
    index_array = [SiO2(i) for i in lambdas]

    print(index_array)

**Output**

    [1.4446167941939, 1.4444864184114001, 1.4443539800979162, 1.444221362433912, 1.4440887447699078, 1.4439561271059036, 1.4438231150141243, 1.4436878678316192, 1.4435526206491143, 1.4434173734666091]

## Mode

    class Mode(object):

`Mode` is a class that contains the information for an eigenmode. It stores Ex, Ey, Ez, Hx, Hy, Hz, and neff (the effective index). 

    def __init__(
        self, 
        x, 
        y, 
        wl, 
        neff, 
        Hx, 
        Hy, 
        Hz, 
        Ex, 
        Ey, 
        Ez
    )

- **`x` [list [numbers]]** List of positions in the x direction. 
- **`y` [list [numbers]]** List of positions in the y direction. 
- **`wl` [number]** Wavelength of eigenmode to solve for (m).
- **`neff` [number]** Effective index of the eigenmode. 
- **`Hx` [ndarray]** len(x) x len(y) matrix representing the Hx field. 
- **`Hy` [ndarray]** len(x) x len(y) matrix representing the Hy field. 
- **`Hz` [ndarray]** len(x) x len(y) matrix representing the Hz field. 
- **`Ex` [ndarray]** len(x) x len(y) matrix representing the Ex field. 
- **`Ey` [ndarray]** len(x) x len(y) matrix representing the Ey field. 
- **`Ez` [ndarray]** len(x) x len(y) matrix representing the Ez field. 

**Methods**

    def plot(self, value_type="Real", colorbar=True):

- **`value_type ` [string]** Operation of field to look at. Options: 'Real', 'Imaginary', 'Abs', 'Abs^2'. [default: 'Real']
- **`colorbar` [bool]** If True, includes a colorbar on each subplot. [default: True]

Plots all 6 fields into subplots with pyplot.

    def inner_product(self, mode2):

- **`mode2 ` [Mode]** A second mode to overlap with. 

Takes the inner product (overlap) with another mode. If both fields are normalized to power 1, then an overlap of 1 means perfect overlap and an overlap of 0 means no overlap. 

    def normalize(self):

Normalizes the Mode to power 1. 

    def zero_phase(self):

Changes the phase such that the z components are all imaginary and the xy components are all real. 

    def get_fields(self):

Returns an array [self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez].

    def get_H(self):

Returns an array [self.Hx, self.Hy, self.Hz].

    def get_E(self):

Returns an array [self.Ex, self.Ey, self.Ez].

    def get_neff(self):

Returns the effective index as a complex number. 

    def get_wavelength(self):

Returns the wavelength. 

    def save(self, path=None):

- **`path` [string]** The path (including name) to save the file. [default: "./ModeObject_" + str(random.random())]

Saves the `Mode` object into a pickled file. Can be reloaded using the `ModeSolver_Pickle` class.

**Example**

## ModeSolvers

    class ModeSolver_EMpy(object):

`ModeSolver_EMpy` is based on the electromagnetic python module. It's open-source and fairly easy to use. This is the recommended class for users who want to use a finite difference solver. 

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

- **`wl` [number]** Wavelength of eigenmode to solve for (m).
- **`width` [number]** Width of the core in the rectangular cross section (m). 
- **`thickness` [number]** Thickness of the core in the rectangular cross section (m). 
- **`num_modes` [int]** Number of modes to solve for. [default: 1]
- **`cladding_width` [number]** Width of the cladding in the rectangular cross section (m). [default: 2.5e-6]
- **`cladding_thickness` [number]** Thickness of the cladding in the rectangular cross section (m). [default: 2.5e-6]
- **`core_index` [number]** Index of refraction of the cross sectional core. [default: Si(wl*1e6)]
- **`cladding_index` [number]** Index of refraction of the cross sectional cladding. [default: SiO2(wl*1e6)]
- **`x` [list [numbers]]** List of positions in the x direction. [default: np.linspace(0,cladding_width,mesh)]
- **`y` [list [numbers]]** List of positions in the y direction. [default: np.linspace(0,cladding_thickess,mesh)]
- **`mesh` [int]** If provided, provides an equally spaced x,y grid with `mesh` number of points.  [default: 300]
- **`accuracy` [number]** Accuracy of the EMpy solver, smaller `accuracy` is more accurate. [default: 1e-8]
- **`boundary` [string]** EMpy boundary type "NESW". Users should only change if they've read the EMpy documentation for boundaries. [default: "0000"]
- **`epsfunc` [function]** Function that provides a mapping of the index of refraction based on a given grid. See `get_epsfunc`. [default: get_epsfunc(width, thickness, cladding_width, cladding_thickness, core_index, cladding_index)]

**Methods**

    def solve(self):

Calls the `ModeSolver` to actually solve for the modes. 

    def clear(self):

Clear the modes inside the `ModeSolver` to open memory. 

    def get_mode(self, mode_num=0):

Get the nth order `Mode` given by mode_num.

- **`mode_num` [int]** The index of the nth order `Mode` to get. 

**Example**

    from emepy.FD_modesolvers import ModeSolver_EMpy
    from emepy.mode import Mode
    from matplotlib import pyplot as plt

    modesolver = ModeSolver_EMpy(
        wl=1.55e-6,
        width=.5e-6,
        thickness=.22e-6,
        mesh = 128
    )

    modesolver.solve()
    mode = modesolver.get_mode()

    plt.figure()
    mode.plot()
    plt.show()

![](images/plot_mode.png)


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

- **`wl` [number]** Wavelength of eigenmode to solve for (m).
- **`width` [number]** Width of the core in the rectangular cross section (m). 
- **`thickness` [number]** Thickness of the core in the rectangular cross section (m). 
- **`num_modes` [int]** Number of modes to solve for. [default: 1]
- **`cladding_width` [number]** Width of the cladding in the rectangular cross section (m). [default: 2.5e-6]
- **`cladding_thickness` [number]** Thickness of the cladding in the rectangular cross section (m). [default: 2.5e-6]
- **`core_index` [number]** Index of refraction of the cross sectional core. [default: Si(wl*1e6)]
- **`cladding_index` [number]** Index of refraction of the cross sectional cladding. [default: SiO2(wl*1e6)]
- **`mesh` [int]** If provided, provides an equally spaced x,y grid with `mesh` number of points.  [default: 300]
- **`lumapi_location` [string]** If the Lumerical Python API is not already in the user's path, they may add the path here. [default: None]. Ubuntu example: "/opt/lumerical/v202/api/python" .

**Methods**

    def solve(self):

Calls the `ModeSolver` to actually solve for the modes. 

    def clear(self):

Clear the modes inside the `ModeSolver` to open memory. 

    def get_mode(self, mode_num=0):

Get the nth order `Mode` given by mode_num.

- **`mode_num` [int]** The index of the nth order `Mode` to get. 

**Example**

    from emepy.FD_modesolvers import ModeSolver_Lumerical
    from emepy.mode import Mode
    from matplotlib import pyplot as plt

    modesolver = ModeSolver_Lumerical(
        wl=1.55e-6,
        width=.5e-6,
        thickness=.22e-6,
        mesh = 128,
        lumapi_location = "/opt/lumerical/v202/api/python"
    )

    modesolver.solve()
    mode = modesolver.get_mode()

    plt.figure()
    mode.plot(value_type="abs^2", colorbar=False)
    plt.show()

![](images/plot_lumapi.png)


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

- **`filename` [string]** Location of where the pickled file is located. 
- **`index` [int]** If the pickle file has an list of Modes saved as opposed to the default singular Mode saved, provide an index of the list for which Mode the user wants. 
- **`width` [number]** Width of the core in the rectangular cross section (m). Only used for drawing EME geometry. Optional.
- **`thickness` [number]** Thickness of the core in the rectangular cross section (m). Only used for drawing EME geometry. Optional.

**Methods**

    def solve(self):

Calls the `ModeSolver` to actually solve for the modes. 

    def clear(self):

Clear the modes inside the `ModeSolver` to open memory. 

    def get_mode(self, mode_num=0):

Get the nth order `Mode` given by mode_num.

- **`mode_num` [int]** The index of the nth order `Mode` to get. 

**Example**

    from emepy.FD_modesolvers import ModeSolver_EMpy
    from emepy.FD_modesolvers import ModeSolver_Pickle
    from emepy.mode import Mode
    from matplotlib import pyplot as plt
    import pickle as pk

    modesolver = ModeSolver_EMpy(wl=1.55e-6, width=0.5e-6, thickness=0.22e-6, mesh=128)
    modesolver.solve()
    mode = modesolver.get_mode()
    pk.dump(mode, open("./example_file.pk", "wb+"))

    # Separate instance

    modesolver = ModeSolver_Pickle(filename="./example_file.pk", width=0.5e-6, thickness=0.22e-6)

    modesolver.solve()
    mode = modesolver.get_mode()

    plt.figure()
    mode.plot(value_type="Imaginary")
    plt.show()

![](images/plot_pickle.png)


---

    class ModeSolver_ANN(object):

ModeSolver_ANN is an example class for what users may design if they chose to use neural networks to generate modes. This is computationally the fasted of the solvers themselves, but requires a pretrained network. **Currently being finished, users should use the other ModeSolvers for now. **

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

- **`wl` [number]** Wavelength of eigenmode to solve for (m).
- **`width` [number]** Width of the core in the rectangular cross section (m). 
- **`thickness` [number]** Thickness of the core in the rectangular cross section (m). 
- **`sklearn_save` [string]** Sklearn save location for the effective index polynomial regression. [default: ?]
- **`torch_save_x` [string]** Pytorch save location for the Hx field neural network. [default: ?]
- **`torch_save_y` [string]** Pytorch save location for the HY field neural network. [default: ?]
- **`num_modes` [int]** Number of modes to solve for. (Don't change for this specific set of networks.) [default: 1]
- **`cladding_width` [number]** Width of the cladding in the rectangular cross section (m). (Don't change for this specific set of networks.) [default: 2.5e-6]
- **`cladding_thickness` [number]** Thickness of the cladding in the rectangular cross section (m). (Don't change for this specific set of networks.) [default: 2.5e-6]
- **`x` [list [numbers]]** List of positions in the x direction. [default: np.linspace(0,cladding_width,mesh)]
- **`y` [list [numbers]]** List of positions in the y direction. [default: np.linspace(0,cladding_thickess,mesh)]

**Methods**

    def solve(self):

Calls the `ModeSolver` to actually solve for the modes. 

    def clear(self):

Clear the modes inside the `ModeSolver` to open memory. 

    def get_mode(self, mode_num=0):

Get the nth order `Mode` given by mode_num.

- **`mode_num` [int]** The index of the nth order `Mode` to get. 

**Example**

## EME Simulation

    class EME(object):

The `EME` class is the heart of the package. It provides the algorithm that cascades sections modes together to provide the s-parameters for a geometric structure. The object is dependent on the `Layer` objects that are fed inside. 

    def __init__(
        self, 
        layers=[],
        keep_modeset=False
    )

- **`layers` [list [`Layer`]]** An list of Layer objects, arranged in the order they belong geometrically. [default: []]
- **`keep_modeset` [bool]** If true, will keep the first and last layers' modes in the system after the simulation is complete. This is utilized by the `PeriodicEME` object. It can also be used by users who wish to examine the first of last modes after the simulation is complete, but usually this is kept false. [default: False].

**Methods**

    def add_layer(self, layer):

The `add_layer` method will add a `Layer` object to the `EME` object. The object will be geometrically added to the very right side of the structure. Using this method after `propagate` is useless as the solver has already been called. 

- **`layer` [`Layer`]** `Layer` object to be appended to the list of `Layers` inside the EME object. 

    def propagate(self):

The `propagate` method should be called once all `Layer` objects have been added. This method will call the `EME` solver and produce s-parameters. 

    def s_parameters(self):

The `s_parameters` method returns an MxN numpy array of s-parameters where each index representing the transmission/reflection from mode N to mode M. M = Number of output modes on the right + Number of output modes on the left. M = Number of input modes on the right + Number of input modes on the left. For each, the order of modes as they correspond to the port number are the left modes from most fundamental to least, and then the right modes. 

    def draw(self):

The `draw` method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. 

**Example**

---

    class PeriodicEME(object):

The `PeriodicEME` class works similarly to the `EME` class. Users specificy the number of periods of repeated geometry and the solving process will significatnly decrease in time when compared to running a full simulation using `EME`. 

    def __init__(
        self, 
        layers=[], 
        num_periods=1
    )

- **`layers` [list [`Layer`]]** An list of Layer objects, arranged in the order they belong geometrically. [default: []]
- **`num_periods` [int]** Number of periods in the repeated geometry. If num_periods == 1, `PeriodicEME` becomes the same as `EME`. [default: 1]

**Methods**

    def add_layer(self, layer):

The `add_layer` method will add a `Layer` object to the `EME` object. The object will be geometrically added to the very right side of the structure. Using this method after `propagate` is useless as the solver has already been called. 

- **`layer` [`Layer`]** `Layer` object to be appended to the list of `Layers` inside the EME object. 

    def propagate(self):

The `propagate` method should be called once all `Layer` objects have been added. This method will call the `EME` solver and produce s-parameters. 

    def s_parameters(self):

The `s_parameters` method returns an MxN numpy array of s-parameters where each index representing the transmission/reflection from mode N to mode M. M = Number of output modes on the right + Number of output modes on the left. M = Number of input modes on the right + Number of input modes on the left. For each, the order of modes as they correspond to the port number are the left modes from most fundamental to least, and then the right modes. 

    def draw(self):

The `draw` method sketches a rough approximation for the xz geometry of the structure using pyplot where x is the width of the structure and z is the length. 

**Example**

---

    class Layer(object):

`Layer` objects form the building blocks inside of an `EME` or `PeriodicEME`. These represent geometric layers of rectangular waveguides that approximate continuous structures. 

    def __init__(
        self, 
        mode_solvers, 
        num_modes, 
        wavelength, 
        length
    )   

- **`mode_solvers` [list [`Modesolver`] or `Modesolver`]** List of `Modesolver` objects. Should be in order from fundamental mode to least significant mode. If singular `Modesolver`, can leave alone without a list. 
- **`num_modes` [int]** Number of total modes for the layer.
- **`wavelength` [number]** Wavelength of eigenmode to solve for (m).
- **`length` [number]** Geometric length of the Layer (m). The length affects the phase of the eigenmodes inside the layer via the complex phasor $e^(jβz)$.

**Example**

---

