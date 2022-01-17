import emepy
from emepy.lumerical import MSLumerical, LumEME
from emepy.mode import Mode
from emepy.eme import Layer
import numpy as np
from matplotlib import pyplot as plt
from emepy.tools import Si, SiO2
from emepy.tools import create_polygon, get_epsfunc

# Geometric parameters
wavelength = 1.55e-6  # Wavelength
num_inputs = 1
num_outputs = 2

input_width = 0.4e-6
output_width = 0.4e-6
input_gap = 2e-6
output_gap = 2e-6
input_length = 2e-6
output_length = 2e-6
mmi_width = 5e-6
mmi_length = 5e-6
input_taper_width = 1e-6
input_taper_length = 0.5e-6
input_taper_num_steps = 0
output_taper_width = 1e-6
output_taper_length = 0.5e-6
output_taper_num_steps = 0

thickness = 0.22e-6
mesh = 100  # Number of mesh points
num_pml_layers = int(mesh / 8.0)
core_index = Si(wavelength * 1e6)  # Silicon core
cladding_index = SiO2(wavelength * 1e6)  # Oxide cladding
num_modes_mmi = 40
num_modes_input = 20
num_modes_output = 20
x = np.linspace(-4e-6, 4e-6, mesh)
y = x.copy()
PML = True
masks=[]

# Create simulation
eme = LumEME()

# Create source input
polygons = []
for inp in range(num_inputs):
    starting_center = -0.5 * (num_inputs - 1) * (input_gap + input_width)
    n_input = np.ones(mesh) * cladding_index
    center = starting_center + inp * (input_gap + input_width)
    left_edge = center - 0.5 * input_width
    right_edge = center + 0.5 * input_width
    n_input = np.where((left_edge <= x) * (x <= right_edge), core_index, n_input)
    masks.append((left_edge <= x) * (x <= right_edge))

    eps = get_epsfunc(
        width=None,
        thickness=thickness,
        cladding_width=8e-6,
        cladding_thickness=8e-6,
        core_index=core_index,
        cladding_index=cladding_index,
        profile=n_input,
        nx=x
    )(x, y)

    polygons.append(create_polygon(x, y, eps, detranslate=False))

input_channel = MSLumerical(
    wavelength,
    mode=eme.mode,
    thickness=thickness,
    core_index=core_index,
    cladding_index=cladding_index,
    cladding_width=8e-6,
    cladding_thickness=8e-6,
    num_modes=num_modes_input,
    mesh=mesh,
    x=x,
    y=x,
    polygons=polygons[:],
    PML=PML,
)

eme.add_layer(Layer(input_channel, num_modes_input, wavelength, input_length))


# Create taper into MMI
for i in range(input_taper_num_steps):
    polygons = []
    for out in range(num_inputs):
        starting_center = -0.5 * (num_inputs - 1) * (input_gap + input_width)
        n_input = np.ones(mesh) * cladding_index
        center = starting_center + out * (input_gap + input_width)
        width = input_width + (input_taper_width - input_width) * (i / input_taper_num_steps)
        left_edge = center - 0.5 * width
        right_edge = center + 0.5 * width
        n_input = np.where((left_edge <= x) * (x <= right_edge), core_index, n_input)

        eps = get_epsfunc(
            width=None,
            thickness=thickness,
            cladding_width=8e-6,
            cladding_thickness=8e-6,
            core_index=core_index,
            cladding_index=cladding_index,
            profile=n_input,
            nx=x
        )(x, y)

        polygons.append(create_polygon(x, y, eps,detranslate=False))

    input_taper_channel = MSLumerical(
        wavelength,
        mode=eme.mode,
        thickness=thickness,
        cladding_width=8.0e-6,
        cladding_thickness=8.0e-6,
        cladding_index=cladding_index,
        num_modes=num_modes_input,
        mesh=mesh,
        x=x,
        y=x,
        polygons=polygons[:],
        PML=PML,
    )

    eme.add_layer(Layer(input_taper_channel, num_modes_input, wavelength, input_taper_length / input_taper_num_steps))

# Create mmi middle
n_middle = np.ones(mesh) * cladding_index
left_edge = -0.5 * mmi_width
right_edge = 0.5 * mmi_width
n_middle = np.where((left_edge <= x) * (x <= right_edge), core_index, n_middle)


eps = get_epsfunc(
            width=None,
            thickness=thickness,
            cladding_width=8e-6,
            cladding_thickness=8e-6,
            core_index=core_index,
            cladding_index=cladding_index,
            profile=n_middle,
            nx=x
        )(x, y)

polygons = [create_polygon(x, y, eps,detranslate=False)]


middle = MSLumerical(
    wavelength,
    mode=eme.mode,
    cladding_width=8.0e-6,
    cladding_thickness=8.0e-6,
    thickness=thickness,
    cladding_index=cladding_index,
    num_modes=num_modes_mmi,
    mesh=mesh,
    x=x,
    y=x,
    polygons=polygons[:],
    PML=PML,
)

eme.add_layer(Layer(middle, num_modes_mmi, wavelength, mmi_length))


# Create taper out of MMI
for i in range(output_taper_num_steps)[::-1]:
    polygons = []
    for out in range(num_outputs):
        starting_center = -0.5 * (num_outputs - 1) * (output_gap + output_width)
        n_output = np.ones(mesh) * cladding_index
        center = starting_center + out * (output_gap + output_width)
        width = output_width + (output_taper_width - output_width) * (i / output_taper_num_steps)
        left_edge = center - 0.5 * width
        right_edge = center + 0.5 * width
        n_output = np.where((left_edge <= x) * (x <= right_edge), core_index, n_output)
        
        eps = get_epsfunc(
            width=None,
            thickness=thickness,
            cladding_width=8e-6,
            cladding_thickness=8e-6,
            core_index=core_index,
            cladding_index=cladding_index,
            profile=n_output,
            nx=x
        )(x, y)

        polygons.append(create_polygon(x, y, eps,detranslate=False))

    output_taper_channel = MSLumerical(
        wavelength,
        mode=eme.mode,
        thickness=thickness,
        cladding_index=cladding_index,
        num_modes=num_modes_output,
        cladding_width=8.0e-6,
        cladding_thickness=8.0e-6,
        mesh=mesh,
        x=x,
        y=x,
        polygons=polygons[:],
        PML=PML,
    )


    eme.add_layer(
        Layer(output_taper_channel, num_modes_output, wavelength, output_taper_length / output_taper_num_steps)
    )


# Create output
polygons = []
for out in range(num_outputs):
    starting_center = -0.5 * (num_outputs - 1) * (output_gap + output_width)
    n_output = np.ones(mesh) * cladding_index
    center = starting_center + out * (output_gap + output_width)
    left_edge = center - 0.5 * output_width
    right_edge = center + 0.5 * output_width
    n_output = np.where((left_edge <= x) * (x <= right_edge), core_index, n_output)
    masks.append((left_edge <= x) * (x <= right_edge))

    eps = get_epsfunc(
        width=None,
        thickness=thickness,
        cladding_width=8e-6,
        cladding_thickness=8e-6,
        core_index=core_index,
        cladding_index=cladding_index,
        profile=n_output,
        nx=x
    )(x, y)

    polygons.append(create_polygon(x, y, eps,detranslate=False))

output_channel = MSLumerical(
    wavelength,
    mode=eme.mode,
    thickness=thickness,
    cladding_index=cladding_index,
    cladding_width=8.0e-6,
    cladding_thickness=8.0e-6,
    num_modes=num_modes_output,
    mesh=mesh,
    x=x,
    y=x,
    polygons=polygons[:],
    PML=PML,
)

eme.add_layer(Layer(output_channel, num_modes_output, wavelength, output_length))


# # Add a monitor
monitor = eme.add_monitor(axes="xyz", mesh_z=200)

# # Visualize the layout
# # plt.figure()
# # eme.draw()
# # plt.show()

input_array = np.zeros(num_modes_input+num_modes_output)
input_array[0] = 1
eme.propagate(input_array=input_array)#

plt.figure()
monitor.visualize(axes="xz", component="n")
plt.show()

plt.figure()
monitor.visualize(axes="xz", component="Hy")
plt.show()

plt.figure()
monitor.visualize(axes="xz", component="E")
plt.show()


in_mask = get_epsfunc(
    width=None,
    thickness=thickness,
    cladding_width=8e-6,
    cladding_thickness=8e-6,
    core_index=1,
    cladding_index=0,
    profile=masks[0],
    nx=x
)(x, y)

x_, y_, inp = monitor.get_array(axes="xy", location=0.5e-6, component="E")
power_in = np.sum(in_mask * inp[num_pml_layers:-num_pml_layers+1,num_pml_layers:-num_pml_layers+1])

x_, y_, outp = monitor.get_array(axes="xy", location=0.5e-6, component="E")
for i in range(num_outputs)[::-1]:
    out_mask = get_epsfunc(
        width=None,
        thickness=thickness,
        cladding_width=8e-6,
        cladding_thickness=8e-6,
        core_index=1,
        cladding_index=0,
        profile=masks[1+i],
        nx=x
    )(x, y)
    power_out = np.sum(out_mask * outp[num_pml_layers:-num_pml_layers+1,num_pml_layers:-num_pml_layers+1]) / power_in
    print("Power {}: {}".format(i, power_out))
