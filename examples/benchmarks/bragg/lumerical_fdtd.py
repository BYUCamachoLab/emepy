import lumapi as lm
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

def lumerical_fdtd(args):

    # open file
    if os.path.isfile("api.lms"):
        os.remove("api.lms")

    fdtd = lm.FDTD(hide=True)
    fdtd.save("api.lms")

    fdtd.deleteall()

    # define parameters
    vlength = args.length * 1e-6
    vwidth1 = args.width1 * 1e-6
    vwidth2 = args.width2 * 1e-6
    vthickness = args.thickness * 1e-6
    vnum_periods = args.num_periods
    vnum_wavelengths = args.num_wavelengths

    # define cladding
    fdtd.addrect()
    fdtd.set("name","cladding")
    fdtd.set("x",vlength*vnum_periods)
    fdtd.set("x span",4*vlength*20+vnum_periods*vlength*2)
    fdtd.set("y",0)
    fdtd.set("y span", 10*np.max([vwidth1,vwidth2]))
    fdtd.set("z",0)
    fdtd.set("z span", 10*vthickness)
    fdtd.set("material","SiO2 (Glass) - Palik")

    # define input waveguide
    fdtd.addrect()
    fdtd.set("name","core1_input")
    fdtd.set("x",-vlength- vlength*10)  
    fdtd.set("x span",vlength*20)
    fdtd.set("y",0)
    fdtd.set("y span", vwidth1)
    fdtd.set("z",0)
    fdtd.set("z span", vthickness)
    fdtd.set("material","Si (Silicon) - Palik")
        
    # Define grating
    for i in range(vnum_periods):
        
        # define core block 1
        fdtd.addrect()
        fdtd.set("name","core1")
        fdtd.set("x",-0.5*vlength + 2*vlength*i)
        fdtd.set("x span",vlength)
        fdtd.set("y",0)
        fdtd.set("y span", vwidth1)
        fdtd.set("z",0)
        fdtd.set("z span", vthickness)
        fdtd.set("material","Si (Silicon) - Palik")
        
        # define core block 2
        fdtd.addrect()
        fdtd.set("name","core2")
        fdtd.set("x",0.5*vlength + 2*vlength*i)
        fdtd.set("x span",vlength)
        fdtd.set("y",0)
        fdtd.set("y span", vwidth2)
        fdtd.set("z",0)
        fdtd.set("z span", vthickness)
        fdtd.set("material","Si (Silicon) - Palik")

    # define output waveguide
    fdtd.addrect()
    fdtd.set("name","core1_input")
    fdtd.set("x",-vlength + 2*vlength*vnum_periods+ vlength*10)  
    fdtd.set("x span",vlength*20)
    fdtd.set("y",0)
    fdtd.set("y span", vwidth2)
    fdtd.set("z",0)
    fdtd.set("z span", vthickness)
    fdtd.set("material","Si (Silicon) - Palik")

    # setup source
    fdtd.addmode()
    fdtd.set("wavelength start",1.45e-6)
    fdtd.set("wavelength stop",1.65e-6)
    fdtd.set("frequency dependent profile",1)
    fdtd.set("number of field profile samples",vnum_wavelengths)
    fdtd.set("x",-vlength- vlength*10)  
    fdtd.set("y",0)
    fdtd.set("y span", np.max([vwidth1,vwidth2])*3)
    fdtd.set("z",0)
    fdtd.set("z span", vthickness*3)

    # setup fdtd
    fdtd.addfdtd()
    fdtd.set("x",vlength*vnum_periods-1*vlength)
    fdtd.set("x span",2*vlength*15+vnum_periods*vlength*2)
    fdtd.set("y",0)
    fdtd.set("y span", 5*np.max([vwidth1,vwidth2]))
    fdtd.set("z",0)
    fdtd.set("z span", 5*vthickness)
    fdtd.set("mesh accuracy",4)

    # setup monitor
    fdtd.addpower()
    fdtd.setglobalmonitor("frequency points",vnum_wavelengths)
    fdtd.set("monitor type","2D X-normal")
    fdtd.set("x",-vlength + 2*vlength*vnum_periods+ vlength*10)  
    fdtd.set("y",0)
    fdtd.set("y span", np.max([vwidth1,vwidth2])*3)
    fdtd.set("z",0)
    fdtd.set("z span", vthickness*3)

    # run
    fdtd.run()

    # postprocess
    f = fdtd.getdata("monitor","f")
    l = 3e8 / f

    spect = fdtd.getresult("source","spectrum")
    spectrum = np.abs(spect['spectrum'])**2
    lambd = spect['lambda']
    spectrum = fdtd.interp(spectrum, lambd, l)

    power = np.abs(fdtd.getdata("monitor","power"))**2 * spectrum
    power = power / np.max(power)

    new_lambd = np.linspace(1.5,1.6,args.num_wavelengths)*1e-6
    power = fdtd.interp(power, l, new_lambd)

    return power





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_periods',type=int, default="50", help='Number of Periods for Bragg Grating (default: 50)')
    parser.add_argument('-length',type=float, default="0.159", help='Length of each segment of BG, Period = Length * 2 (microns) (default: 0.159)')
    parser.add_argument('-num_wavelengths',type=int, default="100", help='Number of wavelengths to sweep (default: 30)')
    parser.add_argument('-wl_lower',type=float, default="1.5", help='Lower wavelength bound (microns) (default: 1.5)')
    parser.add_argument('-wl_upper',type=float, default="1.6", help='Upper wavelength bound (microns) (default: 1.6)')
    parser.add_argument('-num_modes',type=int, default="1", help='Number of Modes (default: 1)')
    parser.add_argument('-mesh',type=int, default="128", help='Number of mesh points (default: 128)')
    parser.add_argument('-width1',type=float, default="0.46", help='Width of first core block (microns) (default: 0.46)')
    parser.add_argument('-width2',type=float, default="0.54", help='Width of second core block  (microns) (default: 0.54)')
    parser.add_argument('-thickness',type=float, default="0.22", help='Thickness of the core (microns) (default: 0.22)')

    args = parser.parse_args()
    t = lumerical_fdtd(args)

    #Plot the results
    plt.plot(np.linspace(args.wl_lower,args.wl_upper, args.num_wavelengths), t)
    plt.title("Grating freq sweep nperiods=" + str(args.num_periods))
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("t")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    
    main()
