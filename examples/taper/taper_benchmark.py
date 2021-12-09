import numpy as np
from matplotlib import pyplot as plt
from taper_ann import taper_ann
from taper_empy import taper_empy
from taper_lumerical import taper_lumerical

def main():

    start = 5
    stop = 30

    taper_lumerical_dict = taper_lumerical(False,start,stop)
    taper_ann_dict = taper_ann(False,start,stop)
    # taper_empy_dict = taper_empy(False,start,stop)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(taper_ann_dict["density"],np.abs(taper_ann_dict["time"]), label="ANN")
    plt.plot(taper_lumerical_dict["density"],np.abs(taper_lumerical_dict["time"]), label="Lumerical FD")
    # plt.plot(taper_empy_dict["density"],np.log10(taper_empy_dict["time"]), label="Electromagnetic Python")
    plt.xlabel("Taper Density")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(taper_ann_dict["density"],np.abs(np.array(taper_ann_dict["s_params"])[:,0,0,1])**2, label="ANN")
    plt.plot(taper_lumerical_dict["density"],np.abs(np.array(taper_lumerical_dict["s_params"])[:,0,0,1])**2, label="Lumerical FD")
    # plt.plot(taper_empy_dict["density"],np.abs(np.array(taper_empy_dict["s_params"])[:,0,0,1])**2, label="Electromagnetic Python")
    plt.xlabel("Taper Density")
    plt.ylabel("Transmission ")
    plt.legend()
    plt.grid()

    plt.savefig('taper.eps', format='eps')
    plt.savefig('taperpng.png')
    plt.show()

if __name__ == "__main__":
    main()