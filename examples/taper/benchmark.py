import numpy as np
from matplotlib import pyplot as plt
from taper_ann import taper_ann
from taper_empy import taper_empy
from taper_lumerical import taper_lumerical

start = 0
stop = 6

taper_ann_dict = taper_ann(False,start,stop)
taper_lumerical_dict = taper_lumerical(False,start,stop)
taper_empy_dict = taper_empy(False,start,stop)

plt.figure()
plt.subplot(2,1,1)
plt.plot(taper_ann_dict["density"],np.log10(taper_ann_dict["time"]), label="ANN")
plt.plot(taper_lumerical_dict["density"],np.log10(taper_lumerical_dict["time"]), label="Lumerical FD")
plt.plot(taper_empy_dict["density"],np.log10(taper_empy_dict["time"]), label="Electromagnetic Python")
plt.xlabel("Taper Density")
plt.ylabel("Time (log10 s)")
plt.legend()
plt.show()