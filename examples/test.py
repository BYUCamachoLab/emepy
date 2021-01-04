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
