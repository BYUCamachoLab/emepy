from emepy.fd import MSEMpy
from emepy.fd import MSPickle
from matplotlib import pyplot as plt
import pickle as pk

modesolver = MSEMpy(wl=1.55e-6, width=0.5e-6, thickness=0.22e-6, mesh=128)
modesolver.solve()
mode = modesolver.get_mode()
pk.dump(mode, open("./example_file.pk", "wb+"))

# Separate instance

modesolver = MSPickle(filename="./example_file.pk", width=0.5e-6, thickness=0.22e-6)

modesolver.solve()
mode = modesolver.get_mode()

plt.figure()
mode.plot(value_type="Imaginary")
plt.show()
