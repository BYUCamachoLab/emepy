from emepy.tools import get_epsfunc
from matplotlib import pyplot as plt
import numpy as np

index_func = get_epsfunc(
    width=0.5e-6,
    thickness=0.22e-6,
    cladding_width=5e-6,
    cladding_thickness=5e-6,
    core_index=np.sqrt(3.5),
    cladding_index=np.sqrt(1.4),
)

x = np.linspace(0, 5e-6, 128)
y = np.linspace(0, 5e-6, 128)

index = index_func(x, y)

plt.imshow(np.real(index), extent=[0, 5, 0, 5])
plt.colorbar()
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("index")
plt.show()
