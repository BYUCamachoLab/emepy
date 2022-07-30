import meep as mp
import emepy as em
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk
from meep_sim import *

def main():

    # Load data
    data = pk.load(open(DATA_FILE, "rb"))

    n = data["n"] #  (250, 250, 250)
    eme_n = data["eme_n"] #  (250, 250, 250)
    A_u = data["A_u"] #  (3, 3, 250, 250, 200, 2)
    em_forward_fields = data["em_forward_fields"] #  (7, 250, 250, 250)
    em_backward_fields = data["em_backward_fields"] #  (7, 250, 250, 250)
    eme_grid_x = data["eme_grid_x"] #  (250,)
    eme_grid_y = data["eme_grid_y"] #  (250,)
    eme_grid_z = data["eme_grid_z"] #  (250,)
    emx = data["emx"] #  (250,)
    emy = data["emy"] #  (250,)
    emz = data["emz"] #  (250,)
    forward_meep_fields = data["forward_meep_fields"] #  (6, 252, 252, 252)
    backward_meep_fields = data["backward_meep_fields"] #  (6, 252, 252, 252)
    mpx = data["mpx"] #  (252,)
    mpy = data["mpy"] #  (252,)
    mpz = data["mpz"] #  (252,)

    plt.figure()
    plt.imshow(forward_meep_fields[4, :, 126, :].real, cmap="RdBu")
    plt.show()


if __name__ == "__main__":
    main()






    # # Downsample function
    # def downsample(fields):
    #     return 1 / 8 * (fields[:, 1:, 1:, 1:] + fields[:, 1:, 1:, :-1] + fields[:, 1:, :-1, 1:] + fields[:, 1:, :-1, :-1] + fields[:, :-1, 1:, 1:] + fields[:, :-1, 1:, :-1] + fields[:, :-1, :-1, 1:] + fields[:, :-1, :-1, :-1])

    # # Downsample the fields if needed
    # size_A_u = lambda x: np.array(list(x.shape)[2:5])
    # size_fields = lambda x: np.array(list(x.shape)[1:])
    # A_u_size = size_A_u(A_u)
    # fields_size = size_fields(fields)
    # while (A_u_size != fields_size).any():
    #     arr = A_u_size - fields_size
    #     if not np.all(arr == arr[0]):
    #         raise Exception("Something went wrong with the downsampling")
    #     fields = downsample(fields)
    #     fields_size = size_fields(fields)