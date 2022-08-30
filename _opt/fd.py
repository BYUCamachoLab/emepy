import numpy as np
import emepy as em
import meep as mp
import matplotlib.pyplot as plt
import pickle as pk
from both import *

VISUALIZE_N = False

# Function to perform finite difference on eme
def eme_fd():

    # Get FOM
    lower_n, f_x_lower, fom_lower = eme("lower")
    upper_n, f_x_upper, fom_upper = eme("upper")

    return lower_n, f_x_lower, fom_lower, upper_n, f_x_upper, fom_upper


# Function to perform finite difference on meep
def meep_fd(n_lower, n_upper):

    # Get fields
    mpx, mpy, mpz, fields_lower = meep(n_lower, True)
    _, _, _, fields_upper = meep(n_upper, True)

    return mpx, mpy, mpz, fields_lower, fields_upper

# Main function
def main():
    fd_data = {}

    # EME finite difference
    lower_n, f_x_lower, fom_lower, upper_n, f_x_upper, fom_upper = eme_fd()
    fd_data["eme_lower_n"] = lower_n
    fd_data["eme_lower_f_x"] = f_x_lower
    fd_data["eme_lower_fom"] = fom_lower
    fd_data["eme_upper_n"] = upper_n
    fd_data["eme_upper_f_x"] = f_x_upper
    fd_data["eme_upper_fom"] = fom_upper

    # # MEEP finite difference
    # mpx, mpy, mpz, fields_lower, fields_upper = meep_fd(lower_n, upper_n)
    # fd_data["meep_lower_fields"] = fields_lower
    # fd_data["meep_upper_fields"] = fields_upper
    # fd_data["meep_mpx"] = mpx
    # fd_data["meep_mpy"] = mpy
    # fd_data["meep_mpz"] = mpz

    if em.am_master():
        pk.dump(fd_data, open("fd_data2.pk", "wb+"))


if __name__ == "__main__":
    main()
