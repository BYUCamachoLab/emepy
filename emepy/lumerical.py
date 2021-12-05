"""
    This file interfaces with the Lumerical API for modesolving
"""

from emepy.eme import EME
from emepy.fd import ModeSolver
import lumapi as lm
import numpy as np
from scipy.interpolate import interp1d
import os

from emepy.mode import Mode
from emepy import tools

import sys
import EMpy

import pickle

class LumEME(EME):
    """
        This class is a wrapper for EME, it performs the same operations but uses Lumerical MODE to solve for the modes at the interfaces
    """

    def __init__(self,  layers=[], num_periods=1):
        super().__init__(layers, num_periods)

        # open file
        if os.path.isfile("api.lms"):
            os.remove("api.lms")
        self.mode = lm.MODE(hide=True)
        self.mode.save("api.lms")


    def close(self):
        if os.path.isfile("api.lms"):
            os.remove("api.lms")

class MSLumerical(ModeSolver):
    """Outdated Lumerical Modesolver. Uses the lumapi Lumerical API. See Modesolver. Parameterizes the cross section as a rectangular waveguide. 
    """
    def __init__(
        self,
        wl,
        width,
        thickness,
        num_modes=1,
        cladding_width=10e-6,
        cladding_thickness=10e-6,
        core_index=None,
        cladding_index=None,
        mesh=300,
        mode=None,
        **kwargs
    ):
        """MSLumerical class constructor

        Parameters
        ----------
        wl : number 
            wavelength of the eigenmodes
        width : number
            width of the core in the cross section
        thickness : number
            thickness of the core in the cross section
        num_modes : int
            number of modes to solve for (default:1)
        cladding_width : number
            width of the cladding in the cross section (default:5e-6)
        cladding_thickness : number
            thickness of the cladding in the cross section (default:5e-6)
        core_index : number
            refractive index of the core (default:Si)
        cladding_index : number
            refractive index of the cladding (default:SiO2)
        mesh : int
            number of mesh points in each direction (xy)
        mode : lumapi.MODE
            MODE object that contains the file information
        """

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.mesh = mesh #- 1
        self.mode = mode
        self.close_after = False

        if core_index is None:
            self.core_index = tools.Si(wl * 1e6)
        if cladding_index is None:
            self.cladding_index = tools.SiO2(wl * 1e6)
        if self.mode is None:
            self.close_after = True

            # open file
            if os.path.isfile("api.lms"):
                os.remove("api.lms")

            self.mode = lm.MODE(hide=True)

    def solve(self):
        """Solves for the eigenmodes
        """

        core_width = self.width
        core_thickness = self.thickness
        clad_width = self.cladding_width
        clad_thickness = self.cladding_thickness
        length = 10.0e-6
        mesh = self.mesh
        num_modes = self.num_modes

        # Restart everything in case this isn't the first go
        self.mode.switchtolayout()
        self.mode.deleteall()

        cladding = self.mode.addrect()
        cladding.name = "cladding"
        cladding.x = 0
        cladding.x_span = length
        cladding.y = 0
        cladding.y_span = clad_width
        cladding.z = 0
        cladding.z_span = clad_thickness
        # cladding.index = clad_index
        cladding.material = "SiO2 (Glass) - Palik"

        # add core x=prop y=width z=thickness
        core = self.mode.addrect()
        core.name = "core"
        core.x = 0
        core.x_span = length
        core.y = 0
        core.y_span = core_width
        core.z = 0
        core.z_span = core_thickness
        # core.index = core_index
        core.material = "Si (Silicon) - Palik"

        # set up FDE
        fde = self.mode.addfde()
        fde.y = 0
        fde.y_span = 2e-6#clad_width / 2
        fde.solver_type = "2D X normal"
        fde.x = 0
        fde.z = 0
        fde.z_span = 2e-6#clad_thickness / 2
        fde.mesh_cells_y = mesh
        fde.mesh_cells_z = mesh
        self.mode.run

        self.mode.set("number of trial modes", num_modes)
        self.mode.set("wavelength", self.wl)
        self.mode.findmodes()
        field = []
        gridx = self.mode.getresult("FDE::data::mode1", "y")
        gridx = gridx.reshape(gridx.shape[0])
        gridy = self.mode.getresult("FDE::data::mode1", "z")
        gridy = gridy.reshape(gridy.shape[0])
        grid = [gridx.tolist(), gridy.tolist()]
        neff = []
        for modeNum in range(1, num_modes + 1):
            mode_field = []
            neff.append(self.mode.getdata("FDE::data::mode" + str(modeNum), "neff")[0][0])
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Hy").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hy").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hy").shape[2],
                    )
                )
            )
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Hz").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hz").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hz").shape[2],
                    )
                )
            )
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Hx").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hx").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Hx").shape[2],
                    )
                )
            )
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Ey").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ey").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ey").shape[2],
                    )
                )
            )
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Ez").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ez").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ez").shape[2],
                    )
                )
            )
            mode_field.append(
                self.mode.getdata("FDE::data::mode" + str(modeNum), "Ex").reshape(
                    (
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ex").shape[1],
                        self.mode.getdata("FDE::data::mode" + str(modeNum), "Ex").shape[2],
                    )
                )
            )
            field.append(mode_field)

        field = np.array(field).tolist()

        self.x = grid[0]
        self.y = grid[1]
        self.neffs = neff
        self.fields = field

        if self.close_after and os.path.isfile("api.lms"):
            os.remove("api.lms")

    def clear(self):
        """Clears the modesolver's eigenmodes to make memory
        """

        self.x = None
        self.y = None
        self.neffs = None
        self.fields = None

    def get_mode(self, mode_num=0):
        """Get the indexed mode number

        Parameters
        ----------
        mode_num : int 
            index of the mode of choice

        Returns
        -------
        Mode 
            the eigenmode of index mode_num
        """

        field = self.fields[mode_num]
        Ex = field[3]
        Ey = field[4]
        Ez = field[5]
        Hx = field[0]
        Hy = field[1]
        Hz = field[2]
        neff = self.neffs[mode_num]
        # print(len(self.x))
        mode = Mode(x=self.x, y=self.y, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez,width=self.width,thickness=self.thickness)
        # mode.normalize()
        # from matplotlib import pyplot as plt
        # plt.figure()
        # mode.plot()
        # plt.show()
        # quit()

        return mode
      