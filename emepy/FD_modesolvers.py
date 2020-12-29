import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

from emepy.mode import Mode
from emepy import tools

import sys
import EMpy

import pickle


class ModeSolver_Lumerical(object):
    def __init__(
        self,
        wl,
        width,
        thickness,
        num_modes=1,
        cladding_width=5e-6,
        cladding_thickness=5e-6,
        core_index=None,
        cladding_index=None,
        mesh=300,
        lumapi_location=None,
    ):

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.mesh = mesh - 1
        self.lumapi_location = lumapi_location

        if core_index == None:
            self.core_index = tools.Si(wl * 1e6)
        if cladding_index == None:
            self.cladding_index = tools.SiO2(wl * 1e6)

    def solve(self):

        core_width = self.width
        core_thickness = self.thickness
        clad_width = self.cladding_width
        clad_thickness = self.cladding_thickness
        length = 10.0e-6
        mesh = self.mesh
        num_modes = self.num_modes

        sys.path.append(self.lumapi_location)
        import lumapi as lm

        with lm.MODE(hide=True) as mode:

            # open file
            if os.path.isfile("api.lms"):
                os.remove("api.lms")
            mode.save("api.lms")
            cladding = mode.addrect()
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
            core = mode.addrect()
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
            fde = mode.addfde()
            fde.y = 0
            fde.y_span = clad_width / 2
            fde.solver_type = "2D X normal"
            fde.x = 0
            fde.z = 0
            fde.z_span = clad_thickness / 2
            fde.mesh_cells_y = mesh
            fde.mesh_cells_z = mesh
            mode.run

            mode.set("number of trial modes", num_modes)
            mode.set("wavelength", self.wl)
            mode.findmodes()
            field = []
            gridx = mode.getresult("FDE::data::mode1", "y")
            gridx = gridx.reshape(gridx.shape[0])
            gridy = mode.getresult("FDE::data::mode1", "z")
            gridy = gridy.reshape(gridy.shape[0])
            grid = [gridx.tolist(), gridy.tolist()]
            neff = []
            for modeNum in range(1, num_modes + 1):
                mode_field = []
                neff.append(mode.getdata("FDE::data::mode" + str(modeNum), "neff")[0][0])
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Hy").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hy").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hy").shape[2],
                        )
                    )
                )
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Hz").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hz").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hz").shape[2],
                        )
                    )
                )
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Hx").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hx").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Hx").shape[2],
                        )
                    )
                )
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Ey").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ey").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ey").shape[2],
                        )
                    )
                )
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Ez").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ez").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ez").shape[2],
                        )
                    )
                )
                mode_field.append(
                    mode.getdata("FDE::data::mode" + str(modeNum), "Ex").reshape(
                        (
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ex").shape[1],
                            mode.getdata("FDE::data::mode" + str(modeNum), "Ex").shape[2],
                        )
                    )
                )
                field.append(mode_field)

            field = np.array(field).tolist()

            if os.path.isfile("api.lms"):
                os.remove("api.lms")
        self.x = grid[0]
        self.y = grid[1]
        self.neffs = neff
        self.fields = field

    def clear(self):

        self.x = None
        self.y = None
        self.neffs = None
        self.fields = None

    def getMode(self, mode_num=0):

        field = self.fields[mode_num]
        Ex = field[3]
        Ey = field[4]
        Ez = field[5]
        Hx = field[0]
        Hy = field[1]
        Hz = field[2]
        neff = self.neffs[mode_num]

        return Mode(x=self.x, y=self.y, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez)


class ModeSolver_EMpy(object):
    def __init__(
        self,
        wl,
        width,
        thickness,
        num_modes=1,
        cladding_width=2.5e-6,
        cladding_thickness=2.5e-6,
        core_index=None,
        cladding_index=None,
        x=None,
        y=None,
        mesh=300,
        accuracy=1e-8,
        boundary="0000",
        epsfunc=None,
    ):

        self.wl = wl
        self.width = width
        self.thickness = thickness
        self.num_modes = num_modes
        self.cladding_width = cladding_width
        self.cladding_thickness = cladding_thickness
        self.core_index = core_index
        self.cladding_index = cladding_index
        self.x = x
        self.y = y
        self.mesh = mesh
        self.accuracy = accuracy
        self.boundary = boundary
        self.epsfunc = epsfunc

        if core_index == None:
            self.core_index = tools.Si(wl * 1e6)
        if cladding_index == None:
            self.cladding_index = tools.SiO2(wl * 1e6)
        if x == None:
            self.x = np.linspace(0, cladding_width, mesh)
        if y == None:
            self.y = np.linspace(0, cladding_width, mesh)
        if epsfunc == None:
            self.epsfunc = tools.get_epsfunc(
                self.width,
                self.thickness,
                self.cladding_width,
                self.cladding_thickness,
                self.core_index,
                self.cladding_index,
            )

    def solve(self):

        self.solver = EMpy.modesolvers.FD.VFDModeSolver(self.wl, self.x, self.y, self.epsfunc, self.boundary).solve(
            self.num_modes, self.accuracy
        )

    def clear(self):

        self.solver = None

    def getMode(self, mode_num=0):

        Ex = self.solver.modes[mode_num].get_field("Ex", self.x, self.y)
        Ey = self.solver.modes[mode_num].get_field("Ey", self.x, self.y)
        Ez = self.solver.modes[mode_num].get_field("Ez", self.x, self.y)
        Hx = self.solver.modes[mode_num].get_field("Hx", self.x, self.y)
        Hy = self.solver.modes[mode_num].get_field("Hy", self.x, self.y)
        Hz = self.solver.modes[mode_num].get_field("Hz", self.x, self.y)
        neff = self.solver.modes[mode_num].neff

        return Mode(x=self.x, y=self.y, wl=self.wl, neff=neff, Hx=Hx, Hy=Hy, Hz=Hz, Ex=Ex, Ey=Ey, Ez=Ez)


class ModeSolver_Pickle(object):
    def __init__(self, filename, index=None, width=None, thickness=None):

        self.filename = filename
        self.index = index
        self.width = width
        self.thickness = thickness

    def solve(self):

        with open(self.filename, "rb") as f:
            self.mode = pickle.load(f)[self.index] if self.index != None else pickle.load(f)

        self.x = self.mode.gridX
        self.y = self.mode.gridY

    def clear(self):

        self.x = None
        self.y = None
        self.mode = None

    def getMode(self, mode_num=0):

        return self.mode

