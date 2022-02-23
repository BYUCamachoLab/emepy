from abc import abstractmethod
import numpy as np
from tqdm import tqdm
from simphony.models import Subcircuit
from emepy.monitors import Monitor
from copy import deepcopy

class Source(object):
    """This class defines mode sources that can be created for monitors"""

    def __init__(self, z:int=None, mode_coeffs:list=[], k=1):
        """Constructor for Source. Note: if the modes corresponding to the coefficients defined are not found in the system, the coefficients will be reduced to 0. For now, the wavelength will always be the same as defined in EME.

        Parameters
        ----------
        z : float
            if defined, the z location to define each mode source (default: None)
        mode_coeffs : list[int]
            if z is non-empty, this list represents the modes used for each source. (default: [])
        k : boolean
            if positve, will propagate in the positive direction
        """

        self.z = z
        self.mode_coeffs = mode_coeffs
        self.k = k > 0

    def get_label(self):
        k = "+" if self.k else "-"
        return "{}{}".format(k,self.z)

    def match_label(self, label):
        return label == self.get_label()


    def verify(self):
        """Parses self to ensure proper format
        """
        return True

