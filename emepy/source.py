from abc import abstractmethod
import numpy as np
from tqdm import tqdm
from simphony.models import Subcircuit
from emepy.monitors import Monitor
from copy import deepcopy

class Source(object):
    """This class defines mode sources that can be created for monitors"""

    def __init__(self, left_input=[1.0], right_input=[], z=None, mode_coeffs=[], k=1):
        """Constructor for Source. Note: if the modes corresponding to the coefficients defined are not found in the system, the coefficients will be reduced to 0. For now, the wavelength will always be the same as defined in EME.

        Parameters
        ----------
        left_input : list[float]
            list of the mode coefficients to input on the left side of the system (default: [1.0])
        right_input : list[float]
            list of the mode coefficients to input on the left side of the system (default: [0.0])
        z : float
            if defined, the z location to define each mode source (default: None)
        mode_coeffs : list[int]
            if z is non-empty, this list represents the modes used for each source. (default: [])
        k : boolean
            if positve, will propagate in the positive direction
        """

        self.left_input = left_input
        self.right_input = right_input 
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

    @staticmethod
    def extract_source_locations(*args):
        """Provides all sources as input, will return the locations of custom sources 
        """

        locations = []
        for i in args:
            i.verify()
            locations += i.z


