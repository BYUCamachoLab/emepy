class Source(object):
    """This class defines mode sources that can be created for monitors"""

    def __init__(self, z: float = None, mode_coeffs: list = [], k=1) -> None:
        """Constructor for Source. Note: if the modes corresponding to the coefficients defined are not found in the system, the coefficients will be reduced to 0. For now, the wavelength will always be the same as defined in EME.

        Parameters
        ----------
        z : float
            if defined, the z location to define each mode source (default: None)
        mode_coeffs : list[int]
            if z is non-empty, this list represents the modes used for each source. (default: [])
        k : int
            if positive (>0), will propagate in the positive direction
        """

        self.z = z
        self.mode_coeffs = mode_coeffs
        self.k = k > 0

    def get_label(self) -> str:
        """Returns a string that represents the source"""
        k = "+" if self.k else "-"
        return "{}{}".format(k, self.z)

    def match_label(self, label: str) -> bool:
        """Sees if the provided label matches the current Source object's and returns True if so

        Parameters
        ----------
        label : str
            the label to compare

        Returns
        -------
        bool
            True if label matches self

        """
        return label == self.get_label()

    def __str__(self):
        return self.get_label()
