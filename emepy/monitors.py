import numpy as np
from matplotlib import pyplot as plt
from copy import copy


class Monitor(object):
    """ Monitor objects store fields during propagation for user visualization. Three types of monitors exist: 3D, 2D, and 1D. 
    """

    def __init__(self, axes="xz", dimensions=(1,1), lengths=[], components=["E"]):
        """Monitor class constructor

        Parameters
        ----------
        axes : string
            the spacial axes to capture fields in. Options : 'xz' (default), 'xy', 'yz', 'xyz', 'x', 'y', 'z'. Currently only 'xz' is implemented. Note, propagation is always in z. 
        dimensions : tuple
            the spacial dimensions of the resulting field
        lengths : list
            list of the remaining points in z at which to calculate the fields
        components : list
            list of the field components to store from ('E','H','Ex','Ey','Ez','Hx','Hy','Hz)
        """

        if axes == "xz" or axes == "zx":
            self.axes = "xz"
            self.dimensions = dimensions
            self.field = np.zeros(dimensions).astype(complex)
            self.lengths = copy(lengths)
            self.remaining_lengths = copy(lengths)
        else:
            raise Exception("Monitor setup {} has not yet been implemented. Please choose from the following implemented monitor types: ['yz']".format(axes))

        self.cur_prop_index = [0 for i in range(len(components))]
        self.cur_length = [0 for i in range(len(components))]
        self.components = components

    def __getitem__(self, subscript):
        return self.field[subscript]

    def __setitem__(self, subscript, item):
        self.field[subscript] = item
        if len(self.remaining_lengths[int(subscript[0])]) > 1:
            self.remaining_lengths[int(subscript[0])] = self.remaining_lengths[int(subscript[0])][1:]
            self.cur_prop_index[int(subscript[0])] += 1
        else:
            self.remaining_lengths[int(subscript[0])] = []
            self.cur_prop_index[int(subscript[0])]  += 1

    def __delitem__(self, subscript):
        del self.field[subscript]

    def visualize(self, components=['Ex','Ey','Ez','Hx','Hy','Hz','E','H']):
        results = {}
        for i,c in zip(range(len(self.components)),self.components):
            results[c] = self.field[i]

        results["E"] = np.abs(results["Ex"]) ** 2 + np.abs(results["Ey"]) ** 2 + np.abs(results["Ez"]) ** 2
        results["H"] = np.abs(results["Hx"]) ** 2 + np.abs(results["Hy"]) ** 2 + np.abs(results["Hz"]) ** 2
            
        for c in components:
            plt.figure()
            plt.imshow(np.real(results[c]), extent=[0,self.lengths[0][-1],0,5e-6])
            plt.xlabel("z (µm)")
            plt.ylabel("x (µm)")
            plt.title(c)
            plt.colorbar()
            plt.show()
