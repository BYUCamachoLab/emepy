from emepy.fd import MSEMpy
import importlib
import EMpy_gpu
if not (importlib.util.find_spec("pycuda") is None):
    import pycuda.driver as cuda
    import pycuda.autoinit



class MSEMpyGPU(MSEMpy):
    """ Exactly the same as MSEMpy except allocates space and solves eigenvectors and values on the GPU
    """

    def solve(self):
        """Solves for the eigenmodes"""
        self.solver = EMpy_gpu.modesolvers.FD.VFDModeSolver(self.wl, self.x, self.y, self.epsfunc, self.boundary).solve(
            self.num_modes, self.accuracy
        )
        return self

    