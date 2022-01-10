"""
    EMEPy is currently developing Cuda support. This will simply allow modesolving through the EMpy module to allocate space on the GPU when finding eigen solutions. Effectively, this will increase the xy computational speed but not affect the z speed. 
"""

import pycuda.driver as cuda
import pycuda.autoinit

cuda.mem_alloc()