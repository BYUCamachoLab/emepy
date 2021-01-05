EMEpy
=====

Table of Contents
-----------------
   
.. toctree::
   :maxdepth: 2

   index
   class_func
   taper
   bragg

EMEpy is an open-source eigenmode expansion solver implemented in Python. 

**Key Features**

- Free and open-source
- Easy to use, great for educators
- Computationally enhancing, great for designers
- Complete design capabilities in Python

Eigenmode Expansion
-------------------

Eigenmode Expansion (EME) is a method of simulating light through optical structures that operates in the frequency domain. The algorithm works by utilizing some useful properties of light. First, light exists as a superposition of eigenmodes that satisfy Maxwell's equations inside the structure. The eigenmodes are composed of a field pattern and an eigenvalue, $\beta$ proportional to the effective index of refraction of the structure. As these eigenmodes propagate through a structure that changes shape or material along the direction of propagation, the effective index and field patterns change. However, if the structure does not change in this direction, the eigenmodes remain the same except for the phase. Along these structures, the phase changes according to $e^{j\beta z}$ where z is the distance travelled. 

The EME algorithm utilizes this property by taking geometric structures and representing them as a series of continuous structures in the direction of propagation. This way, each section of the geometry can contain an set of eigenmodes and phase changes. At each intersection between sections, the power of the input eigenmodes transfer into the power of the output eigenmodes. However, unless the to sets of modes are identical, reflection can also occur. 

To calculate the proportion of power that transmits and reflects from any given mode to another, the overlap is calculated and a system of equations is solved. Together, the intersection mode overlap and phase propagation are cascaded and provide a set of s-parameters for the device. EMEpy can be used to calculate these values and produce s-parameters for users' geometry.

Installation
------------

EMEpy can be found on pip. 

.. code-block:: python

   pip install emepy

For the latest version, the source code can be found on `GitHub <https://github.com/BYUCamachoLab/emepy>`_. Clone the directory onto your local desktop. 

.. code-block:: python

   pip install -e .

Tutorials
---------

- :doc:`class_func`
- :doc:`taper`
- :doc:`bragg`

Acknowledgements
----------------

