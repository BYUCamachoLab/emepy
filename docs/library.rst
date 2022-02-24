===============================
EMEPy Complete User Library
===============================

Mode
----

.. autoclass:: emepy.mode.Mode
   :members:

   .. automethod:: __init__

EME
---

.. autoclass:: emepy.eme.EME
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.lumerical.LumEME
   :members:

   .. automethod:: __init__

Models 
------

.. autoclass:: emepy.models.Layer
   :members:

   .. automethod:: __init__

ModeSolver
----------

.. autoclass:: emepy.fd.ModeSolver
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.lumerical.MSLumerical
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.fd.MSEMpy
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.fd.MSPickle
   :members:

   .. automethod:: __init__

Geometry 
--------

EMEPy now offers geometry abstractions that allow users to more easily implement the layers needed for their system. This is currently under development and subject to changing. Check out emepy.geometry.py for more examples available for users. 

.. autoclass:: emepy.geometries.Geometry
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.geometries.Waveguide
   :members:

   .. automethod:: __init__

Monitors
--------

.. autoclass:: emepy.monitors.Monitor
   :members:

   .. automethod:: __init__


Neural Network Acceleration
---------------------------

.. autoclass:: emepy.ann.MSNeuralNetwork
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.ann.ANN
   :members:

   .. automethod:: __init__

.. autoclass:: emepy.ann.Network
   :members:

   .. automethod:: __init__

Tools
-----

EMEPy offers functions to the user that can be called. These are mostly important for the library backend however. 

.. autofunction:: emepy.tools.get_epsfunc

.. autofunction:: emepy.tools.get_epsfunc_epsfunc

.. autofunction:: emepy.tools.create_polygon

.. autofunction:: emepy.tools.interp

.. autofunction:: emepy.tools.interp1d

.. autofunction:: emepy.tools.into_chunks

.. autofunction:: emepy.tools.from_chunks

.. autofunction:: emepy.tools._get_eps

.. autofunction:: emepy.tools.compute_other_fields_2D
