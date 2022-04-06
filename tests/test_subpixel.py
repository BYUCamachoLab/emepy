from emepy import rectangle_to_n, circle_to_n
import numpy as np
from matplotlib import pyplot as plt
import unittest

# Global params
center = (0,0)
x=np.linspace(-1.5,1.5,128)
y=np.linspace(-1.5,1.5,128)
core_index=3.4
cladding_index=1.4

# Waveguide params
width=0.8
thickness=0.22

# Circle params
radius=1

# Difference params
dp = 1e-8

def rectangle_difference(subpixel:bool, dp:float):

    # Initial rect
    rect_i = rectangle_to_n(
        center=center,
        width=width,
        thickness=thickness,
        x=x,
        y=y,
        subpixel=subpixel,
        core_index=3.4,
        cladding_index=1.4,
    )

    # Final rect
    rect_f = rectangle_to_n(
        center=center,
        width=width+dp,
        thickness=thickness+dp,
        x=x,
        y=y,
        subpixel=subpixel,
        core_index=3.4,
        cladding_index=1.4,
    )
    return rect_f-rect_i

def circle_difference(subpixel:bool, dp:float):

    # Initial rect
    rect_i = circle_to_n(
        center=center,
        radius=radius,
        x=x,
        y=y,
        subpixel=subpixel,
        core_index=3.4,
        cladding_index=1.4,
    )

    # Final rect
    rect_f = circle_to_n(
        center=center,
        radius=radius+dp,
        x=x,
        y=y,
        subpixel=subpixel,
        core_index=3.4,
        cladding_index=1.4,
    )
    return rect_f-rect_i

class TestSubpixel(unittest.TestCase):
    
    def test_subpixel_rectangle(self):
        print("Testing rectagular subpixel")

        # With subpixel
        delta_subpixel = rectangle_difference(subpixel=True, dp=dp)
        self.assertFalse(np.all(delta_subpixel == 0))

        # Without subpixel
        delta_no_subpixel = rectangle_difference(subpixel=False, dp=dp)
        self.assertTrue(np.all(delta_no_subpixel == 0))

        # Compare the two
        self.assertTrue(np.sum(delta_no_subpixel) < np.sum(delta_subpixel))
        
    def test_subpixel_circle(self):
        print("Testing circular subpixel")
        
        # With subpixel
        delta_subpixel = circle_difference(subpixel=True, dp=dp)
        self.assertFalse(np.all(delta_subpixel == 0))

        # Without subpixel
        delta_no_subpixel = circle_difference(subpixel=False, dp=dp)
        self.assertTrue(np.all(delta_no_subpixel == 0))

        # Compare the two
        self.assertTrue(np.sum(delta_no_subpixel) < np.sum(delta_subpixel))
        

if __name__ == '__main__':
    unittest.main()
