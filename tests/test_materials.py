from emepy.materials import Si, SiO2
import numpy as np
import unittest

# Wavelengths
wavelengths = np.linspace(1.3, 1.7, 100)


class TestMaterials(unittest.TestCase):
    def test_Si(self):

        # Ensure that the index of refraction is within the range of the material
        for wl in wavelengths:
            self.assertTrue(3 < Si(wl) < 4)

    def test_SiO2(self):

        # Ensure that the index of refraction is within the range of the material
        for wl in wavelengths:
            self.assertTrue(1 < SiO2(wl) < 2)


if __name__ == "__main__":
    unittest.main()
