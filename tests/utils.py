import unittest
import numpy as np

class ApproxComparisonTestCase(unittest.TestCase):
    """A mixin for adding proper floating point value and vector comparison."""

    def assertClose(self, x, y, epsilon = 1e-2, msg = ''):
        """Asserts that two values or vectors satisfy ‖x-y‖ ≤ ε * max(‖x‖, ‖y‖)."""
        x = np.atleast_1d(x).ravel()
        y = np.atleast_1d(y).ravel()
        x_norm = np.linalg.norm(x, ord=np.inf)
        y_norm = np.linalg.norm(y, ord=np.inf)
        diff_norm = np.linalg.norm(x - y, ord=np.inf)
        self.assertLessEqual(diff_norm, epsilon * np.maximum(x_norm, y_norm), msg)