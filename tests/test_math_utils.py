import unittest
from core.math_utils import MathUtils, SmoothingFilter

class TestMathUtils(unittest.TestCase):
    def test_calculate_angle_90_deg(self):
        # L shape: A(0, 1), B(0, 0), C(1, 0)
        a = [0, 1]
        b = [0, 0]
        c = [1, 0]
        angle = MathUtils.calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_angle_180_deg(self):
        # Straight line: A(-1, 0), B(0, 0), C(1, 0)
        a = [-1, 0]
        b = [0, 0]
        c = [1, 0]
        angle = MathUtils.calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_calculate_angle_0_deg(self):
        # Folded line (same direction): A(1, 0), B(0, 0), C(2, 0)
        a = [1, 0]
        b = [0, 0]
        c = [2, 0]
        angle = MathUtils.calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 0.0, places=1)

    def test_calculate_angle_3d_input(self):
        # MediaPipe provides 3D coords, we only use X,Y in arctan2 
        # (the 3rd coordinate is just ignored by indexing [0] and [1])
        a = [0, 1, 5]
        b = [0, 0, 5]
        c = [1, 0, 5]
        angle = MathUtils.calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_distance(self):
        a = [0, 0]
        b = [3, 4]
        dist = MathUtils.calculate_distance(a, b)
        self.assertEqual(dist, 5.0)

class TestSmoothingFilter(unittest.TestCase):
    def test_filter_smoothing(self):
        f = SmoothingFilter(alpha=0.5)
        self.assertEqual(f.update(100), 100) # El primer valor es exacto
        self.assertEqual(f.update(50), 75)   # (0.5 * 50) + (0.5 * 100) = 75
        self.assertEqual(f.update(75), 75)   # Se estabiliza
        
    def test_filter_reset(self):
        f = SmoothingFilter(alpha=0.5)
        f.update(100)
        f.update(50)
        f.reset()
        self.assertEqual(f.update(10), 10)   # Tras reset, toma el valor exacto

if __name__ == '__main__':
    unittest.main()
