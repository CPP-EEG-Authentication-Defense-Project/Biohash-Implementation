import unittest
import numpy as np

from auth_biohash import normalization, random_token


class DummyMatrixGenerator(random_token.MatrixGenerator):
    def generate(self, dimension: int) -> np.ndarray:
        return np.array([[2] * dimension] * dimension)


class NormalizationTestCase(unittest.TestCase):
    def test_normalization_method(self):
        test_data = np.ones((2,))
        generator = DummyMatrixGenerator('fake')
        normalizer = normalization.TokenMatrixNormalization(generator)
        expected_output = np.array([4, 4])

        normalized_data = normalizer.normalize(test_data)

        self.assertEqual(len(normalized_data), len(expected_output))
        for actual, expected in zip(normalized_data, expected_output):
            self.assertEqual(actual, expected)
