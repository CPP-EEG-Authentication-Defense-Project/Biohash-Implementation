import unittest
import random
import string

from auth_biohash import random_token


class RandomTokenTestCase(unittest.TestCase):
    def test_random_token_generation(self):
        token = random_token.generate_token(size=256)

        self.assertIsInstance(token, str)
        # The hex token generation doubles the size used, so we expect the token to be double the given size.
        self.assertEqual(len(token), 256 * 2)

    def test_random_matrix_generation(self):
        token = ''.join(random.choice(string.ascii_lowercase) for _ in range(32))
        generator = random_token.MatrixGenerator(token)

        matrix = generator.generate(4)

        self.assertEqual(matrix.shape, (4, 4))
