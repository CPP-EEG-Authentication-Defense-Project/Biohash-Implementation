import unittest
import unittest.mock
import random
import string
import numpy as np

from auth_biohash import bio_hash, protocols, exceptions


class DummyEncoder(protocols.EncoderProtocol):
    def encode(self, data: np.ndarray) -> str:
        return ''.join('1' for _ in range(len(data)))


class HashTestCase(unittest.TestCase):
    def test_hash_instance_builtins(self):
        fake_hash_data = ''.join(random.choice(('0', '1')) for _ in range(32))
        instance = bio_hash.BioHash(fake_hash_data)

        self.assertEqual(str(instance), fake_hash_data)
        self.assertEqual(len(instance), len(fake_hash_data))

    def test_invalid_hash_data(self):
        fake_hash_data = ''.join(random.choice(string.ascii_lowercase) for _ in range(32))

        def create_instance(data: str):
            return bio_hash.BioHash(data)

        self.assertRaises(exceptions.InvalidHashException, create_instance, fake_hash_data)

    def test_generate_hash(self):
        fake_data = np.ones((4,))
        token = ''.join(random.choice(string.ascii_lowercase) for _ in range(32))
        encoder = DummyEncoder()
        with unittest.mock.patch('auth_biohash.random_token.MatrixGenerator'):
            with unittest.mock.patch('auth_biohash.normalization.TokenMatrixNormalization') as FakeNormalization:
                FakeNormalization.return_value.normalize.return_value = np.ones((4,))
                hash_instance = bio_hash.BioHash.generate_hash(fake_data, token, encoder)

        self.assertIsInstance(hash_instance, bio_hash.BioHash)
        self.assertEqual(len(hash_instance.content), len(fake_data))

    def test_compare_equal_instances(self):
        fake_hash_string = ''.join('1' for _ in range(32))
        hash_a = bio_hash.BioHash(fake_hash_string)
        hash_b = bio_hash.BioHash(fake_hash_string)

        difference = bio_hash.BioHash.compare(hash_a, hash_b)

        self.assertEqual(difference, 0)

    def test_compare_non_equal_instances(self):
        fake_hash_string_a = ''.join('1' for _ in range(32))
        fake_hash_string_b = ''.join('0' for _ in range(32))
        hash_a = bio_hash.BioHash(fake_hash_string_a)
        hash_b = bio_hash.BioHash(fake_hash_string_b)

        difference = bio_hash.BioHash.compare(hash_a, hash_b)

        self.assertEqual(difference, 1)

    def test_compare_similar_instances(self):
        hash_a = bio_hash.BioHash('1111111111')
        hash_b = bio_hash.BioHash('1111111110')

        difference = bio_hash.BioHash.compare(hash_a, hash_b)

        self.assertEqual(difference, 0.1)
