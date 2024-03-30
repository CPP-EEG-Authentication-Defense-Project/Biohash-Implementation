import numpy as np
import typing
from feature_encoding import base

from . import random_token, exceptions


class TokenMatrixNormalization:
    """
    Specialized object which mixes feature data with an orthogonal matrix generated using a unique random
    token.
    """
    def __init__(self, matrix_generator: random_token.MatrixGenerator):
        self._matrix_generator = matrix_generator

    def normalize(self, data: np.ndarray) -> np.ndarray:
        token_matrix = self._matrix_generator.generate(len(data))
        return self.mix_token_matrix(data, token_matrix)

    @staticmethod
    def mix_token_matrix(feature_data: np.ndarray, token_matrix: np.ndarray) -> np.ndarray:
        """
        Combines the given feature vector with the given token matrix
        using the inner product of each row in the token matrix with the given feature vector.

        :param feature_data: the feature data to combine with the token matrix.
        :param token_matrix: the token matrix containing at least enough rows to combine with the given feature vector.
        :return: the mixed feature vector.
        """
        try:
            rows, columns = token_matrix.shape
            if len(feature_data) > rows:
                raise ValueError(
                    f'Count of feature data elements must be '
                    f'<= to the number of rows in token matrix! '
                    f'(expected at least {len(feature_data)} rows, got {rows})'
                )
        except ValueError as e:
            raise exceptions.CalculationException('Invalid token matrix shape') from e
        mixed_data = []
        for vector in token_matrix:
            mixed_data.append(
                np.inner(feature_data, vector)
            )
        return np.array(mixed_data)


class BioHash:
    """
    Core BioHash class representing a single hash instance.
    """
    def __init__(self, content: str):
        self.content = content

    @classmethod
    def generate_hash(cls,
                      features: np.ndarray,
                      token: typing.Union[int, float, str],
                      encoder: base.BinaryEncoder) -> 'BioHash':
        """
        Generates a BioHash instance using the provided key data components.

        :param features: the features to use to generate the hash.
        :param token: A user-specific token to use during hash generation.
        :param encoder: a feature encoder to use to encode the feature data into a binary string.
        :return: the BioHash instance.
        """
        matrix_generator = random_token.MatrixGenerator(token)
        normalizer = TokenMatrixNormalization(matrix_generator)
        normalized_features = normalizer.normalize(features)
        binary_data = encoder.encode(normalized_features)
        return cls(binary_data)

    @staticmethod
    def compare(hash_a: 'BioHash', hash_b: 'BioHash') -> float:
        """
        Compares two BioHash instances, returning a float number between 0 and 1 (inclusive) that represents
        the percentage difference between the two hashes.
        This similarity is based on hamming distance,
        in a manner described in https://doi.org/10.1109/TIFS.2022.3204222.

        :param hash_a: The first hash to compare.
        :param hash_b: The second hash to compare.
        :return: The percentage difference between the two hashes.
        """
        xor_result = int(str(hash_a), 2) ^ int(str(hash_b), 2)
        xor_bin = '{0:b}'.format(xor_result)
        bits_changed = sum([int(bit) for bit in xor_bin])
        max_len = max(len(hash_a), len(hash_b))
        percentage_change = bits_changed / max_len
        return percentage_change

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)
