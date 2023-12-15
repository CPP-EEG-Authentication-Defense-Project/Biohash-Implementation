import numpy as np

from . import random_token, exceptions, binary_encoding


class BioHash:
    """
    Core BioHash class representing a single hash instance.
    """
    def __init__(self, content: str):
        self.content = content

    @classmethod
    def generate_hash(cls,
                      token: str,
                      features: np.ndarray,
                      encoder: binary_encoding.BinaryEncoder) -> 'BioHash':
        """
        Generates a BioHash instance using the provided key data components.

        :param token: the token to use during random data generation.
        :param features: the features to use to generate the hash.
        :param encoder: an encoder that is used to convert data to binary format.
        :return: the BioHash instance.
        """
        matrix_generator = random_token.MatrixGenerator(token)
        token_matrix = matrix_generator.generate(len(features))
        mixed_data = cls.mix_token_matrix(features, token_matrix)
        binary_data = encoder.encode(mixed_data)
        return cls(binary_data)

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
