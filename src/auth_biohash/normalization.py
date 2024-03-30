import numpy as np

from . import random_token, exceptions


class TokenMatrixNormalization:
    """
    Specialized object which mixes feature data with an orthogonal matrix generated using a unique random
    token.
    """
    def __init__(self, matrix_generator: random_token.MatrixGenerator):
        self._matrix_generator = matrix_generator

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Executes token-matrix normalization on the given data vector. Effectively "mixing" the data vector
        with a token-seeded random matrix of data.

        :param data: The data to normalize.
        :return: The normalized data vector.
        """
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
