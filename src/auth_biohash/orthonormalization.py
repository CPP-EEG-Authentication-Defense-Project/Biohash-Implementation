import secrets
import typing
import random
import sys
import numpy as np


class TokenDataGenerator:
    """
    Manages the generation of user tokens as well as the generation of matrix data
    to be used for normalization operations.
    """
    def __init__(self, token: typing.Union[int, float, str]):
        self._token = token

    @staticmethod
    def generate_random_token(size: int = 256) -> str:
        """
        Generates a secure random token to be used as the basis for matrix normalization.

        :param size: The size, in bytes, of the token to be generated.
        :returns: The generated token.
        """
        return secrets.token_hex(size)

    def generate_matrix(self, dimension: int) -> np.ndarray:
        """
        Generates an orthagonalized random matrix based on the current token.

        :param dimension: the dimension of the matrix (e.g., 2 will produce a 2x2 matrix).
        :returns: the generated matrix.
        """
        random_source = random.Random(self._token)
        matrix = []
        for _ in range(dimension):
            basis = np.array([
                random_source.randrange(1, sys.maxsize)
                for _ in range(dimension)
            ])
            matrix.append(basis)
        orthogonalized_matrix, triangular = np.linalg.qr(matrix)
        return orthogonalized_matrix


class TokenMatrixNormalization:
    """
    Specialized object which mixes feature data with an orthogonal matrix generated using a unique random
    token.
    """
    def __init__(self, matrix_generator: TokenDataGenerator):
        self._matrix_generator = matrix_generator

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Executes token-matrix normalization on the given data vector. Effectively "mixing" the data vector
        with a token-seeded random matrix of data.

        :param data: The data to normalize.
        :return: The normalized data vector.
        """
        token_matrix = self._matrix_generator.generate_matrix(len(data))
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
        rows, columns = token_matrix.shape
        if len(feature_data) > rows:
            raise ValueError(
                f'Count of feature data elements must be '
                f'<= to the number of rows in token matrix! '
                f'(expected at least {len(feature_data)} rows, got {rows})'
            )
        mixed_data = []
        for vector in token_matrix:
            mixed_data.append(
                np.inner(feature_data, vector)
            )
        return np.array(mixed_data)
