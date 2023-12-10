import secrets
import sys
import typing
import random
import numpy as np


def generate_token(size: int = 256) -> str:
    """
    Helper function to generate a secure random token to be used as the basis for
    matrix generations.

    :param size: The size of the generated token.
    :return: the generated token.
    """
    return secrets.token_hex(size)


class MatrixGenerator:
    """
    Helper class wrapping the matrix generation process, based on a given random token.
    """
    def __init__(self, token: typing.Union[int, float, str]):
        self._token = token

    def generate(self, dimension: int) -> np.ndarray:
        """
        Generates an orthagonalized random matrix based on the current token.

        :param dimension: the dimension of the matrix (e.g., 2 will produce a 2x2 matrix).
        :return: the generated matrix.
        """
        random_source = random.Random(self._token)
        matrix = []
        for _ in range(dimension):
            basis = np.ndarray([
                random_source.randrange(1, sys.maxsize)
                for _ in range(dimension)
            ])
            matrix.append(basis)
        orthogonalized_matrix, triangular = np.linalg.qr(matrix)
        return orthogonalized_matrix
