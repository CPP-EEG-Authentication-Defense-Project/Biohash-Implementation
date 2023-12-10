import enum
import numpy as np

from . import random_token, exceptions


class ThresholdStrategy(enum.Enum):
    """
    Constants representing different strategies to use for the biohash binarization threshold value.
    """
    MEDIAN = enum.auto()
    MEAN = enum.auto()
    ZERO = enum.auto()


class BioHash:
    def __init__(self, content: str):
        self.content = content

    @classmethod
    def generate_hash(cls,
                      token: str,
                      features: np.ndarray,
                      strategy: ThresholdStrategy = ThresholdStrategy.MEDIAN) -> 'BioHash':
        matrix_generator = random_token.MatrixGenerator(token)
        token_matrix = matrix_generator.generate(len(features))
        threshold = cls.get_threshold(features, strategy)
        # TODO: mix token matrix with feature vector, binarize data

    @staticmethod
    def get_threshold(feature_data: np.ndarray, strategy: ThresholdStrategy) -> float:
        """
        Retrieves the threshold for determining bits during binarization, based on the given strategy.

        :param feature_data: the feature data to use to calculate the threshold.
        :param strategy: the strategy to use.
        :return: the threshold value.
        """
        match strategy:
            case ThresholdStrategy.MEDIAN:
                return np.median(feature_data)
            case ThresholdStrategy.MEAN:
                return np.mean(feature_data)
            case ThresholdStrategy.ZERO:
                return 0
        raise TypeError('Invalid strategy')

    @staticmethod
    def mix_token_matrix(feature_data: np.ndarray, token_matrix: np.ndarray) -> np.ndarray:
        """
        Combines the given feature vector with the given token matrix.

        :param feature_data:
        :param token_matrix:
        :return:
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
