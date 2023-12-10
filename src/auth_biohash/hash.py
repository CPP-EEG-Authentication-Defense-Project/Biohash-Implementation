import enum
import io
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
    """
    Core BioHash class representing a single hash instance.
    """
    def __init__(self, content: str):
        self.content = content

    @classmethod
    def generate_hash(cls,
                      token: str,
                      features: np.ndarray,
                      strategy: ThresholdStrategy = ThresholdStrategy.MEDIAN) -> 'BioHash':
        """
        Generates a BioHash instance using the provided key data components.

        :param token: the token to use during random data generation.
        :param features: the features to use to generate the hash.
        :param strategy: the threshold strategy to use for binarization.
        :return: the BioHash instance.
        """
        matrix_generator = random_token.MatrixGenerator(token)
        token_matrix = matrix_generator.generate(len(features))
        threshold = cls.get_threshold(features, strategy)
        mixed_data = cls.mix_token_matrix(features, token_matrix)
        binary_data = cls.binarize_data(mixed_data, threshold)
        return cls(binary_data)

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

    @staticmethod
    def binarize_data(data: np.ndarray, threshold: float) -> str:
        """
        Converts the given 1-D data array into a binary string, using the given threshold to determine when to use
        a '0' or a '1' in the result.

        :param data: the data to binarize.
        :param threshold: the threshold to use during encoding.
        :return: the resulting binary string.
        """
        if data.ndim > 1:
            raise exceptions.BinarizationException(f'Expected data to be a 1D array, got {data.shape}')
        with io.StringIO() as stream:
            for element in data:
                if element <= threshold:
                    stream.write('0')
                else:
                    stream.write('1')
            return stream.getvalue()
