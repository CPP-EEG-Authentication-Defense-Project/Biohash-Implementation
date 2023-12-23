import typing
import numpy as np

from . import random_token, exceptions, binary_encoding


QuantizationStep = typing.Callable[[np.ndarray], np.ndarray]


class BioHashPipeline:
    """
    A pipeline used for building BioHash representations of feature data.
    """
    def __init__(self,
                 token: typing.Union[int, float, str],
                 encoder: binary_encoding.BinaryEncoder,
                 quantization_steps: typing.List[QuantizationStep] = None):
        self._matrix_generator = random_token.MatrixGenerator(token)
        self.encoder = encoder
        self.quantization_steps = quantization_steps or []

    def transform(self, features: np.ndarray) -> str:
        """
        Transforms the given feature vector into a raw binary encoded BioHash template string.

        :param features: the feature vector to transform.
        :return: the raw template string.
        """
        token_matrix = self._matrix_generator.generate(len(features))
        mixed_data = self.mix_token_matrix(features, token_matrix)
        step: QuantizationStep
        for step in self.quantization_steps:
            mixed_data = step(mixed_data)
        return self.encoder.encode(mixed_data)

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
    def __init__(self, content: str, validation_threshold: float):
        if validation_threshold < 0 or validation_threshold > 1:
            raise ValueError('validation_threshold must be between 0 and 1')
        self.content = content
        self.validation_threshold = validation_threshold

    @classmethod
    def generate_hash(cls,
                      features: np.ndarray,
                      validation_threshold: float,
                      pipeline: BioHashPipeline) -> 'BioHash':
        """
        Generates a BioHash instance using the provided key data components.

        :param features: the features to use to generate the hash.
        :param validation_threshold: the threshold to use in the generated BioHash instance.
        :param pipeline: a BioHashPipeline to use to build the BioHash template string.
        :return: the BioHash instance.
        """
        binary_data = pipeline.transform(features)
        return cls(binary_data, validation_threshold)

    def __str__(self):
        return str(self.content)

    def __eq__(self, other):
        if not isinstance(other, BioHash):
            return False
        xor_result = int(str(self), 2) ^ int(str(other), 2)
        xor_bin = '{0:b}'.format(xor_result)
        bits_changed = sum([int(bit) for bit in xor_bin])
        percentage_change = bits_changed / len(str(self))
        return percentage_change <= self.validation_threshold
