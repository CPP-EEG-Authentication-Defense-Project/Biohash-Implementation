import abc
import numpy as np

from .. import exceptions


class BinaryEncoder(metaclass=abc.ABCMeta):
    """
    Base class defining the interface for classes implementing binary encoding of feature data.
    """
    def encode(self, data: np.ndarray) -> str:
        """
        Takes the given data vector and encodes it into a binary string.

        :param data: the vector to encode.
        :return: the encoded data.
        """
        if data.ndim > 1:
            raise exceptions.BinarizationException(f'Expected data to be a 1D array, got {data.shape}')
        return self.convert_to_binary(data)

    @abc.abstractmethod
    def convert_to_binary(self, data: np.ndarray) -> str:
        """
        Implements the process of converting the given data vector into a binary string.

        :param data: the data vector to convert.
        :return: the binary data string.
        """
        pass
