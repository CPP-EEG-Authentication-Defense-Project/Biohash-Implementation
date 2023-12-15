import abc
import numpy as np


class BinaryEncoder(metaclass=abc.ABCMeta):
    """
    Base class defining the interface for classes implementing binary encoding of feature data.
    """
    @abc.abstractmethod
    def encode(self, data: np.ndarray) -> str:
        """
        Takes the given data vector and encodes it into a binary string.

        :param data: the vector to encode.
        :return: the encoded data.
        """
        pass
