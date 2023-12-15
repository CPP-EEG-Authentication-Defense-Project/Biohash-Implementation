import enum
import io

from . import base
from .. import exceptions


class ThresholdBinaryEncoder(base.BinaryEncoder):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def encode(self, data):
        """
        Converts the given 1-D data array into a binary string, using the current threshold to determine when to use
        a '0' or a '1' in the result.

        :param data: the data to binarize.
        :return: the resulting binary string.
        """
        if data.ndim > 1:
            raise exceptions.BinarizationException(f'Expected data to be a 1D array, got {data.shape}')
        with io.StringIO() as stream:
            for element in data:
                if element <= self.threshold:
                    stream.write('0')
                else:
                    stream.write('1')
            return stream.getvalue()
