import copy
import typing

from . import base
from .. import exceptions


class GrayCodeBinaryEncoder(base.BinaryEncoder):
    """
    Encodes data into binary using a Gray Code codebook. Note that the codebook is generated upfront using a given
    target bit size. Data will most likely need to be quantized before being passed through the encoder, to avoid
    values outside the codebook's range.
    """
    def __init__(self, n: int):
        self.gray_codes = self._generate_gray_codes(n)

    def convert_to_binary(self, data):
        """
        Converts the given data vector into binary using Gray Codes found in the current codebook.

        :param data: the data vector to encode.
        :return: the encoded data vector.
        """
        gray_codes = []
        codebook_size = len(self.gray_codes)
        for value in data:
            value_index = value - 1
            if value_index < 0 or value_index > codebook_size:
                raise exceptions.BinarizationException(
                    f'Value outside of codebook range (0, {codebook_size + 1}), quantization may be needed'
                )
            gray_codes.append(self.gray_codes[value])
        return ''.join(gray_codes)

    @staticmethod
    def _generate_gray_codes(n: int) -> typing.List[str]:
        """
        Generates a list of gray codes for the given bit size.

        :param n: the number of bits to generate.
        :return: the list of generated gray codes.
        """
        if n <= 0:
            return ['0']
        gray_codes = ['0', '1']
        for i in range(2, n + 1):
            previous_codes = copy.copy(gray_codes)
            gray_codes.clear()
            for code in previous_codes:
                gray_codes.append('0' + code)
            for code in reversed(previous_codes):
                gray_codes.append('1' + code)
        return gray_codes
