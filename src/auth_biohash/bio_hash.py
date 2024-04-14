import auth_token_normalization
import numpy as np
import re
import typing

from . import protocols, exceptions


VALID_HASH_PATTERN = re.compile(r'^[01]+$')


class BioHash:
    """
    Core BioHash class representing a single hash instance.
    """
    def __init__(self, content: str):
        check = VALID_HASH_PATTERN.match(content)
        if not check:
            raise exceptions.InvalidHashException(f'{content} is not a valid BioHash.')
        self.content = content

    @classmethod
    def generate_hash(cls,
                      features: np.ndarray,
                      token: typing.Union[int, float, str],
                      encoder: protocols.EncoderProtocol,
                      normalization_pipeline: protocols.NormalizationPipelineProtocol = None) -> 'BioHash':
        """
        Generates a BioHash instance using the provided key data components.

        :param features: the features to use to generate the hash.
        :param token: A user-specific token to use during hash generation.
        :param encoder: A feature encoder to use to encode the feature data into a binary string.
        :param normalization_pipeline: Additional normalization steps to run against the hash data.
        :return: the BioHash instance.
        """
        normalized_features = auth_token_normalization.normalize_data(features, token)
        if normalization_pipeline is not None:
            normalized_features = normalization_pipeline.run(normalized_features)
        binary_data = encoder.encode(normalized_features)
        return cls(binary_data)

    @staticmethod
    def compare(hash_a: 'BioHash', hash_b: 'BioHash') -> float:
        """
        Compares two BioHash instances, returning a float number between 0 and 1 (inclusive) that represents
        the percentage difference between the two hashes.
        This similarity is based on hamming distance,
        in a manner described in https://doi.org/10.1109/TIFS.2022.3204222.

        :param hash_a: The first hash to compare.
        :param hash_b: The second hash to compare.
        :return: The percentage difference between the two hashes.
        """
        xor_result = int(str(hash_a), 2) ^ int(str(hash_b), 2)
        xor_bin = '{0:b}'.format(xor_result)
        bits_changed = sum([int(bit) for bit in xor_bin])
        max_len = max(len(hash_a), len(hash_b))
        percentage_change = bits_changed / max_len
        return percentage_change

    def __str__(self):
        return str(self.content)

    def __len__(self):
        return len(self.content)
