import typing
import numpy as np


class EncoderProtocol(typing.Protocol):
    """
    Protocol for binary encoders, which can be used with the BioHash interface.
    """
    def encode(self, data: np.ndarray) -> str:
        pass
