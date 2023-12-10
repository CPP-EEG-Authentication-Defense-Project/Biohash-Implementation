# Biohash Implementation

This repository contains a Python-based implementation of the Biohash algorithm originally
proposed by [Andrew Teoh and David Ngo](https://doi.org/10.1016/j.patrec.2004.11.021). This 
implementation of the Biohash algorithm exposes a simple interface capable of transforming or
comparing generic feature vector data. Therefore, this module is agnostic to the specific type
of biometrics being used, as long as feature data can be expressed in a vector of float values.

## Disclaimer

This implementation is intended for research purposes and is not considered production ready.
