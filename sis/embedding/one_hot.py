from typing import Any, Callable, Iterable, List, Optional
import numpy as np

# STANDARD_AMINO_ACID_MAPPING_TO_1_3: Dict[str, str] = {
#     "A": "ALA",
#     "C": "CYS",
#     "D": "ASP",
#     "E": "GLU",
#     "F": "PHE",
#     "G": "GLY",
#     "H": "HIS",
#     "I": "ILE",
#     "K": "LYS",
#     "L": "LEU",
#     "M": "MET",
#     "N": "ASN",
#     "O": "PYL",
#     "P": "PRO",
#     "Q": "GLN",
#     "R": "ARG",
#     "S": "SER",
#     "T": "THR",
#     "U": "SEC",
#     "V": "VAL",
#     "W": "TRP",
#     "Y": "TYR",
#     "X": "UNK",
# }


"""Vocabulary of 20 standard amino acids."""


class embedding(object):
    def __init__(self):
        self.dim = None
        self.dtype = None

    def encode(self):
        NotImplementedError()


class amino_acid_one_hot(object):
    def __init__(self, alphabet: Optional[List[str]] = None) -> None:
        self.alphabet = alphabet if alphabet is not None else BASE_AMINO_ACIDS

        self.dim = len(self.alphabet)
        self.dtype = "float32"

    def encode(self, seq: str):
        """
        amino_acid_one_hot Adds a one-hot encoding of seq by alphabet

        Args:
            seq (str): _description_

        Returns:
            _type_: _description_
        """

        return np.array([[aa == s for s in self.alphabet] for aa in seq]).astype(
            self.dtype
        )
