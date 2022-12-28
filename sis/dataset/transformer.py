from torch.nn import Module
import torch
from typing import Any, List, Optional, Union


class AminoAcidsTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additiona details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SentencePieceTokenizer
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(list(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return list(input)
        else:
            raise TypeError("Input type not supported")
