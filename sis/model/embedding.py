""" Define the Embedding """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Tingfeng Xu"


class OnehotLayer(nn.Module):
    def __init__(self, num_classes):
        super(OnehotLayer, self).__init__()
        self.d_model = num_classes

    def forward(self, x):
        return F.one_hot(x, self.d_model).float()


class EmbeddingLayer(nn.Module):
    """
    EmbeddingLayer 传入aa_embeding_dict和aa_vocab
    aa_embedding_dict is {"A":[1, 2, 3], ...}
    aa_vocab is `torchtext.vocab.vocab.Vocab`

    这两个包含的token应该是一致的

    Example:
        from sis.dataset.vocab import build_vocab_from_alphabet_dict
        from sis.dataset.constants import BASE_AMINO_ACIDS

        aa_vocab = build_vocab_from_alphabet_dict(BASE_AMINO_ACIDS)
        aa_embedding_dict = {aa:np.random.randint(0, 5, size=(5)) for aa in aa_vocab.get_itos()}

        EMLayer = EmbeddingLayer(aa_embedding_dict=aa_embedding_dict, aa_vocab=aa_vocab)

        EMLayer(torch.tensor([1, 3, 5, 1, 2, 4]))
    """

    def __init__(self, aa_embedding_dict, aa_vocab):
        super(EmbeddingLayer, self).__init__()

        if len(aa_embedding_dict.keys()) != len(aa_vocab):
            raise ValueError(
                "aa_embedding_dict should contain all token in aa_vocab, check whether special tokens are in aa_embedding_dict!"
            )

        aa_token_embedding_dict = {
            aa_vocab.lookup_indices([aa])[0]: value
            for aa, value in aa_embedding_dict.items()
        }

        sorted_embeeding_tuple = sorted(
            aa_token_embedding_dict.items(), key=lambda x: x[0]
        )  # (tokens, values) sorted by token at ascending order
        sorted_embedding_array = np.stack(
            [value for token, value in sorted_embeeding_tuple]
        ).astype(np.float32)

        self.sorted_embeeding_tuple = sorted_embeeding_tuple

        self.params = nn.ParameterDict(
            {
                "w": torch.tensor(sorted_embedding_array),
            }
        )

        self.d_model = self.params["w"].shape[1]
        self.tokens_num = self.params["w"].shape[0]

    def forward(self, x):
        x_onehot = F.one_hot(x, self.tokens_num).float()
        return x_onehot @ self.params["w"]
