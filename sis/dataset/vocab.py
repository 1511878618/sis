from torchtext.vocab import vocab
from collections import Counter

from sis.dataset.constants import BASE_AMINO_ACIDS


def build_vocab_from_alphabet_dict(alphabet=BASE_AMINO_ACIDS):
    unk = "<unk>"
    padding = "<pad>"

    aaVocab = vocab(Counter(alphabet), specials=[unk, padding], special_first=True)
    aaVocab.set_default_index(aaVocab[unk])
    return aaVocab
