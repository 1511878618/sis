import pandas as pd
from sis.dataset.constants import STANDARD_AMINO_ACID_MAPPING_TO_3_1
from pathlib import Path

expasy_path = Path(__file__).parent / "amino_acid_properties.csv"
meiler_path = Path(__file__).parent / "meiler_embeddings.csv"


def load_expasy_embedding_dict(path=expasy_path):
    expasy = pd.read_csv(path, sep=",", index_col=0).T
    state_dict = {
        STANDARD_AMINO_ACID_MAPPING_TO_3_1[idx]: value.values
        for idx, value in expasy.iterrows()
    }
    return state_dict


def load_meiler_embedding_dict(path=meiler_path):

    meiler = pd.read_csv(path, sep=",", index_col=0).T
    state_dict = {
        STANDARD_AMINO_ACID_MAPPING_TO_3_1[idx]: value.values
        for idx, value in meiler.iterrows()
    }
    return state_dict


print(load_meiler_embedding_dict())
print(load_expasy_embedding_dict())
