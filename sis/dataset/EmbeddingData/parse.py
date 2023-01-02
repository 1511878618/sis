import pandas as pd
from sis.dataset.constants import STANDARD_AMINO_ACID_MAPPING_TO_3_1, UNK, PAD
from pathlib import Path
import numpy as np

expasy_path = Path(__file__).parent / "amino_acid_properties.csv"
meiler_path = Path(__file__).parent / "meiler_embeddings.csv"


def load_expasy_embedding_dict(path=expasy_path):
    expasy = pd.read_csv(path, sep=",", index_col=0).T

    state_dict = {
        STANDARD_AMINO_ACID_MAPPING_TO_3_1[idx]: value.values
        for idx, value in expasy.iterrows()
    }
    state_dict[UNK] = expasy.mean().values
    state_dict[PAD] = np.zeros((expasy.shape[1]))

    return state_dict


def load_meiler_embedding_dict(path=meiler_path):

    meiler = pd.read_csv(path, sep=",", index_col=0).T
    state_dict = {
        STANDARD_AMINO_ACID_MAPPING_TO_3_1[idx]: value.values
        for idx, value in meiler.iterrows()
    }

    state_dict[UNK] = meiler.mean().values
    state_dict[PAD] = np.zeros((meiler.shape[1]))
    return state_dict
