from functools import partial

import pandas as pd
import torch
import torchtext.transforms as T
from datasets import load_dataset

from sis.dataset.constants import BASE_AMINO_ACIDS, PAD
from sis.dataset.sis_dataset import SISDataset
from sis.dataset.transform import AminoAcidsTokenizer
from sis.dataset.vocab import build_vocab_from_alphabet_dict


class SISDataset(object):
    def __init__(
        self,
        SLF_max_length=420,
        SRnase_max_length=230,
        alphabet=BASE_AMINO_ACIDS,
        root_dir="./total_data.csv",
        test_size=0.2,
        padding="<pad>",
        device="cpu",
    ):
        self.aa_vocab = build_vocab_from_alphabet_dict(alphabet)
        self.padding = padding
        self.SLF_max_length = SLF_max_length
        self.SRnase_max_length = SRnase_max_length
        self.alphabet = alphabet
        self.device = device

        self.SLF_transform = T.Sequential(
            AminoAcidsTokenizer(),
            T.VocabTransform(self.aa_vocab),
            T.Truncate(self.SLF_max_length),
            T.ToTensor(self.aa_vocab[self.padding]),
            T.PadTransform(self.SLF_max_length, pad_value=self.aa_vocab[self.padding]),
        )
        self.SRnase_transform = T.Sequential(
            AminoAcidsTokenizer(),
            T.VocabTransform(self.aa_vocab),
            T.Truncate(self.SRnase_max_length),
            T.ToTensor(self.aa_vocab[self.padding]),
            T.PadTransform(
                self.SRnase_max_length, pad_value=self.aa_vocab[self.padding]
            ),
        )

        self.test_size = test_size
        self.data_dir = root_dir
        self.dataset_dict = self._load()

    def _load(self):
        dataset_dict = load_dataset("csv", data_files=self.data_dir)

        dataset = dataset_dict["train"]

        # step1 preprocessing

        preprocessing_partial = partial(
            self.preproces,
            SLF_transform=self.SLF_transform,
            SRnase_transform=self.SRnase_transform,
            padding_id=self.aa_vocab[self.padding],
        )

        dataset = dataset.map(preprocessing_partial, num_proc=4)

        # step2 to torch type
        selected_columns = [
            "SLF_Seq_token",
            "SRnase_Seq_token",
            "label",
            "SLF_Seq_mask",
            "SRnase_Seq_mask",
        ]

        dataset = dataset.with_format(
            type="torch",
            columns=selected_columns,
            output_all_columns=True,
            device=self.device,
        )

        # step3 train_test_split
        dataset_dict = dataset.train_test_split(test_size=self.test_size, seed=20)
        return dataset_dict

    @staticmethod
    def preproces(item, SLF_transform, SRnase_transform, padding_id=1):

        # use label:-1 data or not , 0 meas use while 1 means do not use
        item["label"] = 0 if item["label"] == -1 else item["label"]

        # tokenize
        item["SLF_Seq_token"] = SLF_transform(item["SLF_Seq"])
        item["SRnase_Seq_token"] = SRnase_transform(item["SRnase_Seq"])

        item["SLF_Seq_mask"] = item["SLF_Seq_token"] == padding_id
        item["SRnase_Seq_mask"] = item["SRnase_Seq_token"] == padding_id

        return item


class SIS_MSADataset(SISDataset):
    def __init__(
        self,
        data_dir,
        device="cpu",
        test_size=0.2,
        padding=PAD,
        alphabet=BASE_AMINO_ACIDS,
    ):
        SLF_max_length, SRnase_max_length = self.__get_max_length__(data_dir=data_dir)

        super(SIS_MSADataset, self).__init__(
            SLF_max_length=SLF_max_length,
            SRnase_max_length=SRnase_max_length,
            alphabet=alphabet,
            root_dir=data_dir,
            device=device,
            padding=padding,
            test_size=test_size,
        )

    def __get_max_length__(self, data_dir):
        data = pd.read_csv(data_dir, sep=",")
        SLF_max_length = len(data.iloc[0, :]["SLF_Seq"])
        SRnase_max_length = len(data.iloc[0, :]["SRnase_Seq"])
        return SLF_max_length, SRnase_max_length

    @staticmethod
    def preproces(item, SLF_transform, SRnase_transform, padding_id=1):

        # use label:-1 data or not , 0 meas use while 1 means do not use
        item["label"] = 0 if item["label"] == -1 else item["label"]

        # tokenize
        item["SLF_Seq_token"] = SLF_transform(item["SLF_Seq"])
        item["SRnase_Seq_token"] = SRnase_transform(item["SRnase_Seq"])

        item["SLF_Seq_mask"] = torch.zeros(len(item["SLF_Seq_token"])).bool()
        item["SRnase_Seq_mask"] = torch.zeros(len(item["SRnase_Seq_token"])).bool()

        return item
