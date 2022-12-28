from datasets import load_dataset, Array2D
from functools import partial
import torchtext.transforms as T


# from constant import BASE_AMINO_ACIDS
# from vocab import build_vocab_from_alphabet_dict
# from transformer import AminoAcidsTokenizer


class SISDataset(object):
    def __init__(
        self,
        SLF_max_length=420,
        SRnase_max_length=230,
        alphabet=BASE_AMINO_ACIDS,
        root_dir="../data/total_data.csv",
        test_size=0.2,
    ):
        self.aa_vocab = build_vocab_from_alphabet_dict(alphabet)

        self.SLF_transform = T.Sequential(
            AminoAcidsTokenizer(),
            T.VocabTransform(self.aa_vocab),
            T.Truncate(SLF_max_length),
            T.ToTensor(self.aa_vocab["<pad>"]),
            T.PadTransform(SLF_max_length, pad_value=self.aa_vocab["<pad>"]),
        )
        self.SRnase_transform = T.Sequential(
            AminoAcidsTokenizer(),
            T.VocabTransform(self.aa_vocab),
            T.Truncate(SRnase_max_length),
            T.ToTensor(self.aa_vocab["<pad>"]),
            T.PadTransform(SRnase_max_length, pad_value=self.aa_vocab["<pad>"]),
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
        )

        dataset = dataset.map(preprocessing_partial, num_proc=4)

        # step2 to torch type
        selected_columns = ["SLF_Seq_token", "SRnase_Seq_token", "label"]

        dataset = dataset.with_format(
            type="torch", columns=selected_columns, output_all_columns=True
        )

        # step3 train_test_split
        dataset_dict = dataset.train_test_split(test_size=self.test_size, seed=20)
        return dataset_dict

    @staticmethod
    def preproces(item, SLF_transform, SRnase_transform):

        # use label:-1 data or not , 0 meas use while 1 means do not use
        item["label"] = 0 if item["label"] == -1 else item["label"]

        # tokenize
        item["SLF_Seq_token"] = SLF_transform(item["SLF_Seq"])
        item["SRnase_Seq_token"] = SRnase_transform(item["SRnase_Seq"])

        return item


# class sis_dataset(object):
#     def __init__(
#         self,
#         embedding_func,
#         root_dir="../data/total_data.csv",
#         mode=0,
#         max_len=None,
#         test_size=0.2,
#         col=None,
#     ):
#         self.embedding_func = embedding_func
#         self.max_len = max_len
#         self.test_size = test_size
#         self.col = (
#             col if col is not None else ["SLF_Seq", "SRnase_Seq"]
#         )  # str or list default ["SLF_Seq", "SRnase_Seq"]
#         self.data_dir = root_dir
#         self.mode = mode  # 0 means label:-1 as 0  while 1 means do not use -1

#         self.dataset_dict = self._load()

#     def _load(self):
#         dataset_dict = load_dataset("csv", data_files=self.data_dir)

#         dataset = dataset_dict["train"]

#         # step1 preprocessing

#         preprocessing_partial = partial(
#             self.preproces,
#             embedding_func=self.embedding_func,
#             col=self.col,
#             max_len=self.max_len,
#         )

#         dataset = dataset.map(preprocessing_partial, num_proc=4)

#         embedding_columns = [
#             i + f"_{self.embedding_func.__class__.__name__}" for i in self.col
#         ]
#         embedding_dim = self.embedding_func.dim
#         dtype = self.embedding_func.dtype
#         for columns in embedding_columns:
#             dataset = dataset.cast_column(
#                 columns, Array2D((self.max_len, embedding_dim), dtype=dtype)
#             )

#         # step2 to torch type
#         selected_columns = [
#             i + f"_{self.embedding_func.__class__.__name__}" for i in self.col
#         ] + ["label"]

#         dataset = dataset.with_format(
#             type="torch", columns=selected_columns, output_all_columns=True
#         )
#         # step3 train_test_split
#         dataset_dict = dataset.train_test_split(test_size=self.test_size, seed=20)
#         return dataset_dict

#     @staticmethod
#     def preproces(item, embedding_func, col, max_len):

#         # use label:-1 data or not , 0 meas use while 1 means do not use
#         item["label"] = 0 if item["label"] == -1 else item["label"]

#         # padding
#         if max_len is not None:
#             pass

#         # embedding
#         if isinstance(col, str):
#             item[col] = embedding_func.encode(item[col])
#         elif isinstance(col, list):
#             for i in col:
#                 item[
#                     i + f"_{embedding_func.__class__.__name__}"
#                 ] = embedding_func.encode(item[i])
#         else:
#             raise ValueError("col:{col} is not str or list")
#         return item
