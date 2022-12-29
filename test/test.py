# import sys

# sys.path.append("/p300s/wangmx_group/xutingfeng/SIS/")

from sis.dataset import SISDataset

a = SISDataset()
print(a.dataset_dict)
print(a.dataset_dict["train"][0])
