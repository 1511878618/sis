# import sys
# sys.path.append("/p300s/wangmx_group/xutingfeng/SIS/")

from sis.dataset import SISDataset

a = SISDataset("/p300s/wangmx_group/xutingfeng/SIS/sis/dataset/total_data.csv")
print(a.dataset_dict)
print(a.dataset_dict["train"][0])
