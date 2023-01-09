from Bio import AlignIO
import os.path as osp


def load_MSA(msa_path)

root_dir = "/p300s/wangmx_group/xutingfeng/SIS/data/alignment"

SRnase_aln = AlignIO.read(osp.join(root_dir, "SRnase.aln"), "clustal")
SLF_aln = AlignIO.read(osp.join(root_dir, "SLF.aln"), "clustal")
