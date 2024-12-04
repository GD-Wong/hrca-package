import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
import random
import torch
import torch.nn.functional as F
from scipy import sparse
from . import utils

###################################
# 抽取空间转录组与单细胞转录组数据
###################################
class STandSC(Dataset):
    def __init__(self, st_file, sc_file):
        spatial_adata = sc.read(st_file)
        singlecell_adata = sc.read(sc_file)
        self.shared_genes = list(singlecell_adata.var.index)
        singlecell_adata = singlecell_adata[:, self.shared_genes]
        all_adata = sc.concat({"st": spatial_adata, "sc": singlecell_adata}, label="data_type", join='outer', axis=0, fill_value=0)[:,self.shared_genes]
        spatial_adata = all_adata[all_adata.obs.data_type == "st", :]
        sc.pp.filter_cells(spatial_adata, min_genes=80)
        # 单细胞数据标准化
        sc.pp.normalize_total(singlecell_adata, target_sum=10000)
        # sc.pp.log1p(singlecell_adata)
        # 空间转录组数据标准化
        # sc.pp.normalize_total(spatial_adata, target_sum = scale_factor)
        self.singlecell = singlecell_adata.X
        self.spatial = spatial_adata.X

    def __len__(self):
        return self.spatial.shape[0]

    def __getitem__(self, idx):
        spot = self.spatial[idx, :]
        cell = self.singlecell[random.randint(0, self.singlecell.shape[1] - 1), :]
        if sparse.issparse(cell):
            cell = cell.toarray()
        if sparse.issparse(spot):
            spot = spot.toarray()
        cell = torch.squeeze(torch.FloatTensor(cell))
        spot = torch.squeeze(torch.FloatTensor(spot))
        return {"spot": spot, "cell": cell}


# 输入空间转录组数据准备转换
class InputLoader(Dataset):
    def __init__(self, input_adata, ref_adata, norm_flag = False, dropout_flag = False, dropout_rate=0.7):
        genes = list(ref_adata.var.index)
        all_adata = sc.concat({"input":input_adata, "ref":ref_adata}, label = "data_type", join="outer", axis=0)[:, genes]
        input_adata = all_adata[all_adata.obs.data_type=="input", :]
        sc.pp.filter_cells(input_adata, min_genes=80)
        if norm_flag:
            sc.pp.normalize_total(input_adata, target_sum=1e4)
            sc.pp.log1p(input_adata)
        self.input = input_adata.X
        self.obs = input_adata.obs
        self.var = input_adata.var
        self.obsm = input_adata.obsm
        self.dropout_rate = dropout_rate
        self.dropout_flag = dropout_flag
    def __len__(self):
        return self.input.shape[0]
    def __getitem__(self, idx):
        spot = self.input[idx, :]
        if sparse.issparse(spot):
            spot = spot.toarray()
        spot = torch.squeeze(torch.FloatTensor(spot))
        spot = utils.dropout(spot, dropout_rate=self.dropout_rate, dropout_flag=self.dropout_flag)
        return spot

#输入文件名
class AnnDataLoader(Dataset):
    def __init__(self, adata, obs_name = None, dropout = False, dropout_rate = 0.7, return_label = False):
        self.obs_name = obs_name
        if self.obs_name != None:
            self.label, self.dict = utils.label_str_to_int(list(adata.obs[[self.obs_name]].iloc[:,0]))
            adata.obs["label_int"] = self.label
        self.input = adata
        self.obs = adata.obs
        self.var = adata.var
        self.obsm = adata.obsm
        self.drop_flag = dropout
        self.dropout_rate = dropout_rate
        self.return_label = return_label
    def __len__(self):
            return self.input.shape[0]
    def __getitem__(self, idx):
        sub_adata = self.input[idx, :]
        x = sub_adata.X
        if sparse.issparse(x):
            x = x.toarray()
        x = torch.squeeze(torch.FloatTensor(x))
        x = utils.dropout(x, dropout_rate=self.dropout_rate, dropout_flag=self.drop_flag)
        if self.return_label:
            l = np.array(sub_adata.obs.label_int)
            #l = self.label[idx]
            return x,l
        else:
            return x

#直接输入数据
class AdataLoader(Dataset):
    def __init__(self, adata, norm_flag = False):
        if norm_flag:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        self.obs = adata.obs
        self.input = adata.X
    def __len__(self):
        return self.input.shape[0]
    def __getitem__(self, idx):
        x = self.input[idx, :]
        if sparse.issparse(x):
            x = x.toarray()
        x = torch.squeeze(torch.FloatTensor(x))
        return x