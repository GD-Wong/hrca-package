import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import metrics
import scanpy as sc
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import copy
# 转换数据格式(h5)
def transform_to_h5(data, path_store):
    uniq_columns = np.unique(data.columns, return_index=True)[1]
    data = data.iloc[:, uniq_columns]
    print(type(data))
    new = data.loc[:, data.sum(axis=0)>0]
    print("Dimension of the matrix after removing all-zero rows:", new.shape)
    new.to_hdf(path_store, key="dge", mode="w", complevel=3)

def softmax(df, axis=1):
    row_max = np.max(df, axis=axis).reshape(-1, 1)
    df -= row_max
    df_exp = np.exp(df)
    s = df_exp / np.sum(df_exp, axis=axis, keepdims=True)
    return s

def get_proportion(df, axis=1):
    df_abs = abs(df)
    df += np.sum(df_abs, axis=axis, keepdims=True)
    return df / np.sum(df, axis=axis, keepdims=True)

# Turn labels into dataframe
def one_hot_dataframe(labels):
    one_hot_df = pd.get_dummies(labels)
    dict_idx = one_hot_df.columns.to_list()
    return one_hot_df, dict_idx

# 将label字符转为数字
def label_str_to_int(labels):
    dict_idx = np.unique(labels)
    labels = pd.DataFrame(labels)
    for i in range(len(dict_idx)):
        labels = labels.replace(dict_idx[i], i)
    labels = list(labels.iloc[:,0])
    return labels, dict_idx

# 将label数字转为字符,label_int为列表
def label_int_to_str(label_int, dict_idx):
    labels = pd.DataFrame(label_int)
    for i in range(len(dict_idx)):
        labels = labels.replace(i, dict_idx[i])
    labels = list(labels.iloc[:, 0])
    return labels


# 统计错误数目,输入向量,行为真实，列为预测
def static_err_and_corr(y_ture, y_pred):
    celltype1 = sorted(np.unique(y_ture))
    celltype2 = sorted(np.unique(y_pred))
    df = pd.DataFrame(0, columns=celltype2, index=celltype1)
    len_num = len(y_ture)
    for i in range(len_num):
        df.loc[y_ture[i], y_pred[i]] += 1
    return df

# 得到字符结果
def get_strRes(Y_verify, dict_idx, threshold=0.35):
    from collections import Counter
    b = Y_verify.shape
    res = []
    for i in range(b[0]):
        a = Y_verify.iloc[i]
        a = a.tolist()
        max_a = max(a)
        if max_a < threshold:
            tmp_res = 'Unknown'
        else:
            idx = a.index(max(a))
            tmp_res = dict_idx[idx]
        res.append(tmp_res)
    print(Counter(res))
    return res

def get_accuracy(res, test_label, str=''):
    count = 0
    for i in range(test_label.shape[0]):
        if res[i] == test_label.iloc[i, 0] or res[i] == str:
            count = count + 1
    print(count / test_label.shape[0])
    return count / test_label.shape[0]

# 将空间转录组bin转化为矩阵
def gemb2table(gem_data):
    gem_data_table = gem_data.pivot(index='geneID', columns='bin_id', values='MIDCounts')
    gem_data_table.fillna(0, inplace=True)
    return gem_data_table

# 对数据进行对数标准化
def logNormalize(df, scalefactor=10000):
    sum_counts = df.sum(axis=1)
    df = df * scalefactor
    df = df.div(sum_counts, axis=0)
    df = np.log1p(df)
    return df

#对tensor数据进行标准化
def normalize_torch(x, scalefactor = 10000, log_flag = True):
    sum_counts = x.sum(axis=1)
    x = (x*scalefactor).div(sum_counts.unsqueeze(1))
    if log_flag:
        x = x.log1p()
    return x

def get_pca(K, data):
    model = decomposition.PCA(n_components=int(K)).fit(data)
    data = model.transform(data)
    return data

def sample_layers(label, each_sample_count):
    import random
    label_data_unique = np.unique(label)
    label = pd.DataFrame(label, columns=['celltype'])
    label['index'] = label.index
    sample_data_idx = [] # 定义空列表，用于存放最终抽样数据
    sample_dict = {} # 定义空字典，用来显示各分层样本数量
    for label_data in label_data_unique: # 遍历每个分层标签
        label_tmp = label.loc[label['celltype']==label_data] # 得到一层的数据
        label_tmp = label_tmp.reset_index()
        if label_tmp.shape[0] > each_sample_count:
            label_tmp = label_tmp.loc[random.sample(range(label_tmp.shape[0]), each_sample_count)] # 对每层数据都随机抽样
        sample_data_idx.extend(list(label_tmp['index'])) # 将抽样数据追加到总体样本集
        sample_dict[label_data] = len(list(label_tmp['index'])) # 样本集统计结果
    print(sample_dict) # 打印输出样本集统计结果
    return sample_data_idx

#def over_sample_SMOTE(X, y):
    #from imblearn.over_sampling import SMOTE
    #X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    #Counter(y_resampled)
    #return X_resampled, y_resampled

def get_celltype_marker_df(celltype_marker, train_label, train):
    marker_gene = list(celltype_marker.values())
    tmp = []
    for i in marker_gene:
        tmp.extend(i)
    marker_gene = list(np.unique(tmp))
    celltype_marker_df = pd.DataFrame(index=list(train_label), columns=marker_gene)
    for key in celltype_marker.keys():
        celltype_marker_df.loc[key, celltype_marker[key]] = 1
    celltype_marker_df.index = list(train.index)
    # celltype_marker_df = pd.DataFrame(index=list(train.index), columns=marker_gene)
    # for i in range(len(train_label)):
    #     tmp_celltype = train_label[i]
    #     celltype_marker_df.loc[i, celltype_marker[tmp_celltype]] = 1
    celltype_marker_df = celltype_marker_df.replace(np.nan, 0)
    return celltype_marker_df

def reset_adata_features(adata, features):
    tmp = sc.AnnData(pd.DataFrame(columns=features))
    tmp = sc.concat({"org":adata, "tmp":tmp}, label = "source", join="outer", axis=0, fill_value=0)[:, features]
    tmp = tmp[tmp.obs.source == "org",:]
    return tmp


# seuratv3找高变基因只能输入counts矩阵
def adata_umap(adata, norm_flag = False, use_layer = "raw"):
    if norm_flag:
        #输入数据为原始counts则先找高变基因再标准化
        sc.pp.highly_variable_genes(adata, n_top_genes = 4000, flavor='seurat_v3')
        sc.pp.normalize_total(adata, target_sum = 10000)
        sc.pp.log1p(adata)
    else:
        #输入数据已经标准化，则需要提供原始counts矩阵在layers中的位置
        sc.pp.highly_variable_genes(adata, use_layer, n_top_genes = 4000, flavor='seurat_v3')
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata)
    adata = adata[:,np.isnan(adata.X).any(axis = 0)==False]
    sc.tl.pca(adata)
    ch = metrics.calinski_harabasz_score(adata.obsm["X_pca"], adata.obs.celltype)
    dbi = metrics.davies_bouldin_score(adata.obsm["X_pca"], adata.obs.celltype)
    sh = metrics.silhouette_score(adata.obsm["X_pca"], adata.obs.celltype)
    sc.pp.neighbors(adata, n_pcs=50)
    sc.tl.umap(adata) 
    r = sc.pl.umap(adata, color = ["celltype", "label"], legend_loc = "on data", legend_fontweight = "normal", show = False, return_fig = True)
    return ch, dbi, sh, r
    
## 定义参数初始化函数
def weights_init_normal(m, init_type = 'kaiming', gain = 0.02):
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:    ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, gain)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain = gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, gain)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.

    ## 设置学习率为初始学习率乘以给定lr_lambda函数的值

#对adata按by_obs分层抽样
def sample_adata(adata, by_obs, n = 1000):
    val_count = adata.obs[by_obs].value_counts()
    groups = list(val_count.index)
    sub_list = list()
    for g in groups:
        sub_adata = adata[adata.obs[by_obs] == g]
        if val_count[g] > n:
            sub_adata = sub_adata[random.sample(range(0,val_count[g]), n)]
        sub_list.append(sub_adata)
    return sc.concat(sub_list, axis = 0)

#复制实例
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#指数衰减函数()
def e_decay(t, init, end, l):
    alpha = np.log(init/end)/l
    mv = -np.log(init)/alpha
    decay = np.exp(-alpha*(t+mv))
    return decay

def dropout(x, dropout_rate = 0.7, dropout_flag = True, norm_flag = True):
    mask = torch.rand(*x.shape, device=x.device)>dropout_rate
    if dropout_rate == 0:
        return x
    if dropout_flag:
        if norm_flag:
            return normalize_torch(x*mask, scalefactor=10000, log_flag=True)
        else:
            return x*mask
    else:
        return x


# 学习率更新系数

#简单系数更新
class SimpleLambda:
    def __init__(self, l = 0.1, flag = False, max_epoch = 10000):
        self.lbd = l
        self.flag = flag
        self.max_epoch = max_epoch
    def step(self, epoch):
        if epoch > self.max_epoch:
            assert epoch > self.max_epoch, "Out of max epoch"
            if self.flag:
                return self.lbd
            else:
                return 1.0
        else:
            return 1.0


#线性逐步递减更新学习率
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  ## 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs + 1 - self.decay_start_epoch)

# 指数更新学习率
class LambdaFindLR:
    def __init__(self, min_lr, max_lr, n_step):
        self.start_lr = min_lr
        self.end_lr = max_lr
        self.n_step = n_step
    def step(self, stp):
        return (self.end_lr/self.start_lr)**(stp/self.n_step)

##标签平滑化的交叉熵函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, temperature = 1.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim
        self.temperature = temperature

    def forward(self, pred, target):
        pred = pred/self.temperature
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

## 先前生成的样本的缓冲区，利用历史样本，防止模式坍塌
class ReplayBuffer:
    def __init__(self, max_size=1000):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  ## 放入一张图像，再从buffer里取一张出来
        to_return = []  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))