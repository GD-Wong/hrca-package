import math
from . import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
##  残差块儿ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features, batchnorm = True):
        super(ResidualBlock, self).__init__()
        # Res层标准化方法
        if batchnorm:
            res_norm_layer = nn.BatchNorm1d
        else:
            res_norm_layer = nn.LayerNorm
        self.block = nn.Sequential(  ## block = [Linear  + norm + relu + Linear + norm]
            nn.Linear(in_features, in_features),
            res_norm_layer(in_features),
            nn.ReLU(inplace=True),  ## 非线性激活
            nn.Linear(in_features, in_features),
            res_norm_layer(in_features),
        )

    def forward(self, x):  ## 输入为 一张图像
        return x+self.block(x)                     ##累加输出
        #return torch.cat([self.block(x), x], dim=1)  ## 连接输出

##############################
#        SE Module
##############################
class SELayer(nn.Module):
    def __init__(self, input_shape, embedding_shape):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, embedding_shape, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_shape, input_shape, bias = False),
            nn.Sigmoid()
            #nn.Softmax()
        )
    def forward(self, x):
        y = self.fc(x)
        return y*x


class SEResBlock(nn.Module):
    def __init__(self, input_shape, output_shape, dropout_rate = 0.0):
        super(SEResBlock, self).__init__()
        self.layer = utils.clones(nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_shape, output_shape),
            nn.BatchNorm1d(output_shape),
            #nn.LayerNorm(output_shape),
            nn.ReLU(inplace=True),
        ), 2)
        self.atten = SELayer(output_shape, output_shape//2)
    def forward(self, x):
        return self.layer[0](x)+self.atten(self.layer[1](x))

class SEFFNet(nn.Module):
    def __init__(self, input_shape, output_shape, dropout_rate = 0.0):
        super(SEFFNet, self).__init__()
        self.attn = SELayer(input_shape, input_shape // 2)
        #self.normLayer = nn.BatchNorm1d
        self.normLayer = nn.LayerNorm
        self.ff = nn.Sequential(
            nn.Dropout(p = dropout_rate),
            nn.Linear(input_shape, input_shape//4),
            self.normLayer(input_shape//4),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout_rate),
            nn.Linear(input_shape//4, input_shape//16),
            self.normLayer(input_shape // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_shape//16, output_shape)
        )
    def forward(self, x):
        out = self.attn(x)
        out = self.ff(out)
        return out

class SE_Classifier(nn.Module):
    def __init__(self, output_shape, input_shape = 1024, n_layers = 3):
        super(SE_Classifier, self).__init__()
        SERes_Layer = [SEResBlock(input_shape, input_shape//2, dropout_rate=0.0)]
        self.attn = SELayer(input_shape, input_shape//2)
        emb_shape = input_shape//2
        for _ in range(n_layers):
            SERes_Layer += [SEResBlock(emb_shape, emb_shape//2, dropout_rate=0.5)]
            emb_shape = emb_shape//2
        SERes_Layer += [nn.Linear(emb_shape, output_shape)]
        self.SEres = nn.Sequential(*SERes_Layer)
    def forward(self, x):
        out = self.attn(x)
        out = self.SEres(out)
        return out







##############################
#        Attention Model
##############################
def attention(q, k, v, train_status, dropout = 0, mask=None):
    dk = q.size(-1)
    # add dim for matmul
    q = q.unsqueeze(dim = -1) #(batch, input_shape, 1)
    k = k.unsqueeze(dim = -1)
    v = v.unsqueeze(dim = -1)
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(dk) #[batch, input_shape, input_shape]
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim = -1)
    if dropout!=0:
        F.dropout(attn, p=dropout, training=train_status, inplace=True)
    r = torch.matmul(attn, v)
    r = r.squeeze() #[batch, input_shape]
    return r

class AttentionModel(nn.Module):
    def __init__(self, input_shape, head_num = 1):
        super(AttentionModel, self).__init__()
        self.input_shape = input_shape
        assert self.input_shape % head_num == 0, "input_shape can't split by head_num"
        self.head_num = head_num
        self.wq = nn.Linear(input_shape, input_shape)
        self.wk = nn.Linear(input_shape, input_shape)
        self.wv = nn.Linear(input_shape, input_shape)
        self.concat_layer = nn.Linear(input_shape, input_shape)
    def forward(self, q, k, v):
        b = q.size(0)
        head = self.head_num
        q = self.wq(q)#[batch_size, input_shape]
        k = self.wk(k)
        v = self.wv(v)
        #multi head
        if head> 1:
            dk = self.input_shape//head
            q = q.view(b, head, dk)#[batch_size,head_num,dk]
            k = k.view(b, head, dk)
            v = v.view(b, head, dk)
            r = attention(q, k, v, train_status=self.training, dropout=0, mask=None) #[batch_size, head_num, dk]
            r = r.view(b, self.input_shape)
            r = self.concat_layer(r)
        else:
            r = attention(q, k, v, train_status=self.training, dropout=0, mask=None)
        return r

##############################
#        Feed Forward Layer
##############################
class FeedForwardLayer(nn.Module):
    def __init__(self, input_shape, output_shape, activate = None, norm_layer = None, dropout = 0.):
        super(FeedForwardLayer, self).__init__()
        self.layer = nn.Linear(input_shape, output_shape)
        self.dropout = nn.Dropout(p=dropout)   #输入dropout率
        if norm_layer is None:
            self.norm_layer=None
        else:
            self.norm_layer = norm_layer(output_shape) #输入标准化方法， 如nn.LayerNorm
        self.activate = activate  #输入激活函数实例，如nn.Relu(inplace = True)
    def forward(self, x):
        if self.activate is not None:
            x = self.activate(x)
        x = self.layer(self.dropout(x))
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
##############################
#        Residual Connection
##############################
class ResConnect(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(ResConnect, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
##############################
#        Transformer Encoder
##############################
class TransformerEncoder(nn.Module):
    def __init__(self, size, ff_size, head = 1):
        super(TransformerEncoder, self).__init__()
        self.size = size
        self.attn = AttentionModel(size, head_num=head)
        self.connect = utils.clones(ResConnect(size, dropout=0.1), 2)
        ff_list = [
            FeedForwardLayer(size, ff_size),
            FeedForwardLayer(ff_size, size,
                             activate=nn.ReLU(inplace=True),
                             norm_layer=None, dropout=0)
        ]
        self.ffLayer = nn.Sequential(*ff_list)
        self.norm = nn.LayerNorm(size)
    def forward(self, x):
        x = self.connect[0](x, lambda x: self.attn(x, x, x))
        return self.norm(self.connect[1](x, self.ffLayer))
##############################
#        Transformer Decoder
##############################
class TransformerDecoder(nn.Module):
    def __init__(self, size, ff_size, head = 1):
        super(TransformerDecoder, self).__init__()
        self.size = size
        self.self_attn = AttentionModel(size, head_num=head)
        self.cross_attn = AttentionModel(size, head_num=head)
        self.connect = utils.clones(ResConnect(size, dropout=0.1), 3)
        ff_list = [
            FeedForwardLayer(size, ff_size),
            FeedForwardLayer(ff_size, size,
                             activate=nn.ReLU(inplace=True),
                             norm_layer=None, dropout=0)
        ]
        self.ffLayer = nn.Sequential(*ff_list)
        self.norm = nn.LayerNorm(size)
    def forward(self, x, memory):
        x = self.connect[0](x, lambda x: self.self_attn(x, x, x))
        x = self.connect[1](x, lambda x: self.cross_attn(x, memory, memory))
        return self.norm(self.connect[2](x, self.ffLayer))

##############################
#        Transformer Model
##############################
class Transformer(nn.Module):
    def __init__(self, input_size, head_number = 1, reverse = False):
        super(Transformer, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 1024, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]#[Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(1024, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]#dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        self.input_layer = utils.clones(nn.Sequential(*input_layer_list),2)
        self.enc = TransformerEncoder(1024, 2048, head = head_number)
        self.dec = TransformerDecoder(1024, 2048, head = head_number)
        self.output_layer = nn.Sequential(*output_layer_list)
        self.reverse = reverse
    def forward(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        if self.reverse:
            output = self.output_layer(self.dec(source_emb, self.enc(target_emb)))
        else:
            output = self.output_layer(self.dec(target_emb, self.enc(source_emb)))
        return output
    def encode(self, source, target):
        if self.reverse:
            enc_out = self.enc(self.input_layer[1](target))
        else:
            enc_out = self.enc(self.input_layer[0](source))
        return enc_out
    def decode(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        if self.reverse:
            dec_out = self.dec(source_emb, self.enc(target_emb))
        else:
            dec_out = self.dec(target_emb, self.enc(source_emb))
        return dec_out
#多层编码解码的Transformer
class TransformerMultiLayer(nn.Module):
    def __init__(self, input_size, head_number = 1, n_layers = 1):
        super(TransformerMultiLayer, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 2048, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]#[Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(2048, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]#dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        #在此处编辑编码器层数
        self.enc_list = utils.clones(TransformerEncoder(2048,1024, head = head_number),n_layers)
        self.dec_list = utils.clones(TransformerDecoder(2048,1024, head=head_number), n_layers)
        self.input_layer = utils.clones(nn.Sequential(*input_layer_list),2)
        self.output_layer = nn.Sequential(*output_layer_list)
    def forward(self, source, target):
        source_emb = self.input_layer[0](source)
        target_emb = self.input_layer[1](target)
        for enc in self.enc_list:
            source_emb = enc(source_emb)
        for dec in self.dec_list:
            target_emb = dec(target_emb, source_emb)
        output = self.output_layer(target_emb)
        return output
    def encode(self, source):
        out = self.input_layer[0](source)
        for enc in self.enc_list:
            out = enc(out)
        return out
    def decode(self, source, target):
        enc_out = self.input_layer[0](source)
        dec_out = self.input_layer[1](target)
        for enc in self.enc_list:
            enc_out = enc(enc_out)
        for dec in self.dec_list:
            dec_out = dec(dec_out, enc_out)
        return dec_out

###
# Gene_AE
###
class Gene_AE(nn.Module):
    def __init__(self, input_size, head_number = 1, n_layers = 1):
        super(Gene_AE, self).__init__()
        input_layer_list = [
            FeedForwardLayer(input_size, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, 1024, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
        ]  # [Linear+Norm+Relu+Drop+Linear]  enc[Norm+...]
        output_layer_list = [
            FeedForwardLayer(1024, 5000, activate=None, norm_layer=nn.LayerNorm, dropout=0),
            FeedForwardLayer(5000, input_size, activate=nn.ReLU(inplace=True), norm_layer=None, dropout=0.1),
            nn.Softplus()
        ]  # dec[...+Norm]  [Linear+Norm+ReLu+Drop+Linear+SoftPlus]
        enc_layer = []
        self.input_layer = nn.Sequential(*input_layer_list)
        for _ in range(n_layers):
            enc_layer += [TransformerEncoder(1024, 2048, head=head_number)]
        self.enc = nn.Sequential(*enc_layer)
        self.output_layer = nn.Sequential(*output_layer_list)
    def forward(self, x):
        return self.output_layer(self.enc(self.input_layer(x)))
    def encode(self, x):
        return self.enc(self.input_layer(x))
