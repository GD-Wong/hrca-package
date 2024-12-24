# HRCA Training Document

When there is sufficient annotated single-cell data (it is recommended to have more than 300 cells for each cell type), you can use HRCA to train new models. Please refer to the following content for training the new model.

[HRCA algorithm description](#HRCA-algorithm-description)

The training process is divided into two stages：

[Pre-training](#pre-training)

[Fine-tuning](#Fine-tuning)

The train data and test data used in this example can be downloaded from [GD-Wong/Spatial_celltype at main](https://huggingface.co/datasets/GD-Wong/Spatial_celltype/tree/main)

### HRCA algorithm description

HRCA is a deep neural network-based framework composed of the *Gene-AE* (transfer module) and the *SE-Classifier* (classifier module). The *Gene-AE* consists of an encoder and a decoder. The encoder, inspired by the Transformer architecture, embeds the original gene expression features into low-dimensional spaces. The decoder reconstructs these features back into the original expression space. The *SE-Classifier*  consists of multiple layers of residual structures with feature attention mechanisms, ultimately outputting a cell type probability distribution.

##### *Gene-AE*

In Gene-AE, the encoder takes the gene expression as input to learn an embedding function, including the input layer and the encode layer. The input layer is constructed as a feedforward network, which is expressed as,

$$
I'=FeedForward(I)=w _2 (σ(w _1 I+b _1))+b _2
$$

where *$I$* is the input feature, and $w _1, w _2, b _1, b_2$ are trainable parameters and $σ$ is the ReLU function. The encode layer take *$I’$* as input, its attention function is expressed as:

$$
Atten(Q,K,V) = softmax(((QK^T)/√(d_k ))V)
$$

$$
SelfAtten(x)=Atten(W^Q x,W^K x,W^V x)
$$

$$
MultiHeadSelfAtten(x,n) = Concat(SelfAtten_1 (x),SelfAtten_2 (x)...SelfAtten_n (x))W^O
$$

where $W^O, W^Q, W^K, W^V$ are trainable, and $n$ is the head number of multi-head self-attention mechanism work in encoder. They connected through residual structures as:

$$
TransformerLayer(x) = AttentionFeatures+FeedForward(AttentionFeatures)
$$

where: 

$$
AttentionFeatuers = x+MultiHeadSelfAtten(x,n)
$$

There are $K$ layers in encode layer, the output of layer $k (k = 2, …, K)$ is computed by the output of layer $k-1$,

$$
z_k=TransformerLayer^k (z_k-_1)
$$

where $z_k$ is the output of layer $k$, and

$$
z_1  =TransformerLayer^1 (I')
$$

The decoder is only involved in the training process and mirrors the input layer of encoder. The output layer of decoder is expressed as,

$$
O = SoftPlus(FeedForwad(z_k)) 
$$

##### *SE-Classifier*

The SE-Classifier takes the output of Gene-AE’s encoder $z_k$ as input, and learns cell type probability distribution in low-dimension spaces. SELayer, a feature attention layer, is a key component in this module. It calculates and adds self-attention to the input features in a weighted manner.

$$
SELayer(x) = x*Sigmoid(FeedForward(x))
$$

The embedding features $z'_1$ are generated from $z_K$, 

$$
z'_1  =SELayer(z_k)
$$

There are $J$ hidden layers in SE-Classifier, the output of layer $j (j = 2, …, J)$ is computed by the output of layer $j-1$,

$$
z'_j= SEres(z'_j-_1))
$$

where:      

$$
SEres(x) = W^p x+ SELayer(W^q x)
$$

The result of SE-Classifier is calculated by,

$$
r=W^r z'_j
$$

where , $W^p,W^q,W^r$ are trainable.

##### *The overall architecture*

The dimension of encoder input layer in *Gene-AE* is configured as $nGenes-5000-1024$, followed by multi-head attention modules, with $n$ heads and $k$ TransformerLayers. It is then connected to the decoder of *Gene-AE* with a $1024-5000-nGenes$ layer structure, and the output is passed through a *Softplus* activation layer. Both the encoder and decoder are utilized for feature transfer, with *LayerNorm* applied to normalize the features. The *SE-Classifier* receives the encoder’s output as input, which is then passed through an *SELayer* for feature enhancement. Subsequently, dimensionality reduction is performed through $j$ *Linear + BatchNorm + ReLU* layers, reducing the feature dimensions from 1024 to 512, 256, and finally 128. At each stage of dimensionality reduction, feature values enhanced by *SELayer* are added. Since annotation of $n_c$ cell types is required, the *SE-Classifier* ends with a final Linear layer with a $128-n_c$ configuration. The *Gene-AE* is used for model pre-training, while the structure of the encoder connected with the *SE-Classifier* is employed for model fine-tuning.

### Pre-training

During the pre-training phase, only the Gene-AE is trained, with the Adam optimizer  being used to minimize the reconstruction loss.

$$
l_r=MSE(I,O) 
$$

The learning rate is adjusted based on the decrease in the loss value, with an initial learning rate of 0.001. If the loss does not decrease over 50 epochs, the learning rate is reduced by a factor of 0.1. The batch size is set to 128, and the model is trained for a total of 1,000 epochs. The model at the point where the loss no longer decrease is selected as the final pre-trained model. This pre-training process can be conducted by HRCA as below.

```python
from hrca import hrca
import scanpy as sc

result_path = "path_to/save_result"
# load train data (anndata)
train = sc.read("path_to/train.h5ad")
Gene_AE = hrca.pretrain(train, save_path = result_path,
                        n_epochs = 1000,
                        batch_size = 128,
                        n_cpu = 5,
                        head_num = 8,
                        n_layers = 8, 
                        lr = 0.001,
                        decay_epoch=200,
                        lr_decrease_factor=0.1,
                        lr_decrease_patience=50,
                        checkpoint_interval=100,
                        tb_writer = True,
                        use_device=0)
```

### Fine-tuning

The pre-trained model is then connected to the SE-Classifier for fine-tuning, during which the model is trained for an additional 200 epochs. The same training strategy as in pre-training is employed for fine-tuning, with the exception that the learning rate is kept constant for the first 20 epochs. This fine-tuning process can be conducted by HRCA as below.

```python
test = sc.read("test.h5ad")
Gene_AE, SE_Classifier = hrca.fine_tune(train, test, Gene_AE,
                                        save_path = result_path,
                                        obs_name = "cell_type",
                                        n_epochs = 200,
                                        batch_size = 128,
                                        n_cpu = 5,
                                        n_layers = 3,
                                        lr = 0.001,
                                        decay_epoch = 20,
                                        lr_decrease_patience = 10,
                                        checkpoint_interval = 20,
                                        tb_writer = True,
                                        use_device = 0)
```
