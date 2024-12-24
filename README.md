# HRCA

HRCA (High-Resolution Cell Annotation) is a deep learning framework specifically designed for cell type annotation in high-resolution spatial transcriptomics (hr-stRNA). This novel method integrates transfer learning from single-cell RNA sequencing (scRNA-seq) data to hr-stRNA. It uses two key components: Gene-AE, for feature adaptation, and SE-Classifier, for cell typing. This method enables accurate cell type annotation even in the presence of sparse gene expression data.

![workflow.jpg](https://github.com/GD-Wong/hrca-package/blob/main/Fig/workflow.jpg)

### Installation

The HRCA can be installed by conda.

```bash
conda create -n hrca
conda activate hrca
conda config --add channels conda-forge
conda install gd-wong::hrca
```

### Download pre-trained model

There is a pre-trained model derived from tumor data [GD-Wong/hrca at main](https://huggingface.co/GD-Wong/hrca/tree/main). Users can utilize this model to identify cell types within the spatial transcriptomic data of tumors.

The model can be downloaded by [Git](https://git-scm.com/).

```git
git lfs install
git clone https://huggingface.co/GD-Wong/hrca
# or
git clone https://hf-mirror.com/GD-Wong/hrca
```

In the future, we will train additional models that offer more comprehensive annotations.

### Example

Here, we will use a pre-trained model and [example data](https://huggingface.co/datasets/GD-Wong/Spatial_celltype/blob/main/example.h5ad) to demonstrate a case of annotating cell types.

```python
# requirements
from hrca import hrca
import scanpy as sc
import numpy as np
import torch
# load input_data: It should be anndata to be annotated.
input_data = sc.read("path_to/example.h5ad")
# load model
# features: The genes will be used in model.
# dictionary: The dictionary about cell type.
features = np.load("path_to/hrca/features.npy")
dictionary = np.load("path_to/hrca/dictionary.npy")
# Gene_AE: The Gene-AE for genes embed.
# SE_Classifier: The SE-Classifier for cell type identification.
Gene_AE = hrca.model.Gene_AE(input_size=21333, head_number=8, n_layers=8)
Gene_AE.load_state_dict(torch.load("path_to/hrca/AE_200.pth"))
# reply: <All keys matched successfully>
SE_Classifier = hrca.model.SE_Classifier(output_shape=36)
SE_Classifier.load_state_dict(torch.load("path_to/hrca/predictor_200.pth"))
# reply: <All keys matched successfully>

# Accelerated by GPU.
# Note: Do not run the following 2 code if you do not have a GPU.
Gene_AE=Gene_AE.cuda()
SE_Classifier=SE_Classifier.cuda()

# Annotate example
result = hrca.annotate(input_data, features, 
                       encoder=Gene_AE, 
                       classifier=SE_Classifier, 
                       l_dict=dictionary,
                       workers = 1)
# The result is a DataFrame
```

![annotation_result.jpg](https://github.com/GD-Wong/hrca-package/blob/main/Fig/annotation_result.jpg)

Show result of annotation,

```python
import pandas as pd
# Add result to input_data
input_data.obs = pd.concat([input_data.obs, result], axis=1)
# preprocess input_data
sc.pp.normalize_total(input_data)
sc.pp.log1p(input_data)
sc.pp.highly_variable_genes(input_data, n_top_genes=4000)
sc.pp.scale(input_data)
sc.tl.pca(input_data)
sc.pp.neighbors(input_data)
sc.tl.umap(input_data)
sc.pl.umap(input_data, color="predicted", legend_loc="on data", size=5)
# The umap will show as below.
```

![Snipaste_2024-12-11_20-40-35.jpg](https://github.com/GD-Wong/hrca-package/blob/main/Fig/umap.jpg)

With true label of cell type, the annotation result can be validated by confusion_matrix.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true = list(input_data.obs.cell_type),
                      y_pred = list(input_data.obs.predicted), 
                      labels=list(dictionary), normalize='true')
fig = plt.figure(figsize=(10, 10), dpi=200)
sns.heatmap(cm, fmt='.2g', cmap="Blues", annot=True, cbar=False,
            xticklabels=list(dictionary), yticklabels=list(dictionary),
            annot_kws={"fontsize": 3})
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
plt.show()
# The confusion matrix will show as below
```

![confusion_matrix.jpg](https://github.com/GD-Wong/hrca-package/blob/main/Fig/confusion_matrix.jpg)

### Train new models

If sufficient annotation data is available, HRCA can also utilize them to train new models. It requires some machine learning knowledge. If you want, please refer to the [training tutorial](https://github.com/GD-Wong/hrca-package/tree/main/training-tutorial).
