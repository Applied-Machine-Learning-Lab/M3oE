import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from .activation import activation_layer
from .features import DenseFeature, SparseFeature, SequenceFeature


class PredictionLayer(nn.Module):
    """Prediction Layer.

    Args:
        task_type (str): if `task_type='classification'`, then return sigmoid(x), 
                    change the input logits to probability. if`task_type='regression'`, then return x.
    """

    def __init__(self, task_type='classification'):
        super(PredictionLayer, self).__init__()
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = torch.sigmoid(x)
        return x


class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.

    
    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output: 
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  #exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[str(fea.name)] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[str(fea.name)] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(self.embed_dict[str(fea.name)](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[str(fea.shared_with)](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." %
                                     (["sum", "mean"], fea.pooling))
                fea_mask = InputMask()(x, fea)
                if fea.shared_with == None:
                    sparse_emb.append(pooling_layer(self.embed_dict[str(fea.name)](x[fea.name].long()), fea_mask).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[str(fea.shared_with)](x[fea.name].long()), fea_mask).unsqueeze(1))  #shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float().unsqueeze(1))  #.unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  #[batch_size, num_features, embed_dim]

        if squeeze_dim:  #Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  #only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(start_dim=1)  #squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values),
                                 dim=1)  #concat dense value with sparse embedding
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  #[batch_size, num_features, embed_dim]
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" %
                    ("SparseFeatures", features))

