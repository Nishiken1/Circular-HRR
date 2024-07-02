"""
Models for training Multilabel classification tasks.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"

import numpy as np
from tqdm import tqdm
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Faiss for MIPS (maximum inner product search)
import faiss
import math
import wandb
# Internal.
from .metrics import compute_prop_metrics
from .embeddings import get_vectors, load_embeddings, get_circular_vectors
from .mathops import get_appx_inv, circular_conv, complexMagProj,circular_circular_conv,dot,custom_dot,sim,custom_sim
from .utils import Measure
from torch.nn.utils import clip_grad_norm_

# Create FFN (feedforward network) for the classification task.
# NOTE: For the baseline the output layer is n binary classification tasks.
# NOTE: Most optimal (when tested with Wiki10.) is size 768
FC_LAYER_SIZE = 768
# FC_LAYER_SIZE = 512
# FC_LAYER_SIZE = 2048
# Network Design


class SemanticPointerNetwork(nn.Module):
    def __init__(self, num_features, num_classes, dims, h_size, Nl_dict, max_pred_size,
                 A=None,B=None,no_grad=False, load_vec=None, debug=False, use_fc3=False):
        super(SemanticPointerNetwork, self).__init__()

        # Initialization Parameters.
        self.num_classes = num_classes # Number of labels in the datasets.
        self.num_features = num_features
        # self.fc_layer_size = FC_LAYER_SIZE
        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1
        self.dims = dims
        self.debug = debug
        self.load_vec = load_vec
        self.fc_layer_size = h_size
        self.use_fc3 = use_fc3
        self.Nl_dict = Nl_dict
        self.A = A
        self.B = B

        # NOTE: Defines the maximum number of positive labels in a sample.
        self.max_label_size = max_pred_size

        if self.debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))
            print("Class vector dimension: {}".format(self.dims))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims*2)
        #self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims)
        #init.normal_(self.olayer.weight, mean=0, std=1)
        #init.uniform_(self.fc1.weight, -1, 1)
        #init.normal_(self.olayer.weight, mean=0, std=1)
        #init.uniform_(self.fc2.weight, -1, 1)
        #init.uniform_(self.olayer.weight, -1, 1)
        #uniformにしたり、平均を変えてみたりする。

        # Create a label embedding layer.
        # self.create_label_embedding()

        # P & N vectors.
        #p_n_vec = get_vectors(2, self.dims, ortho=True)
        # Circular
        #p_n_vec = get_circular_vectors(2, self.dims, ortho=True)
        p_n_vec = get_circular_vectors(2, self.dims)

        if no_grad:
            if self.debug:
                print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            if self.debug:
                print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)
            # print(p_n_vec[0])

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'faiss_inference': Measure("Faiss Inference"),
            'data_load': Measure("Data Loader"),
        }

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        if self.use_fc3:
            x = F.leaky_relu(self.fc3(x))
        x = self.olayer(x)
        x = x.view(-1, self.dims, 2)
        # print(x.shape)
        # exit()
        # xをnormalize
        x = F.normalize(x,p=2, dim=2)
        #座標から角度に変換
        x = torch.atan2(x[:,:,0],x[0:,:,1])
        return x

    def create_label_embedding(self):
        if self.load_vec is not None:
            if self.debug:
                print("Loading Pretrained Embeddings...")

            # Class labels.
            self._class_vectors = load_embeddings(self.load_vec, self.num_classes - 1)
        else:
            if self.debug:
                print("Generate new label embeddings...")

            # Class labels.
            # self._class_vectors = get_vectors(self.num_classes, self.dims)
            # Circular
            self._class_vectors = get_circular_vectors(self.num_classes, self.dims)

        if self.debug:
            print("Label Vectors: {}".format(self._class_vectors.shape))

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_classes, self.dims)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self._class_vectors = None ### self._class_vectors is not required after this line
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_classes, 1), dtype=torch.int8)
        weights[self.num_classes - 1] = 0 # NOTE IMPORTANT: Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_classes, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False


    def inference(self, s, size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)

        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def circular_inference(self, s, size, positive=True,test = False):
        #(batch, dims)

        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        # if test:
        #     print(-1*vec[0])
            # print(torch.mean(torch.abs(vec)))
            # print(torch.mean(torch.abs(s)))
            # print(vec.shape)
        y = circular_circular_conv(-1*vec, s) #(batch, dims)
        return y

class SemanticPointerNetworkHalf(nn.Module):
    def __init__(self, num_features, num_classes, dims, h_size, max_pred_size, A, B,
                 no_grad=False, load_vec=None, debug=False,use_fc3=False):
        super(SemanticPointerNetworkHalf, self).__init__()

        # Initialization Parameters.
        self.num_classes = num_classes # Number of labels in the datasets.
        self.num_features = num_features
        # self.fc_layer_size = FC_LAYER_SIZE
        # self.fc_layer_size_half = int(FC_LAYER_SIZE / 2)
        self.fc_layer_size = h_size
        self.fc_layer_size_half = int(h_size / 2)

        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1
        self.dims = dims
        self.debug = debug
        self.load_vec = load_vec

        # NOTE: Defines the maximum number of positive labels in a sample.
        self.max_label_size = max_pred_size

        if self.debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))
            print("Class vector dimension: {}".format(self.dims))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        # self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer1 = nn.Linear(self.fc_layer_size_half, self.dims)
        self.olayer2 = nn.Linear(self.fc_layer_size_half, self.dims)
        #self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims)
        #init.normal_(self.olayer.weight, mean=0, std=1)
        #init.uniform_(self.fc1.weight, -1, 1)
        #init.normal_(self.olayer.weight, mean=0, std=1)
        #init.uniform_(self.fc2.weight, -1, 1)
        #init.uniform_(self.olayer.weight, -1, 1)
        #uniformにしたり、平均を変えてみたりする。

        # Create a label embedding layer.
        # self.create_label_embedding()

        # P & N vectors.
        #p_n_vec = get_vectors(2, self.dims, ortho=True)
        # Circular
        #p_n_vec = get_circular_vectors(2, self.dims, ortho=True)
        p_n_vec = get_circular_vectors(2, self.dims)

        if no_grad:
            if self.debug:
                print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            if self.debug:
                print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)
            # print(p_n_vec[0])

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'faiss_inference': Measure("Faiss Inference"),
            'data_load': Measure("Data Loader"),
        }

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = x.view(-1, self.fc_layer_size_half, 2)

        d1 = self.olayer1(x[:,:,0])
        d2 = self.olayer2(x[:,:,1])

        x = torch.stack([d1, d2], dim=2)

        x = F.normalize(x,p=2, dim=2)
        #座標から角度に変換
        x = torch.atan2(x[:,:,0],x[0:,:,1])
        return x

    def create_label_embedding(self):
        if self.load_vec is not None:
            if self.debug:
                print("Loading Pretrained Embeddings...")

            # Class labels.
            self._class_vectors = load_embeddings(self.load_vec, self.num_classes - 1)
        else:
            if self.debug:
                print("Generate new label embeddings...")

            # Class labels.
            # self._class_vectors = get_vectors(self.num_classes, self.dims)
            # Circular
            self._class_vectors = get_circular_vectors(self.num_classes, self.dims)

        if self.debug:
            print("Label Vectors: {}".format(self._class_vectors.shape))

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_classes, self.dims)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self._class_vectors = None ### self._class_vectors is not required after this line
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_classes, 1), dtype=torch.int8)
        weights[self.num_classes - 1] = 0 # NOTE IMPORTANT: Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_classes, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False


    def inference(self, s, size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)

        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def circular_inference(self, s, size, positive=True,test = False):
        #(batch, dims)

        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        # if test:
        #     print(-1*vec[0])
            # print(torch.mean(torch.abs(vec)))
            # print(torch.mean(torch.abs(s)))
            # print(vec.shape)
        y = circular_circular_conv(-1*vec, s) #(batch, dims)
        return y

class SemanticPointerNetworkSin(nn.Module):
    def __init__(self, num_features, num_classes, dims, h_size, max_pred_size,
                 no_grad=False, load_vec=None, debug=False, initialize=False,use_fc3=False):
        super(SemanticPointerNetworkSin, self).__init__()

        # Initialization Parameters.
        self.num_classes = num_classes # Number of labels in the datasets.
        self.num_features = num_features
        # self.fc_layer_size = FC_LAYER_SIZE
        self.fc_layer_size = h_size
        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1
        self.dims = dims
        self.debug = debug
        self.load_vec = load_vec

        # NOTE: Defines the maximum number of positive labels in a sample.
        self.max_label_size = max_pred_size

        if self.debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))
            print("Class vector dimension: {}".format(self.dims))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        # self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims)
        if initialize:
            for i in range(self.dims):
                abs_u = np.sqrt(6/self.fc_layer_size)
                torch.nn.init.uniform_(self.olayer.weight[i, :], -abs_u, abs_u)
        # Create a label embedding layer.
        # self.create_label_embedding()

        # P & N vectors.
        #p_n_vec = get_vectors(2, self.dims, ortho=True)
        # Circular
        #p_n_vec = get_circular_vectors(2, self.dims, ortho=True)
        p_n_vec = get_circular_vectors(2, self.dims)

        if no_grad:
            if self.debug:
                print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            if self.debug:
                print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)
            # print(p_n_vec[0])

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'faiss_inference': Measure("Faiss Inference"),
            'data_load': Measure("Data Loader"),
        }

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = self.olayer(x)

        x = torch.sin(x) * np.pi
        return x

    def create_label_embedding(self):
        if self.load_vec is not None:
            if self.debug:
                print("Loading Pretrained Embeddings...")

            # Class labels.
            self._class_vectors = load_embeddings(self.load_vec, self.num_classes - 1)
        else:
            if self.debug:
                print("Generate new label embeddings...")

            # Class labels.
            # self._class_vectors = get_vectors(self.num_classes, self.dims)
            # Circular
            self._class_vectors = get_circular_vectors(self.num_classes, self.dims)

        if self.debug:
            print("Label Vectors: {}".format(self._class_vectors.shape))

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_classes, self.dims)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self._class_vectors = None ### self._class_vectors is not required after this line
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_classes, 1), dtype=torch.int8)
        weights[self.num_classes - 1] = 0 # NOTE IMPORTANT: Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_classes, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False


    def inference(self, s, size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)

        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def circular_inference(self, s, size, positive=True,test = False):
        #(batch, dims)

        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        # if test:
        #     print(-1*vec[0])
            # print(torch.mean(torch.abs(vec)))
            # print(torch.mean(torch.abs(s)))
            # print(vec.shape)
        y = circular_circular_conv(-1*vec, s) #(batch, dims)
        return y


class SemanticPointerNetworkTanh(nn.Module):
    def __init__(self, num_features, num_classes, dims, h_size, max_pred_size,
                 no_grad=False, load_vec=None, debug=False, initialize=False, use_fc3=False):
        super(SemanticPointerNetworkTanh, self).__init__()

        # Initialization Parameters.
        self.num_classes = num_classes # Number of labels in the datasets.
        self.num_features = num_features
        # self.fc_layer_size = FC_LAYER_SIZE
        self.fc_layer_size = h_size
        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1
        self.dims = dims
        self.debug = debug
        self.load_vec = load_vec

        # NOTE: Defines the maximum number of positive labels in a sample.
        self.max_label_size = max_pred_size

        if self.debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))
            print("Class vector dimension: {}".format(self.dims))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        # self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims)
        if initialize:
            for i in range(self.dims):
                abs_u = np.sqrt(6/self.fc_layer_size)
                torch.nn.init.uniform_(self.olayer.weight[i, :], -abs_u, abs_u)
        # Create a label embedding layer.
        # self.create_label_embedding()

        # P & N vectors.
        #p_n_vec = get_vectors(2, self.dims, ortho=True)
        # Circular
        #p_n_vec = get_circular_vectors(2, self.dims, ortho=True)
        p_n_vec = get_circular_vectors(2, self.dims)

        if no_grad:
            if self.debug:
                print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            if self.debug:
                print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)
            # print(p_n_vec[0])

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'faiss_inference': Measure("Faiss Inference"),
            'data_load': Measure("Data Loader"),
        }

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = self.olayer(x)
        x = torch.tanh(x) * np.pi
        return x

    def create_label_embedding(self):
        if self.load_vec is not None:
            if self.debug:
                print("Loading Pretrained Embeddings...")

            # Class labels.
            self._class_vectors = load_embeddings(self.load_vec, self.num_classes - 1)
        else:
            if self.debug:
                print("Generate new label embeddings...")

            # Class labels.
            # self._class_vectors = get_vectors(self.num_classes, self.dims)
            # Circular
            self._class_vectors = get_circular_vectors(self.num_classes, self.dims)

        if self.debug:
            print("Label Vectors: {}".format(self._class_vectors.shape))

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_classes, self.dims)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self._class_vectors = None ### self._class_vectors is not required after this line
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_classes, 1), dtype=torch.int8)
        weights[self.num_classes - 1] = 0 # NOTE IMPORTANT: Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_classes, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False


    def inference(self, s, size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)

        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def circular_inference(self, s, size, positive=True,test = False):
        #(batch, dims)

        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)
        # vec = complexMagProj(vec)
        # if test:
        #     print(-1*vec[0])
            # print(torch.mean(torch.abs(vec)))
            # print(torch.mean(torch.abs(s)))
            # print(vec.shape)
        y = circular_circular_conv(-1*vec, s) #(batch, dims)
        return y


def compute_mrr(predicted_outputs, actual_outputs):
    mrr = 0.0
    num_queries = 0

    for i in range(len(actual_outputs)):
        true_labels = set(np.where(actual_outputs[i] == 1)[0])
        if true_labels:
            rank_list = np.argsort(predicted_outputs[i])[::-1]  # 予測値に基づいてランクリストを作成
            num_queries += 1
            for rank, label in enumerate(rank_list):
                if label in true_labels:
                    mrr += 1.0 / (rank + 1)
                    break

    return mrr / num_queries if num_queries > 0 else 0.0


# wandb.init(
#      project="231015",

#      config={
#           "architecture":"HRR",
#      }
# )


def spp_loss(s, model, data, target, device, without_negative=False, normalize=False):
    """
    Train with SPP.
    """

    pos_classes = model.class_vec(target)
    pos_classes = pos_classes * model.class_weights(target)

    # Remove the padding idx vectors.
    pos_classes = pos_classes.to(device)

    # Positive prediction loss
    convolve = model.circular_inference(s, data.size(0))
    cosine = dot(pos_classes, convolve)
    J_p = torch.mean(torch.sum(1 - torch.abs(cosine), dim=-1))

    # Negative prediction loss.
    J_n = 0.0
    if without_negative is False:
        convolve = model.circular_inference(s, data.size(0), positive=False)
        cosine = dot(pos_classes, convolve)
        J_n = torch.mean(torch.sum(torch.abs(cosine), dim=-1))

    # Total Loss.
    loss = J_p
    return loss, J_n, J_p

def spp_pro_loss(s, model, data, target, device, N, A, B, Nl_dict, without_negative=False, normalize=False):
    """
    Train with SPP.
    """
    C = (torch.log(torch.tensor(N, dtype=torch.float)) - 1) * (B + 1) * A


    # 各ラベルの傾向スコアの逆数を計算
    # weights = torch.tensor([1 + C * ((Nl_dict[l] + B) ** -A) for l in target.tolist()], dtype=torch.float).to(device)
    weights = []
    for label_row in target:
        row_weights = []
        for l in label_row:
            # ラベルを CPU に移動し、整数型に変換
            l_int = l.item()
            weight = 1 + C * ((Nl_dict[l_int] + B) ** -A)
            row_weights.append(weight)
        weights.append(row_weights)
    weights = torch.tensor(weights, dtype=torch.float).to(device)


    pos_classes = model.class_vec(target)  # (batch, no_label, dims)
    pos_classes_weighted = pos_classes * weights.unsqueeze(-1)
    pos_classes_weighted = pos_classes_weighted.to(device)

    # Positive prediction loss
    convolve = model.circular_inference(s, data.size(0))
    cosine = dot(pos_classes_weighted, convolve)
    J_p = torch.mean(torch.sum(1 - torch.abs(cosine), dim=-1))

    # Negative prediction loss.
    J_n = 0.0
    if without_negative is False:
        convolve = model.circular_inference(s, data.size(0), positive=False)
        cosine = dot(pos_classes_weighted, convolve) 
        J_n = torch.mean(torch.sum(torch.abs(cosine), dim=-1))

    # Total Loss.
    loss = J_p
    return loss, J_n, J_p


def spp_train(log_interval, model, device, train_loader, optimizer, epoch,
              without_negative=False, proloss=False, Nl_dict=None, N=None, A=None, B=None):
    total_loss = 0; total_j_n = 0.0; total_j_p = 0.0
    model.train()
    model = model.to(device)

    train_iter = iter(train_loader)
    next_data, next_target = next(train_iter)
    next_data = next_data.to(device, non_blocking=True).float()
    next_target = next_target.to(device, non_blocking=True)

    model.time['train'].start()
    model.time['data_load'].start()

    for batch_idx in tqdm(range(len(train_loader))):

        data = next_data
        target = next_target

        # Prefetch
        if batch_idx + 2 != len(train_loader):
            next_data, next_target = next(train_iter)
            next_data = next_data.to(device, non_blocking=True).float()
            next_target = next_target.to(device, non_blocking=True)

        model.time['data_load'].end()

        # Select.
        # Train with actual negative samples.
        optimizer.zero_grad()

        model.time['train_forward_pass'].start()
        s = model(data)

        #s = torch.tanh(s) * math.pi
        #s = torch.atan2(torch.sin(s), torch.cos(s))
        model.time['train_forward_pass'].end()

        model.time['train_loss'].start()
        # target = target.to(device)
        if proloss is True:
            loss, J_n, J_p = spp_pro_loss(s, model,data, target, device, N, A, B, 
                                          Nl_dict, without_negative=without_negative)
        else:
            loss, J_n, J_p = spp_loss(s, model, data, target, device,
                                  without_negative=without_negative)
        
        model.time['train_loss'].end()
        # Send to GPU.
        #target = target.to(device)

        model.time['optimization'].start()
        loss.backward()
        # clip_grad_norm_(model.parameters(), clip_value)
        total_norm = 0.0
        # for p in model.parameters():
        #     if p.grad is not None: # ここで勾配がNoneでないかチェック
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print("Total gradient norm:", total_norm) # またはロギング

        optimizer.step()
        model.time['optimization'].end()

        # Losses.
        total_loss += loss.item()
        total_j_p += J_p.item()
        if without_negative is False:
            total_j_n += J_n.item()

        model.time['data_load'].start()

    model.time['data_load'].end()
    model.time['train'].end()
    # wandb.log({"loss":total_loss/(batch_idx+1)})
    return total_loss/(batch_idx + 1), total_j_n/(batch_idx + 1), total_j_p/(batch_idx + 1)


def superpose(s, c):
    scos = np.cos(s) + np.cos(c)
    ssin = np.sin(s) + np.sin(c)
    s = torch.atan2(ssin, scos)
    return s

def modulo_pi_to_negpi(theta):
    result = torch.fmod(theta + np.pi, 2.0 * np.pi) - np.pi
    return result

def circular_binding(theta,phi):
  return modulo_pi_to_negpi(theta+phi)

def spp_test(model, device, test_loader, threshold=0.25, propensity=None,
             topk=5, without_negative=False, proloss=False, Nl_dict=None, N=None, A=None, B=None):
    """
    Threshold defines the decision point for a binary classification task.
    """
    # Before evaluation, build the index for current class vectors.
    ### model.build_class_index()

    # Start evaluation.

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        total_pr = 0.0; total_rec = 0.0; total_f1 = 0.0; total_loss = 0.0;
        all_acc = []

        test_iter = iter(test_loader)
        next_data, next_target = next(test_iter)
        next_data = next_data.to(device, non_blocking=True).float()
        next_target = next_target.to(device, non_blocking=True)

        total_mrr = 0

        for idx in tqdm(range(len(test_loader))):

            data = next_data
            target = next_target

            # Prefetch
            if idx + 2 != len(test_loader):
                next_data, next_target = next(test_iter)
                next_data = next_data.to(device, non_blocking=True).float()
                next_target = next_target.to(device, non_blocking=True)

            model.time['test_forward_pass'].start()
            s = model(data) # Y are the predictions.

            batch_size = s.shape[0]
            y = model.circular_inference(s, batch_size,test = True)

            model.time['test_forward_pass'].end()

            # # Inference with faiss.
            y_cpu = y.cpu()

            # # (batch, no_classes)
            # #weights = complexMagProj(model.class_vec.weight)
            model.time['inference'].start()

            #y_cpu = y.cpu()

            y_cpu = torch.abs(custom_dot(y,model.class_vec.weight)).cpu()
            #y_cpu = torch.abs(custom_dot(model.class_vec.weight,y)).cpu()

            #torch.set_printoptions(edgeitems=1000)
            predictions = (y_cpu >= threshold).long()

            # Loss.
            if proloss is True:
                loss, _, _  = spp_pro_loss(s, model, data, target, device, N, A, B, Nl_dict, without_negative=without_negative)
            else:
                loss, _, _ = spp_loss(s, model, data, target, device,
                                  without_negative=without_negative)
            

            total_loss += loss.item()
            target = target.cpu()

            # et one-hot vector.
            y_onehot = torch.LongTensor(batch_size, model.num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, target, 1)


            # Correct predictions
            correct_pred = torch.sum(predictions & y_onehot, axis=1).float()

            # Recall.
            if torch.sum(y_onehot, axis=1).eq(0).any():  # check if there are any zero sums
                print("No labels for this batch, precision is undefined.")
            else:
                rec = torch.mean(correct_pred / torch.sum(y_onehot, axis=1)).item()
                total_rec += rec

            # Precision.
            ind_pr = correct_pred / torch.sum(predictions, axis=1)
            ind_pr[ind_pr != ind_pr] = 0.0
            pr = torch.mean(ind_pr).item()
            total_pr += pr


            # F-1 Score.
            f1 = torch.mean(2 * correct_pred / (torch.sum(predictions, axis=1) + torch.sum(y_onehot, axis=1))).item()
            total_f1 += f1

            if propensity is not None:
                # NOTE: fout is an ordered list of labels based on probability scores.
                # This is not ideal but is a requirement for the xmetrics package.
                actual_outputs = y_onehot.numpy()[:, :-1] # Remove the last column (padding_idx)
                predicted_outputs = y_cpu.numpy()[:, :-1]
                acc = compute_prop_metrics(sparse.csr_matrix(actual_outputs),
                                           sparse.csr_matrix(predicted_outputs), propensity,
                                           topk=topk)
                all_acc.append(acc)

                np.set_printoptions(threshold=np.inf)

                batch_mrr = compute_mrr(y_cpu.numpy(), y_onehot.numpy())
        
                total_mrr += batch_mrr

    # Compute metrics for current threshold.
    num_itr = idx + 1

    # Reset the index.
    ###model.reset_index()

    #戻り値の順番
    # loss , f1, precision, recall

    if propensity is not None:
        return total_loss/num_itr, total_f1/num_itr,\
               total_pr/num_itr, total_rec/num_itr, all_acc, total_mrr/num_itr
    else:
        return total_loss/num_itr, total_f1/num_itr, \
               total_pr/num_itr, total_rec/num_itr

