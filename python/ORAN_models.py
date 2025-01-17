import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math


# Define model
class ConvNN(nn.Module):
    def __init__(self, numChannels=1, slice_len=4, num_feats=17, classes=3):
        super(ConvNN, self).__init__()

        self.numChannels = numChannels

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
                               kernel_size=(4, 1))
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))
        ##  initialize second set of CONV => RELU => POOL layers
        # self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
        #                    kernel_size=(5, 5))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## initialize first (and only) set of FC => RELU layers

        # pass a random input
        rand_x = torch.Tensor(np.random.random((1, 1, slice_len, num_feats)))
        output_size = torch.flatten(self.conv1(rand_x)).shape
        self.fc1 = nn.Linear(in_features=output_size.numel(), out_features=512)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=512, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.reshape \
            ((x.shape[0], self.numChannels, x.shape[1], x.shape[2]))   # CNN 2D expects a [N, Cin, H, W] size of data
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)
        ## pass the output from the previous layer through the second
        ## set of CONV => RELU => POOL layers
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        ## flatten the output from the previous layer and pass it
        ## through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class TransformerNN(nn.Module):
    def __init__(self, classes: int = 3, num_feats: int = 17, slice_len: int = 32, nhead: int = 1, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False, custom_enc: bool = False):
        super(TransformerNN, self).__init__()

        if use_pos and not custom_enc:
            num_feats = num_feats - 1 # exclude the timestamp column (0) that will be used for positional encoding

        self.norm = nn.LayerNorm(num_feats)
        # create the positional encoder
        self.use_positional_enc = use_pos

        # TODO not entirely sure why we need d_model = num_feats + 1 for tradtional pos. encoder
        self.pos_encoder = PositionalEncoding(num_feats + 1, dropout, custom_enc=custom_enc) if use_pos else None
        # define the encoder layers

        encoder_layers = TransformerEncoderLayer(d_model=num_feats, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats*slice_len, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(256, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        # src = self.norm(src) should not be necessary since output can be already normalized
        # apply positional encoding if decided
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # if torch.isnan(t_out).any():
        #     print("NaN detected after encoder")
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # if torch.isnan(t_out).any():
        #     print("NaN detected after flatten")
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        # if torch.isnan(pooler).any():
        #     print("NaN detected after pre_classifier")
        pooler = torch.nn.ReLU()(pooler)
        # if torch.isnan(pooler).any():
        #     print("NaN detected after ReLU")
        pooler = self.dropout(pooler)
        # if torch.isnan(pooler).any():
        #     print("NaN detected after dropout")
        output = self.classifier(pooler)
        # if torch.isnan(output).any():
        #     print("NaN detected after classifier")
        # output = self.logSoftmax(output)
        # if torch.isnan(output).any():
        #     print("NaN detected after logSoftmax")
        return output
