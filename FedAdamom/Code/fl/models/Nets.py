import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_fed import ResNet18 as FedResNet18



class client_model(nn.Module):
    def __init__(self, name,classes, args=True):
        super(client_model, self).__init__()
        self.name = name
        self.classes = classes
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

        if self.name == 'Resnet18':
            resnet18 = FedResNet18(num_classes=self.classes, use_bn_layer=False)
            self.model = resnet18

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80

            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x):
        if self.name == 'Linear':
            x = x.view(-1, 1 * 28 * 28)
            x = self.fc(x)

        if self.name == 'Resnet18':
            x = self.model(x)

        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  
            self.stacked_LSTM.flatten_parameters()
            output, (h_, c_) = self.stacked_LSTM(x)
            
            last_hidden = output[-1, :, :]
            x = self.fc(last_hidden)

        return x
