import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F
import random
import numpy as np

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L1 = 256
        self.L2 = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv3d(3, 20, kernel_size=(1,3,3)),
            nn.ReLU(),
            nn.Dropout3d(p=0.25),
            nn.MaxPool3d(2, stride=1),

            nn.Conv3d(20, 50, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.Dropout3d(p=0.25),
            nn.MaxPool3d(2, stride=1)
        )

        self.feature_extractor_part2 = nn.Sequential(
            # nn.Linear(50 * 12*10*10, self.L1),
            nn.Linear(50 * 4*2*2, self.L1),
            nn.ReLU(),
            nn.Dropout3d(p=0.25),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L1, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        # self.attention = nn.Sequential(
        #     nn.Linear(self.L1, self.L2),
        #     nn.Tanh(),
        #     nn.Dropout2d(0.5),
        #     nn.Linear(self.L2, self.D),
        #     nn.Tanh(),
        #     nn.Dropout2d(0.5),
        #     nn.Linear(self.D, self.K)
        # )


        self.classifier = nn.Sequential(
            nn.Linear(self.L1*self.K, 1)#,
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4*2*2)#H.view(-1, 50 * 12*10*10) # 
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.L1 = 256
        self.L2 = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part1_ws = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1,3,3)),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2, stride=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2, stride=(2,2, 2)), 
            # nn.Conv3d(128, 256, kernel_size=(3,3,3)),
            # nn.ReLU(),
            # nn.Dropout3d(0.2),
            # nn.MaxPool3d(2, stride=2), 
            # nn.Conv3d(64, 128, kernel_size=(3,3,3)),
            # nn.ReLU(),
            # nn.Dropout3d(0.2),
            # nn.MaxPool3d(2, stride=2)
        )

        self.feature_extractor_part2_ws = nn.Sequential(
            nn.Linear(128 * 7*14*14, self.L1),
            nn.ReLU(),
        )


        # self.classifier = nn.Sequential(
        #     nn.Linear(256, 1)#,
        #     #nn.Sigmoid()
        # )

        self.classifier2 = nn.Sequential(
            nn.Linear(256, 1)#,
            #nn.Sigmoid()
        )

    def forward(self, x_ws):
        H1 = self.feature_extractor_part1_ws(x_ws)
        #print('H1', H1.shape)

        H = H1.view(-1, 128*7*14*14)#12*10*10)
        out_after_pooling = self.feature_extractor_part2_ws(H)  # NxL   

        out = self.classifier2(out_after_pooling)

        return  out, out_after_pooling, H1

