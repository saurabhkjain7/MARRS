import numpy as np
import random
import os
import torch
import torch.nn as nn
from scipy.linalg import sqrtm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

def coral(source_features, val_target_features, unlabeled_target_features, labeled_target_features):
    x_s = torch.cat((source_features.to(device), labeled_target_features.to(device)))
    x_t = unlabeled_target_features.to(device)
    x_tv = val_target_features.to(device)
    x_s_n = x_s - x_s.mean(0) # centered source data
    x_t_n = x_t - x_t.mean(0) # centered target data
    x_tv = x_tv - x_t.mean(0) # centered validation data

    x_s_cov = torch.matmul(x_s_n.T, x_s_n) / (x_s_n.shape[0] - 1.)
    x_s_cov = x_s_cov + 0.01 * torch.eye(x_s_cov.shape[0]).to(device)
    x_t_cov = torch.matmul(x_t_n.T, x_t_n) / (x_t_n.shape[0] - 1.)
    x_t_cov = x_t_cov + 0.01 * torch.eye(x_t_cov.shape[0]).to(device)

    x_s_cov_sqrt = torch.tensor(sqrtm(x_s_cov.cpu())).to(device)
    x_s_cov_sqrt_inv = x_s_cov_sqrt.inverse()
    x_s_whitened = torch.matmul(x_s_n, x_s_cov_sqrt_inv.float()) # whiten
    x_t_cov_sqrt = torch.tensor(sqrtm(x_t_cov.cpu())).to(device)
    x_s = torch.matmul(x_s_whitened, x_t_cov_sqrt.float()) # recolor with target variance

    x_tu = x_t_n # centered target
    x_t = x_s[source_features.shape[0]:] # target
    x_s = x_s[:source_features.shape[0]] # source 
    
    # target unlabeled, target labeled, source, target validation
    return x_tu, x_t, x_s, x_tv


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets,T=1.0):
        log_probs = self.logsoftmax(inputs/T)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss
