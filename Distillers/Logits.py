from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import Distillers.cfg as CFG

class Logits(nn.Module):
    '''
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    '''

    def __init__(self):
        super(Logits, self).__init__()
        self.alpha = CFG.lamda_kd
        self.hard_loss = nn.CrossEntropyLoss()  # task loss

    def forward(self, out_s, out_t, hard_label):
        distillation_loss = 0.01 * F.mse_loss(out_s, out_t)
        students_loss = self.hard_loss(out_s, hard_label.long())
        loss = self.alpha * students_loss + (1 - self.alpha) * distillation_loss

        return loss, students_loss, distillation_loss