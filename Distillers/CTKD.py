import torch
import torch.nn as nn
import torch.nn.functional as F
import Distillers.cfg as CFG
from torch.autograd import Function
import numpy as np

class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()

        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

class CTKD(nn.Module):
    """Curriculum Temperature for Knowledge Distillation (AAAI 2023)"""

    def __init__(self):
        super(CTKD, self).__init__()
        self.temp = CFG.CKD_temp
        self.alpha = CFG.lamda_kd
        self.start = CFG.CTKD_start
        self.end = CFG.CTKD_end
        self.hard_loss = nn.CrossEntropyLoss()  # task loss
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')  # distillation loss

    def forward(self, students_preds, teachers_preds, hard_label, mlp_net, cos_value):
        mlp_net.eval()
        temp = mlp_net(teachers_preds, students_preds, cos_value)  # (teacher_output, student_output)
        temp = self.start + self.end * torch.sigmoid(temp)
        temp = temp.cuda()
        # temp = temp.to(torch.device('cuda', CFG.devices_id[0]))

        # if np.isnan(temp.cpu().detach().numpy()):
        #     print('teachers_preds: ' + str(teachers_preds))
        #     print('students_preds: ' + str(students_preds))
        #     print('Temp: '+str(temp))
        #     assert 1 == 2

        distillation_loss = self.soft_loss(
            F.log_softmax(students_preds / temp, dim=1),
            F.softmax(teachers_preds / temp, dim=1)
        )
        students_loss = self.hard_loss(students_preds, hard_label.long())

        loss = 0.1 * students_loss + 0.9 * distillation_loss

        return loss, students_loss, distillation_loss
