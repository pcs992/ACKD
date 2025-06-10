import torch
import torch.nn as nn
import torch.nn.functional as F
import Distillers.cfg as CFG

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

class LSKD(nn.Module):
    """Logit standardization in knowledge distillation (CVPR 2024)"""

    def __init__(self):
        super(LSKD, self).__init__()
        self.temp = CFG.CKD_temp
        self.alpha = CFG.lamda_kd
        self.hard_loss = nn.CrossEntropyLoss()  # task loss
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')  # distillation loss

    def forward(self, students_preds, teachers_preds, hard_label):
        logits_student = normalize(students_preds)
        logits_teacher = normalize(teachers_preds)

        students_loss = self.hard_loss(students_preds, hard_label.long())

        log_pred_student = F.log_softmax(logits_student / self.temp, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temp, dim=1)
        distillation_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        distillation_loss *= self.temp ** 2

        loss = self.alpha * students_loss + (1 - self.alpha) * distillation_loss

        return loss, students_loss, distillation_loss