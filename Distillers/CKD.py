import torch.nn as nn
import torch.nn.functional as F
import Distillers.cfg as CFG

class CKD(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(CKD, self).__init__()
        self.temp = CFG.CKD_temp
        self.hard_loss = nn.CrossEntropyLoss()  # task loss
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')  # distillation loss

    def forward(self, students_preds, teachers_preds, hard_label):
        distillation_loss = self.soft_loss(
            F.log_softmax(students_preds / self.temp, dim=1),
            F.softmax(teachers_preds / self.temp, dim=1)
        )
        students_loss = self.hard_loss(students_preds, hard_label.long())

        loss = students_loss + distillation_loss

        return loss, students_loss, distillation_loss