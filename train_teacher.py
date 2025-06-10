import random
import logging
import sys
import os
import numpy as np
import math
from tqdm import tqdm
from val import val_mlp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model.ResNet import resnet34
from model.MobileNetV2 import MobileNetV2
from model.ShuffleNet import shufflenet_v2_x0_5 as shufflenet
from sklearn.model_selection import KFold
from utils.tools import AvgMeter, get_lr
from main_config import run_config as config
from utils.dataset import S2S_EU, S2S_CN, Multi_label_Dataset

import argparse
from osgeo import gdal
gdal.PushErrorHandler("CPLQuietErrorHandler")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner

class CosineDecay(object):
    def __init__(self, max_value, min_value, num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class Trainer(object):
    def __init__(self, CFG, model_type, kd_mode, teacher_pth_root, student_pth_root):
        self.CFG = CFG
        self.model_list = model_type
        self.kd_mode = kd_mode
        self.teacher_pth_root = teacher_pth_root
        self.student_pth_root = student_pth_root

        self.dataset_classes = {
            'S2S_EU': S2S_EU,
            'S2S_CN': S2S_CN,
            'default': Multi_label_Dataset
        }

        self.task = {
            'S2S_EU': nn.CrossEntropyLoss(), # single-label classification
            'S2S_CN': nn.CrossEntropyLoss(), # single-label classification
            'default': nn.BCEWithLogitsLoss() # multi-label classification
        }

        self.cls_losses = AvgMeter()

        self.dataset_class = self.dataset_classes.get(self.CFG.data_name, self.dataset_classes['default'])
        self.criterions = self.task.get(self.CFG.data_name, self.task['default'])

        self.rs_dataset = self.dataset_class(json_dir=self.CFG.json_dir,
                                             category=self.CFG.category_list,
                                             train_flag=True,
                                             modality="UNI",
                                             teacher_dir=self.CFG.train_teacher_root,
                                             student_dir=None
                                             )

        kfold = KFold(n_splits=self.CFG.K_fold, shuffle=True)
        splits = kfold.split(self.rs_dataset)
        train_indices, val_indices = next(splits)
        logging.info('----------- Dataset Initialization --------------')
        self.train_dataset = torch.utils.data.Subset(self.rs_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.rs_dataset, val_indices)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.CFG.batch,
                                       shuffle=self.CFG.train_data_shuffle,
                                       num_workers=self.CFG.num_workers
                                       )
        self.val_loader = DataLoader(dataset=self.val_dataset,
                                     batch_size=self.CFG.batch,
                                     shuffle=self.CFG.test_data_shuffle,
                                     num_workers=self.CFG.num_workers
                                     )

        self.teacher = self._init_model(model_type[0], self.CFG.train_num_class, self.CFG.msi_channel)
        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.teacher)

        self.optimizer = torch.optim.Adam(self.trainable_list.parameters(), lr=self.CFG.learning_rate)

        self.indicator_history = 0
        self.gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=5)

    def _init_model(self, model_name, num_classes, in_channels):
        if model_name == 'ResNet':
            logging.info(f'----------- {"T" if in_channels == self.CFG.msi_channel else "S"}: ResNet --------------')
            return resnet34(num_classes, in_channels).cuda()
        elif model_name == 'MobileNet':
            logging.info(f'----------- {"T" if in_channels == self.CFG.msi_channel else "S"}: MobileNet --------------')
            return MobileNetV2(num_classes=num_classes, in_c=in_channels).cuda()
        elif model_name == 'ShuffleNet':
            logging.info(
                f'----------- {"T" if in_channels == self.CFG.msi_channel else "S"}: ShuffleNet --------------')
            return shufflenet(num_classes=num_classes, in_c=in_channels).cuda()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def training(self, epoch):
        """Train the student model using knowledge distillation"""
        # for epoch in range(self.CFG.epochs):
        logging.info('Epoch: %s', str(epoch + 1))

        self.teacher.train()

        self.decay_value = self.gradient_decay.get_value(epoch)

        trainloader = tqdm(self.train_loader)
        for i, data in enumerate(trainloader, 0):
            msi_img, label, subject, _ = data

            msi_img = msi_img.float().cuda()

            if (label.size(0) == 1):
                labels = label[0].cuda().float()
            else:
                labels = torch.squeeze(label).cuda().float()

            if self.CFG.data_name == 'S2S_EU' or self.CFG.data_name == 'S2S_CN':
                labels = labels.long()

            self.optimizer.zero_grad()

            msi_embedding, msi_logits = self.teacher(msi_img)

            loss = self.criterions(msi_logits, labels)

            self.cls_losses.update(loss.item(), msi_img.size(0))

            loss.backward()

            self.optimizer.step()

        self.indicator_history = self.model_evaluation(self.teacher, self.val_loader, self.indicator_history, self.teacher_pth_root)


    def model_evaluation(self, student, val_loader, indicator_history, teacher_pth_root):
        """Evaluate the student model"""
        OA, precision, recall, F1_score = val_mlp(student, val_loader, mode='Teacher', label_type=self.CFG.label_type)

        if self.CFG.indicator_prior == 'OA':
            indicator = OA
        elif self.CFG.indicator_prior == 'Precision':
            indicator = precision
        elif self.CFG.indicator_prior == 'Recall':
            indicator = recall
        else:
            indicator = F1_score

        if indicator_history < indicator:
            indicator_history = indicator
            torch.save(student.state_dict(), teacher_pth_root)

        logging.info('Best ACC: %s', str(indicator_history))
        logging.info(' ')

        return indicator_history

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='M2S_GL', help='S2S_EU, S2S_CN, M2S_GL')

    args = parser.parse_args()

    CFG = config(args.dataset)

    # Setup logging
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(CFG.save_log, CFG.log_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    setup_seed(CFG.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = CFG.cuda_id

    # Main flow
    trainer = Trainer(CFG=CFG,
                      model_type=CFG.model_list[0],
                      kd_mode=CFG.kd_mode[5],
                      teacher_pth_root=CFG.teacher_list[0],
                      student_pth_root=CFG.student_list[0]
                      )

    for epoch in range(CFG.epochs):
        trainer.training(epoch)
