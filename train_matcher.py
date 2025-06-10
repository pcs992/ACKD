import random
import logging
import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.dataset import S2S_EU, S2S_CN, M2S_GL
from configration.config_matcher import run_config as config
from model.CLIP import CLIPModel
from utils.matcher_utils import AvgMeter, get_lr
from osgeo import gdal
import argparse
gdal.PushErrorHandler("CPLQuietErrorHandler")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner

class Trainer(object):
    def __init__(self, CFG):
        self.CFG = CFG
        self.dataset_classes = {
            'S2S_EU': S2S_EU,
            'S2S_CN': S2S_CN,
            'M2S_GL': M2S_GL
        }
        self.dataset_class = self.dataset_classes.get(self.CFG.data_name, self.dataset_classes['M2S_GL'])
        self.rs_dataset = self.dataset_class(json_dir=self.CFG.train_teacher_root,
                                             category=self.CFG.category_list,
                                             train_flag=True,
                                             modality="CLIP",
                                             teacher_dir=None,
                                             student_dir=None)

        kfold = KFold(n_splits=self.CFG.folds, shuffle=True)
        splits = kfold.split(self.rs_dataset)
        train_indices, val_indices = next(splits)
        self.train_dataset = torch.utils.data.Subset(self.rs_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.rs_dataset, val_indices)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.CFG.batch,
                                       shuffle=self.CFG.train_data_shuffle,
                                       num_workers=self.CFG.num_workers)
        self.valid_loader = DataLoader(dataset=self.val_dataset,
                                       batch_size=self.CFG.batch,
                                       shuffle=self.CFG.test_data_shuffle,
                                       num_workers=self.CFG.num_workers)

        self.model = CLIPModel(temperature=self.CFG.temperature,
                               image_embedding=self.CFG.image_embedding,
                               text_embedding=self.CFG.text_embedding,
                               projection_dim=self.CFG.projection_dim,
                               dropout=self.CFG.dropout,
                               train_num_class=self.CFG.train_num_class,
                               msi_channel=self.CFG.msi_channel,
                               rgb_channel=self.CFG.rgb_channel,
                               ).to(self.CFG.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.CFG.lr,
            weight_decay=self.CFG.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=self.CFG.patience, factor=self.CFG.factor
        )
        self.step = "epoch"
        self.best_loss = float('inf')

        self.trian_loss_meter = AvgMeter()
        self.val_loss_meter = AvgMeter()

    def train_epoch(self):
        self.model.train()
        tqdm_object = tqdm(self.train_loader, total=len(self.train_loader))

        for i, data in enumerate(tqdm_object, 0):
            msi_img, rgb_img, label, subject = data
            msi_img = msi_img.float().to(CFG.device)
            rgb_img = rgb_img.float().to(CFG.device)

            loss = self.model(msi_img, rgb_img)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.step == "batch":
                self.lr_scheduler.step()

            count = msi_img.size(0)
            self.trian_loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(train_loss=self.trian_loss_meter.avg, lr=get_lr(self.optimizer))

        with torch.no_grad():
            self.valid_epoch()

    def valid_epoch(self):
        self.model.eval()

        loss_meter = AvgMeter()
        tqdm_object = tqdm(self.valid_loader, total=len(self.valid_loader))

        for i, data in enumerate(tqdm_object, 0):
            msi_img, rgb_img, label, subject = data
            msi_img = msi_img.float().to(CFG.device)
            rgb_img = rgb_img.float().to(CFG.device)

            loss = self.model(msi_img, rgb_img)

            count = msi_img.size(0)
            loss_meter.update(loss.item(), count)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        if self.val_loss_meter.avg < self.best_loss:
            self.best_loss = self.val_loss_meter.avg
            torch.save(self.model.state_dict(), self.CFG.save_model_path)
            print("Saved Best Model!")

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='S2S_CN', help='S2S_EU, S2S_CN, M2S_GL')
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
    trainer = Trainer(CFG=CFG)

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        trainer.train_epoch()