import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import rasterio
import json
from osgeo import gdal

class S2S_EU(torch.utils.data.Dataset):

    def __init__(self,
                 json_dir,
                 category,
                 train_flag=True,
                 modality=None,
                 teacher_dir=None,
                 student_dir=None):
        super(S2S_EU, self).__init__()
        self.label_name = category
        self.label_class = {category: idx for idx, category in enumerate(self.label_name)}

        self.input_type = modality
        self.teacher_dir = teacher_dir
        self.student_dir = student_dir

        if self.input_type == "CLIP":
            self.img_path = self.get_teacher_info(json_dir)
        elif self.input_type == "UNI":
            if teacher_dir is not None:
                self.img_path = self.get_teacher_info(self.teacher_dir)
            if student_dir is not None:
                self.img_path = self.get_student_info(self.student_dir)
        else:
            with open(json_dir, 'r') as file:
                self.img_path = json.load(file)

        self.msi_mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.msi_std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.normalize = transforms.Normalize(mean=self.msi_mean, std=self.msi_std)
        self.trans_tensor = transforms.ToTensor()

        self.rgb_transform_t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.rgb_transform_v = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_flag = train_flag

    def __getitem__(self, index):
        if self.input_type == "CLIP":
            path_msi = self.img_path[index][0]
            subject = self.img_path[index][1]

            with rasterio.open(path_msi, "r") as src:
                # Read in the red, green, and blue bands
                msi_2_rgb = src.read([1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
                msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
                msi_img = msi_2_rgb.transpose(1, 2, 0)

                rgb_ssl = src.read([1, 2, 3])
                rgb_ssl = (rgb_ssl / rgb_ssl.max() * 255).astype(np.uint8)

            msi_img = self.trans_tensor(msi_img)
            normalized_msi = self.normalize(msi_img)

            label = torch.Tensor([self.label_class[subject]])
            return normalized_msi, rgb_ssl, label, subject

        elif self.input_type == "UNI":
            # Teacher Only
            if self.teacher_dir is not None:
                path_msi = self.img_path[index][0]
                subject = self.img_path[index][1]
                label = torch.Tensor([self.label_class[subject]])
                with rasterio.open(path_msi, "r") as src:
                    # Read in the red, green, and blue bands
                    msi_2_rgb = src.read([1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
                    msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
                    msi_img = msi_2_rgb.transpose(1, 2, 0)
                msi_img = self.trans_tensor(msi_img)
                normalized_msi = self.normalize(msi_img)

                return normalized_msi, label, subject, path_msi
            # Student Only
            if self.student_dir is not None:
                path_rgb = self.img_path[index][0]
                subject = self.img_path[index][-1]
                rgb_img = Image.open(path_rgb)
                label = torch.Tensor([self.label_class[subject]])
                if self.train_flag:
                    rgb_img = self.rgb_transform_t(rgb_img)
                else:
                    rgb_img = self.rgb_transform_v(rgb_img)

                return rgb_img, label, subject, path_rgb

        else:
            path_msi = self.img_path[index][0]
            path_rgb = self.img_path[index][-1]
            # subject = path_msi.split('/')[5]
            subject = path_msi.split('/')[-2]

            with rasterio.open(path_msi, "r") as src:
                # Read in the red, green, and blue bands
                msi_2_rgb = src.read([1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
                msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
                msi_img = msi_2_rgb.transpose(1, 2, 0)

            rgb_img = Image.open(path_rgb)
            msi_img = self.trans_tensor(msi_img)
            normalized_msi = self.normalize(msi_img)

            if self.train_flag:
                rgb_img = self.rgb_transform_t(rgb_img)
            else:
                rgb_img = self.rgb_transform_v(rgb_img)

            label = torch.Tensor([self.label_class[subject]])

            return normalized_msi, rgb_img, label, subject, path_rgb, index

    def __len__(self):
        return len(self.img_path)

    def get_teacher_info(self, teacher_dir):
        data_info_t = list()
        for subjects in (self.label_name):
            path_t = os.listdir(os.path.join(teacher_dir, str(subjects)))
            msi_img_names = list(filter(lambda x: x.endswith('.tif'), path_t))
            root_t = os.path.join(teacher_dir, str(subjects))
            min_data = len(msi_img_names)
            for i in range(min_data):
                msi_root = root_t + '/' + msi_img_names[i]
                data_info_t.append((msi_root, subjects))

        return data_info_t

    def get_student_info(self, student_dir):
        data_info_s = list()
        for subjects in (self.label_name):
            path_s = os.listdir(os.path.join(student_dir, str(subjects)))
            rgb_img_names = list(filter(lambda x: x.endswith('.jpg'), path_s))
            root_s = os.path.join(student_dir, str(subjects))
            min_data = len(rgb_img_names)

            for i in range(min_data):
                rgb_root = root_s + '/' + rgb_img_names[i]
                data_info_s.append((rgb_root, subjects))

        return data_info_s

class S2S_CN(torch.utils.data.Dataset):

    def __init__(self, json_dir, category, train_flag=True, modality=None, teacher_dir=None,
                 student_dir=None):
        super(S2S_CN, self).__init__()
        self.label_name = category
        self.label_class = {category: idx for idx, category in enumerate(self.label_name)}

        self.input_type = modality
        self.teacher_dir = teacher_dir
        self.student_dir = student_dir

        if self.input_type == "CLIP":
            self.img_path = self.get_teacher_info(json_dir)
        elif self.input_type == "UNI":
            if teacher_dir is not None:
                self.img_path = self.get_teacher_info(self.teacher_dir)
            if student_dir is not None:
                self.img_path = self.get_student_info(self.student_dir)
        else:
            with open(json_dir, 'r') as file:
                self.img_path = json.load(file)

        self.msi_mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.msi_std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.normalize = transforms.Normalize(mean=self.msi_mean, std=self.msi_std)
        self.trans_tensor = transforms.ToTensor()

        self.rgb_transform_t = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.rgb_transform_v = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_flag = train_flag

    def __getitem__(self, index):

        if self.input_type == "CLIP":
            path_msi = self.img_path[index][0]
            subject = self.img_path[index][1]

            msi_2_rgb = gdal.Open(path_msi).ReadAsArray()
            rgb_ssl = msi_2_rgb[0:3]
            msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
            msi_img = msi_2_rgb.transpose(1, 2, 0)

            rgb_ssl = (rgb_ssl / rgb_ssl.max() * 255).astype(np.uint8)

            msi_img = self.trans_tensor(msi_img)
            normalized_msi = self.normalize(msi_img)

            label = torch.Tensor([self.label_class[subject]])

            return normalized_msi, rgb_ssl, label, subject

        elif self.input_type == "UNI":
            if self.teacher_dir is not None:
                path_msi = self.img_path[index][0]
                subject = self.img_path[index][1]
                msi_2_rgb = gdal.Open(path_msi).ReadAsArray()
                msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
                msi_img = msi_2_rgb.transpose(1, 2, 0)
                msi_img = self.trans_tensor(msi_img)
                normalized_msi = self.normalize(msi_img)
                label = torch.Tensor([self.label_class[subject]])

                return normalized_msi, label, subject, path_msi

            if self.student_dir is not None:
                path_rgb = self.img_path[index][0]
                subject = self.img_path[index][-1]
                rgb_img = Image.open(path_rgb)
                if self.train_flag:
                    rgb_img = self.rgb_transform_t(rgb_img)
                else:
                    rgb_img = self.rgb_transform_v(rgb_img)

                label = torch.Tensor([self.label_class[subject]])

                return rgb_img, label, subject, path_rgb

        else:
            path_msi = self.img_path[index][0]
            path_rgb = self.img_path[index][-1]
            # subject = path_msi.split('/')[6]
            subject = path_msi.split('/')[-2]

            msi_2_rgb = gdal.Open(path_msi).ReadAsArray()
            msi_2_rgb = (msi_2_rgb / msi_2_rgb.max() * 255).astype(np.uint8)
            msi_img = msi_2_rgb.transpose(1, 2, 0)

            rgb_img = Image.open(path_rgb)
            msi_img = self.trans_tensor(msi_img)
            normalized_msi = self.normalize(msi_img)

            if self.train_flag:
                rgb_img = self.rgb_transform_t(rgb_img)
            else:
                rgb_img = self.rgb_transform_v(rgb_img)

            label = torch.Tensor([self.label_class[subject]])

            return normalized_msi, rgb_img, label, subject, path_rgb, index

    def __len__(self):
        return len(self.img_path)

    def get_teacher_info(self, teacher_dir):
        data_info_t = list()

        for subjects in (self.label_name):
            path_t = os.listdir(os.path.join(teacher_dir, str(subjects)))
            msi_img_names = list(filter(lambda x: x.endswith('.tif'), path_t))
            root_t = os.path.join(teacher_dir, str(subjects))
            min_data = len(msi_img_names)

            for i in range(min_data):
                msi_root = root_t + '/' + msi_img_names[i]
                data_info_t.append((msi_root, subjects))

        return data_info_t

    def get_student_info(self, student_dir):
        data_info_s = list()
        for subjects in (self.label_name):
            path_s = os.listdir(os.path.join(student_dir, str(subjects)))
            rgb_img_names = list(filter(lambda x: x.endswith('.jpg'), path_s))
            root_s = os.path.join(student_dir, str(subjects))
            min_data = len(rgb_img_names)

            for i in range(min_data):
                rgb_root = root_s + '/' + rgb_img_names[i]
                data_info_s.append((rgb_root, subjects))

        return data_info_s


class M2S_GL(torch.utils.data.Dataset):
    def __init__(self, json_dir, category, train_flag=True, modality=None, teacher_dir=None,
                 student_dir=None):
        super(M2S_GL, self).__init__()

        self.label_name = category
        self.label_class = {category: idx for idx, category in enumerate(self.label_name)}

        self.input_type = modality
        self.teacher_dir = teacher_dir
        self.student_dir = student_dir

        if self.input_type == "CLIP":
            self.img_path = self.get_teacher_info(json_dir)
        elif self.input_type == "UNI":
            if teacher_dir is not None:
                self.img_path = self.get_teacher_info(self.teacher_dir)
            if student_dir is not None:
                self.img_path = self.get_student_info(self.student_dir)
        else:
            with open(json_dir, 'r') as file:
                self.img_path = json.load(file)

        self.msi_mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.msi_std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.normalize = transforms.Normalize(mean=self.msi_mean, std=self.msi_std)
        self.trans_tensor = transforms.ToTensor()

        self.rgb_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.rgb_transform_clip = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_flag = train_flag

    def __getitem__(self, index):
        if self.input_type == "CLIP":
            path_msi = self.img_path[index][0]
            subject = self.img_path[index][1]

            msi_img = scio.loadmat(path_msi)['msi']
            msi_img = self.trans_tensor(msi_img)
            normalized_msi = self.normalize(msi_img)

            rgb_ssl = msi_img[0:3, :, :]
            rgb_ssl = self.rgb_transform_clip(rgb_ssl)

            label = torch.Tensor([self.label_class[subject]])
            return normalized_msi, rgb_ssl, label, subject

        elif self.input_type == "UNI":
            if self.teacher_dir is not None:
                path_msi = self.img_path[index][0]
                subject = self.img_path[index][1]
                msi_img = scio.loadmat(path_msi)['msi']
                msi_img = self.trans_tensor(msi_img)
                normalized_msi = self.normalize(msi_img)
                label = torch.Tensor([self.label_class[subject]])

                return normalized_msi, label, subject, path_msi

            if self.student_dir is not None:
                path_rgb = self.img_path[index][0]
                subject = self.img_path[index][-1]
                rgb_img = Image.open(path_rgb)
                rgb_img = self.rgb_transform(rgb_img)
                label = torch.Tensor([self.label_class[subject]])

                return rgb_img, label, subject, path_rgb

        else:
            path_msi = self.img_path[index][0]
            path_rgb = self.img_path[index][-1]
            # subject = path_msi.split('/')[6]
            subject = path_msi.split('/')[-2]

            msi_2_rgb = scio.loadmat(path_msi)['msi']

            rgb_img = Image.open(path_rgb)
            msi_img = self.trans_tensor(msi_2_rgb)
            normalized_msi = self.normalize(msi_img)

            rgb_img = self.rgb_transform(rgb_img)

            label = torch.Tensor([self.label_class[subject]])

            return normalized_msi, rgb_img, label, subject, path_rgb, index

    def __len__(self):
        return len(self.img_path)

    def get_teacher_info(self, teacher_dir):
        data_info_t = list()
        for subjects in (self.label_name):
            path_t = os.listdir(os.path.join(teacher_dir, str(subjects)))
            msi_img_names = list(filter(lambda x: x.endswith('.mat'), path_t))
            root_t = os.path.join(teacher_dir, str(subjects))
            for i in range(len(msi_img_names)):
                msi_root = root_t + '/' + msi_img_names[i]
                data_info_t.append((msi_root, subjects))

        return data_info_t

    def get_student_info(self, student_dir):
        data_info_s = list()
        for subjects in (self.label_name):
            path_s = os.listdir(os.path.join(student_dir, str(subjects)))
            rgb_img_names = list(filter(lambda x: x.endswith('.jpg'), path_s))
            root_s = os.path.join(student_dir, str(subjects))
            min_data = len(rgb_img_names)
            for i in range(min_data):
                rgb_root = root_s + '/' + rgb_img_names[i]
                data_info_s.append((rgb_root, subjects))

        return data_info_s

class Multi_label_Dataset(torch.utils.data.Dataset):
    def __init__(self, json_dir, category, train_flag=True, modality=None, teacher_dir=None,
                 student_dir=None):
        super(Multi_label_Dataset, self).__init__()
        self.label_name = category
        self.label_class = {category: idx for idx, category in enumerate(self.label_name)}
        self.data_info_t = self.get_img_info(teacher_dir)

        self.msi_mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.msi_std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.normalize = transforms.Normalize(mean=self.msi_mean, std=self.msi_std)
        self.trans_tensor = transforms.ToTensor()

        self.train_flag = train_flag

    def __getitem__(self, index):
        path_msi = self.data_info_t[index][0]
        subject = self.data_info_t[index][1]

        msi_img = scio.loadmat(path_msi)['msi']
        multi_subjects = scio.loadmat(path_msi)['label']
        multi_subjects = [item.strip() for item in multi_subjects]
        multi_label = []

        for s_label in self.label_name:
            if s_label in multi_subjects:
                multi_label.append(1)
            else:
                multi_label.append(0)

        msi_img = self.trans_tensor(msi_img)
        normalized_msi = self.normalize(msi_img)

        label = torch.Tensor(multi_label)

        return normalized_msi, label, subject, path_msi

    def __len__(self):
        return len(self.data_info_t)

    def get_img_info(self, teacher_dir):
        data_info_t = list()
        for subjects in (self.label_name):
            path_t = os.listdir(os.path.join(teacher_dir, str(subjects)))
            msi_img_names = list(filter(lambda x: x.endswith('.mat'), path_t))
            root_t = os.path.join(teacher_dir, str(subjects))

            for i in range(len(msi_img_names)):
                msi_root = root_t + '/' + msi_img_names[i]
                data_info_t.append((msi_root, subjects))

        return data_info_t






