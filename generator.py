import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset import S2S_EU, S2S_CN, M2S_GL
from torch.utils.data import DataLoader
from configration.config_matcher import run_config as config
from model.CLIP import CLIPModel
import json
from osgeo import gdal
import argparse
gdal.PushErrorHandler("CPLQuietErrorHandler")

class Matcher(object):
    def __init__(self, CFG):
        self.CFG = CFG
        self.dataset_classes = {
            'S2S_EU': S2S_EU,
            'S2S_CN': S2S_CN,
            'M2S_GL': M2S_GL
        }

        self.Gallery = {
            category: {
                'embedding': [],
                'image_list': []
            } for category in self.CFG.category_list
        }

        self.dataset_class = self.dataset_classes.get(self.CFG.data_name, self.dataset_classes['M2S_GL'])
        self.t_dataset = self.dataset_class(json_dir=self.CFG.train_teacher_root,
                                            category=self.CFG.category_list,
                                            train_flag=True,
                                            modality="UNI",
                                            teacher_dir=self.CFG.train_teacher_root,
                                            student_dir=None
                                            )
        self.s_dataset = self.dataset_class(json_dir=self.CFG.train_teacher_root,
                                            category=self.CFG.category_list,
                                            train_flag=True,
                                            modality="UNI",
                                            teacher_dir=None,
                                            student_dir=self.CFG.unpaired_root
                                            )
        self.t_valid_loader = DataLoader(dataset=self.t_dataset,
                                         batch_size=1,
                                         shuffle=CFG.train_data_shuffle,
                                         num_workers=self.CFG.num_workers)

        self.s_valid_loader = DataLoader(dataset=self.s_dataset,
                                         batch_size=1,
                                         shuffle=CFG.train_data_shuffle,
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

        self.model.load_state_dict(torch.load(self.CFG.save_model_path, map_location=CFG.device))

        self.model.eval()

    def get_image_embeddings(self):
        with torch.no_grad():
            tqdm_object = tqdm(self.t_valid_loader, total=len(self.t_valid_loader))
            for i, data in enumerate(tqdm_object, 0):
                msi_img, _, subject, msi_path = data
                msi_img = msi_img.float().to(CFG.device)
                msi_features, _ = self.model.image_encoder(msi_img)
                msi_embeddings = self.model.image_projection(msi_features)

                self.Gallery[subject[0]]['embedding'].append(msi_embeddings)
                self.Gallery[subject[0]]['image_list'].append(list(msi_path))

    def find_matches(self):
        paired_list = []
        with torch.no_grad():
            tqdm_object = tqdm(self.s_valid_loader, total=len(self.s_valid_loader))
            for i, data in enumerate(tqdm_object, 0):
                rgb_img, _, subject, rgb_path = data

                rgb_img = rgb_img.float().to(CFG.device)
                rgb_features, _ = self.model.text_encoder(rgb_img)
                rgb_embeddings = self.model.text_projection(rgb_features)
                msi_embeddings = torch.cat(self.Gallery[subject[0]]['embedding'])
                msi_list = self.Gallery[subject[0]]['image_list']

                msi_embeddings_n = F.normalize(msi_embeddings, p=2, dim=-1)
                rgb_embeddings_n = F.normalize(rgb_embeddings, p=2, dim=-1)

                dot_similarity = rgb_embeddings_n @ msi_embeddings_n.T
                _, indices = torch.topk(dot_similarity.squeeze(0), 1)

                matches = [msi_list[idx] for idx in indices[::1]]

                paired_list.append([matches[0][0], list(rgb_path)[0]])

        with open(self.CFG.save_json_path, 'w') as file:
            json.dump(paired_list, file)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='S2S_EU', help='S2S_EU, S2S_CN, M2S_GL')
    args = parser.parse_args()

    CFG = config(args.dataset)

    matcher = Matcher(CFG=CFG)

    matcher.get_image_embeddings()
    matcher.find_matches()