import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import torch.nn as nn

def get_image_embeddings(dataset_path, model, category_list):
    valid_loader = DataLoader(dataset=dataset_path, batch_size=1, shuffle=False, num_workers=8)
    model.eval()

    Gallery = {
        category: {
            'embedding': [],
            'image_list': []
        } for category in category_list
    }
    with torch.no_grad():
        # tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        for i, data in enumerate(valid_loader, 0):
            msi_img, _, subject, msi_path = data
            msi_img = msi_img.float().cuda()
            msi_features, msi_logits = model(msi_img)
            Gallery[subject[0]]['embedding'].append(msi_logits)
            Gallery[subject[0]]['image_list'].append(list(msi_path))

    return Gallery

def find_matches(model, valid_dataset, Gallery, save_file):
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    paired_list = []
    model.eval()

    criterions = nn.KLDivLoss(reduction='none')

    with torch.no_grad():
        # tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        for i, data in enumerate(valid_loader, 0):
            _, rgb_img, _, subject, rgb_path, _ = data
            rgb_img = rgb_img.float().cuda()
            rgb_features, rgb_logits = model(rgb_img)
            msi_embeddings = torch.cat(Gallery[subject[0]]['embedding'])
            msi_list = Gallery[subject[0]]['image_list']
            num_msi = msi_embeddings.size(0)

            rgb_logits_expand = rgb_logits.expand(num_msi, -1)

            ce_score = criterions(
                F.log_softmax(rgb_logits_expand / 3, dim=1),
                F.softmax(msi_embeddings / 3, dim=1))

            score_list = torch.sum(ce_score, dim=1)

            value, indices = torch.topk(score_list, 1, largest=False)
            value_dis, indices_dis = torch.topk(score_list, 1, largest=True) # new
            matches = [msi_list[idx] for idx in indices[::1]]
            matches_dis = [msi_list[idx] for idx in indices_dis[::1]] # new

            # paired_list.append([matches[0][0], list(rgb_path)[0]])
            paired_list.append([matches[0][0], matches_dis[0][0], list(rgb_path)[0]])

    with open(save_file, 'w') as file:
        json.dump(paired_list, file)

def adjusted(student, Train_Student_Dataloader, gallery, save_file):
    # msi_image_dataset = Paired_ImageDataset(data_dir=teacher_root, input_type="MSI")
    # gallery = get_image_embeddings(msi_image_dataset, teacher)
    find_matches(student, Train_Student_Dataloader, gallery, save_file)