import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def val_mlp(model, val_loader, mode, label_type="Single"):
    model.eval() # validation
    valloader = tqdm(val_loader)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            if mode == 'Teacher':
                msi_img, label, subject, _ = data
                msi_img = msi_img.float().cuda()
            else:
                _, rgb_img, label, subject, _, _ = data
                rgb_img = rgb_img.float().cuda()

            if (label.size(0) == 1):
                labels_val = label[0].cuda()
            else:
                labels_val = torch.squeeze(label).cuda()

            if mode == 'Teacher':
                rgb_embedding, rgb_logits = model(msi_img)
            else:
                rgb_embedding, rgb_logits = model(rgb_img)

            labels_val = labels_val.clone().detach().cpu()

            if label_type == "Single":
                if labels_val.size(0) == 1:
                    labels_val = torch.unsqueeze(labels_val, dim=0)
                    y_true.append(int(labels_val.numpy()[0]))
                else:
                    y_true.extend(list(map(int, labels_val.cpu().tolist())))
            else:
                y_true.extend(labels_val.cpu().tolist())

            if mode == 'Teacher' and label_type == "Multi":
                prediction = torch.where(rgb_logits > 0.5, 1, 0)
            else:
                prediction = torch.max(rgb_logits, 1)[1]

            y_pred.extend(prediction.cpu().tolist())

        # print(classification_report(y_true, y_pred))

        OA = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred, average='macro')

        recall = recall_score(y_true, y_pred, average='macro')

        F1_score = f1_score(y_true, y_pred, average='macro')


        return OA, precision, recall, F1_score

