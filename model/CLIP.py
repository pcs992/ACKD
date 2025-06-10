import torch
from torch import nn
import torch.nn.functional as F
from .CLIP_modules import ImageEncoder, TextEncoder, ProjectionHead

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        text_embedding,
        projection_dim,
        dropout,
        train_num_class,
        msi_channel,
        rgb_channel
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(train_num_class, msi_channel)
        self.text_encoder = TextEncoder(train_num_class, rgb_channel)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding,
                                               projection_dim=projection_dim,
                                               dropout=dropout
        )
        self.text_projection = ProjectionHead(embedding_dim=text_embedding,
                                              projection_dim=projection_dim,
                                              dropout=dropout
                                              )
        self.temperature = temperature

    def forward(self, msi_img, rgb_img):
        # Getting Image and Text Features
        image_features, _ = self.image_encoder(msi_img)
        text_features, _ = self.text_encoder(rgb_img)
        # Getting Image and Text Embeddings (with same dimension)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")