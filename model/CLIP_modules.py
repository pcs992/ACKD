from torch import nn
from .CLIP_encoder import resnet34

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
            self,
            train_num_class,
            msi_channel
    ):
        super().__init__()
        self.model = resnet34(train_num_class, msi_channel).cuda()

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self,
                 train_num_class,
                 rgb_channel):
        super().__init__()
        self.model = resnet34(train_num_class, rgb_channel).cuda()

    def forward(self, x):
        self.model(x)
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

