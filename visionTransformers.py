import torch.nn as nn
from torchinfo import summary
from torchvision import models

class ExistingModels:
    class ViT_3(nn.Module):
        model: models.VisionTransformer

        def __init__(self, freeze_upto=6):
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.model.conv_proj = nn.Conv2d(1, self.model.conv_proj.out_channels, kernel_size=16, stride=16)
            self.model.heads = nn.Sequential(
                nn.Linear(self.model.heads[0].in_features, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 3)
            )
            # Freeze Layers
        
        def forward(self, x):
            return self.model(x)
        
        def init_weights(self):
            pass

        def unFreeze():
            pass




