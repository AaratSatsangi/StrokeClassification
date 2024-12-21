from torch import nn
from torchinfo import summary
from torchvision import models

class VIT_B16(nn.Module):
    def __init__(self, input_size = (1, 1, 224, 224), num_classes = 3, freezeToLayer:str = "encoder_layer_7.ln_1"):
        super(VIT_B16, self).__init__()
        self.vitb16 = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.last_freezed_layer = ""

        if freezeToLayer is not None:
            for name, param in self.vitb16.named_parameters():
                if(freezeToLayer in name):
                    self.last_freezed_layer = name
                    break
                param.requires_grad = False

        # Modify input layer for single channel images
        self.vitb16.conv_proj = nn.Conv2d(input_size[1], self.vitb16.conv_proj.out_channels, kernel_size=16, stride=16)

        # Modify the final fully connected layer to match the number of output classes
        in_features = self.vitb16.heads[0].in_features
        self.vitb16.heads = nn.Sequential(nn.Linear(in_features, 3))


    def forward(self, x):
        return self.vitb16(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    
class SWIN_T(nn.Module):
    def __init__(self, input_size=(1, 1, 224, 224), num_classes=3, freezeToLayer: str = "features.3"):
        super(SWIN_T, self).__init__()
        
        # Load the Swin Transformer model
        self.swint = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.last_freezed_layer = ""

        # Optionally freeze layers up to a certain layer
        if freezeToLayer is not None:
            for name, param in self.swint.named_parameters():
                if freezeToLayer in name:
                    self.last_freezed_layer = name
                    break
                param.requires_grad = False

        # Modify the input layer for single-channel images if needed
        self.swint.features[0][0] = nn.Conv2d(input_size[1], self.swint.features[0][0].out_channels, 
                                              kernel_size=4, stride=4, padding=0)

        # Modify the final fully connected layer to match the number of output classes
        in_features = self.swint.head.in_features
        self.swint.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.swint(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer