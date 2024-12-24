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
        self.vitb16.heads = nn.Sequential(nn.Linear(in_features, num_classes))


    def forward(self, x):
        return self.vitb16(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    

class VIT_L32(nn.Module):
    def __init__(self, input_size = (1, 1, 224, 224), num_classes = 3, freezeToLayer:str = "encoder_layer_7.ln_1"):
        super(VIT_L32, self).__init__()
        self.vit = models.vit_l_32(weights = models.ViT_L_32_Weights.IMAGENET1K_V1)
        self.last_freezed_layer = ""

        if freezeToLayer is not None:
            for name, param in self.vit.named_parameters():
                if(freezeToLayer in name):
                    self.last_freezed_layer = name
                    break
                param.requires_grad = False

        # Modify input layer for single channel images
        self.vit.conv_proj = nn.Conv2d(input_size[1], self.vit.conv_proj.out_channels, kernel_size=32, stride=32)

        # Modify the final fully connected layer to match the number of output classes
        in_features = self.vit.heads[0].in_features
        self.vit.heads = nn.Sequential(nn.Linear(in_features, num_classes))


    def forward(self, x):
        return self.vit(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    
class SWIN(nn.Module):
    def __init__(self, swin:str, input_size=(1, 1, 224, 224), num_classes=3, freezeToLayer: str = "features.5.0"): # swin should be [b,s,t]
        super(SWIN, self).__init__()
        
        # Load the Swin Transformer model
        if swin == 'b':
            self.swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        elif swin == 's':
            self.swin = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        elif swin == 't':
            self.swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.last_freezed_layer = ""

        # Optionally freeze layers up to a certain layer
        if freezeToLayer is not None:
            for name, param in self.swin.named_parameters():
                if freezeToLayer in name:
                    self.last_freezed_layer = name
                    break
                param.requires_grad = False

        # Modify the input layer for single-channel images if needed
        self.swin.features[0][0] = nn.Conv2d(input_size[1], self.swin.features[0][0].out_channels, 
                                              kernel_size=4, stride=4, padding=0)

        # Modify the final fully connected layer to match the number of output classes
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.swin(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    
if __name__ == "__main__":
    from torchinfo import summary
    model = VIT_L32()
    summary(model, input_size=(1, 1, 224, 224), depth=8, col_names=["input_size","output_size","num_params"])
