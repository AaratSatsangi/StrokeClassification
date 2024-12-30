import sys
from torch import nn
from torchinfo import summary
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, model_size:str, input_size=(1, 1, 224, 224), num_classes=3, freezeToLayer:str = "layer3.12", pretrained=True): # model size can be [t,s,b]
        super(ResNet, self).__init__()
        try:
        # Load the ResNet101 model
            if model_size == "t":
                self.model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == "s":
                self.model = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == "b":
                self.model = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            else:
                raise ValueError(f"Model Size should be from [t, s, b] i.e. tiny (ResNet50), small (ResNet101), base (ResNet152), but {model_size} was passed")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        self.last_freezed_layer = ""
        
        # Optionally freeze layers up to a certain layer
        if freezeToLayer is not None:
            for name, param in self.model.named_parameters():
                if(freezeToLayer in name):
                    self.last_freezed_layer = name
                    break
                param.requires_grad = False
        
        # Modify the input layer for single-channel images if needed
        self.model.conv1 = nn.Conv2d(input_size[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1.requires_grad_(True)
        
        # Modify the final fully connected layer to match the number of output classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    
class VGG19_BN(nn.Module):
    def __init__(self, input_size=(1, 1, 224, 224), num_classes=3, freezeToLayer="features.33"):
        super(VGG19_BN, self).__init__()
        
        # Load the pre-trained VGG19 with batch normalization
        self.vgg19bn = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        self.last_freezed_layer = ""
        
        # Optionally freeze layers up to a certain layer
        if freezeToLayer is not None:
            for name, param in self.vgg19bn.named_parameters():
                if freezeToLayer in name:
                    self.last_freezed_layer = name
                    break  # Stop after freezing up to the specified layer
                # Print each layer's name for debugging purposes
                param.requires_grad = False
                

        self.vgg19bn.features[0] = nn.Conv2d(input_size[1], 64, kernel_size=3, stride=1, padding=1)
        self.vgg19bn.features[1].requires_grad = True
        # Replace the fully connected layer (classifier) to match your number of classes
        num_ftrs = self.vgg19bn.classifier[6].in_features  # Get input features of the last classifier layer
        self.vgg19bn.classifier[6] = nn.Linear(num_ftrs, num_classes)  # Replace final layer

    def forward(self, x):
        return self.vgg19bn(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    

# Example usage
if __name__ == "__main__":
    model = ResNet(model_size="s", pretrained=False)
    summary(model, input_size=(1, 1, 224, 224), depth=3, col_names=["input_size","output_size","num_params"])
    for name, param in model.model.named_parameters():
        print(f"NAME: {name}")
    # Test with a dummy input
    # dummy_input = torch.randn(1, 1, 224, 224)  # Batch size 1, 1 channel, 224x224 image
    # output = model(dummy_input)
    # print(output.shape)  # Should output torch.Size([1, 3])
