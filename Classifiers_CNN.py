import torch.nn as nn
from torchinfo import summary

class CNN_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_1"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    # def init_weights(self, model: nn.Module):
    #     module = model.children()
    #     for layer in module.children():
    #         for i in range(len())
    #     pass
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN1_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_1"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    # def init_weights(self, model: nn.Module):
    #     module = model.children()
    #     for layer in module.children():
    #         for i in range(len())
    #     pass
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN1SO_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_1SO"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.SELU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.SELU(),

            nn.Linear(in_features=128, out_features=64),
            nn.SELU(),

            nn.Linear(in_features=64, out_features=32),
            nn.SELU(),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN2_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_2"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),



            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN2SO_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_2SO"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.SELU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.SELU(),



            nn.Linear(in_features=128, out_features=32),
            nn.SELU(),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN3_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_3"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=128, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN3SO_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_3SO"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.SELU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.SELU(),

            nn.Linear(in_features=128, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN4_Classifier(nn.Module):
    def __init__(self, input_shape, name = "CNN_4"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN4T_Classifier(nn.Module):
    """CNN with Tanh in Last Conv Block"""
    def __init__(self, input_shape, name = "CNN_4T"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Tanh(),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN4TC1_Classifier(nn.Module):
    """CNN4 with Tanh in first Conv Block"""
    def __init__(self, input_shape, name = "CNN_4TC1"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.Tanh(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN4TC2_Classifier(nn.Module):
    """CNN4 with Tanh in first 2 Conv Blocks"""
    def __init__(self, input_shape, name = "CNN_4TC2"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.Tanh(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Tanh(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class CNN4SO_Classifier(nn.Module):
    """CNN4 with all SeLU activations"""
    def __init__(self, input_shape, name = "CNN_4SO"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.SELU(inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])

class FCNN_Classifier(nn.Module):
    def __init__(self, input_shape, name = "FCNN"):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=1),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=1),
            
            nn.Linear(in_features=32, out_features=3)
        )
        self.name = name

    def forward(self, input):
        return self.model(input)
    
    # def init_weights(self, model: nn.Module):
    #     module = model.children()
    #     for layer in module.children():
    #         for i in range(len())
    #     pass
    
    def getSummary(self):
        return summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])
