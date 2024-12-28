import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Logger import MyLogger
from Preprocessor import CTPreprocessor

def plot_losses(fold, training_losses, validation_losses, save_path: str, logger:MyLogger):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='x', linestyle='--', color='orange')

    # Add titles and labels
    plt.title(f'Fold: {fold+1}\nTraining and Validation Losses Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    
    # Set y-axis limits
    plt.ylim(0, max(max(training_losses), max(validation_losses)) * 1.1)  # Slightly higher than max loss

    # Adding a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.legend(fontsize=12)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    if logger is not None:
        logger.log(f"\tPlot saved as {save_path}")
    else:
        print(f"\tPlot saved as {save_path}")
    # ---------------------------------------------
    # Add a combined loss for all files with log axis
    # ----------------------------------------------

def is_decreasing_order(lst: list):
    for i in range(len(lst) - 1):
        if lst[i] <= lst[i + 1]:
            return False
    return True

def get_min_val_loss(path_model_save: str):
    loss_files = []
    for file_name in os.listdir(path_model_save):
        if "LOSSES" in file_name:
            loss_files.append(file_name)

    min_val_loss = float('inf')
    for loss_file in loss_files:
        try:
            val_losses = np.loadtxt(path_model_save + loss_file, delimiter=",")[1]
            min_val_loss = min(min_val_loss, np.min(val_losses))
        except Exception as e:
            print(f"Unable to process file: {loss_file}")
            print(f"Error: {e}")
    
    return min_val_loss

def get_sample_weights(dataset, indices, name, logger: MyLogger):
    if indices is not None:
        targets = torch.tensor([dataset.targets[i] for i in indices])
        log_string = "\t" + name 
    else:
        targets = torch.tensor(dataset.targets)
        log_string = name
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    logger.log(log_string + f" Class Counts: {class_counts}, weights: {class_weights}")
    return class_weights, sample_weights

def logTime(start_time, end_time, logger:MyLogger):
    
    elapsed_time = end_time - start_time
    # Convert to HH:MM:SS format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    logger.log("\t"*10 + f"  EPOCH TIME: [{time_formatted}]")

def verify_lengths(l1:list, l2:list):
    # Determine the difference in lengths
    len_diff = abs(len(l1) - len(l2))
    
    # Append 1000 to the smaller list
    if len(l1) < len(l2):
        l1.extend([1000] * len_diff)
    elif len(l2) < len(l1):
        l2.extend([1000] * len_diff)
    
    return l1, l2

class Config:
    def __init__(self, path_init_file="init.json"):
        
        model_type_dict = {
            "conv": "Convolutional Networks",
            "trans": "Transformer Networks",
            "hybrid": "Hybrid Networks"
        }
        print("Initializing Training Variables...")
        try:
            # Training Variables: Mostly Constant
            with open(path_init_file, "r") as file:
                train_vars:dict = json.load(file)

        except Exception as e:
            print(f"Error decoding JSON from the file: {path_init_file} ==> {e}")
            exit(1)
        
        
        # ============= TRAINING VARIABLES ==================
        self.CURRENT_FOLD = 0
        self.MODEL_NAME = train_vars["MODEL_NAME"]
        self.MODEL_TYPE = train_vars["MODEL_TYPE"]
        self.MODEL_SIZE = (self.MODEL_NAME[-1]).lower()
        self.K_FOLD = train_vars["K_FOLD"]
        self.TRAIN_EPOCHS = train_vars["TRAIN_EPOCHS"]
        self.FINE_TUNE_EPOCHS = train_vars["FINE_TUNE_EPOCHS"]
        self.PERSIST = train_vars["PERSISTENCE"]
        self.IMG_SIZE = tuple(train_vars["IMG_SIZE"])
        self.AUTO_BREAK = train_vars["AUTO_BREAK"]
        self.LRS_PATIENCE = train_vars["LRS_PATIENCE"]
        self.LRS_FACTOR = train_vars["LRS_FACTOR"]

        self.SERVER_USERNAME = train_vars["SERVER_USERNAME"]
        self.SERVER_FOLDER = train_vars["SERVER_FOLDER"]
        self.SERVER_URL = train_vars["SERVER_URL"]
        self.PATH_DATASET_TRAIN = train_vars["PATH_DATASET_TRAIN"]
        self.PATH_DATASET_TEST = train_vars["PATH_DATASET_TEST"]
        
        self.RANDOM_STATE = 26
        self.WORKERS = os.cpu_count()
        self.GENERATOR = torch.Generator().manual_seed(26)
        self.TRANSFORMS_TRAIN = CTPreprocessor(
                img_size=self.IMG_SIZE[1:],
                transformations=[
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
                    transforms.RandomRotation(90),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip()
                ],
                use_mask=False
        )
        self.TRANSFORMS_TEST = IMG_TRANSFORMS_TEST = CTPreprocessor(
                img_size=self.IMG_SIZE[1:],
                transformations=[
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
                ],
                use_mask=False
        )
        self.CRITERION_TRAIN = torch.nn.CrossEntropyLoss()
        self.CRITERION_VAL:torch.nn.CrossEntropyLoss = None
        self.TRAIN_DATA = ImageFolder(self.PATH_DATASET_TRAIN, self.TRANSFORMS_TRAIN)
        self.CLASS_NAMES = self.TRAIN_DATA.classes

        self.PATH_MODEL_FOLDER = f"Classifiers/{model_type_dict[self.MODEL_TYPE]}/{self.MODEL_NAME}/"
        self.PATH_MODEL_LOG_FOLDER = f"{self.PATH_MODEL_FOLDER}Logs/"
        self.EXPERIMENT_NUMBER = str(sum(1 for file_name in os.listdir(self.PATH_MODEL_FOLDER) if "architecture" in file_name))
        self.PATH_MODEL_LOG_FILE = f"{self.PATH_MODEL_LOG_FOLDER}/architecture_{self.EXPERIMENT_NUMBER}.txt"
        self.PATH_LOSSPLOT_FOLDER = f"{self.PATH_MODEL_FOLDER}Plots/"
        self.PATH_MODEL_SAVE = None 
        self.PATH_LOSSES_SAVE = None
        self.PATH_LOSSPLOT_SAVE = None
        if(not os.path.exists(self.PATH_MODEL_LOG_FOLDER)): os.makedirs(self.PATH_MODEL_LOG_FOLDER)
        if(not os.path.exists(self.PATH_LOSSPLOT_FOLDER)): os.makedirs(self.PATH_LOSSPLOT_FOLDER)
        
        # Model Specific Variables
        print("Initializing Model Specific Variables...")    
        try:
            with open(self.PATH_MODEL_FOLDER + "init.json", "r") as file:
                model_vars:dict = json.load(file)
        except Exception as e:
            print(f"Error decoding JSON from the file: {self.PATH_MODEL_FOLDER + "init.json"} ==> {e}")
            exit(1)

        # =========== MODEL SPECIFIC VARIABLES =================
        self.DEVICE = model_vars["DEVICE"]
        self.BATCH_SIZE = model_vars["BATCH_SIZE"]
        self.BATCH_LOAD = model_vars["BATCH_LOAD"]
        self.LEARNING_RATE = model_vars["LEARNING_RATE"]
        self.FREEZE_TO_LAYER = model_vars["FREEZE_TO_LAYER"]

    def updateFold(self, fold:int):
        self.CURRENT_FOLD = fold
        self.PATH_MODEL_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Checkpoint.pth"
        self.PATH_LOSSES_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Losses.txt"
        self.PATH_LOSSPLOT_SAVE = f"{self.PATH_LOSSPLOT_FOLDER}F{self.CURRENT_FOLD+1}_lossplot.png"


if __name__ == "__main__":
    consts = Config()
    print(json.dumps(str(vars(consts)), indent=4))