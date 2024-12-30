import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Logger import MyLogger
from Preprocessor import CTPreprocessor

def plot_losses(fold, training_losses, validation_losses, save_path: str, logger:MyLogger):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Train Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, validation_losses, label='Val Loss', marker='x', linestyle='--', color='orange')

    # start = len(ft_training_losses) - 1 - ft_training_losses[::-1].index(-1)
    # epochs = range(start+1+1, len(ft_training_losses) + 1)
    # plt.plot(epochs, ft_training_losses[start+1:], label='FT Train Loss', marker='o', linestyle='-', color='green')
    # plt.plot(epochs, ft_validation_losses[start+1:], label='FT Val Loss', marker='x', linestyle='--', color='red')


    # Add titles and labels
    plt.title(f'Fold: {fold+1}\nTraining and Validation Losses Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    
    # Set y-axis limits
    plt.ylim(0.0001, max(max(training_losses), max(validation_losses)) * 1.2)  # Slightly higher than max loss

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


def saveAsTable(path_save: str):
    with open(path_save, 'r') as f:
        data = json.load(f)
    path_save = path_save[:-5] + ".png"
    # Convert JSON data into a DataFrame
    df: pd.DataFrame
    df = pd.DataFrame(data).T  # Transpose to get categories as rows
    df = df.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)
    df = df.iloc[:3,:3]

    # Set up a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.8))  # Adjust figure height based on rows

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Render the DataFrame as a table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust table scale
    plt.savefig(path_save, dpi=300, bbox_inches='tight')

def _binarizeUsingMax(t:torch.tensor):
        max_values, _ = t.max(dim=1, keepdim=True)
        return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

def _calcPerformMetrics(y_pred, y_true, class_names, path_save):
    y_pred = _binarizeUsingMax(y_pred)
    y_true = _binarizeUsingMax(y_true)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(path_save, 'w') as f:
        json.dump(report, f, indent=4)
    saveAsTable(path_save)
    return report


def test_model(t_model: torch.nn.Module, test_loader:ImageFolder,test_class_weights, device, path_save, class_names, logger:MyLogger):
    y_trueTensor = torch.empty(0,3)
    y_predTensor = torch.empty(0,3)
    CRITERION_TEST = torch.nn.CrossEntropyLoss(weight=test_class_weights.to(device))
    with torch.no_grad():
        test_loss = 0.0
        for test_XY in test_loader:
            x = test_XY[0].to(device)
            y = test_XY[1].to(device)

            y_pred =  t_model(x)
            test_loss += CRITERION_TEST(y_pred, y).item()

            y_true = torch.zeros(y.shape[0],3)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    report = _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor, class_names=class_names, path_save=path_save)
    logger.log(f"\tFinal Test Loss:{round(test_loss,5)}")
    return report


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
        self.SERVER_URL = train_vars["SERVER_URL"] if  train_vars["SERVER_URL"] != "" else None
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
        self.TEST_DATA = None
        self.CLASS_NAMES = self.TRAIN_DATA.classes

        self.PATH_MODEL_FOLDER = f"Classifiers/{model_type_dict[self.MODEL_TYPE]}/{self.MODEL_NAME}/"
        self.PATH_MODEL_LOG_FOLDER = f"{self.PATH_MODEL_FOLDER}Logs/"
        self.EXPERIMENT_NUMBER = str(sum(1 for file_name in os.listdir(self.PATH_MODEL_FOLDER) if "architecture" in file_name))
        self.PATH_MODEL_LOG_FILE = f"{self.PATH_MODEL_LOG_FOLDER}/architecture_{self.EXPERIMENT_NUMBER}.txt"
        self.PATH_LOSSPLOT_FOLDER = f"{self.PATH_MODEL_FOLDER}Plots/"
        self.PATH_PERFORMANCE_FOLDER = f"{self.PATH_MODEL_FOLDER}Performance/"
        self.PATH_MODEL_SAVE = None 
        self.PATH_LOSSES_SAVE = None
        self.PATH_LOSSPLOT_SAVE = None
        self.PATH_PERFORMANCE_SAVE = None
        if(not os.path.exists(self.PATH_MODEL_LOG_FOLDER)): os.makedirs(self.PATH_MODEL_LOG_FOLDER)
        if(not os.path.exists(self.PATH_LOSSPLOT_FOLDER)): os.makedirs(self.PATH_LOSSPLOT_FOLDER)
        if(not os.path.exists(self.PATH_PERFORMANCE_FOLDER)): os.makedirs(self.PATH_PERFORMANCE_FOLDER)
        
        # Model Specific Variables
        print("Initializing Model Specific Variables...")    
        try:
            with open(self.PATH_MODEL_FOLDER + "init.json", "r") as file:
                model_vars:dict = json.load(file)
        except Exception as e:
            print(f"Error decoding JSON from the file: {self.PATH_MODEL_FOLDER + 'init.json'} ==> {e}")
            exit(1)

        # =========== MODEL SPECIFIC VARIABLES =================
        self.DEVICE = train_vars["DEVICE"]
        self.BATCH_SIZE = model_vars["BATCH_SIZE"]
        self.BATCH_LOAD = model_vars["BATCH_LOAD"]
        self.LEARNING_RATE = model_vars["LEARNING_RATE"]
        self.FREEZE_TO_LAYER = model_vars["FREEZE_TO_LAYER"]

    def updateFold(self, fold:int):
        self.CURRENT_FOLD = fold
        self.PATH_MODEL_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Checkpoint.pth"
        self.PATH_LOSSES_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Losses.txt"
        self.PATH_LOSSPLOT_SAVE = f"{self.PATH_LOSSPLOT_FOLDER}F{self.CURRENT_FOLD+1}_lossplot.png"
        self.PATH_PERFORMANCE_SAVE = f"{self.PATH_PERFORMANCE_FOLDER}F{self.CURRENT_FOLD+1}_performance.json"


if __name__ == "__main__":
    consts = Config()
    print(json.dumps(str(vars(consts)), indent=4))
