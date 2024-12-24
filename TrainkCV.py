import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pandas as pd
from Classifiers import ConvNets, TransNets
from Log import Logger
from torchvision import models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import os
import shutil
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import json
import matplotlib.pyplot as plt
from Preprocessor import CTPreprocessor
from torchinfo import summary
import sys
import requests

# ========= HYPER-PARAMETERS ============
K_FOLD = 5
AUTO_BREAK = True # Auto Stop training if overfitting is detected
BATCH_SIZE = 128
BATCH_LOAD = 128 # Batch load must be less than batch size
LEARNING_RATE = 1e-3
PERSISTANCE = 15
WORKERS = os.cpu_count()
EPOCHS = 200 
IMG_SIZE = (1, 224, 224)
LRS_PATIENCE = 5 # If LR Schedular requires patience parameter {ReduceLRonPlataue}
LRS_FACTOR = 0.5 # IF LR Schedular required Decrease by Factor {ReduceLRonPlataue}
# =======================================

# ======== DO NOT TOUCH PARAMETERS ==========
SERVER_USERNAME = "AaratSatsangi"
SERVER_FOLDER = "StrokeClassification"
SERVER_URL = "https://www.aaratsatsangi.in/logger.php"
PATH_DATASET_TRAIN = "Data/Compiled/Split/Train"
PATH_DATASET_TEST = "Data/Compiled/Split/Test"
MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
IMG_TRANSFORMS_TRAIN = CTPreprocessor(
    img_size=IMG_SIZE[1:],
    transformations=[
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ]
)
IMG_TRANSFORMS_TEST = CTPreprocessor(
    img_size=IMG_SIZE[1:],
    transformations=[
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
    ]
)
LOSS = torch.nn.CrossEntropyLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA = ImageFolder(PATH_DATASET_TRAIN, IMG_TRANSFORMS_TRAIN)
TEST_DATA = ImageFolder(PATH_DATASET_TEST, IMG_TRANSFORMS_TEST)
CLASS_NAMES = TRAIN_DATA.classes
GENERATOR = torch.Generator().manual_seed(26)
# ===========================================

# ======= SOME CONSTANTS NEEDED =============
MODEL_NAME: str = None
PATH_MODEL_SAVE: str = None
PATH_MODEL_LOG_FOLDER: str = None
PATH_MODEL_LOG: str = None
OPTIMIZER = None
LR_SCHEDULER = None
KF: KFold = None
SAMPLER: WeightedRandomSampler = None
LOGGER: Logger = None
model: torch.nn.Module = None
# =========================================== 

def setup(model:nn.Module, fine_tine = False, openTillLayer:str = None):
    global PATH_MODEL_LOG, BATCH_LOAD, BATCH_SIZE
    if BATCH_LOAD > BATCH_SIZE: 
        print("Setting Batch Load = Batch Size")
        print(f"ERROR: Batch Load {BATCH_LOAD} is greater than {BATCH_SIZE}")
        BATCH_LOAD = BATCH_SIZE
        
    torch.cuda.empty_cache()
    last_freezed_layer = ""
    if fine_tine:
        req_grad = True
        flag = True
        
        for name, param in reversed(list(model.named_parameters())):
            param.requires_grad = req_grad
            if (flag and openTillLayer != "" and openTillLayer in name):
                req_grad = False
                flag = False
                last_freezed_layer = name
    else:
        last_freezed_layer = model.get_last_freezed_layer()
    # Create the folder
    if not os.path.exists(PATH_MODEL_SAVE):
        os.makedirs(PATH_MODEL_SAVE)
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("architecture" in file_name): 
            count +=1
    
    # Writing Architecture
    with open(PATH_MODEL_SAVE + "architecture_"+ str(count) + ".txt", "w") as f:
        f.write("="*25 + "Layer Names" + "="*25 + "\n")
        for i, (name, param) in enumerate(model.named_parameters()):
            if last_freezed_layer in name and last_freezed_layer != "":
                f.write(str(i) + ": " + name + "\t\t(freezed till here)\n")
            else:
                f.write(str(i) + ": " + name + "\n")
        f.write("="*61 + "\n")
        f.write("\n\n")
        f.write(str(summary(model, (1,) + IMG_SIZE , col_names=["input_size","output_size","num_params"], verbose=0)))
        
def _isDecreasingOrder(lst: list):
    for i in range(len(lst) - 1):
        if lst[i] <= lst[i + 1]:
            return False
    return True

def plot_losses(training_losses, validation_losses):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='x', linestyle='--', color='orange')

    # Add titles and labels
    plt.title('Training and Validation Losses Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Set y-axis limits
    plt.ylim(0, max(max(training_losses), max(validation_losses)) * 1.1)  # Slightly higher than max loss

    # Adding a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.legend(fontsize=12)

    # Save the figure
    if(not os.path.exists(PATH_MODEL_SAVE + "Plots/")):
        os.makedirs(PATH_MODEL_SAVE + "Plots/", exist_ok=True)
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE + "Plots/"):
        if("loss_plot" in file_name): 
            count +=1
    plt.savefig(PATH_MODEL_SAVE + "Plots/loss_plot_" + str(count) + ".png", bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

    LOGGER.log(f"Plot saved as {PATH_MODEL_SAVE + 'Plots/loss_plot.png'}")

def get_sample_weights(dataset, indices, name):
    targets = torch.tensor([dataset.targets[i] for i in indices])
    class_counts = torch.bincount(targets)
    LOGGER.log("\t" + name + f" Class Counts: {class_counts}")
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    return sample_weights

def _get_min_val_loss():
    loss_files = []
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if "LOSSES" in file_name:
            loss_files.append(file_name)

    min_val_loss = float('inf')
    for loss_file in loss_files:
        try:
            val_losses = np.loadtxt(PATH_MODEL_SAVE + loss_file, delimiter=",")[1]
            min_val_loss = min(min_val_loss, np.min(val_losses))
        except Exception as e:
            print(f"Unable to process file: {loss_file}")
            print(f"Error: {e}")
    
    return min_val_loss

def train_KCV():
    global LEARNING_RATE
    lr = LEARNING_RATE
    LOGGER.log("\n" + "#"*115 + "\n")
    LOGGER.log("\t\t\t\t\tTraining: " + MODEL_NAME)
    LOGGER.log("\n" + "#"*115)

    training_losses = []
    validation_losses = []
    p_counter = 1
    
    if FINE_TUNE:
        min_val_loss = _get_min_val_loss()
        if isinstance(LR_SCHEDULER, ReduceLROnPlateau): LR_SCHEDULER.step(min_val_loss)
        LOGGER.log(f"\tLoaded Last Min Val Loss: {min_val_loss}")
    else:
        min_val_loss = float('inf')

    try: 
        # Loop Here
        for fold, (train_idx, val_idx) in enumerate(KF.split(TRAIN_DATA)):
            LOGGER.log("\t" + "="*100)
            LOGGER.log(f"\tFold {fold+1}/{K_FOLD}")
            LOGGER.log("\t" + "="*100)

            _train = Subset(TRAIN_DATA, train_idx)
            _val = Subset(TRAIN_DATA, val_idx)

            sample_weights_train = get_sample_weights(TRAIN_DATA, train_idx, "Train")
            sample_weights_val = get_sample_weights(TRAIN_DATA, val_idx, "Val")

            SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=GENERATOR)
            SAMPLER_VAL = WeightedRandomSampler(weights=sample_weights_val , num_samples=len(sample_weights_val), replacement=True, generator=GENERATOR)

            train_loader = DataLoader(dataset = _train, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True, sampler=SAMPLER_TRAIN, generator=GENERATOR)
            val_loader = DataLoader(dataset = _val, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True, sampler=SAMPLER_VAL, generator=GENERATOR)
            for epoch in range(EPOCHS):
                LOGGER.log("\t" + "-"*100)
                LOGGER.log(("\t" + "FOLD: [%d/%d]") % (fold+1, K_FOLD))
                LOGGER.log(("\t" + "EPOCH: [%d/%d]" + "\t"*8 + "PERSISTANCE: [%d/%d]") % (epoch+1, EPOCHS, p_counter, PERSISTANCE))
                LOGGER.log("\t" +"-"*100)

                train_loss = 0.0
                accum_loss = 0.0
                count = 0
                # Training
                model.train()
                for step, train_XY in enumerate(train_loader, 0):
                    
                    # Extract X and Y
                    imgs = train_XY[0].to(DEVICE)
                    labels = train_XY[1].to(DEVICE)
                    
                    # Predict labels 
                    y_pred = model(imgs)

                    # Calculate Error
                    error = LOSS(y_pred, labels)
                    error.backward()
                    accum_loss += error.item()
                    
                    print("\t" +"\tSTEP: [%d/%d]" % (step+1,len(train_loader)), end= "\r")
                    if(count*BATCH_LOAD >= BATCH_SIZE):    
                        OPTIMIZER.step()
                        OPTIMIZER.zero_grad()
                        train_loss += accum_loss
                        print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),accum_loss/(count)), end = "\r") # Print avg batch loss instead of total accum loss
                        accum_loss = 0.0
                        count = 0
                    count += 1
                # avg epoch loss
                train_loss /= len(train_loader)
                training_losses.append(train_loss)
                LOGGER.log("\n\n\t" +"\tTraining Loss: [%0.5f]" % (training_losses[-1]))

                # Validation
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_XY in val_loader:
                        imgs = val_XY[0].to(DEVICE)
                        labels = val_XY[1].to(DEVICE)

                        y_pred = model(imgs)
                        val_loss += LOSS(y_pred, labels).item()
                val_loss /= len(val_loader)
                validation_losses.append(val_loss)
                LOGGER.log("\t" +"\tValidation Loss: [%0.5f]" % (val_loss))

                # Save Best Model
                p_counter += 1
                if(val_loss < min_val_loss):
                    min_val_loss = val_loss
                    torch.save(model, PATH_MODEL_SAVE + MODEL_NAME  + ".pt")
                    p_counter = 1
                LOGGER.log("\t" +"\tMin Validation Loss: [%0.5f]" % (min_val_loss))

                # Learning Rate Schedular Step
                if(isinstance(LR_SCHEDULER, CosineAnnealingWarmRestarts)): 
                    LR_SCHEDULER.step()
                    if(lr > LR_SCHEDULER.get_last_lr()[-1]):
                        LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{LR_SCHEDULER.get_last_lr()[-1]: 0.2e}]")
                        lr = LR_SCHEDULER.get_last_lr()[-1]
                    elif(lr < LR_SCHEDULER.get_last_lr()[-1]):
                        LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{LR_SCHEDULER.get_last_lr()[-1]: 0.2e}]")
                        lr = LR_SCHEDULER.get_last_lr()[-1]
                elif(isinstance(LR_SCHEDULER, ReduceLROnPlateau)): 
                    LR_SCHEDULER.step(val_loss)
                    if(lr > LR_SCHEDULER.get_last_lr()[-1]):
                        LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{LR_SCHEDULER.get_last_lr()[-1]: 0.2e}]")
                        lr = LR_SCHEDULER.get_last_lr()[-1]
                    elif(lr < LR_SCHEDULER.get_last_lr()[-1]):
                        LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{LR_SCHEDULER.get_last_lr()[-1]: 0.2e}]")
                        lr = LR_SCHEDULER.get_last_lr()[-1]
                else: raise Exception("LR Schedular not recognized!\nType: " + type(LR_SCHEDULER))
                
                
                # Early Stopping for Overfitting Stopping
                if(p_counter-1 >= PERSISTANCE):
                    LOGGER.log("\t" + "\tValidation Loss Constant for %d Epochs at EPOCH %d" % (PERSISTANCE, epoch+1))
                    if(_isDecreasingOrder(training_losses[-PERSISTANCE:])):
                        LOGGER.log("\t" + f"\tStopping Training: Overfitting Detected at EPOCH {epoch+1}")
                        # Break out of Training Loop
                        if(AUTO_BREAK): 
                            p_counter = 1
                            break
                    else:
                        LOGGER.log("\t" + "\tTraining Loss Fluctuating")
                        # Unsure about Overfitting, ask the user to continue
                    while(True):
                        if(AUTO_BREAK):
                            p_counter = 1
                            flag = "n"
                            break
                        flag = input("\t" + "Keep Training? (y/n) : ")
                        if(flag == "y" or flag == "n"):
                            p_counter = 1
                            break
                        else:
                            LOGGER.log("\t" + "Wrong Input!!\n")
                    if(flag == "n"):
                        break
                LOGGER.log("") # Add New Line

    except KeyboardInterrupt:
        # Exit Loop code
        LOGGER.log("\t" + "Keyboard Interrupt: Exiting Loop...")

    LOGGER.log("\n" + "#"*100 + "\n" + "#"*100 + "\n")
    if(len(training_losses) and len(validation_losses)):
        count = 0
        for file_name in os.listdir(PATH_MODEL_SAVE):
            if("LOSSES" in file_name): 
                count +=1
        path_loss_save = PATH_MODEL_SAVE + "LOSSES_" + str(count) + ".txt"
        np.savetxt(path_loss_save, (training_losses, validation_losses), fmt="%0.5f" , delimiter=",")
        plot_losses(training_losses, validation_losses)

def _binarizeUsingMax(t:torch.tensor):
        max_values, _ = t.max(dim=1, keepdim=True)
        return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

def _calcPerformMetrics(y_pred, y_true, class_names, path_saveDict):
    y_pred = _binarizeUsingMax(y_pred)
    y_true = _binarizeUsingMax(y_true)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(path_saveDict, 'w') as f:
        json.dump(report, f, indent=4)
        LOGGER.log("Results Written in: " + str(path_saveDict))

    saveAsTable(path_saveDict)
    return

def saveAsTable(json_file_path: str):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

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

    # Save the table as an image
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("performance" in file_name and ".png" in file_name): 
            count +=1
    output_path = PATH_MODEL_SAVE + "performance_" + str(count) + ".png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    LOGGER.log(f"Table image saved to {output_path}")

def testModel(t_model: nn.Module):
    y_trueTensor = torch.empty(0,3)
    y_predTensor = torch.empty(0,3)
    with torch.no_grad():
        test_loss = 0.0
        for test_XY in test_loader:
            x = test_XY[0].to(DEVICE)
            y = test_XY[1].to(DEVICE)

            y_pred =  t_model(x)
            error = LOSS(y_pred, y)
            test_loss += error.item()

            y_true = torch.zeros(y.shape[0],3)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("performance" in file_name and ".json" in file_name): 
            count +=1
    path_perfom_save = PATH_MODEL_SAVE + "performance_" + str(count) + ".json"
    _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=CLASS_NAMES, path_saveDict=path_perfom_save)
    LOGGER.log(f"Test Loss:{round(test_loss,5)}")
    return


if __name__ == "__main__":
    
    # ====================================================================
    # ==================== CHANGE HERE ===================================
    FINE_TUNE = True
    MODEL_FOLDER_NAME = "SWIN_S"
    MODEL_TYPE = MODEL_TYPE_DICT["trans"]
    OPEN_TILL_LAYER = ""
    # ====================================================================
    MODEL_NAME = MODEL_FOLDER_NAME if(not FINE_TUNE) else MODEL_FOLDER_NAME + "_FT"
    PATH_MODEL_SAVE = "./Classifiers/" + MODEL_TYPE + "/" + MODEL_FOLDER_NAME + "/"
    PATH_MODEL_LOG = PATH_MODEL_SAVE + "Logs/" 
    if(not os.path.exists(PATH_MODEL_LOG)): os.makedirs(PATH_MODEL_LOG)
    PATH_MODEL_LOG += MODEL_NAME + "_architecture_" + str(sum(1 for file_name in os.listdir(PATH_MODEL_SAVE) if "architecture" in file_name)) + ".txt"
    LOGGER = Logger(
        server_url = SERVER_URL,
        server_username= SERVER_USERNAME,
        server_folder = SERVER_FOLDER,
        model_name = MODEL_NAME,
        path_localFile = PATH_MODEL_LOG
    )
    # ====================================================================
    
    LOGGER.log("\n\n" + "="*54 + " START " + "="*54)
    KF = KFold(n_splits=K_FOLD, shuffle=True, random_state=26)
    if(FINE_TUNE):
        if(not os.path.exists(PATH_MODEL_SAVE + MODEL_NAME)):
            LOGGER.log("Loading Model: " + MODEL_NAME[:-3] + ".pt")
            model = torch.load(PATH_MODEL_SAVE + MODEL_NAME[:-3] + ".pt")
        else:
            src = PATH_MODEL_SAVE + MODEL_NAME
            dst = PATH_MODEL_SAVE + MODEL_NAME + "_previous.pt"
            LOGGER.log("Loading Model: " + MODEL_NAME)
            model = torch.load(src)
            shutil.copy(src, dst)
    else:
        # =================================================================
        # ================= THE NEW MODEL TO TRAIN ========================
        
        model = TransNets.SWIN(swin="s", input_size=(BATCH_SIZE,) + IMG_SIZE)
        
        # =================================================================
        # =================================================================

    
    model.to(DEVICE)
    # OPTIMIZER = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,  weight_decay=1e-4)
    OPTIMIZER = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005, dampening=0, momentum=0.9, nesterov=True)
    # LR_SCHEDULER = CosineAnnealingWarmRestarts(OPTIMIZER, T_0=10, T_mult=2)
    LR_SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, mode='min',  factor=LRS_FACTOR, patience=LRS_PATIENCE)
    
    LOGGER.log(f"Batch Size: {BATCH_SIZE}")
    LOGGER.log(f"Learning Rate: {LEARNING_RATE}")
    LOGGER.log(f"Early Stopping with Persistence: {PERSISTANCE}")
    LOGGER.log(f"LR Schedular: {type(LR_SCHEDULER).__name__}")
    # Add if statement
    if isinstance(LR_SCHEDULER, ReduceLROnPlateau):
        LOGGER.log(f"Patience: {LRS_PATIENCE}")
        LOGGER.log(f"Factor: {LRS_FACTOR}")

    setup(model, FINE_TUNE, OPEN_TILL_LAYER)
    # exit()
    train_KCV()
    
    LOGGER.log("Testing model: " + PATH_MODEL_SAVE + MODEL_NAME + ".pt")
    test_loader = DataLoader(dataset = TEST_DATA, batch_size = BATCH_LOAD, num_workers=WORKERS)
    testModel(torch.load(PATH_MODEL_SAVE + MODEL_NAME + ".pt"))
    LOGGER.log("="*54 + " END " + "="*54 + "\n\n")
