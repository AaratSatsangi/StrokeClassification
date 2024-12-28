import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pandas as pd
from Utils.Helpers import *
from Classifiers import ConvNets, TransNets
from Logger import Logger
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
import time

# ========= HYPER-PARAMETERS ============
K_FOLD = 5
AUTO_BREAK = True # Auto Stop training if overfitting is detected
BATCH_SIZE = 128
BATCH_LOAD = 16 # Batch load must be less than batch size
LEARNING_RATE = 1e-3
PERSISTANCE = 15 
WORKERS = os.cpu_count()
EPOCHS = 1 
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
    ],
    use_mask=False
)
IMG_TRANSFORMS_TEST = CTPreprocessor(
    img_size=IMG_SIZE[1:],
    transformations=[
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
    ],
    use_mask=False
)
CRITERION_TRAIN = torch.nn.CrossEntropyLoss()
CRITERION_VAL: torch.nn.CrossEntropyLoss = None
CRITERION_TEST: torch.nn.CrossEntropyLoss = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        min_val_loss = get_min_val_loss(path_model_save=PATH_MODEL_SAVE)
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

            _, sample_weights_train = get_sample_weights(TRAIN_DATA, train_idx, "Train", logger = LOGGER)
            val_class_weights, sample_weights_val = get_sample_weights(TRAIN_DATA, val_idx, "Val", logger = LOGGER)

            
            CRITERION_VAL = nn.CrossEntropyLoss(weight=val_class_weights.to(DEVICE))
            SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=GENERATOR)
            
            train_loader = DataLoader(dataset = _train, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True, sampler=SAMPLER_TRAIN, generator=GENERATOR)
            val_loader = DataLoader(dataset = _val, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True, generator=GENERATOR)


            for epoch in range(EPOCHS):
                start_time = time.time()
                LOGGER.log("\t" + "-"*100)
                LOGGER.log("\t" + f"FOLD: [{fold+1}/{K_FOLD}]")
                LOGGER.log("\t" + f"EPOCH: [{epoch+1}/{EPOCHS}]" + "\t"*8 + f"PERSISTANCE: [{p_counter}/{PERSISTANCE}]")
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
                    error = CRITERION_TRAIN(y_pred, labels)
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
                        val_loss += CRITERION_VAL(y_pred, labels).item()
                val_loss /= len(val_loader)
                validation_losses.append(val_loss)
                LOGGER.log("\t" +"\tWeighted Val Loss: [%0.5f]" % (val_loss))

                # Save Best Model
                p_counter += 1
                if(val_loss < min_val_loss):
                    min_val_loss = val_loss
                    torch.save(model, PATH_MODEL_SAVE + MODEL_NAME  + ".pt")
                    p_counter = 1
                LOGGER.log("\t" +"\tMinimum Val Loss: [%0.5f]" % (min_val_loss))

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
                    if(is_decreasing_order(training_losses[-PERSISTANCE:])):
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
                end_time = time.time()
                logTime(start_time, end_time, logger=LOGGER)
                
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
        plot_losses(training_losses, validation_losses, PATH_MODEL_SAVE, LOGGER)

if __name__ == "__main__":
    
    # ====================================================================
    # ==================== CHANGE HERE ===================================
    FINE_TUNE = False
    MODEL_FOLDER_NAME = "SWIN_S_trial"
    MODEL_TYPE = MODEL_TYPE_DICT["trans"]
    OPEN_TILL_LAYER = ""
    # ====================================================================
    MODEL_NAME = MODEL_FOLDER_NAME if(not FINE_TUNE) else MODEL_FOLDER_NAME + "_FT"
    PATH_MODEL_SAVE = "./Classifiers/" + MODEL_TYPE + "/" + MODEL_FOLDER_NAME + "/"
    PATH_MODEL_LOG = PATH_MODEL_SAVE + "Logs/" 
    if(not os.path.exists(PATH_MODEL_LOG)): os.makedirs(PATH_MODEL_LOG)
    PATH_MODEL_LOG += MODEL_NAME + "_architecture_" + str(sum(1 for file_name in os.listdir(PATH_MODEL_SAVE) if "architecture" in file_name)) + ".txt"
    LOGGER = Logger(
        server_url = "" , #SERVER_URL,
        server_username= SERVER_USERNAME,
        server_folder = SERVER_FOLDER,
        model_name = MODEL_NAME,
        path_localFile = "" #PATH_MODEL_LOG
    )
    # ====================================================================
    
    LOGGER.log("\n\n" + "="*54 + " START " + "="*54)
    KF = KFold(n_splits=K_FOLD, shuffle=True, random_state=26)
    if(FINE_TUNE):
        if(not os.path.exists(PATH_MODEL_SAVE + MODEL_NAME + ".pt")):
            LOGGER.log("Loading Model: " + MODEL_NAME[:-3] + ".pt")
            model = torch.load(PATH_MODEL_SAVE + MODEL_NAME[:-3] + ".pt")
        else:
            src = PATH_MODEL_SAVE + MODEL_NAME + ".pt"
            dst = PATH_MODEL_SAVE + MODEL_NAME + "_previous.pt"
            LOGGER.log("Loading Model: " + MODEL_NAME + ".pt")
            model = torch.load(src)
            shutil.copy(src, dst)
    else:
        # =================================================================
        # ================= THE NEW MODEL TO TRAIN ========================
        
        model = TransNets.SWIN(model_size="s")
        # model = TransNets.CvT(model_size="s", freezeToLayer="blocks.9.mlp.fc2.bias")
        # model = TransNets.MaxViT()

        # =================================================================
        # =================================================================

    
    model.to(DEVICE)
    # OPTIMIZER = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,  weight_decay=1e-4)
    OPTIMIZER = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005, dampening=0, momentum=0.9, nesterov=True)
    LR_SCHEDULER = CosineAnnealingWarmRestarts(OPTIMIZER, T_0=10, T_mult=2)
    # LR_SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, mode='min',  factor=LRS_FACTOR, patience=LRS_PATIENCE)
    LOGGER.log(f"Using GPU: {DEVICE}")
    LOGGER.log(f"Batch Size: {BATCH_SIZE}")
    LOGGER.log(f"Learning Rate: {LEARNING_RATE}")
    LOGGER.log(f"Early Stopping with Persistence: {PERSISTANCE}")
    LOGGER.log(f"LR Schedular: {type(LR_SCHEDULER).__name__}")
    # Add if statement
    if isinstance(LR_SCHEDULER, ReduceLROnPlateau):
        LOGGER.log(f"|---Patience: {LRS_PATIENCE}")
        LOGGER.log(f"|---Factor: {LRS_FACTOR}")

    setup(model, FINE_TUNE, OPEN_TILL_LAYER)
    # exit()
    train_KCV()
    LOGGER.log("="*54 + " END " + "="*54 + "\n\n")
