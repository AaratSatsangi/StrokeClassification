import torch
import sys
import pandas as pd
from Classifiers_CNN import *
from torchvision import models
# from transformers import CvtConfig, CvtModel
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import json
import matplotlib.pyplot as plt
from preprocessor import CTPreprocessor

RANDOM_STATE = 26
K_FOLD = 5
AUTO_BREAK = True #Auto Stop training if overfitting is detected
PATH_DATASET = "./Data/Compiled/PNG/"
PATH_DATASET_TRAIN = "Data/Compiled/Split/Train"
PATH_DATASET_TEST = "Data/Compiled/Split/Test"
MODEL_NAME: str = None
MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
PATH_MODEL_SAVE:str = None
BATCH_SIZE = 128
BATCH_LOAD = 8
LEARNING_RATE = 1e-3
PERSISTANCE = 5
WORKERS = os.cpu_count()
EPOCHS = 50
IMG_SIZE = (1, 224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
GENERATOR = torch.manual_seed(RANDOM_STATE)

CLASS_NAMES = None
OPTIMIZER = None
LR_SCHEDULER = None

TRAIN_dataset: ImageFolder = None
TEST_dataset: ImageFolder = None
test_loader : DataLoader = None
KF : KFold = None
model: torch.nn.Module = None

def setup(model:nn.Module, model_name:str, model_type:str, fine_tine = False, freeze_upto = -1):
    global TRAIN_dataset, test_loader, KF, OPTIMIZER, CLASS_NAMES, MODEL_NAME, PATH_MODEL_SAVE, LR_SCHEDULER
    
    TRAIN_dataset = ImageFolder(PATH_DATASET_TRAIN, IMG_TRANSFORMS_TRAIN)
    # TRAIN_dataset, _ = random_split(dataset, [0.8, 0.2], generator=GENERATOR)
    KF = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    CLASS_NAMES = TRAIN_dataset.classes
    
    
    torch.cuda.empty_cache()

    # ================CHANGE HERE=====================
    if(not fine_tine):
        freeze_layers = freeze_upto if(freeze_upto > 0) else len(list(model.parameters())) // 2 #?????
        
        if("ViT" in model_name):
            # Freeeze starting Layers
            for layer in list(model.encoder.layers[:6]):  # Freezing the first half of the transformer layers (e.g., first 6 out of 12)
                for param in layer.parameters():
                    param.requires_grad = False

        elif ("RESNET" in model_name or "VGG" in model_name):
            # Calculate half of the layers
            print("Freezing half layers...")
            num_layers = len(list(model.parameters()))
            half_layers = num_layers // 2

            # Freeze the first half of the layers
            for i, param in enumerate(model.parameters()):
                if i < half_layers:
                    param.requires_grad = False
        elif("SWIN" in model_name):
            # Freeze the Patch Embedding Layer and the first stage (Stage 1)
            for param in model.features[0].parameters():  # Patch Embedding
                param.requires_grad = False
            # Unfreeze the modified first layer
            for param in model.features[0][0].parameters():
                param.requires_grad = True

            # model.features[0].parameters()[0].requires_grad = True

            for block in model.features[1:5]:
                for param in block.parameters():
                    param.requires_grad = False
            
            for param in model.features[5][:12].parameters(): 
                param.requires_grad = False
            for param in model.features[5][12:].parameters():
                param.requires_grad = True

            # for param in model.features[6][:5].parameters():  # Stage 1
            #     param.requires_grad = False

            # for param in model.features[2].parameters():  # Stage 2
            #     param.requires_grad = False
    else:
        if(model_type == MODEL_TYPE_DICT["trans"] and "ViT" in model_name):
            # UnFreeze all layers
            for layer in list(model.encoder.layers): 
                for param in layer.parameters():
                    param.requires_grad = True
        
        else:
            for layer in model.children():
                for param in layer.parameters():
                    param.requires_grad = True
    # ================CHANGE HERE=====================
    
    
    model.to(DEVICE)
    # Create the folder
    if not os.path.exists(PATH_MODEL_SAVE):
        os.makedirs(PATH_MODEL_SAVE)
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("architecture" in file_name): 
            count +=1
    with open(PATH_MODEL_SAVE + "architecture_"+ str(count) + ".txt", "w") as f:
        f.write(str(summary(model, (BATCH_SIZE,) + IMG_SIZE , col_names=["input_size","output_size","num_params"], verbose=0)))
    # print(summary(model, (BATCH_SIZE,) + IMG_SIZE , col_names=["input_size","output_size","num_params"]))
    OPTIMIZER = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,  weight_decay=1e-4)
    LR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER, T_0=10, T_mult=1, eta_min=1e-8)

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

    print(f"Plot saved as {PATH_MODEL_SAVE + "Plots/loss_plot.png"}")

def train_KCV():
    print("\n" + "#"*100 + "\n")
    print("\t\t\t\tTraining: " + MODEL_NAME)
    print("\n" + "#"*100)

    training_losses = []
    validation_losses = []
    p_counter = 1
    min_val_loss = 1000

    try: 
        # Loop Here
        for fold, (train_idx, val_idx) in enumerate(KF.split(TRAIN_dataset)):
            print("\t" + "="*100)
            print(f"\tFold {fold+1}/{K_FOLD}")
            print("\t" + "="*100)

            _train = Subset(TRAIN_dataset, train_idx)
            _val = Subset(TRAIN_dataset, val_idx)

            train_loader = DataLoader(dataset = _train, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True)
            val_loader = DataLoader(dataset = _val, batch_size = BATCH_LOAD, num_workers=WORKERS-1, pin_memory=True)
            for epoch in range(EPOCHS):
                print("\t" + "-"*100)
                print(("\t" + "FOLD: [%d/%d]") % (fold+1, K_FOLD))
                print(("\t" + "EPOCH: [%d/%d]" + "\t"*8 + "PERSISTANCE: [%d/%d]") % (epoch+1, EPOCHS, p_counter-1, PERSISTANCE))
                print("\t" +"-"*100)

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
                        print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),accum_loss/(count*BATCH_LOAD)), end = "\r") # Print avg batch loss instead of total accum loss
                        accum_loss = 0.0
                        count = 0
                    count += 1
                # avg epoch loss
                train_loss /= len(train_loader)
                training_losses.append(train_loss)
                print("\n\n\t" +"\tTraining Loss: [%0.5f]" % (training_losses[-1]))

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
                print("\t" +"\tValidation Loss: [%0.5f]" % (val_loss))
                LR_SCHEDULER.step()
                
                

                # Save Best Model
                p_counter += 1
                if(val_loss < min_val_loss):
                    min_val_loss = val_loss
                    torch.save(model, PATH_MODEL_SAVE + MODEL_NAME  + ".pt")
                    p_counter = 1
                print("\t" +"\tMin Validation Loss: [%0.5f]" % (min_val_loss), end = "\n\n")
                
                # Early Stopping for Overfitting Stopping
                if(p_counter-1 >= PERSISTANCE):
                    print("\t" + "Validation Loss Constant for %d Epochs at EPOCH %d" % (PERSISTANCE, epoch+1))
                    if(_isDecreasingOrder(training_losses[-PERSISTANCE:])):
                        print("\t" + "Stopping Training: Overfitting Detected at EPOCH", epoch)
                        # Break out of Training Loop
                        if(AUTO_BREAK): 
                            p_counter = 1
                            break
                    else:
                        print("\t" + "Training Loss Fluctuating -- " , training_losses[-PERSISTANCE:])
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
                            print("\t" + "Wrong Input!!\n")
                    if(flag == "n"):
                        break

    except KeyboardInterrupt:
        # Exit Loop code
        print("\t" + "Keyboard Interrupt: Exiting Loop...")

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
        print("Results Written in:", path_saveDict)

    print("\nResults:")
    print(json.dumps(report, indent=4))
    saveAsTable(path_saveDict)
    return

def saveAsTable(json_file_path: str):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Convert JSON data into a DataFrame
    df: pd.DataFrame
    df = pd.DataFrame(data).T  # Transpose to get categories as rows
    df = df.map(lambda x: round(x, 3) if isinstance(x, float) else x)
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
    print(f"Table image saved to {output_path}")

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

            y_true = torch.zeros(y.shape[0],3)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    print("\t" +"Test Loss:", round(test_loss,5))
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("performance" in file_name and ".json" in file_name): 
            count +=1
    path_perfom_save = PATH_MODEL_SAVE + "performance" + str(count) + ".json"
    _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=CLASS_NAMES, path_saveDict=path_perfom_save)
    return

            

if __name__ == "__main__":
    print("\n\n" + "="*50 + " START " + "="*50)
    # ============================================
    # ============================================
    
    FINE_TUNE = False
    RETRAIN = False
    MODEL_NAME = "SWIN_B_4"
    MODEL_TYPE = MODEL_TYPE_DICT["trans"]

    # ============================================
    # ============================================
    
    FT = "_FT" if (FINE_TUNE) else ""
    MODEL_NAME = MODEL_NAME + FT if("FT" not in MODEL_NAME) else MODEL_NAME
    PATH_MODEL_SAVE = "./Classifiers/" + MODEL_TYPE + "/" + MODEL_NAME + "/"
    
    # ============ CHANGE HERE===================
    # ============================================
    # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    

    if(RETRAIN):
        print("Loading Model: " + MODEL_NAME + ".pt")
        model = torch.load(PATH_MODEL_SAVE + MODEL_NAME + ".pt")
    else:
        model = models.swin_b(weights = models.Swin_B_Weights.IMAGENET1K_V1)
        model.features[0][0] = nn.Conv2d(
            in_channels=1,  # Change input channels to 1 for grayscale
            out_channels=model.features[0][0].out_channels,
            kernel_size=4,  # Swin default
            stride=4,       # Swin default
            padding=0       # Swin default
        )
        # model.conv_proj = nn.Conv2d(1, model.conv_proj.out_channels, kernel_size=16, stride=16)
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 3)
        )

        model = models.vit_b_16(weights= models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.conv_proj = nn.Conv2d(1, model.conv_proj.out_channels, kernel_size=16, stride=16)
        model.heads = nn.Linear(model.heads.in_features, 3)
    # ============================================
    # ============================================
    setup(model, MODEL_NAME, MODEL_TYPE, FINE_TUNE)
    train_KCV()
    print("Testing model: " + PATH_MODEL_SAVE + MODEL_NAME + ".pt")
    TEST_dataset = ImageFolder(PATH_DATASET_TEST, IMG_TRANSFORMS_TEST)
    # _, TEST_dataset = random_split(dataset, [0.8, 0.2], generator=GENERATOR)
    test_loader = DataLoader(dataset = TEST_dataset, batch_size = BATCH_LOAD, num_workers=WORKERS)
    testModel(torch.load(PATH_MODEL_SAVE + MODEL_NAME + ".pt"))
    print("\n\n" + "="*50 + " END " + "="*50)
    