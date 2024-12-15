import torch
import sys
from Classifiers_CNN import CNN_Classifier, CNN4T_Classifier
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import classification_report
import json
import matplotlib.pyplot as plt

AUTO_BREAK = False #Auto Stop training if overfitting is detected
PATH_DATASET = "./Data/Compiled/PNG/"
MODEL_NAME = "CNN_4T"
PATH_MODEL_SAVE = "./Classifiers/" + MODEL_NAME + "/"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PERSISTANCE = 5
WORKERS = 4
EPOCHS = 50
IMG_SIZE = (1, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE[1:]),
    transforms.Grayscale(),
    transforms.ToTensor()
])
LOSS = torch.nn.CrossEntropyLoss()
torch.manual_seed(26)

CLASS_NAMES = None
OPTIMIZER = None
train_loader : DataLoader = None 
val_loader : DataLoader =  None
test_loader : DataLoader = None
model: CNN_Classifier = None

def setup():
    global train_loader, val_loader, test_loader, model, OPTIMIZER, CLASS_NAMES
    
    dataset = ImageFolder(PATH_DATASET, IMG_TRANSFORMS)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2])
    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE)
    val_loader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE)
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE)
    CLASS_NAMES = dataset.classes

    torch.cuda.empty_cache()
    model = CNN4T_Classifier((BATCH_SIZE, ) + IMG_SIZE, MODEL_NAME)
    model.to(DEVICE)
    model.getSummary()
    OPTIMIZER = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
    plt.savefig(PATH_MODEL_SAVE + "Plots/loss_plot.png", bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved as {PATH_MODEL_SAVE + "Plots/loss_plot.png"}")


def train():
    print("\n" + "#"*80)
    print("\t\t\tTraining: " + model.name)
    print("\n" + "#"*80)

    
    if not os.path.exists(PATH_MODEL_SAVE):
        # Create the folder
        os.makedirs(PATH_MODEL_SAVE)

    training_losses = []
    validation_losses = []
    p_counter = 0
    min_val_loss = 1000

    try:
        for epoch in range(EPOCHS):
            print("\t" + "-"*100)
            print(("\t" + "EPOCH: [%d/%d]" + "\t"*8 + "p_counter: %d") % (epoch+1, EPOCHS, p_counter))
            print("\t" +"-"*100)

            train_loss = 0
            # Training
            model.train()
            for step, train_XY in enumerate(train_loader, 0):
                # Reset Gradient Values
                model.zero_grad()
                
                # Extract X and Y
                imgs = train_XY[0].to(DEVICE)
                labels = train_XY[1].to(DEVICE)

                # Predict labels 
                y_pred = model(imgs)

                # Calculate Error
                error = LOSS(y_pred, labels)
                error.backward()
                OPTIMIZER.step()
                train_loss += error.item()
                print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),error.item()), end = "\r")

            # avg epoch loss
            train_loss /= len(train_loader)
            training_losses.append(train_loss)
            print("\n\n\t" +"\tTraining Loss: %0.5f" % (training_losses[-1]))

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
            print("\t" +"\tValidation Loss: %0.5f" % (val_loss), end = "\n\n")
            
            # Save Best Model
            p_counter += 1
            if(val_loss < min_val_loss):
                min_val_loss = val_loss
                torch.save(model, PATH_MODEL_SAVE + MODEL_NAME  + ".pt")
                p_counter = 0

            # Early Stopping for Overfitting Stopping
            if(p_counter == PERSISTANCE):
                print("Validation Loss Constant for %0.5f Epochs at EPOCH %d" % (len(training_losses), epoch))
                if(_isDecreasingOrder(training_losses[-PERSISTANCE:])):
                    print("Overfitting Detected at EPOCH", epoch)
                    # Break out of Training Loop
                    if (AUTO_BREAK): break
                else:
                    print("Training Loss Fluctuating -- " , training_losses[-PERSISTANCE:])
                    # Unsure about Overfitting, ask the user to continue
                while(True):
                    flag = input("Keep Training? (y/n) : ")
                    if(flag == "y" or flag == "n"):
                        p_counter = 0
                        break
                    else:
                        print("Wrong Input!!\n")
                if(flag == "n"):
                    break

    except KeyboardInterrupt:
        print("Keyboard Interrupt: Exiting Training Loop...")

    np.savetxt(PATH_MODEL_SAVE + "LOSSES.txt", (training_losses, validation_losses), fmt="%0.5f" , delimiter=",")
    plot_losses(training_losses, validation_losses)
    with open(PATH_MODEL_SAVE + "architecture.txt", "w") as f:
        f.write(str(model.getSummary()))

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
    return


def testModel(t_model: CNN_Classifier):
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
    path_perfom_save = PATH_MODEL_SAVE + "/performance.json"
    _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=CLASS_NAMES, path_saveDict=path_perfom_save)
    return

            

if __name__ == "__main__":
    print("\n\n" + "="*50 + " START " + "="*50)
    setup()
    train()
    testModel(model)
    print("\n\n" + "="*50 + " END " + "="*50)
    